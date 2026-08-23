"""
Flow Sampler for FlowSat inference.

References:
    - Flow Matching: https://arxiv.org/abs/2210.02747
    - facebook/flow_matching samplers

Implements ODE integration to go from noise z ~ N(0,I) at t=1
to clean latent x_0 at t=0, by integrating the learned velocity field:

    dx/dt = v_theta(x_t, t, conditions)
    x_0 = x_1 - ∫₁⁰ v_theta(x_t, t) dt

Supports:
    - Euler method (1st order)
    - Midpoint method (2nd order)
    - Configurable number of steps and schedules
"""

from typing import Callable, Optional, List
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from tqdm import tqdm


class FlowSampler(ABC):
    """Base class for flow-based ODE samplers."""

    def __init__(self, num_steps: int = 50):
        self.num_steps = num_steps

    def get_time_schedule(
        self,
        num_steps: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        schedule: str = "linear",
    ) -> Tensor:
        """Generate time schedule from t=1 (noise) to t=0 (data).
        
        Args:
            num_steps: number of integration steps.
            device: target device.
            schedule: "linear" or "cosine" spacing.
        Returns:
            (num_steps + 1,) tensor of timesteps from 1 to 0.
        """
        n = num_steps or self.num_steps

        if schedule == "linear":
            timesteps = torch.linspace(1.0, 0.0, n + 1, device=device)
        elif schedule == "cosine":
            # Cosine schedule puts more steps near t=0 where details emerge
            s = torch.linspace(0, 1, n + 1, device=device)
            timesteps = 1.0 - (1.0 - torch.cos(s * torch.pi / 2))
        elif schedule == "quadratic":
            # Quadratic schedule: more steps near t=0
            s = torch.linspace(0, 1, n + 1, device=device)
            timesteps = 1.0 - s ** 2
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        return timesteps

    @abstractmethod
    def sample(
        self,
        model_fn: Callable,
        z: Tensor,
        num_steps: Optional[int] = None,
        schedule: str = "linear",
        show_progress: bool = True,
        **model_kwargs,
    ) -> Tensor:
        """Run ODE integration from noise to data.
        
        Args:
            model_fn: callable that takes (x_t, t, **kwargs) and returns velocity.
            z: (B, C, H, W) initial noise at t=1.
            num_steps: override default number of steps.
            schedule: time schedule type.
            show_progress: show tqdm progress bar.
            **model_kwargs: additional args passed to model_fn.
        Returns:
            (B, C, H, W) denoised latent at t≈0.
        """
        ...


class EulerSampler(FlowSampler):
    """
    Euler method (1st order) ODE integrator.
    
    Update rule:
        x_{t-dt} = x_t - dt * v_theta(x_t, t)
    
    Simple and stable. Recommended starting point.
    """

    def __init__(self, num_steps: int = 50):
        super().__init__(num_steps)

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable,
        z: Tensor,
        num_steps: Optional[int] = None,
        schedule: str = "linear",
        show_progress: bool = True,
        **model_kwargs,
    ) -> Tensor:
        timesteps = self.get_time_schedule(num_steps, z.device, schedule)
        x_t = z.clone()

        iterator = range(len(timesteps) - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="FlowSat Sampling (Euler)")

        for i in iterator:
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr  # negative (going from 1 → 0)

            # Expand t to batch dimension
            t_batch = t_curr.expand(z.shape[0])

            # Predict velocity
            v_pred = model_fn(x_t, t_batch, **model_kwargs)

            # Euler step: x_{t+dt} = x_t + dt * v(x_t, t)
            x_t = x_t + dt * v_pred

        return x_t


class MidpointSampler(FlowSampler):
    """
    Midpoint method (2nd order) ODE integrator.
    
    Update rule:
        x_mid = x_t - (dt/2) * v_theta(x_t, t)
        x_{t-dt} = x_t - dt * v_theta(x_mid, t - dt/2)
    
    Better accuracy than Euler at the cost of 2x model evaluations per step.
    Use fewer steps (e.g., 25) compared to Euler (50) for similar quality.
    """

    def __init__(self, num_steps: int = 25):
        super().__init__(num_steps)

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable,
        z: Tensor,
        num_steps: Optional[int] = None,
        schedule: str = "linear",
        show_progress: bool = True,
        **model_kwargs,
    ) -> Tensor:
        timesteps = self.get_time_schedule(num_steps, z.device, schedule)
        x_t = z.clone()

        iterator = range(len(timesteps) - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="FlowSat Sampling (Midpoint)")

        for i in iterator:
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr
            t_mid = t_curr + dt / 2.0

            t_batch_curr = t_curr.expand(z.shape[0])
            t_batch_mid = t_mid.expand(z.shape[0])

            # First evaluation at current point
            v1 = model_fn(x_t, t_batch_curr, **model_kwargs)

            # Midpoint evaluation
            x_mid = x_t + (dt / 2.0) * v1
            v2 = model_fn(x_mid, t_batch_mid, **model_kwargs)

            # Full step using midpoint velocity
            x_t = x_t + dt * v2

        return x_t


class HeunSampler(FlowSampler):
    """
    Heun's method (improved Euler / trapezoidal, 2nd order).
    
    Update rule:
        v1 = v_theta(x_t, t)
        x_pred = x_t + dt * v1
        v2 = v_theta(x_pred, t + dt)
        x_{t+dt} = x_t + dt * (v1 + v2) / 2
    
    Generally produces the best quality per NFE among 2nd-order methods.
    """

    def __init__(self, num_steps: int = 25):
        super().__init__(num_steps)

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable,
        z: Tensor,
        num_steps: Optional[int] = None,
        schedule: str = "linear",
        show_progress: bool = True,
        **model_kwargs,
    ) -> Tensor:
        timesteps = self.get_time_schedule(num_steps, z.device, schedule)
        x_t = z.clone()

        iterator = range(len(timesteps) - 1)
        if show_progress:
            iterator = tqdm(iterator, desc="FlowSat Sampling (Heun)")

        for i in iterator:
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            t_batch_curr = t_curr.expand(z.shape[0])
            t_batch_next = t_next.expand(z.shape[0])

            # First velocity evaluation
            v1 = model_fn(x_t, t_batch_curr, **model_kwargs)

            # Predicted next state
            x_pred = x_t + dt * v1

            # Second velocity evaluation at predicted state (skip at last step)
            if i < len(timesteps) - 2:
                v2 = model_fn(x_pred, t_batch_next, **model_kwargs)
                # Trapezoidal average
                x_t = x_t + dt * (v1 + v2) / 2.0
            else:
                x_t = x_pred

        return x_t


class CFGWrapper:
    """
    Classifier-Free Guidance wrapper for flow matching.
    
    Implements: v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
    
    Usage:
        cfg_model = CFGWrapper(model, guidance_scale=7.5)
        sampler.sample(cfg_model, z, encoder_hidden_states=prompt_embeds)
    """

    def __init__(
        self,
        model: Callable,
        guidance_scale: float = 7.5,
    ):
        self.model = model
        self.guidance_scale = guidance_scale

    def __call__(
        self,
        x_t: Tensor,
        t: Tensor,
        encoder_hidden_states: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            x_t: (B, C, H, W) current latent (NOT doubled).
            t: (B,) timestep.
            encoder_hidden_states: (2*B, S, D) concatenated [uncond, cond] embeddings.
            **kwargs: other model inputs (metadata etc.), already doubled if needed.
        """
        # Double the latent and timestep for CFG
        x_t_double = torch.cat([x_t, x_t], dim=0)
        t_double = torch.cat([t, t], dim=0)

        # Forward pass (both unconditional and conditional)
        v_pred = self.model(
            x_t_double, t_double,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        if hasattr(v_pred, "sample"):
            v_pred = v_pred.sample

        # Split and apply guidance
        v_uncond, v_cond = v_pred.chunk(2, dim=0)
        v_guided = v_uncond + self.guidance_scale * (v_cond - v_uncond)

        return v_guided
