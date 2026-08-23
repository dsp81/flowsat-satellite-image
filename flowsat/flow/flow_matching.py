"""
Flow Matching for FlowSat.

References:
    - Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
    - Conditional Flow Matching: https://arxiv.org/abs/2302.00482
    - Stable Diffusion 3 (rectified flow): https://arxiv.org/abs/2403.03206

Implements:
    - Linear interpolation path:  x_t = (1 - t) * x_0 + t * noise
    - Velocity target:            v = noise - x_0
    - Optional logit-normal time sampling (SD3 style)
    - MSE velocity loss
"""

import math
from typing import Optional, Tuple, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class FlowMatchingSample(NamedTuple):
    """Container for a flow matching training sample."""
    x_t: Tensor       # (B, C, H, W) interpolated latent
    v_target: Tensor   # (B, C, H, W) velocity target
    t: Tensor          # (B,) timestep
    noise: Tensor      # (B, C, H, W) sampled noise


def sample_timesteps_uniform(batch_size: int, device: torch.device) -> Tensor:
    """Sample timesteps uniformly from [0, 1].
    
    Avoids exact 0 and 1 for numerical stability.
    """
    return torch.rand(batch_size, device=device).clamp(min=1e-5, max=1.0 - 1e-5)


def sample_timesteps_logit_normal(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tensor:
    """Sample timesteps from logit-normal distribution (SD3 style).
    
    This biases sampling towards intermediate timesteps where the model
    needs to learn the most, improving training efficiency.
    
    Args:
        batch_size: number of timesteps to sample.
        device: target device.
        mean: mean of the underlying normal distribution.
        std: standard deviation of the underlying normal distribution.
    """
    normal_samples = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(normal_samples)
    return t.clamp(min=1e-5, max=1.0 - 1e-5)


def linear_interpolation(
    x_0: Tensor,
    noise: Tensor,
    t: Tensor,
) -> Tensor:
    """Compute linear interpolation between data and noise.
    
    x_t = (1 - t) * x_0 + t * noise
    
    This defines the conditional probability path for flow matching.
    
    Args:
        x_0: (B, C, H, W) clean latent (data).
        noise: (B, C, H, W) Gaussian noise.
        t: (B,) timestep values in [0, 1].
    Returns:
        (B, C, H, W) interpolated latent.
    """
    t_expand = t[:, None, None, None]  # (B, 1, 1, 1)
    return (1.0 - t_expand) * x_0 + t_expand * noise


def compute_velocity_target(x_0: Tensor, noise: Tensor) -> Tensor:
    """Compute the velocity target for flow matching.
    
    v = noise - x_0
    
    The model learns to predict the velocity field that transports
    the data distribution to the noise distribution.
    
    Args:
        x_0: (B, C, H, W) clean latent.
        noise: (B, C, H, W) Gaussian noise.
    Returns:
        (B, C, H, W) velocity target.
    """
    return noise - x_0


def optimal_transport_interpolation(
    x_0: Tensor,
    noise: Tensor,
    t: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute OT-CFM interpolation and velocity target.
    
    Path:   x_t = (1 - (1-sigma_min)*t) * x_0 + t * noise
    Target: v = noise - (1 - sigma_min) * x_0
    
    With sigma_min → 0, this reduces to the standard linear path.
    
    Args:
        x_0: clean latent.
        noise: Gaussian noise.
        t: timestep.
    Returns:
        (x_t, v_target) tuple.
    """
    sigma_min = 1e-4
    t_expand = t[:, None, None, None]
    x_t = (1.0 - (1.0 - sigma_min) * t_expand) * x_0 + t_expand * noise
    v_target = noise - (1.0 - sigma_min) * x_0
    return x_t, v_target


# ---------------------------------------------------------------------------
# Flow Matching Loss
# ---------------------------------------------------------------------------

class FlowMatchingLoss(torch.nn.Module):
    """
    Computes the flow matching training loss.
    
    Given a batch of clean latents, samples noise and timesteps,
    computes interpolated latents and velocity targets, then
    returns the MSE loss between predicted and target velocities.
    
    Supports:
        - Uniform time sampling
        - Logit-normal time sampling (SD3-style, better convergence)
        - Standard linear interpolation
        - OT-CFM interpolation
        - Optional SNR weighting
    """

    def __init__(
        self,
        time_sampling: str = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        use_ot_cfm: bool = False,
        snr_weighting: bool = False,
    ):
        super().__init__()
        self.time_sampling = time_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.use_ot_cfm = use_ot_cfm
        self.snr_weighting = snr_weighting

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        if self.time_sampling == "logit_normal":
            return sample_timesteps_logit_normal(
                batch_size, device,
                mean=self.logit_normal_mean,
                std=self.logit_normal_std,
            )
        elif self.time_sampling == "uniform":
            return sample_timesteps_uniform(batch_size, device)
        else:
            raise ValueError(f"Unknown time sampling: {self.time_sampling}")

    def prepare_training_sample(
        self,
        x_0: Tensor,
        noise: Optional[Tensor] = None,
    ) -> FlowMatchingSample:
        """Prepare a flow matching training sample.
        
        Args:
            x_0: (B, C, H, W) clean latent from VAE.
            noise: optional pre-sampled noise (if None, will sample).
        Returns:
            FlowMatchingSample with x_t, v_target, t, noise.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        batch_size = x_0.shape[0]
        t = self.sample_timesteps(batch_size, x_0.device)

        if self.use_ot_cfm:
            x_t, v_target = optimal_transport_interpolation(x_0, noise, t)
        else:
            x_t = linear_interpolation(x_0, noise, t)
            v_target = compute_velocity_target(x_0, noise)

        return FlowMatchingSample(x_t=x_t, v_target=v_target, t=t, noise=noise)

    def forward(
        self,
        v_pred: Tensor,
        v_target: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute flow matching loss.
        
        Args:
            v_pred: (B, C, H, W) predicted velocity from model.
            v_target: (B, C, H, W) target velocity.
            t: (B,) timesteps (used for optional SNR weighting).
        Returns:
            scalar loss.
        """
        if self.snr_weighting and t is not None:
            # Weight loss by inverse SNR proxy: higher weight for harder timesteps
            # For linear path, SNR ∝ (1-t)^2 / t^2
            weight = 1.0 / (1.0 - t + 1e-5)
            weight = weight / weight.mean()  # normalize
            weight = weight[:, None, None, None]
            loss = (weight * (v_pred - v_target) ** 2).mean()
        else:
            loss = F.mse_loss(v_pred, v_target, reduction="mean")

        return loss
