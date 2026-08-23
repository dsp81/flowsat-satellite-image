"""
FlowSat Inference Pipeline.

References:
    - DiffusionSat pipeline.py
    - flow_sampler.py

Provides a clean interface for text-to-satellite-image generation:
    1. Encode text prompt via CLIP
    2. Initialize latent noise z ~ N(0,I)
    3. Integrate velocity field via ODE solver (Euler/Midpoint/Heun)
    4. Decode latent to pixel space via VAE

Usage:
    model = SatSana(...)                     # weights already loaded
    pipeline = FlowSatPipeline.from_pretrained(model, hf_model_id)
    pipeline = pipeline.to("cuda")
    images = pipeline(
        prompt="a satellite image of an airport",
        metadata=[360.0, 130.0, 0.5, 0.0, 30.0, 6.0, 15.0],
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from PIL import Image
from tqdm import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowsat.models.sat_sana import SatSana
from flowsat.flow.flow_sampler import EulerSampler, MidpointSampler, HeunSampler, CFGWrapper
from flowsat.data.sat_data_util import metadata_normalize


class FlowSatOutput:
    """Output container for FlowSat pipeline."""

    def __init__(self, images: List[Image.Image], latents: Optional[Tensor] = None):
        self.images = images
        self.latents = latents


class FlowSatPipeline:
    """
    Text-to-satellite-image generation pipeline using flow matching.
    
    Components:
        - CLIP text encoder + tokenizer
        - SatSana transformer (velocity field)
        - Stable Diffusion VAE (latent ↔ pixel)
        - Flow ODE sampler (Euler / Midpoint / Heun)
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        model: SatSana,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = model
        self._device = device
        self._dtype = dtype

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @classmethod
    def from_pretrained(
        cls,
        model: SatSana,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> "FlowSatPipeline":
        """Assemble a pipeline around an already-constructed model.

        Args:
            model: the velocity-field transformer, with its weights loaded.
            pretrained_model_name_or_path: HF model ID supplying the VAE,
                text encoder and tokenizer.
            torch_dtype: computation dtype.
            device: target device.
        """
        hf_path = pretrained_model_name_or_path

        # Load VAE and text encoder from HuggingFace
        tokenizer = CLIPTokenizer.from_pretrained(hf_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(hf_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(hf_path, subfolder="vae")

        # Move to device
        device = torch.device(device)
        vae = vae.to(device, dtype=torch_dtype)
        text_encoder = text_encoder.to(device, dtype=torch_dtype)
        model = model.to(device, dtype=torch_dtype)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        model.eval()

        return cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            model=model,
            device=device,
            dtype=torch_dtype,
        )

    def to(self, device: Union[str, torch.device]) -> "FlowSatPipeline":
        """Move pipeline to device."""
        device = torch.device(device)
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.model = self.model.to(device)
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tensor:
        """Encode text prompt to CLIP embeddings.
        
        If CFG enabled, returns concatenated [uncond, cond] embeddings.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        # Encode
        prompt_embeds = self.text_encoder(text_input_ids)[0]  # (B, S, D)

        # Repeat for multiple images per prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        # Unconditional embeddings for CFG
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeds = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            if num_images_per_prompt > 1:
                uncond_embeds = uncond_embeds.repeat_interleave(num_images_per_prompt, dim=0)

            # Concatenate: [uncond, cond]
            prompt_embeds = torch.cat([uncond_embeds, prompt_embeds], dim=0)

        return prompt_embeds

    def _prepare_metadata(
        self,
        metadata: Optional[List[float]],
        batch_size: int,
        do_classifier_free_guidance: bool,
    ) -> Optional[Tensor]:
        """Prepare metadata tensor for model input."""
        if metadata is None:
            return None

        md = torch.tensor(metadata, dtype=self._dtype, device=self.device)
        if len(md.shape) == 1:
            md = md.unsqueeze(0).expand(batch_size, -1)

        if do_classifier_free_guidance:
            # Concatenate: [zeros (uncond), metadata (cond)]
            md = torch.cat([torch.zeros_like(md), md], dim=0)

        return md

    def _decode_latents(self, latents: Tensor) -> np.ndarray:
        """Decode latents to images via VAE."""
        latents = latents / self.vae.config.scaling_factor

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[Tensor] = None,
        metadata: Optional[List[float]] = None,
        output_type: str = "pil",
        sampler: str = "euler",
        schedule: str = "linear",
        show_progress: bool = True,
    ) -> FlowSatOutput:
        """
        Generate satellite images from text prompts.
        
        Args:
            prompt: text prompt(s).
            height, width: output image dimensions.
            num_inference_steps: number of ODE integration steps.
            guidance_scale: CFG scale (1.0 = no guidance).
            negative_prompt: negative prompt(s) for CFG.
            num_images_per_prompt: number of images per prompt.
            generator: random generator for reproducibility.
            latents: optional pre-sampled latent noise.
            metadata: normalized metadata vector [lon, lat, gsd, cc, year, month, day].
            output_type: "pil" or "latent" or "np".
            sampler: "euler", "midpoint", or "heun".
            schedule: time schedule ("linear", "cosine", "quadratic").
            show_progress: show progress bar.
        Returns:
            FlowSatOutput with generated images.
        """
        # Defaults
        height = height or self.model.latent_size * self.vae_scale_factor
        width = width or self.model.latent_size * self.vae_scale_factor

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        do_cfg = guidance_scale > 1.0

        # 1. Encode prompt
        prompt_embeds = self._encode_prompt(
            prompt, num_images_per_prompt, do_cfg, negative_prompt
        )

        # 2. Prepare metadata
        md_tensor = self._prepare_metadata(
            metadata,
            batch_size * num_images_per_prompt,
            do_cfg,
        )

        # 3. Initialize latent noise at t=1
        num_channels = self.model.in_channels
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor

        if latents is None:
            shape = (batch_size * num_images_per_prompt, num_channels, latent_h, latent_w)
            latents = torch.randn(shape, generator=generator, device=self.device, dtype=self._dtype)
        else:
            latents = latents.to(self.device)

        # 4. Create sampler
        if sampler == "euler":
            flow_sampler = EulerSampler(num_steps=num_inference_steps)
        elif sampler == "midpoint":
            flow_sampler = MidpointSampler(num_steps=num_inference_steps)
        elif sampler == "heun":
            flow_sampler = HeunSampler(num_steps=num_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        # 5. Define model function (with optional CFG)
        if do_cfg:
            def model_fn(x_t, t, encoder_hidden_states, metadata=None, **kwargs):
                # Double inputs for CFG
                x_double = torch.cat([x_t, x_t], dim=0)
                t_double = torch.cat([t, t], dim=0)

                out = self.model(
                    x_double, t_double,
                    encoder_hidden_states=encoder_hidden_states,
                    metadata=metadata,
                ).sample

                v_uncond, v_cond = out.chunk(2, dim=0)
                return v_uncond + guidance_scale * (v_cond - v_uncond)

            model_kwargs = {
                "encoder_hidden_states": prompt_embeds,
                "metadata": md_tensor,
            }
        else:
            def model_fn(x_t, t, encoder_hidden_states, metadata=None, **kwargs):
                return self.model(
                    x_t, t,
                    encoder_hidden_states=encoder_hidden_states,
                    metadata=metadata,
                ).sample

            model_kwargs = {
                "encoder_hidden_states": prompt_embeds,
                "metadata": md_tensor,
            }

        # 6. Run ODE integration (t=1 → t=0)
        latents = flow_sampler.sample(
            model_fn=model_fn,
            z=latents,
            num_steps=num_inference_steps,
            schedule=schedule,
            show_progress=show_progress,
            **model_kwargs,
        )

        # 7. Decode
        if output_type == "latent":
            return FlowSatOutput(images=None, latents=latents)

        images_np = self._decode_latents(latents)

        if output_type == "pil":
            images_pil = [
                Image.fromarray((img * 255).astype(np.uint8))
                for img in images_np
            ]
            return FlowSatOutput(images=images_pil, latents=latents)
        else:
            return FlowSatOutput(images=images_np, latents=latents)
