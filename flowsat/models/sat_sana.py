"""
SatSana: Sana-0.6B-based architecture for FlowSat.

Why Sana?
  - Native flow matching pretrained (FlowMatchEulerDiscreteScheduler)
  - QK-norm built into attention (no stability issues like PixArt-Sigma)
  - AdaLN-single conditioning identical to PixArt — our metadata graft approach
    transfers directly with minimal change
  - 0.6B params: fits on A5000 (24GB) with batch=2-4
  - Linear attention in self-attn, vanilla cross-attn with QK-norm
  - Smaller text encoder (Gemma-2-2B, ~5GB vs T5-XXL's ~11GB)

Architecture (from Sana_600M_512px_diffusers):
  - in_channels:        32 (DC-AE compressed latents, NOT SD VAE's 4!)
  - patch_size:         1 (vs PixArt's 2 — because DC-AE already compresses 32x)
  - hidden_size:        1152
  - num_layers:         28
  - num_attention_heads: ~16 (head_dim=72)
  - cross_attention_dim: 2304 (Gemma-2-2B output dim)
  - out_channels:       32 (matches in_channels for direct velocity output)
  - AdaLN-single:       same as PixArt — linear(1152) -> 6912 (6*1152)

DC-AE VAE specifics:
  - 32x spatial compression (vs SD VAE's 8x)
  - For 512px image: latent is 16x16x32 (= 8192 elements)
  - For 1024px image: latent is 32x32x32 (= 32768 elements)
  - The transformer sees 16x16 = 256 patch tokens for 512px (with patch_size=1)

Metadata conditioning: SAME approach as SatPixArt
  - PerFieldMetadataEmbed or external (SatCLIP) -> (B, 1152) embedding
  - metadata_mod: zero-init Linear(1152, 6*1152) parallel to time_embed.linear
  - MetadataModulatedLinear wrapper adds md_contrib to AdaLN output

This file is structurally a port of sat_pixart.py with three differences:
  1. SanaTransformer2DModel instead of Transformer2DModel
  2. Different config defaults (in_channels=32, patch_size=1)
  3. Different attribute name for time embedding (time_embed vs adaln_single)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SatDiTOutput:
    sample: Tensor


# --------------------------------------------------------------------------
# Per-field sinusoidal -> MLP metadata embedder (fallback)
# --------------------------------------------------------------------------

def _sinusoidal(x: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=x.device) / half
    )
    args = x.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PerFieldMetadataEmbed(nn.Module):
    def __init__(self, num_metadata: int, embed_dim: int,
                 frequency_embedding_size: int = 256):
        super().__init__()
        self.num_metadata = num_metadata
        self.embed_dim = embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(frequency_embedding_size, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(num_metadata)
        ])

    def forward(self, metadata: Tensor) -> Tensor:
        out = None
        for i, layer in enumerate(self.layers):
            field = metadata[:, i]
            freq = _sinusoidal(field, self.frequency_embedding_size).to(
                dtype=next(self.parameters()).dtype)
            emb = layer(freq)
            out = emb if out is None else out + emb
        return out


# --------------------------------------------------------------------------
# SatSana
# --------------------------------------------------------------------------

class SatSana(nn.Module):
    """Sana-0.6B fine-tuned for satellite flow matching with metadata."""

    def __init__(
        self,
        latent_size: int = 16,   # 512px / 32 = 16 for DC-AE
        in_channels: int = 32,    # DC-AE latent channels
        num_metadata: int = 7,
        use_metadata: bool = True,
        cross_attention_dim: int = 2304,  # Gemma-2-2B
        pretrained_sana_id: Optional[str] = "Efficient-Large-Model/Sana_600M_512px_diffusers",
        load_pretrained: bool = True,
    ):
        super().__init__()

        self.config = {
            "latent_size": latent_size,
            "in_channels": in_channels,
            "num_metadata": num_metadata,
            "use_metadata": use_metadata,
            "cross_attention_dim": cross_attention_dim,
            "pretrained_sana_id": pretrained_sana_id,
            "load_pretrained": False,
        }

        self.latent_size = latent_size
        self.in_channels = in_channels
        self.num_metadata = num_metadata
        self.use_metadata = use_metadata
        self.cross_attention_dim = cross_attention_dim
        self.out_channels = in_channels
        self.gradient_checkpointing = False

        # ---------------------------------------------------------------
        # Build SanaTransformer2DModel
        # ---------------------------------------------------------------
        from diffusers import SanaTransformer2DModel

        if load_pretrained and pretrained_sana_id is not None:
            self.transformer = SanaTransformer2DModel.from_pretrained(
                pretrained_sana_id,
                subfolder="transformer",
                use_safetensors=True,
            )
        else:
            # Manual config (defaults match Sana_600M_512px_diffusers)
            self.transformer = SanaTransformer2DModel(
                in_channels=in_channels,
                out_channels=in_channels,
                num_attention_heads=16,
                attention_head_dim=72,
                num_layers=28,
                num_cross_attention_heads=16,
                cross_attention_head_dim=72,
                # diffusers>=0.38: captions are projected to inner_dim before attn2,
                # so cross_attention_dim must equal inner_dim (16*72=1152); the raw
                # text-encoder width goes in caption_channels.
                cross_attention_dim=16 * 72,
                caption_channels=cross_attention_dim,
                mlp_ratio=2.5,
                dropout=0.0,
                attention_bias=False,
                sample_size=latent_size,
                patch_size=1,
                norm_elementwise_affine=False,
                norm_eps=1e-6,
            )

        # Sana already outputs the right number of channels for flow matching
        # velocity (in_channels == out_channels). No proj_out surgery needed!
        # The pretrained model was trained for epsilon prediction but the
        # FlowMatchEulerDiscreteScheduler reinterprets the output as velocity.

        hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.embed_dim = hidden_size

        # ---------------------------------------------------------------
        # Metadata embedder + external-encoder hook
        # ---------------------------------------------------------------
        if use_metadata and num_metadata > 0:
            self.metadata_embed = PerFieldMetadataEmbed(num_metadata, hidden_size)
        else:
            self.metadata_embed = None

        self.external_metadata_encoder = None

        # ---------------------------------------------------------------
        # Metadata modulation: zero-init parallel path to time_embed.linear
        # ---------------------------------------------------------------
        if use_metadata and num_metadata > 0:
            self.metadata_mod = nn.Linear(hidden_size, 6 * hidden_size, bias=False)
            nn.init.zeros_(self.metadata_mod.weight)
        else:
            self.metadata_mod = None

        # Install the wrapper around time_embed.linear so metadata_mod's
        # contribution gets added to the modulation scalars at runtime.
        self._install_metadata_hook()

    # ------------------------------------------------------------------
    # Metadata hook into Sana's AdaLayerNormSingle (called time_embed in Sana)
    # ------------------------------------------------------------------
    def _install_metadata_hook(self):
        """Wrap time_embed.linear with a module that adds metadata_mod(md_emb)
        to its output. Mathematically:
            modulation = W_t @ silu(t_emb) + b_t  +  W_md @ md_emb
        where W_md is zero-init. At step 0, metadata contribution is exactly 0
        and the model is byte-identical to vanilla Sana.

        Sana's AdaLN module is at self.transformer.time_embed (NOT adaln_single
        like PixArt). Same structure though: it has .linear that produces
        6*hidden_size modulation scalars.
        """
        self._current_metadata_emb = None
        parent = self  # capture SatSana instance for closure

        class MetadataModulatedLinear(nn.Module):
            def __init__(self, original_linear: nn.Linear):
                super().__init__()
                self.linear = original_linear
                self.in_features = original_linear.in_features
                self.out_features = original_linear.out_features

            def forward(self, h: torch.Tensor) -> torch.Tensor:
                out = self.linear(h)
                md = parent._current_metadata_emb
                if md is not None and parent.metadata_mod is not None:
                    md_normed = md / (md.norm(dim=-1, keepdim=True) + 1e-6) * md.shape[-1] ** 0.5  # ~RMS scaling
                    md_contrib = parent.metadata_mod(md_normed.to(dtype=out.dtype))
                    out = out + md_contrib
                return out

        # Find the right AdaLN module. Sana calls it time_embed.
        if hasattr(self.transformer, 'time_embed'):
            ada = self.transformer.time_embed
        elif hasattr(self.transformer, 'adaln_single'):
            ada = self.transformer.adaln_single
        else:
            raise RuntimeError(
                "Could not find time_embed or adaln_single in SanaTransformer2DModel. "
                "Inspect with: [n for n,_ in model.transformer.named_children()]"
            )

        if not hasattr(ada, 'linear'):
            raise RuntimeError(
                f"AdaLN module ({type(ada).__name__}) has no .linear attribute. "
                "Sana's diffusers implementation may have changed."
            )

        ada.linear = MetadataModulatedLinear(ada.linear)
        self._ada_module_name = 'time_embed' if hasattr(self.transformer, 'time_embed') else 'adaln_single'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_metadata_encoder(self, encoder: nn.Module):
        self.external_metadata_encoder = encoder

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        encoder_hidden_states: Tensor,
        metadata: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> SatDiTOutput:
        if self.use_metadata and metadata is not None:
            if self.external_metadata_encoder is not None:
                md_emb = self.external_metadata_encoder(metadata)
            elif self.metadata_embed is not None:
                md_emb = self.metadata_embed(metadata)
            else:
                md_emb = None
        else:
            md_emb = None
        self._current_metadata_emb = md_emb

        # Sana was trained on continuous t in [0, 1] (flow matching native).
        # NO need to scale to 1000 like we did for PixArt's DDPM features.
        # (If validation outputs are garbage, try t * 1000 here.)

        try:
            # Sana's forward signature uses 'encoder_attention_mask'
            result = self.transformer(
                hidden_states=x,
                encoder_hidden_states=encoder_hidden_states,
                timestep=t,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )
        finally:
            self._current_metadata_emb = None

        v = result.sample
        if return_dict:
            return SatDiTOutput(sample=v)
        return (v,)


def SatSana_600M(**kwargs):
    return SatSana(**kwargs)


SATSANA_MODELS = {
    "SatSana-600M": SatSana_600M,
}