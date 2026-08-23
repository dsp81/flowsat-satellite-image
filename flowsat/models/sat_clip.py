"""
SatCLIP-inspired Metadata Encoder for FlowSat.

References:
    - SatCLIP: https://arxiv.org/abs/2311.17179
    - DiffusionSat metadata embedding (data_util.py)

Encodes satellite metadata (lat, lon, gsd, timestamp, cloud cover, etc.)
into a dense vector for conditioning the SatDiT backbone.

Metadata fields (7 values, normalized):
    [0] longitude (+ base_lon offset)
    [1] latitude  (+ base_lat offset)
    [2] gsd (ground sampling distance)
    [3] cloud_cover
    [4] year
    [5] month
    [6] day
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional Encoding Components (inspired by SatCLIP geo_embedding)
# ---------------------------------------------------------------------------

class SphericalHarmonicEncoding(nn.Module):
    """
    Encodes geographic coordinates (lon, lat) using multi-frequency sinusoidal
    features, inspired by SatCLIP's spherical harmonics approach.

    Converts (lon, lat) → 3D unit sphere → multi-frequency sinusoidal encoding.
    """

    def __init__(self, num_frequencies: int = 32):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Output dim: 3 (xyz) * 2 (sin, cos) * num_frequencies + 3 (raw xyz)
        self.out_dim = 3 + 3 * 2 * num_frequencies

    def forward(self, lon: Tensor, lat: Tensor) -> Tensor:
        """
        Args:
            lon: (B,) longitude in degrees, normalized.
            lat: (B,) latitude in degrees, normalized.
        Returns:
            (B, out_dim) spherical harmonic features.
        """
        # Convert to radians (denormalize from [0, scale] to degrees first)
        lon_rad = lon * math.pi / 180.0
        lat_rad = lat * math.pi / 180.0

        # Convert to 3D unit sphere
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)

        xyz = torch.stack([x, y, z], dim=-1)  # (B, 3)

        # Multi-frequency encoding
        freqs = 2.0 ** torch.arange(
            self.num_frequencies, dtype=torch.float32, device=xyz.device
        )  # (F,)

        # (B, 3, 1) * (1, 1, F) → (B, 3, F)
        scaled = xyz.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        sin_feat = torch.sin(scaled).flatten(-2)  # (B, 3*F)
        cos_feat = torch.cos(scaled).flatten(-2)  # (B, 3*F)

        return torch.cat([xyz, sin_feat, cos_feat], dim=-1)


class TemporalEncoding(nn.Module):
    """
    Encodes temporal information (year, month, day) using cyclical features
    for month/day and linear features for year.
    """

    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies
        # year: num_freq * 2, month: 2 (sin/cos cycle) + num_freq*2, day: same
        # Simplified: 3 raw + 3 * 2 * num_freq
        self.out_dim = 3 + 3 * 2 * num_frequencies

    def forward(self, year: Tensor, month: Tensor, day: Tensor) -> Tensor:
        """
        Args:
            year: (B,) normalized year.
            month: (B,) normalized month.
            day: (B,) normalized day.
        Returns:
            (B, out_dim) temporal features.
        """
        raw = torch.stack([year, month, day], dim=-1)  # (B, 3)

        freqs = 2.0 ** torch.arange(
            self.num_frequencies, dtype=torch.float32, device=raw.device
        )

        scaled = raw.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        sin_feat = torch.sin(scaled).flatten(-2)
        cos_feat = torch.cos(scaled).flatten(-2)

        return torch.cat([raw, sin_feat, cos_feat], dim=-1)


class ScalarEncoding(nn.Module):
    """Encodes a single scalar value using sinusoidal frequencies."""

    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.out_dim = 1 + 2 * num_frequencies

    def forward(self, val: Tensor) -> Tensor:
        """
        Args:
            val: (B,) scalar values.
        Returns:
            (B, out_dim) encoded features.
        """
        freqs = 2.0 ** torch.arange(
            self.num_frequencies, dtype=torch.float32, device=val.device
        )
        scaled = val.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, F)
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        return torch.cat([val.unsqueeze(-1), sin_feat, cos_feat], dim=-1)


# ---------------------------------------------------------------------------
# SatCLIP Metadata Encoder
# ---------------------------------------------------------------------------

class SatCLIPMetadataEncoder(nn.Module):
    """
    Full metadata encoder that combines geographic, temporal, and sensor
    metadata into a single conditioning vector.

    Input: (B, 7) normalized metadata
        [lon, lat, gsd, cloud_cover, year, month, day]

    Output: (B, embed_dim) conditioning vector
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        geo_frequencies: int = 32,
        temporal_frequencies: int = 16,
        scalar_frequencies: int = 16,
        hidden_dim: int = 512,
        num_metadata: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_metadata = num_metadata

        # Component encoders
        self.geo_encoder = SphericalHarmonicEncoding(geo_frequencies)
        self.temporal_encoder = TemporalEncoding(temporal_frequencies)
        self.gsd_encoder = ScalarEncoding(scalar_frequencies)
        self.cloud_encoder = ScalarEncoding(scalar_frequencies)

        # Compute total input dimension
        total_input_dim = (
            self.geo_encoder.out_dim
            + self.temporal_encoder.out_dim
            + self.gsd_encoder.out_dim
            + self.cloud_encoder.out_dim
        )

        # Projection MLP
        self.projection = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, metadata: Tensor) -> Tensor:
        """
        Args:
            metadata: (B, 7) normalized metadata tensor.
                [lon, lat, gsd, cloud_cover, year, month, day]
        Returns:
            (B, embed_dim) metadata conditioning vector.
        """
        lon = metadata[:, 0]
        lat = metadata[:, 1]
        gsd = metadata[:, 2]
        cloud_cover = metadata[:, 3]
        year = metadata[:, 4]
        month = metadata[:, 5]
        day = metadata[:, 6]

        geo_feat = self.geo_encoder(lon, lat)
        temporal_feat = self.temporal_encoder(year, month, day)
        gsd_feat = self.gsd_encoder(gsd)
        cloud_feat = self.cloud_encoder(cloud_cover)

        combined = torch.cat([geo_feat, temporal_feat, gsd_feat, cloud_feat], dim=-1)
        combined = combined.to(dtype=next(self.projection.parameters()).dtype)
        return self.projection(combined)


class SimpleMetadataEncoder(nn.Module):
    """
    Simpler metadata encoder matching DiffusionSat's approach more closely.
    Each metadata field gets its own sinusoidal projection + MLP, then they
    are summed together.

    This provides backward compatibility with DiffusionSat's metadata format.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_metadata: int = 7,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_metadata = num_metadata

        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(frequency_embedding_size, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(num_metadata)
        ])
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def sinusoidal_embedding(val: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=val.device) / half
        )
        args = val[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, metadata: Tensor) -> Tensor:
        """
        Args:
            metadata: (B, num_metadata) normalized metadata.
        Returns:
            (B, embed_dim) summed metadata embedding.
        """
        emb = torch.zeros(
            metadata.shape[0], self.embed_dim,
            device=metadata.device, dtype=metadata.dtype,
        )
        for i, embed_layer in enumerate(self.embeddings):
            freq_emb = self.sinusoidal_embedding(
                metadata[:, i], self.frequency_embedding_size
            )
            freq_emb = freq_emb.to(dtype=metadata.dtype)
            emb = emb + embed_layer(freq_emb)
        return emb
