"""
FMoW Dataset Loader for FlowSat.

Reads the standard licensed FMoW dataset directory format:

    fmow/
      train/
        airport/
          airport_0/
            airport_0_0_rgb.tif
            airport_0_0_rgb.json
            airport_0_1_rgb.tif
            airport_0_1_rgb.json
          airport_1/
            ...
        amusement_park/
          ...

Each sample consists of:
    - *_rgb.tif: RGB satellite image (GeoTIFF)
    - *_rgb.json: metadata JSON with fields:
        bounding_box  (WKT POLYGON string)
        raw_location  ([lat, lon])
        timestamp     (ISO format string)
        gsd           (float, meters per pixel)
        cloud_cover   (float, percentage 0-100)
        country_code  (2 or 3 letter code)
        img_filename  (original filename)

Optionally, a separate caption directory mirroring the data structure provides
rich VLM-generated captions (one .txt per image). If `caption_dir` is set, the
dataset reads captions from disk instead of using procedural template captions.

    {caption_dir}/airport/airport_0/airport_0_0_rgb.txt

Coordinate extraction priority:
    1. Optional CSV file (fmow-train-meta.csv) via category+location_id+image_id lookup
    2. bounding_box WKT polygon centroid from JSON
    3. raw_location field from JSON ([lat, lon])
    4. Falls back to (0, 0)

The dataset returns batches of:
    {
        "pixel_values": (B, 3, H, W) normalized images,
        "input_ids": (B, max_len) tokenized captions,
        "metadata": (B, 7) normalized metadata vectors,
    }
"""

import os
import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .sat_data_util import (
    FMOW_CATEGORIES,
    CATEGORY_TO_IDX,
    generate_fmow_caption,
    generate_fmow_caption_with_metadata,
    generate_short_caption,
    extract_fmow_metadata,
    metadata_normalize,
    percentile_normalization,
)

logger = logging.getLogger(__name__)


def _extract_month(metadata_dict: Dict[str, Any]) -> int:
    """Extract calendar month (1-12) from FMoW JSON metadata.

    Returns 0 if timestamp is missing or unparseable.
    """
    try:
        timestamp = metadata_dict.get("timestamp", "")
        if timestamp and len(timestamp) >= 7:
            return int(timestamp[5:7])
    except (ValueError, TypeError):
        pass
    return 0


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _read_image(path: str) -> np.ndarray:
    """Read an image file, supporting .tif (via rasterio or PIL) and common formats.

    Returns:
        (H, W, 3) numpy array in uint8 [0, 255].
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
            if img.shape[0] >= 3:
                img = img[:3]
            elif img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)

            if img.dtype == np.uint16:
                img = percentile_normalization(img, axis=(0, 1))
                img = (img * 255).astype(np.uint8)
            elif img.dtype in (np.float32, np.float64):
                if img.max() > 1.0:
                    img = percentile_normalization(img, axis=(0, 1))
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.float32)
                img = percentile_normalization(img, axis=(0, 1))
                img = (img * 255).astype(np.uint8)

            return img
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"rasterio failed for {path}: {e}, falling back to PIL")

    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise RuntimeError(f"Failed to read image {path}: {e}")


def _read_metadata(path: str) -> Dict[str, Any]:
    """Read JSON metadata file."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Coordinate Extraction from FMoW JSON
# ---------------------------------------------------------------------------

def _parse_wkt_polygon_centroid(wkt_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse a WKT POLYGON string and return centroid (lon, lat).

    Handles formats:
        POLYGON ((-71.23 42.56, -71.23 42.57, ...))
        POLYGON((-71.23 42.56, -71.23 42.57, ...))
        MULTIPOLYGON (((-71.23 42.56, ...)))

    Returns:
        (lon, lat) or (None, None) on failure.
    """
    try:
        # Try shapely first for robust parsing
        from shapely.wkt import loads as wkt_loads
        geom = wkt_loads(wkt_str)
        centroid = geom.centroid
        lon, lat = centroid.x, centroid.y
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            return lon, lat
        return None, None
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: regex-based extraction
    try:
        coord_pattern = re.compile(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)')
        matches = coord_pattern.findall(wkt_str)
        if not matches:
            return None, None

        lons = [float(m[0]) for m in matches]
        lats = [float(m[1]) for m in matches]
        lon = sum(lons) / len(lons)
        lat = sum(lats) / len(lats)

        if -180 <= lon <= 180 and -90 <= lat <= 90:
            return lon, lat
        return None, None
    except (ValueError, TypeError):
        return None, None


def _extract_coords_from_metadata(metadata: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Extract lon/lat from FMoW JSON metadata.

    Priority:
        1. bounding_box WKT polygon centroid
        2. raw_location field ([lat, lon] — NOTE: lat is first in FMoW format)
        3. Direct lat/lon fields
    """
    # 1. bounding_box WKT polygon
    bbox = metadata.get("bounding_box", None)
    if bbox is not None and isinstance(bbox, str) and len(bbox) > 10:
        lon, lat = _parse_wkt_polygon_centroid(bbox)
        if lon is not None:
            return lon, lat

    # 2. raw_location — FMoW stores as [lat, lon]
    raw_loc = metadata.get("raw_location", None)
    if raw_loc is not None:
        try:
            if isinstance(raw_loc, (list, tuple)) and len(raw_loc) >= 2:
                lat_val, lon_val = float(raw_loc[0]), float(raw_loc[1])
                if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                    return lon_val, lat_val
        except (ValueError, TypeError, IndexError):
            pass

    # 3. Direct fields
    for lat_key in ("lat", "latitude"):
        for lon_key in ("lon", "longitude"):
            lat = metadata.get(lat_key)
            lon = metadata.get(lon_key)
            if lat is not None and lon is not None:
                try:
                    lat_val, lon_val = float(lat), float(lon)
                    if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                        return lon_val, lat_val
                except (ValueError, TypeError):
                    pass

    return None, None


def _extract_ids_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract location_id and image_id from FMoW filename.

    Examples:
        airport_0_0_rgb.tif → location_id=0, image_id=0
        airport_0_1_rgb.jpg → location_id=0, image_id=1
        multi-unit_residential_42_3_rgb.tif → location_id=42, image_id=3
    """
    stem = Path(filename).stem  # e.g. airport_0_0_rgb
    stem = stem.replace("_rgb", "").replace("_ms", "")
    parts = stem.split("_")

    try:
        image_id = int(parts[-1])
        location_id = int(parts[-2])
        return location_id, image_id
    except (ValueError, IndexError):
        return None, None


# ---------------------------------------------------------------------------
# Optional CSV Coordinate Lookup (DiffusionSat compatible)
# ---------------------------------------------------------------------------

class FMoWMetadataCSV:
    """Loads fmow-train-meta.csv for coordinate lookup.

    The CSV has columns: category, location_id, image_id, polygon (WKT), ...
    Used by DiffusionSat to get precise polygon centroids.
    """

    def __init__(self, csv_path: str):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded FMoW metadata CSV: {len(self.df)} rows from {csv_path}")

        # Build lookup index
        self.df["_key"] = (
            self.df["category"].astype(str) + "_"
            + self.df["location_id"].astype(str) + "_"
            + self.df["image_id"].astype(str)
        )
        self._lookup = dict(zip(self.df["_key"], self.df.index))

    def get_coords(
        self, category: str, location_id: int, image_id: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """Look up polygon centroid from CSV."""
        key = f"{category}_{location_id}_{image_id}"
        idx = self._lookup.get(key, None)
        if idx is None:
            return None, None

        try:
            polygon_wkt = self.df.loc[idx, "polygon"]
            if isinstance(polygon_wkt, str):
                return _parse_wkt_polygon_centroid(polygon_wkt)
        except (KeyError, TypeError):
            pass

        return None, None


# ---------------------------------------------------------------------------
# FMoW Dataset
# ---------------------------------------------------------------------------

class FMoWDataset(Dataset):
    """
    PyTorch Dataset for the FMoW (Functional Map of the World) dataset.

    Scans the FMoW directory structure:
        root_dir / category / location / *_rgb.{tif,jpg,png} + *_rgb.json

    If `caption_dir` is provided, rich VLM captions are read from
    `{caption_dir}/<category>/<location>/<stem>.txt` (mirroring root_dir
    structure). Otherwise, procedural template captions are used.

    Args:
        root_dir: path to the FMoW split root (e.g., "fmow/train").
        tokenizer: CLIP tokenizer for caption encoding.
        resolution: target image resolution (default 512).
        transform: optional torchvision transform for images.
        num_metadata: number of metadata fields (default 7).
        text_metadata: if True, embed metadata in caption text instead.
        caption_drop_pct: probability of dropping caption (set to empty string).
        max_gsd: maximum GSD for normalization.
        categories: optional subset of categories to use.
        meta_csv_path: optional path to fmow-train-meta.csv for coordinate lookup.
        caption_dir: optional path to a parallel directory of .txt captions.
            When provided, this overrides the procedural caption generator
            (template captions). Falls back to template if a .txt is missing.
    """

    def __init__(
        self,
        root_dir: str,
        tokenizer: Any,
        resolution: int = 512,
        transform: Optional[Callable] = None,
        num_metadata: int = 7,
        text_metadata: bool = False,
        caption_drop_pct: float = 0.03,
        max_gsd: float = 1.0,
        categories: Optional[List[str]] = None,
        meta_csv_path: Optional[str] = None,
        caption_dir: Optional[str] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.num_metadata = num_metadata
        self.text_metadata = text_metadata
        self.caption_drop_pct = caption_drop_pct
        self.max_gsd = max_gsd

        # Rich-caption directory (optional). When set, prefers .txt over template.
        self.caption_dir = Path(caption_dir) if caption_dir is not None else None
        self._cap_total = 0
        self._cap_missing = 0
        self._cap_log_warned = False

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop(resolution),
                transforms.Normalize([0.5], [0.5]),
            ])

        # Optional CSV for coordinate lookup
        self.meta_csv = None
        if meta_csv_path is not None and os.path.exists(meta_csv_path):
            try:
                self.meta_csv = FMoWMetadataCSV(meta_csv_path)
            except Exception as e:
                logger.warning(f"Failed to load metadata CSV {meta_csv_path}: {e}")

        # Scan directory structure
        self.samples = self._scan_dataset(categories)
        logger.info(f"FMoW dataset loaded: {len(self.samples)} samples from {self.root_dir}")
        if self.caption_dir is not None:
            logger.info(f"  Rich captions enabled: reading .txt files from {self.caption_dir}")

    def _scan_dataset(self, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scan FMoW directory and build sample list."""
        samples = []
        valid_categories = set(categories) if categories else set(FMOW_CATEGORIES)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"FMoW root directory not found: {self.root_dir}")

        for category_dir in sorted(self.root_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            if category not in valid_categories:
                continue

            for location_dir in sorted(category_dir.iterdir()):
                if not location_dir.is_dir():
                    continue

                image_files = (
                    sorted(location_dir.glob("*_rgb.tif"))
                    + sorted(location_dir.glob("*_rgb.jpg"))
                    + sorted(location_dir.glob("*_rgb.png"))
                )

                for img_path in image_files:
                    # airport_0_0_rgb.tif → airport_0_0_rgb.json
                    json_path = img_path.with_suffix(".json")

                    if not json_path.exists():
                        logger.debug(f"No JSON metadata for {img_path}, skipping")
                        continue

                    location_id, image_id = _extract_ids_from_filename(img_path.name)

                    samples.append({
                        "image_path": str(img_path),
                        "metadata_path": str(json_path),
                        "category": category,
                        "location_id": location_id,
                        "image_id": image_id,
                    })

        if len(samples) == 0:
            logger.warning(
                f"No samples found in {self.root_dir}. "
                f"Expected structure: root/category/location/*_rgb.tif + *_rgb.json"
            )

        return samples

    def _get_coordinates(
        self,
        metadata_dict: Dict[str, Any],
        category: str,
        location_id: Optional[int],
        image_id: Optional[int],
    ) -> Tuple[float, float]:
        """Get coordinates using all available sources.

        Priority: CSV → JSON bounding_box → JSON raw_location → (0, 0)
        """
        if self.meta_csv is not None and location_id is not None and image_id is not None:
            lon, lat = self.meta_csv.get_coords(category, location_id, image_id)
            if lon is not None:
                return lon, lat

        lon, lat = _extract_coords_from_metadata(metadata_dict)
        if lon is not None:
            return lon, lat

        return 0.0, 0.0

    # -------------------------------------------------------------------
    # Rich caption I/O
    # -------------------------------------------------------------------

    def _read_rich_caption(self, image_path: str) -> Optional[str]:
        """Resolve image_path → caption file → text.

        Maps:  {root_dir}/category/location/file_rgb.tif
            →  {caption_dir}/category/location/file_rgb.txt

        Returns:
            Caption string if file exists and is non-trivial; None otherwise.
        """
        if self.caption_dir is None:
            return None
        try:
            img_p = Path(image_path)
            rel = img_p.relative_to(self.root_dir)
            cap_p = self.caption_dir / rel.with_suffix(".txt")
            self._cap_total += 1
            if not cap_p.exists():
                self._cap_missing += 1
                if (
                    not self._cap_log_warned
                    and self._cap_total > 100
                    and self._cap_missing / max(self._cap_total, 1) > 0.1
                ):
                    logger.warning(
                        f"Caption-miss rate >10% after {self._cap_total} reads "
                        f"({self._cap_missing} missing). Check caption_dir path."
                    )
                    self._cap_log_warned = True
                return None
            text = cap_p.read_text(encoding="utf-8").strip()
            if len(text) < 10:
                self._cap_missing += 1
                return None
            return text
        except Exception:
            self._cap_missing += 1
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Try the requested sample, skip to nearby indices if corrupted
        candidates = [idx] + list(range(
            min(idx + 1, len(self.samples) - 1),
            min(idx + 10, len(self.samples)),
        ))
        for attempt_idx in candidates:
            try:
                return self._load_sample(attempt_idx)
            except Exception as e:
                logger.warning(
                    f"Skipping corrupted sample "
                    f"{self.samples[attempt_idx]['image_path']}: {e}"
                )
                continue
        # If all nearby samples fail, return a random one
        return self._load_sample(0)

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        # Read image
        img_np = _read_image(sample_info["image_path"])
        h, w = img_np.shape[:2]

        # Read metadata JSON
        metadata_dict = _read_metadata(sample_info["metadata_path"])
        category = sample_info["category"]

        # Extract coordinates
        lon, lat = self._get_coordinates(
            metadata_dict,
            category,
            sample_info.get("location_id"),
            sample_info.get("image_id"),
        )

        # Extract numerical metadata vector
        raw_metadata = extract_fmow_metadata(
            metadata_dict,
            img_h=h,
            img_w=w,
            target_resolution=self.resolution,
            lon=lon,
            lat=lat,
        )

        # Normalize metadata
        norm_metadata = metadata_normalize(raw_metadata, max_gsd=self.max_gsd)

        # ----------------------------------------------------------------
        # Caption resolution — three-way mode sampling
        #
        # When caption_dir is set (rich VLM captions available):
        #   Roll a uniform random number to decide caption mode:
        #     [0.00, 0.70) → VLM rich caption (from .txt file)
        #     [0.70, 0.90) → short location-aware caption
        #     [0.90, 1.00) → empty string (CFG unconditional drop)
        #
        # When caption_dir is NOT set (procedural template only):
        #   Fall back to the original caption_drop_pct logic unchanged.
        #
        # Caption dropout is implemented by setting caption="" here.
        # The training loop zeroes encoder_hidden_states for those samples
        # (embedding-zeroing, NOT attention_mask zeroing — NaN-safe).
        # ----------------------------------------------------------------
        caption: Optional[str] = None

        if self.caption_dir is not None:
            roll = random.random()

            if roll < 0.70:
                # --- Mode A: VLM rich caption ---
                caption = self._read_rich_caption(sample_info["image_path"])
                if caption is None:
                    # Missing .txt → degrade to short caption for this sample
                    caption = generate_short_caption(
                        category=category,
                        lat=lat,
                        lon=lon,
                        month=_extract_month(metadata_dict),
                    )

            elif roll < 0.90:
                # --- Mode B: short location-aware caption ---
                caption = generate_short_caption(
                    category=category,
                    lat=lat,
                    lon=lon,
                    month=_extract_month(metadata_dict),
                )

            else:
                # --- Mode C: unconditional drop ---
                caption = ""

        else:
            # No rich captions — original procedural path, unchanged.
            if self.text_metadata:
                caption = generate_fmow_caption_with_metadata(
                    category, metadata_dict, raw_metadata, self.caption_drop_pct
                )
            else:
                caption = generate_fmow_caption(
                    category, metadata_dict, self.caption_drop_pct
                )

        # Tokenize caption
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.squeeze(0)

        # Apply image transform
        pixel_values = self.transform(img_np)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "metadata": norm_metadata,
        }


# ---------------------------------------------------------------------------
# Collate Function
# ---------------------------------------------------------------------------

def flowsat_collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for FlowSat DataLoader."""
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([ex["input_ids"] for ex in examples])
    metadata = torch.stack([ex["metadata"] for ex in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "metadata": metadata,
    }