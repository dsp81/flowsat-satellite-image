"""
eval_common.py — the metric spine shared by every FlowSat evaluation.

Everything that decides *what a number means* lives here, in one place, so that
two runs of `evaluate_fmow.py` differing only in checkpoint are comparable, and
so that a baseline scored through this module is scored the same way as FlowSat:

  * sample discovery     which test images, in which order
  * image preprocessing  how a GeoTIFF becomes the 512x512 uint8 tensor that FID sees
  * metadata extraction  how an FMoW JSON sidecar becomes the 7-vector the model takes
  * metrics              FID, Inception Score, LPIPS, SSIM (torchmetrics) and CLIP score

FID is only meaningful relative to an identical reference set and an identical
preprocessing path. Changing the resize filter, the crop, or the sample count
changes the number without changing the model. If you modify anything in this
file, every previously published number becomes incomparable -- including the
ones in the paper.

Requires the optional evaluation extra:

    pip install -e ".[eval]"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("flowsat.eval")

# Used when a field is absent from the sidecar. These are mid-range values, not
# zeros: after normalisation zero is not "unknown", it is a confident assertion
# (lon = -180, lat = -90, year = 1980).
DEFAULT_METADATA_FALLBACK = {
    "lon": 0.0, "lat": 0.0, "gsd": 1.0, "cloud": 0.0,
    "year": 2015.0, "month": 6.0, "day": 15.0,
}


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------

@dataclass
class TestSample:
    """One evaluation sample: a real image, its caption, and its metadata.

    FMoW's test split is laid out as one directory per sequence:

        <fmow_test_root>/<seq_id>/<stem>_rgb.tif     the image
        <fmow_test_root>/<seq_id>/<stem>_rgb.json    the metadata sidecar
        <caption_root>/test/<seq_id>/<stem>_rgb.txt  the caption (generated separately)
    """

    image_path: Path
    caption_path: Path
    metadata_path: Path
    seq_id: str
    stem: str
    category: Optional[str] = None

    @property
    def caption(self) -> str:
        try:
            return self.caption_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"failed to read caption {self.caption_path}: {e}")
            return ""

    @property
    def metadata(self) -> Dict[str, Any]:
        try:
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"failed to read metadata {self.metadata_path}: {e}")
            return {}


def _build_seqid_to_category(path: Optional[Path]) -> Dict[str, str]:
    """Parse FMoW's `test_gt_mapping` into {seq_id: category}, if present.

    Categories are optional -- FID, CLIP, SSIM and LPIPS never use them; they
    only enable a per-category breakdown. The file's format varies between FMoW
    releases, so try the common ones and give up quietly.
    """
    if path is None or not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            head = f.read(1)
            f.seek(0)
            if head == "{":
                data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            elif head == "[":
                data = json.load(f)
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    return {str(r["id"]): str(r["category"]) for r in data
                            if "id" in r and "category" in r}
            else:
                first = f.readline().strip()
                f.seek(0)
                if first.startswith("{"):          # JSONL
                    out = {}
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        if "id" in r and "category" in r:
                            out[str(r["id"])] = str(r["category"])
                    return out
                import csv                          # CSV
                reader = csv.reader(f)
                header = next(reader, None)
                id_col, cat_col = 0, 1
                if header:
                    low = [h.lower() for h in header]
                    for name in ("id", "seq_id"):
                        if name in low:
                            id_col = low.index(name)
                            break
                    for name in ("category", "class"):
                        if name in low:
                            cat_col = low.index(name)
                            break
                return {str(r[id_col]): str(r[cat_col]) for r in reader
                        if len(r) > max(id_col, cat_col)}
    except Exception as e:
        logger.warning(f"could not parse test_gt_mapping {path}: {e}")
    return {}


def load_test_samples(
    fmow_test_root: Path,
    caption_root: Path,
    test_gt_mapping_path: Optional[Path] = None,
    limit: int = 0,
    require_caption: bool = True,
) -> List[TestSample]:
    """Enumerate the FMoW test split in a fixed, reproducible order.

    Samples are sorted by (seq_id, stem), so `limit` always selects the same
    prefix: two runs with the same `limit` score the same images. FID is highly
    sensitive to which images are in the reference set, so this ordering is part
    of the protocol, not an implementation detail.

    The scan touches thousands of directories. On network storage that is slow
    enough to dominate a short run, so the full list is cached under
    ~/.cache/flowsat/. Set FLOWSAT_NO_SAMPLE_CACHE=1 to force a rescan (do that
    after adding or removing captions).
    """
    if not fmow_test_root.exists():
        raise FileNotFoundError(f"FMoW test root not found: {fmow_test_root}")

    import hashlib
    key = hashlib.sha1(
        f"{fmow_test_root}|{caption_root}|{require_caption}".encode()).hexdigest()[:16]
    cache = Path.home() / ".cache" / "flowsat" / f"test_samples_{key}.json"
    if not os.environ.get("FLOWSAT_NO_SAMPLE_CACHE"):
        try:
            if cache.exists():
                raw = json.loads(cache.read_text())
                out = [TestSample(image_path=Path(r["i"]), caption_path=Path(r["c"]),
                                  metadata_path=Path(r["m"]), seq_id=r["s"],
                                  stem=r["t"], category=r.get("g")) for r in raw]
                if not limit or limit <= 0 or len(out) >= limit:
                    logger.info(f"loaded {len(out):,} test samples from cache {cache}")
                    return out[:limit] if limit and limit > 0 else out
        except Exception as e:
            logger.warning(f"sample cache unreadable ({e}); rescanning")

    caption_test_root = caption_root / "test"
    seqid_to_cat = _build_seqid_to_category(test_gt_mapping_path)

    samples: List[TestSample] = []
    early_stop = False
    # os.listdir returns every name from a handful of getdents calls, while
    # is_dir() on each entry is one stat per entry -- thousands of round trips on
    # a network filesystem, before a single image has been read. Skip the filter:
    # globbing a non-directory simply yields nothing.
    seq_dirs = [fmow_test_root / n for n in sorted(os.listdir(fmow_test_root))]
    for seq_dir in seq_dirs:
        seq_id = seq_dir.name
        for ext in ("*_rgb.tif", "*_rgb.jpg", "*_rgb.png"):
            try:
                images = sorted(seq_dir.glob(ext))
            except (NotADirectoryError, PermissionError, OSError):
                continue
            for img_path in images:
                json_path = img_path.with_suffix(".json")
                if not json_path.exists():
                    continue
                stem = img_path.stem
                cap_path = caption_test_root / seq_id / f"{stem}.txt"
                if require_caption:
                    try:
                        if not cap_path.exists() or cap_path.stat().st_size == 0:
                            continue
                    except OSError:
                        continue
                samples.append(TestSample(
                    image_path=img_path, caption_path=cap_path,
                    metadata_path=json_path, seq_id=seq_id, stem=stem,
                    category=seqid_to_cat.get(seq_id)))
                # The walk is already in (seq_id, stem) order, so stopping early
                # gives exactly the same prefix a full scan would.
                if limit and limit > 0 and len(samples) >= limit:
                    early_stop = True
                    break
            if early_stop:
                break
        if early_stop:
            break

    samples.sort(key=lambda s: (s.seq_id, s.stem))
    if not os.environ.get("FLOWSAT_NO_SAMPLE_CACHE") and not early_stop:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(
                [{"i": str(x.image_path), "c": str(x.caption_path),
                  "m": str(x.metadata_path), "s": x.seq_id, "t": x.stem,
                  "g": x.category} for x in samples]))
            logger.info(f"cached {len(samples):,} test samples -> {cache}")
        except Exception as e:
            logger.warning(f"could not write sample cache: {e}")
    if limit and limit > 0:
        samples = samples[:limit]
    return samples


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _percentile_normalization(img: np.ndarray, lo=2.0, hi=98.0,
                              axis=(0, 1)) -> np.ndarray:
    """Stretch each band between its 2nd and 98th percentile.

    FMoW GeoTIFFs are 11-bit data stored in 16-bit containers, so a naive
    `/ 65535` leaves everything nearly black. This is the same stretch the
    training dataset applies, which matters: the real images FID compares
    against must be preprocessed exactly as the training images were.
    """
    img = img.astype(np.float32)
    p_lo = np.percentile(img, lo, axis=axis, keepdims=True)
    p_hi = np.percentile(img, hi, axis=axis, keepdims=True)
    return np.clip((img - p_lo) / np.maximum(p_hi - p_lo, 1e-6), 0.0, 1.0)


def read_real_image(path: Path, resolution: int = 512) -> Optional[Image.Image]:
    """Load one real test image as a centre-cropped, resized RGB PIL image.

    rasterio if available (it handles the uint16/float GeoTIFF dtypes), PIL
    otherwise. Returns None if the file cannot be read at all -- corrupt TIFFs
    exist in FMoW and one of them should not end a 10,000-sample run.
    """
    arr: Optional[np.ndarray] = None
    try:
        import rasterio
        with rasterio.open(str(path)) as src:
            img = src.read()
            if img.shape[0] >= 3:
                img = img[:3]
            elif img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))
            if img.dtype == np.uint16:
                img = (_percentile_normalization(img) * 255).astype(np.uint8)
            elif img.dtype in (np.float32, np.float64):
                img = (_percentile_normalization(img) if img.max() > 1.0
                       else np.clip(img, 0.0, 1.0))
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = (_percentile_normalization(img) * 255).astype(np.uint8)
            arr = img
    except Exception:
        pass

    if arr is None:
        try:
            with Image.open(path) as im:
                arr = np.array(im.convert("RGB"))
        except Exception as e:
            logger.warning(f"failed to read real image {path}: {e}")
            return None

    pil = Image.fromarray(arr).convert("RGB")
    w, h = pil.size
    side = min(w, h)
    pil = pil.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    return pil.resize((resolution, resolution), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def extract_metadata_vector(metadata: Dict[str, Any],
                            num_metadata: int = 7) -> List[float]:
    """FMoW JSON sidecar -> the normalised 7-vector the model conditions on.

        lon   -> (lon + 180) / 360 * 1000
        lat   -> (lat + 90)  / 180 * 1000
        gsd   -> gsd / 1.0 * 1000          (metres per pixel; max_gsd = 1.0)
        cloud -> cloud / 100 * 1000
        year  -> (year - 1980) / 120 * 1000
        month -> month / 12 * 1000
        day   -> day / 31 * 1000

    This is the same arithmetic as `flowsat.data.sat_data_util.metadata_normalize`,
    restated here so the evaluation does not depend on the training data stack.

    ON COORDINATES. The FMoW **test** sidecars carry `gsd`, `cloud_cover` and
    `timestamp`, but no latitude or longitude -- unlike the train sidecars, they
    have no `raw_location` and no `bounding_box` WKT. So lon/lat fall back to
    (0, 0) for every test sample, i.e. 500/500 after normalisation. The reported
    FID/CLIP/SSIM/LPIPS were measured under exactly that condition, and it is
    identical for every model scored through this module, so the comparison is
    fair. It does mean the paper's quality table exercises the date, GSD and
    cloud pathways but not the geographic one; geographic control is measured
    separately, by the controllability sweeps.
    """
    raw_lon = raw_lat = None
    try:
        bbox = (metadata.get("bounding_boxes") or [{}])[0]
        if "centroid_lon_lat" in bbox:
            raw_lon = float(bbox["centroid_lon_lat"][0])
            raw_lat = float(bbox["centroid_lon_lat"][1])
    except Exception:
        raw_lon = raw_lat = None
    if raw_lon is None or raw_lat is None:
        raw_lon = float(metadata.get("raw_location_lon") or 0.0)
        raw_lat = float(metadata.get("raw_location_lat") or 0.0)

    gsd = float(metadata.get("gsd", DEFAULT_METADATA_FALLBACK["gsd"]))
    cloud = float(metadata.get("cloud_cover", DEFAULT_METADATA_FALLBACK["cloud"]))

    ts = metadata.get("timestamp", "") or ""
    year = DEFAULT_METADATA_FALLBACK["year"]
    month = DEFAULT_METADATA_FALLBACK["month"]
    day = DEFAULT_METADATA_FALLBACK["day"]
    if isinstance(ts, str) and len(ts) >= 10:
        try:
            year, month, day = float(ts[:4]), float(ts[5:7]), float(ts[8:10])
        except (ValueError, IndexError):
            pass

    norm = [
        (raw_lon + 180.0) / 360.0 * 1000.0,
        (raw_lat + 90.0) / 180.0 * 1000.0,
        gsd / 1.0 * 1000.0,
        cloud / 100.0 * 1000.0,
        (year - 1980.0) / 120.0 * 1000.0,
        month / 12.0 * 1000.0,
        day / 31.0 * 1000.0,
    ]
    return norm[:num_metadata]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def setup_metrics(device: str = "cuda") -> Dict[str, Any]:
    """Build the four torchmetrics modules the paper reports.

    FID uses the standard 2048-dim InceptionV3 pool features. IS is reported for
    completeness only: Inception is ImageNet-trained and overhead imagery is far
    outside that domain, so treat it as a weak signal.

    `normalize=False` means both metrics expect uint8 in [0, 255]; LPIPS and
    SSIM take floats. `update_distribution_metrics` / `update_paired_metrics`
    below encode that split so callers cannot get it wrong.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    return {
        "fid": FrechetInceptionDistance(feature=2048, normalize=False).to(device),
        "is": InceptionScore(normalize=False).to(device),
        "lpips": LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device),
        "ssim": StructuralSimilarityIndexMeasure().to(device),
    }


class ManualCLIPScorer:
    """CLIP score: cosine(image, caption) x 100, averaged over all samples.

    Implemented directly rather than via torchmetrics so the model id, the 77-token
    truncation and the x100 convention are visible and fixed. Papers differ on all
    three; a CLIP score is not comparable across them without saying which was used.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch16",
                 device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.scores: List[float] = []

    @staticmethod
    def _as_tensor(x) -> torch.Tensor:
        """Some transformers versions wrap get_*_features output; unwrap it."""
        if torch.is_tensor(x):
            return x
        for attr in ("pooler_output", "last_hidden_state", "image_embeds", "text_embeds"):
            v = getattr(x, attr, None)
            if v is not None and torch.is_tensor(v):
                return v
        if isinstance(x, tuple) and x and torch.is_tensor(x[0]):
            return x[0]
        raise TypeError(f"cannot extract tensor from CLIP output {type(x).__name__}")

    @torch.no_grad()
    def update(self, images: List[Image.Image], captions: List[str]) -> None:
        if not images:
            return
        inputs = self.processor(text=captions, images=images, return_tensors="pt",
                                padding=True, truncation=True, max_length=77
                                ).to(self.device)
        img_feat = self._as_tensor(self.model.get_image_features(
            pixel_values=inputs["pixel_values"]))
        txt_feat = self._as_tensor(self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]))
        cos = (F.normalize(img_feat, dim=-1) * F.normalize(txt_feat, dim=-1)).sum(-1)
        self.scores.extend((cos * 100.0).cpu().tolist())

    def compute(self) -> float:
        return float(np.mean(self.scores)) if self.scores else float("nan")


def setup_clip_scorer(device: str = "cuda") -> ManualCLIPScorer:
    return ManualCLIPScorer(device=device)


def update_distribution_metrics(metrics: Dict[str, Any],
                                fake_uint8: torch.Tensor,
                                real_uint8: torch.Tensor) -> None:
    """FID + IS. Both tensors (B, 3, H, W) uint8 in [0, 255]."""
    metrics["fid"].update(real_uint8, real=True)
    metrics["fid"].update(fake_uint8, real=False)
    metrics["is"].update(fake_uint8)


def update_paired_metrics(metrics: Dict[str, Any],
                          fake_float: torch.Tensor,
                          real_float: torch.Tensor) -> None:
    """LPIPS + SSIM. Both tensors (B, 3, H, W) float in [0, 1].

    LPIPS wants [-1, 1] and SSIM wants [0, 1]; the conversion happens here so a
    caller cannot silently feed one the other's range.
    """
    metrics["lpips"].update(fake_float * 2.0 - 1.0, real_float * 2.0 - 1.0)
    metrics["ssim"].update(fake_float, real_float)


def compute_final_metrics(metrics: Dict[str, Any],
                          clip_scorer: ManualCLIPScorer) -> Dict[str, float]:
    is_mean, is_std = metrics["is"].compute()
    return {
        "fid": float(metrics["fid"].compute()),
        "clip_score": clip_scorer.compute(),
        "ssim": float(metrics["ssim"].compute()),
        "lpips": float(metrics["lpips"].compute()),
        "is_mean": float(is_mean),
        "is_std": float(is_std),
    }


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def pil_to_uint8_tensor(pil: Image.Image) -> torch.Tensor:
    """PIL RGB -> (3, H, W) uint8 in [0, 255]."""
    arr = np.array(pil.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(torch.uint8)


def uint8_tensor_to_pils(t: torch.Tensor) -> List[Image.Image]:
    """(B, 3, H, W) uint8 -> list of PIL images (the CLIP scorer wants PILs)."""
    return [Image.fromarray(a, mode="RGB")
            for a in t.permute(0, 2, 3, 1).cpu().numpy()]
