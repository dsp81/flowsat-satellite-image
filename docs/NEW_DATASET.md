# Using FlowSat on a New Dataset

FlowSat is not tied to FMoW. Anything that can produce **(image, caption,
metadata)** triples — Sentinel-2 tiles, NAIP, SpaceNet, an internal corpus —
plugs in by implementing **one class with one method**.

This document gives the interface contract, a copy-paste template, the metadata
convention, the commands to train and evaluate, and a list of failure modes that
cost us real time. Read the [pitfalls](#pitfalls-read-this-before-you-train)
section before your first long run; every item there is something that silently
degrades results rather than raising an error.

---

## 1. The contract

Your dataset is a standard `torch.utils.data.Dataset` whose `__getitem__`
returns a dict with exactly three keys:

```python
{
    "pixel_values": torch.Tensor,   # (3, H, W)  float, normalised to [-1, 1]
    "input_ids":    torch.Tensor,   # (L,)       long, tokenised caption
    "metadata":     torch.Tensor,   # (7,)       float, normalised (see §2)
}
```

That is the entire interface. Everything else — VAE encoding, text encoding,
flow-matching, CFG dropout, EMA, checkpointing — is handled by the training
loop and does not need to change.

**If your data has no metadata**, pass zeros for the unused fields *and* train
with `--num_metadata 0`. Do **not** pass a zero vector while metadata
conditioning is enabled: after normalisation, zero is not "unknown", it decodes
to a specific and wrong assertion (lon = −180°, lat = −90°, year = 1980). See
[pitfall 4](#4-zero-is-not-unknown).

---

## 2. Metadata convention

Seven fields, each normalised to approximately `[0, 1000]`:

| idx | field | raw range | normalisation |
|---|---|---|---|
| 0 | longitude | −180 … 180 | `(lon + 180) / 360 * 1000` |
| 1 | latitude | −90 … 90 | `(lat + 90) / 180 * 1000` |
| 2 | GSD (m/px) | 0 … `max_gsd` | `gsd / max_gsd * 1000` |
| 3 | cloud cover | 0 … 1 | `cloud * 1000` |
| 4 | year | 1980 … 2100 | `(year − 1980) / 120 * 1000` |
| 5 | month | 1 … 12 | `month / 12 * 1000` |
| 6 | day | 1 … 31 | `day / 31 * 1000` |

Use the provided helper rather than hand-rolling this:

```python
from flowsat.data.sat_data_util import metadata_normalize
md = metadata_normalize(torch.tensor([lon + 180, lat + 90, gsd, cloud,
                                      year - 1980, month, day]),
                        max_gsd=YOUR_MAX_GSD)
```

**`max_gsd` must match between training, evaluation, and generation.** It is the
one free parameter in the scheme, and a mismatch silently shifts the entire GSD
axis. Record it alongside your checkpoint.

**Fields your dataset lacks.** Keep the 7-dim layout and fill missing fields with
a plausible in-distribution constant (e.g. cloud = 0, day = 15), not zero. If a
field is *always* constant it carries no information and the model will ignore
it, which is the correct outcome.

---

## 3. Template adapter

Copy to `flowsat/data/my_dataset.py` and fill in the three marked sections.

```python
"""Adapter template: MyDataset -> FlowSat."""
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .sat_data_util import metadata_normalize


class MyDataset(Dataset):
    def __init__(self, root_dir, tokenizer, resolution=512,
                 max_gsd=1.0, caption_dir=None, transform=None):
        self.root = Path(root_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.max_gsd = max_gsd
        self.caption_dir = Path(caption_dir) if caption_dir else None

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution, antialias=True),
            transforms.CenterCrop(resolution),
            transforms.Normalize([0.5], [0.5]),          # -> [-1, 1]
        ])

        # ---- (1) BUILD YOUR SAMPLE INDEX ---------------------------------
        # A list of whatever you need to load one example later. Keep it
        # deterministic (sort it) so shards and resumes are reproducible.
        self.samples = sorted(self.root.rglob("*.tif"))
        assert self.samples, f"no samples under {self.root}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rec = self.samples[idx]

        # ---- (2) LOAD IMAGE + RAW METADATA -------------------------------
        # Return an (H, W, 3) uint8 array. For multi-band sources, select or
        # composite to RGB here — the released weights are RGB-only.
        img = self._read_image(rec)                        # (H, W, 3) uint8
        lon, lat, gsd, cloud, year, month, day = self._read_metadata(rec)

        # ---- (3) CAPTION -------------------------------------------------
        # Match the caption style you will use at inference time (pitfall 1).
        caption = self._caption(rec)

        md = metadata_normalize(
            torch.tensor([lon + 180.0, lat + 90.0, gsd, cloud,
                          float(year - 1980), float(month), float(day)]),
            max_gsd=self.max_gsd)

        ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids.squeeze(0)

        return {"pixel_values": self.transform(img),
                "input_ids": ids,
                "metadata": md}

    # ------------------------------------------------------------------
    def _read_image(self, rec):
        import numpy as np, rasterio
        with rasterio.open(rec) as src:
            a = src.read()[:3].transpose(1, 2, 0)
        if a.dtype != np.uint8:                # 16-bit / float -> 8-bit
            lo, hi = np.percentile(a, [2, 98], axis=(0, 1), keepdims=True)
            a = np.clip((a - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
            a = (a * 255).astype(np.uint8)
        return a

    def _read_metadata(self, rec) -> Any:
        raise NotImplementedError("return (lon, lat, gsd, cloud01, year, month, day)")

    def _caption(self, rec) -> str:
        if self.caption_dir:
            p = (self.caption_dir / rec.relative_to(self.root)).with_suffix(".txt")
            if p.exists():
                t = p.read_text(encoding="utf-8").strip()
                if len(t) > 10:
                    return t
        return "a satellite image"


def my_collate_fn(examples):
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples])
                             .to(memory_format=torch.contiguous_format).float(),
        "input_ids":    torch.stack([e["input_ids"] for e in examples]),
        "metadata":     torch.stack([e["metadata"] for e in examples]),
    }
```

### Verify the adapter before training

Five minutes here saves a wasted run:

```python
ds = MyDataset(root, tokenizer, max_gsd=1.0)
b = ds[0]
assert b["pixel_values"].shape == (3, 512, 512)
assert -1.01 <= b["pixel_values"].min() and b["pixel_values"].max() <= 1.01
assert b["metadata"].shape == (7,)
assert (b["metadata"] >= -1e-3).all() and (b["metadata"] <= 1000 + 1e-3).all(), \
    f"metadata outside [0,1000]: {b['metadata']}"     # usually a max_gsd error
print(tokenizer.decode(b["input_ids"], skip_special_tokens=True)[:300])
```

Also print the caption **token-length histogram** across ~1000 samples and
compare it with `--t5_max_length`. If the 90th percentile exceeds it, most of
your captions are being silently truncated (pitfall 2).

---

## 4. Registration

```python
# flowsat/data/__init__.py
from .fmow_dataset import FMoWDataset, flowsat_collate_fn
from .my_dataset  import MyDataset,  my_collate_fn

DATASETS = {
    "fmow": (FMoWDataset, flowsat_collate_fn),
    "mine": (MyDataset,   my_collate_fn),
}
```

Then `--dataset mine` selects it.

---

## 5. Train

Fine-tuning from the released FlowSat checkpoint is strongly preferred over
training from the raw Sana initialisation — the released weights already carry
satellite priors, and convergence is several times faster.

```bash
accelerate launch --num_processes=<N> --mixed_precision=bf16 \
  -m flowsat.training.train \
  --dataset mine --data_root /path/to/data --caption_dir /path/to/captions \
  --pretrained  <sana-snapshot-dir> \
  --resume_from checkpoints/flowsat-fmow-512 \
  --output_dir  runs/mine \
  --resolution 512 --max_gsd <YOUR_MAX_GSD> \
  --num_metadata 7 --use_satclip_encoder \
  --train_batch_size 8 --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 --lr_scheduler cosine --lr_warmup_steps 1000 \
  --max_train_steps 50000 --gradient_checkpointing \
  --checkpointing_steps 5000 --validation_steps 2500 --use_ema
```

Watch in the first 500 steps:

- the load line reports **~0 missing / ~0 unexpected** keys;
- the first loss is finite;
- validation images are coherent (not colour noise) by ~2k steps.

## 6. Evaluate

```bash
python -m flowsat.evaluation.evaluate \
  --ckpt runs/mine/checkpoint-50000 \
  --dataset mine --data_root /path/to/data \
  --metrics fid,clip --n_samples 5000 --steps 20 --cfg 2.5
```

FID is only comparable within a fixed protocol. To compare against a published
number you must match **the same reference statistics, the same sample count,
the same preprocessing, and the same CLIP variant**. Recomputing reference stats
on a different split silently changes the number by several points.

## 7. Measure controllability

The claim FlowSat makes is not only image quality but *metadata control*. To
test it on your data, sweep one field with caption and noise held fixed:

```bash
python -m flowsat.evaluation.controllability \
  --ckpt runs/mine/checkpoint-50000 \
  --axis month --values 1,4,7,10 --n_seeds 4 \
  --prompt-regime rich,short,empty
```

This reports divergence across the sweep, separated into **tonal** (colour and
contrast) and **structural** (content) components. The distinction matters: a
model can tint an image in response to metadata without changing what is
depicted, and only the structural component supports a controllability claim.

---

## Pitfalls (read this before you train)

### 1. Caption style must match between training and inference
The largest single quality factor. If you train predominantly on long dense
captions and then generate from `"a satellite image of a port"`, you are
sampling an under-represented mode and output quality drops visibly. Decide your
caption distribution up front and use the same distribution at inference. If you
need both, train with an explicit mixture (we use ~40% rich / 30% short / 30%
empty) so both modes are supported.

### 2. Token truncation is silent
`--t5_max_length` truncates without warning. A 200-word caption is ~260 tokens;
at the default of 120 more than half of it — typically the spatial and
object-level detail at the end — never reaches the model. Print a token-length
histogram and set the limit above your 90th percentile.

### 3. Empty-caption samples are what train the metadata pathway
If every training sample has a caption, text explains most of the variance and
the metadata pathway is starved of gradient. A meaningful fraction of
caption-free samples (~30%) is what forces metadata to carry information on its
own.

### 4. Zero is not "unknown"
For metadata dropout, do **not** multiply the metadata vector by zero. After
normalisation zero decodes to lon = −180°, lat = −90°, year = 1980 — a confident
wrong assertion, not an absence. Per-field zeroing at p = 0.1 over 7 fields
corrupts ~52% of samples and teaches the model to distrust metadata entirely.
Use the learned null-metadata embedding instead (`--metadata_drop_prob`, applied
per sample).

### 5. Classifier-free guidance: guide text, not metadata magnitude
Pass **real** metadata to both CFG branches and guide only the text direction,
or use a proper three-branch formulation with a learned null. Substituting a
zero metadata vector for the unconditional branch puts that branch out of
distribution and corrupts the guidance direction.

### 6. Caption redundancy suppresses metadata control
If your captions already state the terrain, climate, or season, they cover the
same variance that latitude and month would explain, and metadata conditioning
will appear weak — correctly so, because there is nothing left for it to
explain. Let text describe *content* and metadata supply *context*, or test
controllability with short/empty prompts where text is silent.
See [What Makes a Model Metadata-Controllable](METADATA_CONTROLLABILITY.md)
for the full discussion, including how to choose fields that stay controllable.

### 7. Keep metadata inside its normalised range
Values outside `[0, 1000]` are extrapolation. The most common cause is a GSD
sweep exceeding `max_gsd` (3.0 m with `max_gsd = 1.0` normalises to 3000).
Assert the range in your adapter.

### 8. Validation with a fixed seed measures one sample
A fixed noise seed makes progress easy to eyeball but hides diversity and mode
collapse — you watch one composition sharpen forever. Vary the seed across
validations; keep it fixed *within* a metadata sweep so the metadata remains the
only variable.

---

## Getting help

Open an issue with your adapter's verification output (shapes, metadata range,
caption histogram) and the first 50 lines of your training log. Those three
things identify most integration problems immediately.
