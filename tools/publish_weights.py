#!/usr/bin/env python3
"""
publish_weights.py — upload a FlowSat checkpoint to the Hugging Face Hub.

The weights do not fit anywhere on GitHub: a git file is capped at 100 MB, a
release asset at 2 GiB, and the free LFS tier at 1 GB, while the checkpoint is
2.3 GiB. The Hub is also simply where people doing ablations expect to find a
model, and it gives the weights a citable page of their own.

    huggingface-cli login            # once, with a write token
    python tools/publish_weights.py \
        --checkpoint /path/to/checkpoint-125000 \
        --repo-id    dsp81/flowsat-fmow-512

It checks the state dict before uploading anything, writes a model card naming
the evaluation protocol the published numbers came from, and prints the sha256
so the uploaded file can be tied to a specific local one later.

    --dry-run    do every check and render the card, upload nothing
    --private    create the repo private (make it public before the paper)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

CARD = """---
license: mit
tags:
  - flow-matching
  - diffusion
  - satellite-imagery
  - remote-sensing
  - text-to-image
library_name: flowsat
pipeline_tag: text-to-image
---

# FlowSat — FMoW-RGB 512 px

Flow-matching diffusion transformer that generates satellite imagery from text
**and** acquisition metadata: longitude, latitude, ground sample distance, cloud
cover and date. Trained on FMoW-RGB at 512 px.

Code, documentation and the evaluation that produced the numbers below:
<https://github.com/dsp81/flowsat-satellite-image>

## Using it

```bash
pip install "flowsat @ git+https://github.com/dsp81/flowsat-satellite-image"

python generate.py --ckpt {repo_id} \\
    --prompt "An airport surrounded by dry farmland, long grey runway crossing the centre." \\
    --lon 4.40 --lat 51.92 --gsd 0.5 --month 7
```

`--ckpt` takes this repo id directly; the weights download once into the usual
Hugging Face cache.

## Results

FMoW-RGB test split, {n_samples} samples, 20 Euler steps, text guidance 2.5, seed 42:

| FID ↓ | CLIP ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| **31.10** | **0.3016** | 0.1600 | 0.6853 |

SSIM and LPIPS are *paired* against each caption's own source image, so they
measure conditioning fidelity, not image quality — two real FMoW acquisitions of
the same place score SSIM 0.214 / LPIPS 0.425 through the same pipeline.

FID is only comparable under a matched protocol. Reproduce these exactly, and
read what has to match before comparing anything against them:

```bash
python -m flowsat.evaluation.evaluate_fmow \\
    --checkpoint {repo_id} \\
    --fmow_test_root /path/to/fmow-full/test \\
    --caption_root   /path/to/fmow_captions_test \\
    --output_dir     evaluations/flowsat
```

→ [Reproducing the evaluation](https://dsp81.github.io/flowsat-satellite-image/evaluation.html)

## Architecture

Sana-0.6B backbone (28 layers, linear attention, AdaLN-single) with a DC-AE 32×
latent (16×16×32 at 512 px) and a frozen Gemma-2-2B text encoder. Metadata enters
through a geometry-aware encoder — spherical lift for coordinates, cyclical
encoding for dates — projected into the shared AdaLN modulation by a
**zero-initialised** graft, so the model is byte-identical to its text-to-image
initialisation at step 0 and metadata influence is learned rather than imposed.

## Checkpoint

| | |
|---|---|
| file | `{filename}` ({size_gib:.2f} GiB, fp32) |
| sha256 | `{sha256}` |
| training step | {step} |
| parameters | {n_params:,} across {n_tensors:,} tensors |

Load it with `flowsat.checkpoint.resolve_checkpoint`, which both `generate.py`
and the evaluation script use, or directly with `torch.load(..., weights_only=True)`.

## Citation

```bibtex
@inproceedings{{parihar2026flowsat,
  title     = {{FlowSat: Flow-Matching Diffusion Transformers with Metadata
               Conditioning for Satellite Image Generation}},
  author    = {{Parihar, Digvijay Singh and Mondal, Rishabh and Batra, Nipun}},
  booktitle = {{British Machine Vision Conference (BMVC)}},
  year      = {{2026}}
}}
```

Built on [Sana](https://github.com/NVlabs/Sana),
[DC-AE](https://github.com/mit-han-lab/efficientvit) and
[Gemma-2](https://huggingface.co/google/gemma-2-2b-it); the location encoding
follows [SatCLIP](https://github.com/microsoft/satclip). Trained on
[FMoW](https://github.com/fMoW/dataset), whose own licence governs the data.
"""


def inspect(path: Path) -> dict:
    """Load the state dict and confirm it is the model we think it is.

    Publishing a checkpoint that turns out to be an optimiser shard, or one
    missing the metadata graft, wastes everybody's bandwidth and is invisible
    until somebody tries to load it.
    """
    import torch

    print(f"[1/4] reading {path} ({path.stat().st_size / 2 ** 30:.2f} GiB)")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    groups = {
        "transformer": sum(1 for k in sd if k.startswith("transformer.")),
        "metadata encoder": sum(1 for k in sd if "external_metadata_encoder" in k),
        "metadata graft": sum(1 for k in sd if "metadata_mod" in k),
    }
    print(f"      {len(sd):,} tensors, {n_params:,} parameters")
    for name, count in groups.items():
        print(f"      {name:18s} {count:5,d} tensors")

    missing = [n for n, c in groups.items() if c == 0]
    if missing:
        sys.exit(f"[error] no {' or '.join(missing)} tensors in this file. That is "
                 f"not a complete FlowSat checkpoint — publishing it would give "
                 f"everyone a model that loads partially and generates noise.")

    dtypes = {str(v.dtype) for v in sd.values() if hasattr(v, "dtype")}
    print(f"      dtypes: {', '.join(sorted(dtypes))}")
    return {"n_params": n_params, "n_tensors": len(sd)}


def sha256(path: Path) -> str:
    print("[2/4] hashing")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="local checkpoint directory or weight file")
    p.add_argument("--repo-id", required=True, help="e.g. dsp81/flowsat-fmow-512")
    p.add_argument("--n-samples", default="10,000",
                   help="sample count quoted in the model card's results table")
    p.add_argument("--private", action="store_true")
    p.add_argument("--card-out", default="MODEL_CARD.md",
                   help="where to write the rendered model card")
    p.add_argument("--dry-run", action="store_true",
                   help="check everything and render the card, upload nothing")
    a = p.parse_args()

    ck = Path(a.checkpoint)
    weights = ck if ck.is_file() else next(
        (ck / n for n in ("model_0.pt", "model.safetensors", "model.pt")
         if (ck / n).exists()), None)
    if weights is None:
        sys.exit(f"[error] no weight file in {ck}")

    info = inspect(weights)
    digest = sha256(weights)
    print(f"      sha256 {digest}")

    step = "".join(c for c in weights.parent.name if c.isdigit()) or "unknown"
    card = CARD.format(repo_id=a.repo_id, filename=weights.name,
                       size_gib=weights.stat().st_size / 2 ** 30,
                       sha256=digest, step=step, n_samples=a.n_samples,
                       n_params=info["n_params"], n_tensors=info["n_tensors"])

    # Deliberately not next to the weights: a checkpoint directory is often on
    # shared storage somebody else owns.
    card_path = Path(a.card_out)
    card_path.write_text(card)
    print(f"[3/4] model card -> {card_path}")

    if a.dry_run:
        print("[4/4] --dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print(f"[4/4] uploading as {who['name']} -> {a.repo_id}")
    api.create_repo(a.repo_id, repo_type="model", private=a.private,
                    exist_ok=True)
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                    repo_id=a.repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=str(weights), path_in_repo=weights.name,
                    repo_id=a.repo_id, repo_type="model")
    print(f"\ndone: https://huggingface.co/{a.repo_id}")
    if a.private:
        print("The repo is PRIVATE. Make it public before the paper goes out.")


if __name__ == "__main__":
    main()
