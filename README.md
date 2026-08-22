<div align="center">

# FlowSat

### Flow-Matching Diffusion Transformers with Metadata Conditioning for Satellite Image Generation

**BMVC 2026**

[**Project Page**](https://sustainability-lab.github.io/flowsat-satellite-image/) &nbsp;|&nbsp;
[**Paper**](https://sustainability-lab.github.io/flowsat-satellite-image/) &nbsp;|&nbsp;
[**Quick Start**](#quick-start) &nbsp;|&nbsp;
[**Use on Your Own Dataset**](docs/NEW_DATASET.md) &nbsp;|&nbsp;
[**Captioning**](docs/CAPTIONING.md)

<!-- TODO(camera-ready): confirm author order and affiliations before 28 Aug -->
Digvijay Singh Parihar · Rishabh · Nipun Batra · Shanmuganathan Raman

Sustainability Lab, IIT Gandhinagar

</div>

---

> **Status.** Paper accepted at BMVC 2026. Code and pretrained weights are being
> prepared for public release ahead of the conference (November 2026). The
> dataset-adapter interface and pipeline documentation are already available in
> [`docs/`](docs/) — see [Use on Your Own Dataset](docs/NEW_DATASET.md).

---

## Overview

FlowSat generates satellite imagery conditioned on **text** *and* on the
**acquisition metadata** that defines how an image was captured — where on Earth
(longitude/latitude), at what ground resolution (GSD), on what date, under what
cloud cover.

Most text-to-image models treat such metadata, if they use it at all, as an
undifferentiated vector folded into the timestep embedding. FlowSat instead
respects each field's **geometry**: coordinates are lifted onto the sphere,
periodic fields are encoded cyclically, and scalars are encoded on their natural
scale. The resulting embedding is injected through a **zero-initialised AdaLN
graft**, so the model is byte-identical to its text-to-image initialisation at
step 0 and metadata influence is *learned* rather than imposed.

**Why it matters.** Metadata is the dial that text cannot turn. Two images of the
same place differ by season, sensor resolution, and atmosphere — properties a
caption rarely states and a user often wants to control directly.

### Key ideas

| | |
|---|---|
| **Flow matching, not DDPM** | Velocity-prediction training on a natively flow-matched backbone; high-quality samples in ~20 steps. |
| **Geometry-aware metadata encoder** | Spherical lift for coordinates, cyclical encoding for dates, scale-appropriate encoding for GSD/cloud. ~1.0M parameters — an order of magnitude smaller than a naive per-field MLP baseline, and better. |
| **Zero-initialised AdaLN graft** | Metadata enters the shared modulation pathway with a zero-init projection: no perturbation of the pretrained prior at initialisation. |
| **Efficient backbone** | Sana-0.6B with linear attention, DC-AE 32× latents (16×16×32 at 512 px), Gemma-2-2B text encoder. |

## Results

On FMoW-RGB (512 px):

| Model | FID ↓ | CLIP ↑ | Sampling steps |
|---|---|---|---|
| DiffusionSat (ICLR'24) | 35.27 | 0.1720 | 100 |
| GeoDiT-2Σ | 32.11 | — | — |
| **FlowSat (ours)** | **31.10** | **0.3016** | **20** |

Three-seed variance: FID 31.53 ± 0.32, CLIP 0.3018 ± 0.0007.

> The step-count advantage derives from flow matching and the DC-AE/Sana
> backbone rather than from metadata conditioning; the metadata encoder's
> contribution is measured separately in the encoder ablation (see paper §5.3).

## Architecture

```
        caption ──► Gemma-2-2B (frozen) ──────────► cross-attention (28 blocks)
                                                             │
   metadata (7) ──► geometry-aware encoder ──► zero-init ──► AdaLN modulation
   lon lat gsd            (~1.0M params)         graft              │
   cloud y m d                                                      ▼
                                                       Sana-0.6B DiT (28 blocks)
                                                       linear attn · AdaLN-single
                                                                    │
        512×512 image ◄── DC-AE decoder ◄── 16×16×32 latent ◄────────┘
                            (frozen)         flow matching, 20 Euler steps
```

## Quick start

> Code release in progress. The commands below reflect the intended public
> interface and will work against the released package.

```bash
git clone https://github.com/sustainability-lab/flowsat-satellite-image.git
cd flowsat-satellite-image
conda env create -f environment.yml && conda activate flowsat
```

Generate an image from a caption and metadata:

```bash
python -m flowsat.generate \
    --ckpt  checkpoints/flowsat-fmow-512 \
    --prompt "An airport surrounded by dry farmland, long grey runway crossing the centre." \
    --lon 4.40 --lat 51.92 --gsd 0.5 --cloud 0 --date 2016-07-15 \
    --out sample.png
```

Sweep a single metadata field with caption and noise held fixed — the
controllability demonstration from the paper:

```bash
python -m flowsat.sweep \
    --ckpt checkpoints/flowsat-fmow-512 \
    --prompt "A farmland in a temperate river valley." \
    --lon -0.38 --lat 39.47 --axis month --values 1,3,5,7,9,11 \
    --out sweeps/month/
```

## Repository layout

```
flowsat/
├── flowsat/
│   ├── models/          # SatSana backbone + metadata encoders
│   ├── data/            # dataset adapters and metadata normalisation
│   ├── flow/            # flow-matching loss and samplers
│   ├── training/        # training entry points
│   └── evaluation/      # FID / CLIP / controllability metrics
├── docs/
│   ├── NEW_DATASET.md   # ← plug in your own dataset
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   └── index.html       # project page
├── configs/
└── scripts/
```

## Using FlowSat on your own dataset

FlowSat is not tied to FMoW. Any dataset that can supply **(image, caption,
metadata)** triples can be plugged in by implementing a single adapter class.

**→ [`docs/NEW_DATASET.md`](docs/NEW_DATASET.md)** gives the interface contract,
a copy-paste template, the metadata normalisation convention, and step-by-step
commands to train and evaluate on a new corpus.

## Citation

```bibtex
@inproceedings{parihar2026flowsat,
  title     = {FlowSat: Flow-Matching Diffusion Transformers with Metadata
               Conditioning for Satellite Image Generation},
  author    = {Parihar, Digvijay Singh and Rishabh and Batra, Nipun and
               Raman, Shanmuganathan},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

## Acknowledgements

Built on [Sana](https://github.com/NVlabs/Sana) (backbone),
[DC-AE](https://github.com/mit-han-lab/efficientvit) (latent autoencoder), and
[Gemma-2](https://huggingface.co/google/gemma-2-2b-it) (text encoder). The
geometry-aware location encoding follows
[SatCLIP](https://github.com/microsoft/satclip). Evaluation compares against
[DiffusionSat](https://github.com/samar-khanna/DiffusionSat).

## License

<!-- TODO: confirm with the lab before public release -->
Released under the MIT License. See [`LICENSE`](LICENSE).
