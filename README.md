<div align="center">

# FlowSat

### Flow-Matching Diffusion Transformers with Metadata Conditioning for Satellite Image Generation

**BMVC 2026**

[**Project Page**](https://dsp81.github.io/flowsat-satellite-image/) &nbsp;|&nbsp;
[**Paper**](https://dsp81.github.io/flowsat-satellite-image/) &nbsp;|&nbsp;
[**Quick Start**](#quick-start) &nbsp;|&nbsp;
[**Reproduce the Results**](docs/EVALUATION.md) &nbsp;|&nbsp;
[**Use on Your Own Dataset**](docs/NEW_DATASET.md) &nbsp;|&nbsp;
[**Captioning**](docs/CAPTIONING.md)

Digvijay Singh Parihar · Rishabh Mondal · Nipun Batra

Sustainability Lab, IIT Gandhinagar

</div>

---

> **Status.** Paper accepted at BMVC 2026. Code and pretrained weights are
> public: the weights are on the Hub at
> [`Djisgod/flowsat-fmow-512`](https://huggingface.co/Djisgod/flowsat-fmow-512).
> The dataset-adapter interface and pipeline documentation are in
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
| FlowSat (ours, as submitted) | 31.10 | 0.3016 | 20 |
| **FlowSat (ours, this release)** | **28.74** | **0.3019** | **20** |

Three-seed variance, as submitted: FID 31.53 ± 0.32, CLIP 0.3018 ± 0.0007.

**On the two FlowSat rows.** The released checkpoint and the evaluation code in
this repository give FID 28.74, not the 31.10 printed in the paper. Both are
listed rather than one quietly replacing the other. The gap is not the
evaluation code: three independent runs on the same checkpoint, captions and
protocol — `evaluate_fmow.py` on its defaults (28.74), the same script on the
paper-submission text-encoder path (28.72), and the original unreleased script
that produced the submitted number (28.72) — agree to within 0.02 FID. What
differed in the submitted run has not been identified. Reproduce 28.74; see
[docs/EVALUATION.md](docs/EVALUATION.md).

FlowSat's paired reconstruction metrics on the same run: **SSIM 0.1564**,
**LPIPS 0.6574**. These score each generated image against the one real image
whose caption and metadata produced it, so they measure conditioning fidelity
rather than image quality — two *real* FMoW acquisitions of the same place score
SSIM 0.214 / LPIPS 0.425 through this pipeline, which is the ceiling, not 1.0.

> The step-count advantage derives from flow matching and the DC-AE/Sana
> backbone rather than from metadata conditioning; the metadata encoder's
> contribution is measured separately in the encoder ablation (see paper §5.3).

### Reproducing these numbers

All four metrics come from one script, whose defaults are the published protocol:

```bash
pip install -e ".[eval,data]"

python -m flowsat.evaluation.evaluate_fmow \
    --checkpoint      Djisgod/flowsat-fmow-512 \
    --pretrained_sana Efficient-Large-Model/Sana_600M_512px_diffusers \
    --fmow_test_root  /path/to/fmow-full/test \
    --caption_root    /path/to/fmow_captions_test \
    --output_dir      evaluations/flowsat-125k
```

10,000 test samples, 20 Euler steps, guidance 2.5, seed 42 — about an hour on
one A100. It writes `metrics.json` with the protocol it ran under and a hash of
it, and prints the measured numbers beside the published ones.

Add `--curve_at 5000,6000,7000,8000,9000` to read every metric at intermediate
sample counts as well. The points come from the same generation pass, so the
whole FID-vs-N curve costs nothing extra — useful because FID is biased upward
at small N and a number quoted without its N cannot be compared to anything.

**→ [`docs/EVALUATION.md`](docs/EVALUATION.md)** explains what each metric
measures here, what has to match before two runs can be compared at all, and the
two failure modes that silently produce a plausible-looking but meaningless
number.

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

> The pretrained weights are on the Hub — pass
> `Djisgod/flowsat-fmow-512` anywhere a checkpoint path is expected.

```bash
git clone https://github.com/dsp81/flowsat-satellite-image.git
cd flowsat-satellite-image
pip install -e .
```

Generate an image from a caption and metadata:

```bash
python generate.py \
    --ckpt  checkpoints/flowsat-fmow-512 \
    --prompt "An airport surrounded by dry farmland, long grey runway crossing the centre." \
    --lon 4.40 --lat 51.92 --gsd 0.5 --cloud 0 --year 2016 --month 7 --day 15 \
    --out sample.png
```

Sweep a single metadata field with caption and noise held fixed — the
controllability demonstration from the paper:

```bash
for m in 1 3 5 7 9 11; do
  python generate.py \
      --ckpt checkpoints/flowsat-fmow-512 \
      --prompt "A farmland in a temperate river valley." \
      --lon -0.38 --lat 39.47 --month $m --seed 1234 \
      --out sweeps/month/$m.png
done
```

The seed is pinned, so the only thing changing across the six frames is the
month. That is the sweep shown on the project page.

## Repository layout

```
flowsat-satellite-image/
├── generate.py              # single-image entry point
├── flowsat/
│   ├── models/              # SatSana backbone + metadata encoders
│   ├── data/                # dataset adapters and metadata normalisation
│   ├── flow/                # flow-matching loss and samplers
│   ├── training/            # training entry point
│   ├── inference/           # pipeline wrapper
│   └── evaluation/          # FID / CLIP / SSIM / LPIPS, and the metric spine
└── docs/
    ├── EVALUATION.md        # ← reproduce the reported numbers
    ├── NEW_DATASET.md       # ← plug in your own dataset
    ├── CAPTIONING.md
    ├── METADATA_CONTROLLABILITY.md
    └── index.html           # project page
```

## Pretrained weights

The checkpoint is 2.3 GiB, past every GitHub limit, so it is distributed through
the Hugging Face Hub rather than this repository. It is live at
**[`Djisgod/flowsat-fmow-512`](https://huggingface.co/Djisgod/flowsat-fmow-512)**
(2.28 GiB, fp32, sha256 `b9b64439…0ffb972`). Both entry points take the repo id
wherever they take a path and download once into the usual cache:

```bash
python generate.py --ckpt Djisgod/flowsat-fmow-512 --prompt "..."
python -m flowsat.evaluation.evaluate_fmow --checkpoint Djisgod/flowsat-fmow-512 ...
```

`tools/publish_weights.py` is what uploads them: it checks that the state dict
is a complete FlowSat checkpoint before sending anything, records the sha256 and
renders the model card.

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
  author    = {Parihar, Digvijay Singh and Mondal, Rishabh and Batra, Nipun},
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
