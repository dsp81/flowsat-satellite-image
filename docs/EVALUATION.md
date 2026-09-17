# Reproducing the evaluation

This is the exact procedure behind the FMoW-RGB row in the paper:

| | FID ↓ | CLIP ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| FlowSat, as submitted | 31.10 | 0.3016 | 0.1600 | 0.6853 |
| **FlowSat, this release** | **28.74** | **0.3019** | **0.1564** | **0.6574** |

**Read the second row.** The released checkpoint and the code in this repository
produce FID 28.74, not the 31.10 printed in the paper. Both are given here
rather than one.

Measured on 10,000 FMoW test samples at 512 px, 20 Euler steps, text guidance
2.5, seed 42. One script produces all four numbers:

```bash
python -m flowsat.evaluation.evaluate_fmow \
    --checkpoint      Djisgod/flowsat-fmow-512 \
    --pretrained_sana Efficient-Large-Model/Sana_600M_512px_diffusers \
    --fmow_test_root  /path/to/fmow-full/test \
    --caption_root    /path/to/fmow_captions_test \
    --output_dir      evaluations/flowsat-125k
```

Every other flag defaults to the published setting, so there is nothing else to
match. The script prints its protocol and a hash of it, writes `metrics.json`,
and finishes by printing the measured numbers next to the published ones.

## What you need

```
pip install -e ".[eval,data]"
```

`eval` brings in torchmetrics, torch-fidelity and scipy (FID, Inception Score,
LPIPS, SSIM). `data` brings in rasterio, which reads FMoW's GeoTIFFs — without
it the loader falls back to PIL and the 16-bit files come out nearly black, so
install it.

Four inputs:

| | |
|---|---|
| **Checkpoint** | The released FlowSat weights. A directory holding `model_0.pt`. |
| **Sana snapshot** | `Efficient-Large-Model/Sana_600M_512px_diffusers` — supplies the DC-AE, the Gemma-2 tokenizer and the text encoder. Pass a local path if you are offline. |
| **FMoW test split** | The RGB test split, one directory per sequence: `<seq_id>/<stem>_rgb.tif` beside `<stem>_rgb.json`. |
| **Captions** | One `.txt` per image, mirroring that layout under `<caption_root>/test/<seq_id>/<stem>.txt`. These are the VLM captions described in [Captioning a dataset](CAPTIONING.md); the numbers above were measured with them. |

A 10,000-sample run takes about an hour on one A100, dominated by sampling.
Read the test images from local disk rather than network storage — streaming
GeoTIFFs over NFS inside the loop starves the data loader and can stall a run
outright.

## What the four metrics mean

**FID** and **CLIP** are the generation-quality metrics. FID compares the
distribution of the 10,000 generated images against the distribution of the
10,000 real images they were conditioned on. CLIP score is the cosine
similarity between each generated image and its caption, under
`openai/clip-vit-base-patch16`. The script reports it ×100 (30.19); the tables
on this page report the fraction (0.3019).

**SSIM** and **LPIPS** are *paired*: each generated image is scored against the
one specific real image whose caption and metadata produced it. They measure
whether the conditioning recovers that particular acquisition — not image
quality. Read them against a floor measured the same way:

| | SSIM | LPIPS |
|---|---|---|
| FlowSat vs. its source image | 0.156 | 0.657 |
| **Two real images of the same place** | **0.214** | **0.425** |

Two genuine FMoW acquisitions of one location, taken months apart, only reach
SSIM 0.214 through this same preprocessing. That is the ceiling, not 1.0.
Quoting either number as generation quality overstates what it measures.

**Inception Score** is also reported. Inception is ImageNet-trained and
overhead imagery sits far outside that domain, so treat it as a weak signal.
FID is primary.

## What makes two runs comparable

FID is a property of a protocol as much as of a model. These fields all move it,
several of them by more than the gap between two models, so the script hashes
them into `protocol_hash` and writes the hash into `metrics.json`. **If two runs
disagree on that hash, their FID cannot be compared**, whatever else is true.

| field | published value |
|---|---|
| `num_samples` | 10000 |
| `num_steps` | 20 |
| `guidance_scale` | 2.5 |
| `uncond_metadata` | `zero` |
| `max_caption_len` | 256 |
| `resolution` | 512 |
| `seed` | 42 |
| `dtype` | `bf16` |
| reference set | FMoW test split, first 10,000 captioned samples in `(seq_id, stem)` order |
| preprocessing | 2–98 percentile stretch, centre crop, bilinear resize to 512 px |

Sample count deserves particular care. FID fits a 2048-dimensional covariance;
below a few thousand samples that estimate is rank-deficient and inflated. FID
between two halves of a *single identical pool* measures 189 at N = 66. Small-N
runs are useful for ranking checkpoints against each other and for nothing else.
The script warns when `--num_samples` is below 10,000.

You can see that dependence directly. `--curve_at` reads every metric at
intermediate sample counts during a single generation pass:

```bash
python -m flowsat.evaluation.evaluate_fmow ... \
    --num_samples 10000 --curve_at 5000,6000,7000,8000,9000
```

Each point costs one metric read, not another pass — the metrics accumulate
state and reading them does not disturb it, so the whole curve falls out of the
run that produces the headline number. The curve lands in `metrics.json` under
`curve`, and the shape to expect is FID *falling* as N grows, because the bias
is upward at small N. That is worth internalising before comparing against a
published FID measured at an unstated N: a lower number is not automatically a
better model.

This is the curve the released checkpoint actually produces, every row read from
the same generation pass:

| N scored | FID ↓ | CLIP ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|---:|
| 5,007 | 35.93 | 0.3021 | 0.1585 | 0.6575 |
| 6,007 | 33.64 | 0.3017 | 0.1581 | 0.6579 |
| 7,007 | 31.93 | 0.3017 | 0.1574 | 0.6582 |
| 8,007 | 30.43 | 0.3019 | 0.1575 | 0.6568 |
| 9,007 | 29.42 | 0.3018 | 0.1578 | 0.6572 |
| **9,999** | **28.74** | **0.3019** | **0.1564** | **0.6574** |


## Which text encoder

Gemma-2 can be loaded two ways, and they give measurably different numbers. The
choice is a flag, and it is part of the protocol hash:

```bash
--text_encoder bf16-causal   # what the paper submission used
--text_encoder fp32-eager    # the default; matches training
```

| | `bf16-causal` | `fp32-eager` |
|---|---|---|
| class | `AutoModelForCausalLM` | `AutoModel` |
| dtype | the run dtype (bf16) | float32 |
| attention | `eager` | `eager` |
| read-out | `hidden_states[-1]` | `last_hidden_state` |
| mask | the tokenizer's | built from pad ids, all-pad rows guarded |

Measured on transformers 4.49, three of those five rows turn out not to matter:
`AutoModel` and `AutoModelForCausalLM` return **bit-identical** hidden states
(max difference 0.0), `hidden_states[-1]` *is* `last_hidden_state`, and with
`padding="max_length"` the two masks agree except on an all-pad row. **The only
difference that actually moves the numbers is the dtype.**


Both paths request eager attention explicitly, and that is load-bearing.
Gemma-2 soft-caps its attention logits, and **sdpa + soft-capping in bf16
returns 100% NaN** on this model — NaN conditioning decodes to a black image
rather than raising, so the metrics would be computed on black frames. Older
transformers defaulted Gemma-2 to eager for exactly this reason; 4.49 defaults
to sdpa. The original run produced sane metrics, so it cannot have been on sdpa.
Leaving the attention implementation to the library is what broke, not the
choice of head or dtype.

Measured directly, at 256 tokens on an A100:

| load | result |
|---|---|
| `AutoModelForCausalLM`, bf16, **sdpa** (the 4.49 default) | **all NaN** |
| `AutoModelForCausalLM`, bf16, eager | finite, absmax 75.5, std 3.947 |
| `AutoModelForCausalLM`, fp32, eager | finite, absmax 75.7, std 3.944 |
| `AutoModel`, fp32, eager | identical to the row above |

`fp32-eager` is the default because it matches training and avoids half
precision in a model that soft-caps. The run aborts on non-finite conditioning
either way rather than reporting a number measured on black images.

If you are checking the published row, use `bf16-causal`. If you are measuring a
new model, use the default and say so.

## Two things that will silently ruin a run

**Black images.** Gemma-2 soft-caps its attention logits, and in half precision
off the eager attention path that overflows to NaN — NaN conditioning, NaN
latents, and images that decode to pure black. Nothing raises; the metrics
score the black frames happily. The script therefore loads the text encoder in
fp32 with `attn_implementation="eager"` (matching training), casts the
conditioning to the transformer's dtype at the boundary, and aborts if the
first batch comes back NaN or all-black. The tell-tale signature, if it ever
reappears, is **Inception Score exactly 1.0** on a run large enough that it
should not be — identical inputs, zero KL — together with FID stuck near 750
regardless of sample count. (A handful of images give IS 1.0 for ordinary
reasons; it is only diagnostic at scale.) Look at the images before believing
any metric that pairs those two.

**Partial checkpoint loads.** The backbone is built from the real Sana config,
not a hand-written one, because the two differ slightly and `strict=False` would
quietly discard the mismatched tensors and leave randomly initialised weights in
place. The script aborts if more than a handful of state-dict keys are missing
or unexpected, and specifically if any metadata-pathway key is missing — that
one would produce a complete, plausible-looking run describing a metadata-blind
model.

## Notes on the test split

FMoW's **test** sidecars carry `gsd`, `cloud_cover` and `timestamp`, but no
coordinates — unlike the train sidecars they have neither `raw_location` nor a
`bounding_box` polygon. Longitude and latitude therefore fall back to (0, 0) for
every test sample. This is identical for every model scored through this
pipeline, so the comparison is fair, but it does mean the quality table
exercises the date, GSD and cloud-cover pathways and not the geographic one.
Geographic control is measured separately, by the controllability sweeps on the
project page.

Expect **9,999** samples scored out of 10,000 requested: one GeoTIFF in that
prefix of the split is unreadable. The script counts and reports dropped samples
rather than failing on them — a corrupt file should not end a two-hour run.

The sample list is cached under `~/.cache/flowsat/` because enumerating the
split is slow on network storage. Set `FLOWSAT_NO_SAMPLE_CACHE=1` to force a
rescan after adding or removing captions.

## Getting the weights

`--checkpoint` takes a local directory, a local weight file, or a Hugging Face
repo id:

```bash
--checkpoint Djisgod/flowsat-fmow-512        # downloads once into the HF cache
--checkpoint Djisgod/flowsat-fmow-512@v1.0   # a specific revision
--checkpoint checkpoints/flowsat-fmow-512  # a local directory
```

The checkpoint is 2.3 GiB, which is past every GitHub limit (100 MB per git
file, 2 GiB per release asset, 1 GB on the free LFS tier), so the Hub is where
it will live; the upload happens with the weights release, and until then only
a local path works. `tools/publish_weights.py` is what puts it there: it verifies the
state dict is a complete FlowSat checkpoint before uploading anything, records
the sha256, and renders the model card.

## Evaluating a different model

`flowsat/evaluation/eval_common.py` holds everything that decides what a number
means: sample discovery and ordering, the image preprocessing path, and the
metric definitions. Scoring a baseline through that module makes it comparable
to the table above; scoring it any other way does not. If you change anything in
that file, every number on this page becomes incomparable, including the
published ones.
