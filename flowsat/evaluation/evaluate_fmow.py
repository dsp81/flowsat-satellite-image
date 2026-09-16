#!/usr/bin/env python3
"""
evaluate_fmow.py — reproduce the FlowSat FMoW-RGB numbers reported in the paper.

    FID 31.10   CLIP 0.3016   SSIM 0.1600   LPIPS 0.6853
    (N = 10,000 test samples, 20 Euler steps, text guidance 2.5, seed 42)

Every default in this file is the value that produced those numbers. Running it
with no overrides beyond the four paths re-measures the published row:

    python -m flowsat.evaluation.evaluate_fmow \
        --checkpoint      checkpoints/flowsat-fmow-512 \
        --pretrained_sana Efficient-Large-Model/Sana_600M_512px_diffusers \
        --fmow_test_root  /path/to/fmow-full/test \
        --caption_root    /path/to/fmow_captions_test \
        --output_dir      evaluations/flowsat-125k

It writes `metrics.json` (with the protocol it ran under and a hash of it) and
prints the measured numbers beside the published ones.

--------------------------------------------------------------------------
WHAT THE FOUR METRICS MEAN HERE
--------------------------------------------------------------------------
FID and CLIP are the generation-quality metrics. FID compares the distribution
of 10,000 generated images against the distribution of the 10,000 real test
images they were conditioned on; CLIP measures how well each generated image
matches the caption it was given.

SSIM and LPIPS are PAIRED: each generated image is scored against the specific
real image whose caption and metadata produced it. They answer "does the
conditioning recover this particular acquisition", not "is this a good image".
Read them only against a floor measured the same way -- two *real* FMoW images
of the same place score SSIM 0.214 / LPIPS 0.425 through this exact
preprocessing path, which is the ceiling any generative model is working
towards, not 1.0 / 0.0. Never quote them as image quality.

--------------------------------------------------------------------------
COMPARABILITY
--------------------------------------------------------------------------
FID is a property of a protocol as much as of a model. Reference set, sample
count, sampler, step count, guidance scale, caption length and the metadata
convention in the unconditional branch all move it, some of them by more than
the gap between two models. The `protocol_hash` written into metrics.json is
taken over exactly those fields: if two runs disagree on it, their FID cannot
be compared, whatever else is true.

Two specific traps, both of which have bitten this project:

  * Sample count. FID fits a 2048-dimensional covariance. Below a few thousand
    samples that estimate is rank-deficient and badly inflated -- FID between
    two halves of one identical pool measures 189 at N = 66. Small-N runs rank
    checkpoints; they do not produce quotable numbers. Keep N >= 10,000.

  * The text encoder's precision. Gemma-2's attention soft-capping overflows in
    bf16 unless eager attention is used, producing NaN hidden states, NaN
    latents and pure black images -- which the metrics will happily score. The
    encoder is therefore loaded in fp32 with `attn_implementation="eager"`,
    matching training, and the run aborts on the first NaN or all-black batch
    rather than reporting a number computed on nothing. The signature of that
    failure, if it ever reappears, is Inception Score exactly 1.0 on a run
    large enough that it should not be (a handful of images give IS 1.0 for
    ordinary reasons) together with an FID near 750 whatever N is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from flowsat.checkpoint import resolve_checkpoint
from flowsat.evaluation.eval_common import (
    TestSample, compute_final_metrics, extract_metadata_vector,
    load_test_samples, pil_to_uint8_tensor, read_real_image, setup_clip_scorer,
    setup_metrics, uint8_tensor_to_pils, update_distribution_metrics,
    update_paired_metrics,
)

logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger("flowsat.eval")

# The published row, for the side-by-side print at the end. These came from
# checkpoint-125000 under the defaults below, scoring 9,999 of the 10,000
# requested samples (one test GeoTIFF in that prefix is unreadable).
PUBLISHED = {"fid": 31.10, "clip_score": 30.16, "ssim": 0.1600,
             "lpips": 0.6853, "is_mean": 7.19}

# Measured on 1,500 pairs of *real* FMoW images of the same location through
# this module's preprocessing. The floor for the paired metrics.
REAL_VS_REAL = {"ssim": 0.2138, "lpips": 0.4251}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class EvalSampleDataset(Dataset):
    """One worker, one sample: read the GeoTIFF, stretch it, crop it, resize it.

    All of that is slow enough that doing it on the main thread between batches
    caps the whole run at a fraction of the GPU's throughput. Items that fail to
    load return None and are dropped by the collate function, which counts them.
    """

    def __init__(self, samples: List[TestSample], resolution: int = 512):
        self.samples = samples
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        s = self.samples[idx]
        try:
            real_pil = read_real_image(s.image_path, resolution=self.resolution)
            if real_pil is None:
                return None
            caption = s.caption
            if not caption:
                return None
            return {
                "real_uint8": pil_to_uint8_tensor(real_pil),
                "metadata": torch.tensor(extract_metadata_vector(s.metadata),
                                         dtype=torch.float32),
                "caption": caption,
                "seq_id": s.seq_id,
                "stem": s.stem,
            }
        except Exception as e:
            logger.debug(f"loader error on {s.image_path}: {e}")
            return None


def eval_collate(batch):
    valid = [b for b in batch if b is not None]
    dropped = len(batch) - len(valid)
    if not valid:
        return {"_empty": True, "n_dropped": dropped}
    return {
        "real_uint8": torch.stack([b["real_uint8"] for b in valid]),
        "metadata": torch.stack([b["metadata"] for b in valid]),
        "captions": [b["caption"] for b in valid],
        "seq_ids": [b["seq_id"] for b in valid],
        "stems": [b["stem"] for b in valid],
        "n_dropped": dropped,
        "_empty": False,
    }


# ---------------------------------------------------------------------------
# Model and encoders
# ---------------------------------------------------------------------------

def load_model(args, device, dtype):
    """Build FlowSat and load the checkpoint, refusing a partial load.

    `load_pretrained=True` is required: it takes the architecture from the real
    Sana config. Building from the manual config gives a slightly different
    network, and `strict=False` would then quietly drop the mismatched tensors,
    leaving randomly initialised weights behind -- a full run reporting metrics
    on noise. The key-count guard turns that into an error instead.
    """
    from flowsat.models.sat_sana import SatSana

    logger.info(f"building SatSana from {args.pretrained_sana}")
    model = SatSana(latent_size=args.resolution // 32, in_channels=32,
                    num_metadata=7, use_metadata=True,
                    pretrained_sana_id=args.pretrained_sana, load_pretrained=True)
    if not args.no_satclip:
        from flowsat.models.sat_clip import SatCLIPMetadataEncoder
        model.set_metadata_encoder(
            SatCLIPMetadataEncoder(embed_dim=model.embed_dim, num_metadata=7))

    weights = resolve_checkpoint(args.checkpoint)
    logger.info(f"loading weights from {weights}")
    if weights.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(weights))
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")
    for k in missing:
        if "metadata_mod" in k or "external_metadata_encoder" in k:
            sys.exit(f"[fatal] metadata pathway key missing: {k}\n"
                     f"        The metadata graft did not load, so every number "
                     f"below would describe a metadata-blind model. Pass "
                     f"--no-satclip if this checkpoint was trained without the "
                     f"geometry-aware encoder.")
    if len(missing) > args.load_guard or len(unexpected) > args.load_guard:
        logger.error(f"missing[:6]    = {missing[:6]}")
        logger.error(f"unexpected[:6] = {unexpected[:6]}")
        sys.exit(f"[fatal] partial load (>{args.load_guard} keys). This is an "
                 f"architecture mismatch -- the metrics would be measured on "
                 f"partly random weights. Check that --pretrained_sana points at "
                 f"the same Sana snapshot used for training.")

    return model.to(device=device, dtype=dtype).eval()


def load_encoders(args, device, dtype):
    """Gemma-2 and the DC-AE.

    Two text-encoder paths, because they give measurably different numbers and
    the published row was measured on one of them:

    fp32-eager (default)
        AutoModel, float32, attn_implementation="eager", last_hidden_state.
        What training uses. Gemma-2 soft-caps its attention logits, and in half
        precision off the eager path that can overflow to NaN -- silently, since
        NaN conditioning decodes to a black image rather than raising.

    bf16-causal
        AutoModelForCausalLM at the run dtype, eager attention,
        hidden_states[-1], and the tokenizer's own mask. This is what
        eval_sana.py did, and therefore what the **paper submission** numbers
        were measured with. Kept so those numbers stay reproducible.

        Measured on transformers 4.49: the only difference that remains against
        fp32-eager is the dtype. AutoModel and AutoModelForCausalLM return
        bit-identical hidden states, and hidden_states[-1] is last_hidden_state.

    Either way the conditioning is cast to the transformer's dtype at the
    boundary.
    """
    from diffusers import AutoencoderDC
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_sana,
                                              subfolder="tokenizer")
    if args.text_encoder == "bf16-causal":
        logger.info(f"loading Gemma-2 text encoder (AutoModelForCausalLM, "
                    f"{args.dtype}, default attention) -- the paper-submission path")
        # eager is requested rather than left to the library. transformers
        # used to default Gemma-2 to eager precisely because of the attention
        # soft-capping; 4.49 defaults to sdpa, and sdpa + soft-capping in bf16
        # returns 100% NaN here. The original run produced sane metrics, so it
        # cannot have been on sdpa -- eager is what it effectively used.
        text_encoder = AutoModelForCausalLM.from_pretrained(
            args.pretrained_sana, subfolder="text_encoder", torch_dtype=dtype,
            attn_implementation="eager").to(device).eval()
    else:
        logger.info("loading Gemma-2 text encoder (AutoModel, fp32, eager attention)")
        text_encoder = AutoModel.from_pretrained(
            args.pretrained_sana, subfolder="text_encoder",
            attn_implementation="eager").to(device, torch.float32).eval()

    logger.info("loading DC-AE")
    vae = AutoencoderDC.from_pretrained(args.pretrained_sana, subfolder="vae",
                                        torch_dtype=dtype).to(device).eval()
    return tokenizer, text_encoder, vae


@torch.no_grad()
def encode_captions(captions, tokenizer, text_encoder, device, max_length,
                    mode="fp32-eager"):
    """Tokenise and encode. The two modes differ in mask and readout as well as
    in dtype, so both halves have to match the path being reproduced."""
    tok = tokenizer(captions, padding="max_length", max_length=max_length,
                    truncation=True, return_tensors="pt").to(device)
    if mode == "bf16-causal":
        # eval_sana.py exactly: the tokenizer's own mask, and the last entry of
        # hidden_states off the causal-LM head.
        out = text_encoder(input_ids=tok.input_ids,
                           attention_mask=tok.attention_mask,
                           output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1], tok.attention_mask
    ids = tok.input_ids
    mask = (ids != tokenizer.pad_token_id).long()
    # An all-pad row (the empty unconditional caption can produce one) makes the
    # attention softmax divide by zero, so force its first position visible.
    empty = mask.sum(dim=1) == 0
    if empty.any():
        mask[empty, 0] = 1
    return text_encoder(ids, attention_mask=mask)[0], mask


def check_truncation(captions, tokenizer, max_length) -> None:
    lens = [len(x) for x in tokenizer(captions, truncation=False)["input_ids"]]
    over = [n for n in lens if n > max_length]
    if over:
        logger.warning(f"{len(over)}/{len(lens)} sampled captions exceed "
                       f"--max_caption_len={max_length} (longest {max(over)}) "
                       f"and are being truncated. This changes CLIP score; the "
                       f"published numbers used {max_length}.")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_latents(model, cond_hidden, cond_mask, uncond_hidden, uncond_mask,
                   metadata, args, seed, device, dtype, latent_shape):
    """Euler integration of the flow ODE from t = 1 (noise) to t = 0 (image).

    Two-branch classifier-free guidance, both branches in one forward pass.

    The unconditional branch is given a *zeroed* metadata vector, which is the
    convention the published numbers were measured under. Note that zero is not
    "unknown" here: after normalisation it decodes to lon -180, lat -90, year
    1980. `--uncond_metadata real` instead passes the true metadata to both
    branches, guiding the text direction alone. That is arguably the better
    sampler -- `generate.py` uses it -- but it is a different sampler, and
    mixing the two across runs makes their FIDs incomparable.
    """
    B = metadata.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((B, *latent_shape), generator=gen, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, args.num_steps + 1, device=device, dtype=dtype)

    md_uncond = (torch.zeros_like(metadata) if args.uncond_metadata == "zero"
                 else metadata)
    emb = torch.cat([uncond_hidden, cond_hidden], 0).to(dtype)
    mask = torch.cat([uncond_mask, cond_mask], 0)
    md = torch.cat([md_uncond, metadata], 0)

    for i in range(args.num_steps):
        t_cur = ts[i].expand(B)
        dt = ts[i + 1] - ts[i]
        v = model(torch.cat([x, x], 0), t_cur.repeat(2), encoder_hidden_states=emb,
                  metadata=md, attention_mask=mask).sample
        v_uncond, v_cond = v.chunk(2, 0)
        x = x + (v_uncond + args.guidance_scale * (v_cond - v_uncond)) * dt
    return x


@torch.no_grad()
def decode_latents(vae, latents) -> torch.Tensor:
    """Latents -> (B, 3, H, W) uint8 on device, the form FID and IS expect."""
    out = vae.decode(latents / getattr(vae.config, "scaling_factor", 1.0))
    img = out.sample if hasattr(out, "sample") else out
    img = (img.clamp(-1.0, 1.0) + 1.0) / 2.0
    return (img.float() * 255.0).round().clamp(0, 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# Sample-count curve
# ---------------------------------------------------------------------------

def snapshot_metrics(metrics, clip_scorer, n_scored: int) -> Dict[str, Any]:
    """Read every metric at the current sample count without disturbing it.

    torchmetrics accumulates state and `compute()` does not reset it, so this
    can be called mid-run as often as wanted; the next `update()` invalidates
    the cached value. That makes the whole FID-vs-N curve fall out of a single
    generation pass instead of one full run per point -- which matters, because
    generating is hours and reading the metrics is seconds.
    """
    is_mean, is_std = metrics["is"].compute()
    return {
        "num_scored": n_scored,
        "fid": float(metrics["fid"].compute()),
        "clip_score": float(np.mean(clip_scorer.scores)) if clip_scorer.scores else float("nan"),
        "ssim": float(metrics["ssim"].compute()),
        "lpips": float(metrics["lpips"].compute()),
        "is_mean": float(is_mean),
        "is_std": float(is_std),
    }


def parse_curve_at(spec: str, num_samples: int) -> List[int]:
    """"5000,6000,...' or 'auto' -> the sample counts to report at."""
    if not spec:
        return []
    if spec == "auto":
        step = max(1000, num_samples // 10)
        pts = list(range(step, num_samples, step))
    else:
        pts = [int(x) for x in spec.replace(" ", "").split(",") if x]
    return sorted({p for p in pts if 0 < p < num_samples})


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def build_protocol(args) -> Dict[str, Any]:
    """The fields that must match for two runs' FID/CLIP to be comparable.

    Deliberately excludes the checkpoint: that is the thing being compared.
    Deliberately includes the caption root, because the captions are as much
    part of the protocol as the images.
    """
    return {
        "fmow_test_root": str(args.fmow_test_root),
        "caption_root": str(args.caption_root),
        "num_samples": args.num_samples,
        # Noise is seeded per batch (args.seed + batch index), so the batch size
        # decides which noise each sample gets. It changes the images, therefore
        # the metrics, and so it belongs in the hash.
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "uncond_metadata": args.uncond_metadata,
        "max_caption_len": args.max_caption_len,
        "seed": args.seed,
        "dtype": args.dtype,
        "text_encoder": args.text_encoder,
        "clip_model": "openai/clip-vit-base-patch16",
        "metric_backend": "torchmetrics (flowsat.evaluation.eval_common)",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    if device == "cpu":
        logger.warning("no GPU visible; a 10,000-sample run on CPU is not practical")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    img_root = out_root / "images"
    if args.save_images:
        img_root.mkdir(parents=True, exist_ok=True)

    protocol = build_protocol(args)
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    print("\n=== PROTOCOL (FID/CLIP are comparable only across matching hashes) ===")
    for k, v in protocol.items():
        print(f"  {k:18s} {v}")
    print(f"  {'protocol_hash':18s} {protocol_hash}")
    print(f"  {'checkpoint':18s} {args.checkpoint}   (not hashed: the thing being compared)\n")
    if args.num_samples < 10000:
        print(f"  !! --num_samples {args.num_samples} < 10000. FID is rank-deficient "
              f"and inflated below a few thousand samples; this run ranks "
              f"checkpoints, it does not produce a quotable number.\n")

    samples = load_test_samples(
        fmow_test_root=Path(args.fmow_test_root),
        caption_root=Path(args.caption_root),
        test_gt_mapping_path=(Path(args.test_gt_mapping)
                              if args.test_gt_mapping else None),
        limit=args.num_samples, require_caption=True)
    logger.info(f"found {len(samples):,} test samples with captions")
    if not samples:
        sys.exit("[fatal] no samples. Check --fmow_test_root and --caption_root; "
                 "captions are expected at <caption_root>/test/<seq_id>/<stem>.txt")
    if len(samples) < args.num_samples:
        logger.warning(f"only {len(samples):,} samples available but "
                       f"--num_samples={args.num_samples:,}. FID depends on N: "
                       f"record this before comparing against anything.")

    loader = DataLoader(EvalSampleDataset(samples, resolution=args.resolution),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=eval_collate, shuffle=False, pin_memory=True,
                        prefetch_factor=4 if args.num_workers > 0 else None,
                        persistent_workers=args.num_workers > 0)

    model = load_model(args, device, dtype)
    tokenizer, text_encoder, vae = load_encoders(args, device, dtype)
    latent_shape = (vae.config.latent_channels,
                    args.resolution // 32, args.resolution // 32)
    logger.info(f"latent shape {latent_shape}, "
                f"scaling_factor={getattr(vae.config, 'scaling_factor', 1.0)}")

    check_truncation([s.caption for s in samples[:200]], tokenizer,
                     args.max_caption_len)

    metrics = setup_metrics(device=device)
    clip_scorer = setup_clip_scorer(device=device)
    uncond_full, uncond_mask_full = encode_captions(
        [""] * args.batch_size, tokenizer, text_encoder, device,
        args.max_caption_len, args.text_encoder)
    if not torch.isfinite(uncond_full).all():
        sys.exit(f"[fatal] the text encoder produced NaN/Inf on the empty "
                 f"caption under --text_encoder {args.text_encoder}. Every image "
                 f"would decode to black and every metric below would be "
                 f"meaningless. This is the Gemma-2 half-precision soft-capping "
                 f"failure; re-run with --text_encoder fp32-eager, which does "
                 f"not take that path.")

    curve_at = parse_curve_at(args.curve_at, args.num_samples)
    curve: List[Dict[str, Any]] = []
    if curve_at:
        logger.info(f"will also report metrics at N = {curve_at}")

    n_done = n_dropped = 0
    t0 = time.time()
    with open(out_root / "progress.log", "a") as prog:
        for bi, batch in enumerate(loader):
            n_dropped += batch.get("n_dropped", 0)
            if batch.get("_empty"):
                continue

            real_u8 = batch["real_uint8"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device=device, dtype=dtype,
                                            non_blocking=True)
            captions = batch["captions"]
            B = real_u8.shape[0]

            cond_h, cond_m = encode_captions(captions, tokenizer, text_encoder,
                                             device, args.max_caption_len,
                                             args.text_encoder)
            latents = sample_latents(model, cond_h, cond_m, uncond_full[:B],
                                     uncond_mask_full[:B], metadata, args,
                                     args.seed + bi, device, dtype, latent_shape)
            fake_u8 = decode_latents(vae, latents)
            assert real_u8.shape[-2:] == fake_u8.shape[-2:], \
                f"shape mismatch: real {tuple(real_u8.shape)} fake {tuple(fake_u8.shape)}"

            # Catch a dead generation path on the first batch rather than after
            # eight hours of scoring black frames.
            if bi == 0 and int(fake_u8.max()) == 0:
                sys.exit("[fatal] the first batch decoded to pure black "
                         "(max pixel 0). Metrics on these images are worthless. "
                         "Check the text encoder precision and the checkpoint.")

            update_distribution_metrics(metrics, fake_u8, real_u8)
            update_paired_metrics(metrics, fake_u8.float() / 255.0,
                                  real_u8.float() / 255.0)
            fake_pils = uint8_tensor_to_pils(fake_u8)
            clip_scorer.update(fake_pils, captions)

            if args.save_images:
                for sid, stem, pil in zip(batch["seq_ids"], batch["stems"], fake_pils):
                    d = img_root / sid
                    d.mkdir(parents=True, exist_ok=True)
                    pil.save(d / f"{stem}.png")

            n_done += B
            while curve_at and n_done >= curve_at[0]:
                point = snapshot_metrics(metrics, clip_scorer, n_done)
                point["requested_at"] = curve_at.pop(0)
                curve.append(point)
                logger.info(f"[curve] N={point['num_scored']:,}  "
                            f"FID {point['fid']:.2f}  CLIP {point['clip_score']:.2f}  "
                            f"SSIM {point['ssim']:.4f}  LPIPS {point['lpips']:.4f}")
            if bi % 5 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-6)
                msg = (f"{n_done:,}/{len(samples):,} | {rate:.2f} img/s | "
                       f"dropped={n_dropped} | {elapsed / 60:.1f}min | "
                       f"eta={(len(samples) - n_done) / max(rate, 1e-6) / 60:.1f}min")
                logger.info(msg)
                prog.write(msg + "\n")
                prog.flush()

    logger.info("computing FID / CLIP / SSIM / LPIPS / IS ...")
    final = compute_final_metrics(metrics, clip_scorer)
    results = {
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "checkpoint": str(args.checkpoint),
        "num_scored": n_done,
        "num_dropped": n_dropped,
        "wall_time_seconds": time.time() - t0,
        **final,
        "notes": {
            "clip_score": "cosine x 100; the paper reports it as a fraction "
                          "(30.16 here = 0.3016 in the table).",
            "ssim_lpips": "PAIRED against each caption's own source image. Not "
                          "image quality. Real-vs-real floor through this same "
                          f"pipeline: SSIM {REAL_VS_REAL['ssim']:.4f}, "
                          f"LPIPS {REAL_VS_REAL['lpips']:.4f}.",
            "is_mean": "Inception is ImageNet-trained; IS is weakly meaningful "
                       "for overhead imagery. FID is primary.",
            "coordinates": "FMoW test sidecars carry no lat/lon, so the geo "
                           "fields are (0, 0) for every sample here. Identical "
                           "for every model scored this way.",
        },
    }
    if curve:
        curve.append({**final, "num_scored": n_done, "requested_at": args.num_samples})
        results["curve"] = curve
        results["notes"]["curve"] = (
            "Every point comes from one generation pass, read at increasing "
            "sample counts. FID is biased upward at small N, so the curve falls "
            "as N grows and a number measured at one N is not comparable to a "
            "number measured at another.")
    (out_root / "metrics.json").write_text(json.dumps(results, indent=2))
    logger.info(f"wrote {out_root / 'metrics.json'}")

    print(f"\n=== RESULTS  ({n_done:,} scored, {n_dropped} unreadable) ===")
    print(f"  {'':12s}{'measured':>12s}{'published':>12s}")
    for key, label, scale in (("fid", "FID", 1.0), ("clip_score", "CLIP", 1.0),
                              ("ssim", "SSIM", 1.0), ("lpips", "LPIPS", 1.0),
                              ("is_mean", "IS", 1.0)):
        print(f"  {label:12s}{final[key] * scale:12.4f}{PUBLISHED[key]:12.4f}")
    if curve:
        print(f"\n=== FID vs SAMPLE COUNT (one generation pass) ===")
        print(f"  {'N':>8s}{'FID':>10s}{'CLIP':>9s}{'SSIM':>9s}{'LPIPS':>9s}")
        for pt in curve:
            print(f"  {pt['num_scored']:8,d}{pt['fid']:10.2f}{pt['clip_score']:9.2f}"
                  f"{pt['ssim']:9.4f}{pt['lpips']:9.4f}")
        print("  FID is biased upward at small N, so this curve falls as N grows. "
              "Two FIDs\n  measured at different N are not comparable.")

    print(f"\n  CLIP in the paper's units: {final['clip_score'] / 100:.4f}")
    print(f"  Paired-metric floor (real vs real): "
          f"SSIM {REAL_VS_REAL['ssim']:.4f}, LPIPS {REAL_VS_REAL['lpips']:.4f}")
    print(f"  Compare FID/CLIP only against runs with protocol_hash "
          f"{protocol_hash}.\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce the FlowSat FMoW-RGB evaluation from the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    paths = p.add_argument_group("paths")
    paths.add_argument("--checkpoint", required=True,
                       help="checkpoint directory, weight file, or Hugging Face "
                            "repo id (e.g. dsp81/flowsat-fmow-512)")
    paths.add_argument("--pretrained_sana",
                       default="Efficient-Large-Model/Sana_600M_512px_diffusers",
                       help="Sana snapshot providing the VAE, tokenizer and text encoder")
    paths.add_argument("--fmow_test_root", required=True,
                       help="FMoW test split root, one directory per sequence")
    paths.add_argument("--caption_root", required=True,
                       help="caption root; captions at <caption_root>/test/<seq_id>/<stem>.txt")
    paths.add_argument("--test_gt_mapping", default="",
                       help="optional FMoW test_gt_mapping, for category labels")
    paths.add_argument("--output_dir", required=True,
                       help="where metrics.json, progress.log and --save_images land")

    proto = p.add_argument_group("protocol (defaults reproduce the paper)")
    proto.add_argument("--num_samples", type=int, default=10000,
                       help="keep at 10000 for a quotable FID")
    proto.add_argument("--num_steps", type=int, default=20,
                       help="Euler steps; 20 is the trained operating point")
    proto.add_argument("--guidance_scale", type=float, default=2.5,
                       help="classifier-free guidance on the text direction")
    proto.add_argument("--uncond_metadata", choices=["zero", "real"], default="zero",
                       help="what the unconditional CFG branch sees; 'zero' is "
                            "what the published numbers used")
    proto.add_argument("--text_encoder", default="fp32-eager",
                       choices=["fp32-eager", "bf16-causal"],
                       help="Gemma-2 dtype and read-out. 'bf16-causal' "
                            "reproduces eval_sana.py, which is what the paper "
                            "submission numbers were measured with; "
                            "'fp32-eager' matches training and is the default. "
                            "Both request eager attention -- sdpa returns NaN "
                            "on this model. The choice is in the protocol hash.")
    proto.add_argument("--curve_at", default="",
                       help="also report every metric at these intermediate "
                            "sample counts, e.g. '5000,6000,7000,8000,9000', or "
                            "'auto' for ten evenly spaced points. Costs one "
                            "metric read each, not a second generation pass. "
                            "Does not enter the protocol hash -- it only adds "
                            "readings of the same run.")
    proto.add_argument("--max_caption_len", type=int, default=256)
    proto.add_argument("--resolution", type=int, default=512)
    proto.add_argument("--seed", type=int, default=42)
    proto.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                       help="transformer and VAE dtype; the text encoder is "
                            "always fp32 (see load_encoders)")

    run_grp = p.add_argument_group("run")
    run_grp.add_argument("--batch_size", type=int, default=8)
    run_grp.add_argument("--num_workers", type=int, default=8,
                         help="dataloader workers; raise it if the GPU is idling")
    run_grp.add_argument("--save_images", action="store_true",
                         help="also write every generated PNG (~10k files)")
    run_grp.add_argument("--no-satclip", action="store_true",
                         help="checkpoint was trained without the geometry-aware "
                              "metadata encoder")
    run_grp.add_argument("--load-guard", type=int, default=5,
                         help="abort if more than this many state-dict keys "
                              "miss or are unexpected")
    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
    run(parse_args())
