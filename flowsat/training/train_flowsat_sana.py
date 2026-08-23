"""
FlowSat Training Script with Sana-0.6B backbone: train_flowsat_sana.py

Forked from train_flowsat_pixart.py with these changes:

  1. Backbone: SatSana (Sana-0.6B) instead of SatPixArt.
     ~600M params, loaded from Efficient-Large-Model/Sana_600M_512px_diffusers.
     Native flow-matching pretrained (no DDPM-to-FM mismatch).
     Built-in QK-norm in attention (no gradient explosion).

  2. Text encoder: Gemma-2-2B-IT (decoder LLM, instead of T5-XXL).
     Frozen, kept on GPU in bf16 (uses ~5 GB vs T5's ~11 GB).

  3. VAE: DC-AE 32x (Deep Compression AutoEncoder) instead of SD VAE 8x.
     For 512px image: latent is 16x16x32 (vs SD's 64x64x4).
     Pretrained scaling_factor from vae.config.

  4. Cross-attention dim: 2304 (Gemma-2) instead of 4096 (T5).

  5. Latent channels: 32 (DC-AE) instead of 4 (SD).

Everything else stays IDENTICAL to PixArt training:
  - Flow matching loss (velocity prediction)
  - Logit-normal time sampling
  - Per-field metadata embedding (or SatCLIP)
  - CFG dropout (caption + metadata) — without zeroing attention_mask
  - EMA averaging
  - Checkpoint save/resume
  - WandB / TensorBoard logging
  - NaN guard on every step

Usage (2 GPUs):

    # Paths below are read from the environment so nothing machine-specific is
    # baked into this file. SANA_ID may be a HuggingFace repo id (downloaded on
    # demand) or a local snapshot directory; set HF_HUB_OFFLINE=1 to force the
    # latter.
    export SANA_ID="${SANA_ID:-Efficient-Large-Model/Sana_600M_512px_diffusers}"
    export FMOW_ROOT=/path/to/fmow/train
    export CAPTION_DIR=/path/to/fmow/captions/train
    export OUTPUT_DIR=runs/flowsat-sana-run1
    export CUDA_VISIBLE_DEVICES=0,1

    pip install -e .          # from the repository root

    accelerate launch \\
        --num_processes=2 --mixed_precision=bf16 \\
        flowsat/training/train_flowsat_sana.py \\
        --pixart_id "$SANA_ID" \\
        --pretrained_model_name_or_path "$SANA_ID" \\
        --fmow_root "$FMOW_ROOT" \\
        --caption_dir "$CAPTION_DIR" \\
        --output_dir "$OUTPUT_DIR" \\
        --resolution 512 \\
        --train_batch_size 4 \\
        --gradient_accumulation_steps 3 \\
        --learning_rate 1e-5 \\
        --lr_scheduler cosine \\
        --lr_warmup_steps 1000 \\
        --max_train_steps 100000 \\
        --checkpointing_steps 5000 \\
        --checkpoints_total_limit 5 \\
        --validation_steps 5000 \\
        --mixed_precision bf16 \\
        --gradient_checkpointing \\
        --dataloader_num_workers 4 \\
        --wandb flowsat-pixart \\
        --seed 42
"""

import argparse
import logging
import math
import os
import random
import shutil
import signal
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*geotransform.*")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

import accelerate
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

# Gemma-2 text encoder + DC-AE VAE (Sana's pretrained stack)
from transformers import AutoTokenizer, AutoModel
# SD VAE — PixArt-Sigma is paired with the SDXL VAE in pixart_sigma_sdxlvae_T5_diffusers
from diffusers import AutoencoderDC
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

# FlowSat imports (your existing code, unchanged)
from flowsat.flow.flow_matching import FlowMatchingLoss
from flowsat.flow.flow_sampler import EulerSampler, HeunSampler
from flowsat.data.fmow_dataset import FMoWDataset, flowsat_collate_fn

# NEW: PixArt-based model
from flowsat.models.sat_sana import SatSana, SATSANA_MODELS
# SatCLIP metadata encoder (replaces per-field embedding when --use_satclip_encoder)
from flowsat.models.sat_clip import SatCLIPMetadataEncoder

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = get_logger(__name__, log_level="INFO")


# ============================================================================
# Argument parser (small set vs original; we hardcode some things since
# PixArt-Sigma's choices are fixed)
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    # Model. Both default to the public Sana snapshot this script is built
    # around; override with $SANA_ID to use a local mirror. The flag is named
    # --pixart_id for backwards compatibility with earlier runs, but it selects
    # the Sana transformer weights.
    _sana_default = os.environ.get(
        "SANA_ID", "Efficient-Large-Model/Sana_600M_512px_diffusers")
    p.add_argument("--pixart_id", type=str, default=_sana_default,
                   help="HuggingFace repo id or local dir for the Sana "
                        "transformer weights. [env: SANA_ID]")
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default=_sana_default,
                   help="HuggingFace repo id or local dir for the DC-AE VAE and "
                        "Gemma-2 text encoder. [env: SANA_ID]")
    p.add_argument("--revision", type=str, default=None)

    # Data
    p.add_argument("--fmow_root", type=str, required=True)
    p.add_argument("--caption_dir", type=str, default=None,
                   help="Path to rich VLM captions. Highly recommended for PixArt-Sigma.")
    p.add_argument("--meta_csv_path", type=str, default=None)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--center_crop", action="store_true", default=True)
    p.add_argument("--random_flip", action="store_true", default=False)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--t5_max_length", type=int, default=120,
                   help="Max T5 token length. PixArt-Sigma trained with 120.")

    # Metadata
    p.add_argument("--num_metadata", type=int, default=7)
    p.add_argument("--text_metadata", action="store_true", default=False)
    p.add_argument("--metadata_drop_prob", type=float, default=0.1)
    p.add_argument("--caption_drop_prob", type=float, default=0.1)
    p.add_argument("--use_satclip_encoder", action="store_true", default=False,
                   help="Use SatCLIP metadata encoder (spherical harmonics + temporal) "
                        "instead of per-field sinusoidal+MLP. Wired into SatPixArt "
                        "via set_metadata_encoder() at embed_dim=1152.")

    # Flow matching
    p.add_argument("--time_sampling", type=str, default="logit_normal",
                   choices=["uniform", "logit_normal"])
    p.add_argument("--logit_normal_mean", type=float, default=0.0)
    p.add_argument("--logit_normal_std", type=float, default=1.0)
    p.add_argument("--use_ot_cfm", action="store_true", default=False)

    # Training
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, default=5e-5,
                   help="Lower default than from-scratch (was 1e-4) since this is fine-tuning.")
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=1000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow_tf32", action="store_true")

    # Optimizer
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=0.0)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)

    # Freeze schedule (for PixArt fine-tuning warmup)
    p.add_argument("--freeze_transformer_steps", type=int, default=0,
                   help="If >0, freeze the inner transformer blocks for this many steps. "
                        "Only proj_out, AdaLN-single, patch_embed, and metadata branch train.")

    # EMA
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)

    # Checkpointing
    p.add_argument("--checkpointing_steps", type=int, default=5000)
    p.add_argument("--checkpoints_total_limit", type=int, default=5)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Logging
    p.add_argument("--wandb", type=str, default=None)
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--tracker_project_name", type=str, default="flowsat-pixart")

    # Validation
    p.add_argument("--validation_steps", type=int, default=5000)
    p.add_argument("--early_validation_steps", type=int, default=None,
                   help="If set, validate at this frequency for the first --early_validation_until "
                        "steps, then switch to --validation_steps. Recommended: 500.")
    p.add_argument("--early_validation_until", type=int, default=5000,
                   help="Switch from early_validation_steps to validation_steps after this many steps.")
    p.add_argument("--validation_num_steps", type=int, default=20)
    p.add_argument("--validation_guidance_scale", type=float, default=2.5)
    p.add_argument("--num_validation_images", type=int, default=4)

    return p.parse_args()


# ============================================================================
# Validation prompt builder (same as your existing one, simplified)
# ============================================================================

def build_validation_prompts(args, seed=1234):
    """Return 4 validation prompts. Uses real VLM captions if caption_dir is set."""
    fixed = [
        ("airport", "United States"),
        ("solar_farm", "Morocco"),
        ("stadium", "United States"),
        ("port", "Singapore"),
    ]
    rng = random.Random(seed)
    prompts = []
    if args.caption_dir and Path(args.caption_dir).exists():
        for cat, country in fixed:
            cat_dir = Path(args.caption_dir) / cat
            if cat_dir.exists():
                files = []
                for loc in sorted(cat_dir.iterdir()):
                    if loc.is_dir():
                        files.extend(sorted(loc.glob("*_rgb.txt"))[:5])
                    if len(files) > 20:
                        break
                if files:
                    picked = rng.choice(files)
                    try:
                        txt = picked.read_text(encoding="utf-8").strip()
                        if len(txt) > 30:
                            prompts.append(f"a fmow satellite image of a {cat.replace('_', ' ')}. {txt}")
                            continue
                    except Exception:
                        pass
            prompts.append(f"a fmow satellite image of a {cat.replace('_', ' ')} in {country}.")
    else:
        for cat, country in fixed:
            prompts.append(f"a fmow satellite image of a {cat.replace('_', ' ')} in {country}.")
    return prompts


# ============================================================================
# Metadata sweeps (geographic + seasonal)
#
# Both sweeps hold ONE rich VLM caption fixed and vary only the relevant
# metadata field, so any visible change in the output is attributable to
# metadata conditioning alone (text is identical, text-CFG is identical).
#
# Normalized-metadata layout — see sat_data_util.metadata_normalize (scale=1000):
#   [0] lon   = (real_lon + 180) / 360 * 1000
#   [1] lat   = (real_lat +  90) / 180 * 1000
#   [2] gsd   [3] cloud_cover   [4] year
#   [5] month = month / 12 * 1000
#   [6] day
# ============================================================================

# Geographic sweep: well-separated biomes so the shift is unambiguous in a grid.
# (name, latitude, longitude)
GEO_SWEEP_LOCATIONS = [
    ("Sahara Desert, Egypt",        26.8,  30.8),   # arid, bare sand / dunes
    ("Amazon Basin, Brazil",        -3.1, -60.0),   # dense tropical rainforest
    ("Arctic Norway (Troms\u00f8)", 69.6,  18.9),   # snow / tundra, high latitude
    ("Netherlands (polders)",       52.1,   5.3),   # temperate, geometric farmland
]

# Seasonal sweep: one month per season (Northern-Hemisphere framing).
# (label, month_index 1-12)
SEASON_SWEEP_MONTHS = [
    ("January (winter)", 1),
    ("April (spring)",   4),
    ("July (summer)",    7),
    ("October (autumn)", 10),
]


def _lonlat_to_norm(lon: float, lat: float):
    """Real (lon, lat) -> normalized metadata indices [0], [1]."""
    return ((lon + 180.0) / 360.0 * 1000.0,
            (lat + 90.0) / 180.0 * 1000.0)


def _month_to_norm(month: int) -> float:
    """Calendar month (1-12) -> normalized metadata index [5]."""
    return month / 12.0 * 1000.0


def build_sweep_anchor_prompt(args, seed=1234):
    """Pick ONE rich VLM caption to hold fixed across both metadata sweeps.

    Prefers terrain/vegetation-sensitive categories so that changing only
    lat/lon (biome) or month (season) produces a visible, well-documented
    shift. Falls back to a template caption if no rich .txt is found.
    """
    priority = ["crop_field", "golf_course", "park", "solar_farm", "airport"]
    rng = random.Random(seed)
    if args.caption_dir and Path(args.caption_dir).exists():
        for cat in priority:
            cat_dir = Path(args.caption_dir) / cat
            if not cat_dir.exists():
                continue
            files = []
            for loc in sorted(cat_dir.iterdir()):
                if loc.is_dir():
                    files.extend(sorted(loc.glob("*_rgb.txt"))[:5])
                if len(files) > 20:
                    break
            if files:
                picked = rng.choice(files)
                try:
                    txt = picked.read_text(encoding="utf-8").strip()
                    if len(txt) > 30:
                        return (f"a fmow satellite image of a "
                                f"{cat.replace('_', ' ')}. {txt}")
                except Exception:
                    pass
    return "a fmow satellite image of a crop field."


# ============================================================================
# Validation generation (T5-aware)
# ============================================================================

@torch.no_grad()
def log_validation(model, vae, tokenizer, text_encoder, accelerator, args, step,
                   weight_dtype):
    logger.info(f"Running validation at step {step}")
    model_unwrapped = accelerator.unwrap_model(model)
    model_unwrapped.eval()

    sampler = EulerSampler(num_steps=args.validation_num_steps)
    prompts = build_validation_prompts(args, seed=1234)

    # DC-AE compresses 32x. Latent for 512px = 16x16. For 1024px = 32x32.
    latent_size = args.resolution // 32
    gs = args.validation_guidance_scale

    # Empty (unconditional) text embedding is identical for every image — encode once.
    uncond_inputs = tokenizer(
        "", padding="max_length", max_length=args.t5_max_length,
        truncation=True, return_tensors="pt",
    )
    uncond_ids = uncond_inputs.input_ids.to(accelerator.device)
    uncond_mask = uncond_inputs.attention_mask.to(accelerator.device)
    uncond_embeds = text_encoder(uncond_ids, attention_mask=uncond_mask)[0]

    # Base metadata row: lon 0, lat 0, mid-GSD, no cloud, ~2020, June, mid-month.
    BASE_MD = [500., 500., 500., 0., 333., 500., 500.]

    def _encode_prompt(prompt):
        ti = tokenizer(
            prompt, padding="max_length", max_length=args.t5_max_length,
            truncation=True, return_tensors="pt",
        )
        ids = ti.input_ids.to(accelerator.device)
        mask = ti.attention_mask.to(accelerator.device)
        cond = text_encoder(ids, attention_mask=mask)[0]
        return cond, mask

    def generate(prompt, md_row, seed):
        """Render one image. md_row: list[7] of normalized metadata values."""
        cond_embeds, text_mask = _encode_prompt(prompt)
        prompt_d = torch.cat([uncond_embeds, cond_embeds], dim=0)
        mask_d = torch.cat([uncond_mask, text_mask], dim=0)

        md = torch.tensor([md_row], device=accelerator.device, dtype=weight_dtype)
        # Same metadata on both CFG paths — only text is guided (metadata is NOT
        # CFG-scaled, matching training-time conditioning).
        md_d = torch.cat([md, md], dim=0)

        g = torch.Generator(device=accelerator.device).manual_seed(seed)
        z = torch.randn(1, vae.config.latent_channels, latent_size, latent_size,
                        device=accelerator.device, dtype=weight_dtype, generator=g)

        def model_fn(x_t, t, encoder_hidden_states=None, metadata=None,
                     attention_mask=None, **kwargs):
            x_d = torch.cat([x_t, x_t], dim=0)
            t_d = torch.cat([t, t], dim=0)
            out = model_unwrapped(
                x_d, t_d,
                encoder_hidden_states=encoder_hidden_states,
                metadata=metadata,
                attention_mask=attention_mask,
            ).sample
            v_u, v_c = out.chunk(2, dim=0)
            return v_u + gs * (v_c - v_u)

        latents = sampler.sample(
            model_fn, z, num_steps=args.validation_num_steps,
            show_progress=False,
            encoder_hidden_states=prompt_d,
            metadata=md_d,
            attention_mask=mask_d,
        )
        img = vae.decode(latents / vae.config.scaling_factor).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img[0].cpu().float().permute(1, 2, 0).numpy()
        return (img * 255).astype("uint8")

    # --- (1) Normal validation images (unchanged behaviour) ---
    normal_images = []
    for i, prompt in enumerate(prompts):
        arr = generate(prompt, list(BASE_MD), seed=1234 + i)
        normal_images.append((prompt, arr))

    # --- (2) Geographic sweep: one fixed rich caption, vary only lon/lat ---
    # Shared noise seed across panels so differences come ONLY from lat/lon.
    anchor_prompt = build_sweep_anchor_prompt(args, seed=1234)
    geo_images = []
    for name, lat, lon in GEO_SWEEP_LOCATIONS:
        md_row = list(BASE_MD)
        lon_n, lat_n = _lonlat_to_norm(lon, lat)
        md_row[0], md_row[1] = lon_n, lat_n
        arr = generate(anchor_prompt, md_row, seed=777)
        geo_images.append((f"{name}  (lat={lat}, lon={lon})", arr))

    # --- (3) Seasonal sweep: same fixed caption, vary only month ---
    season_images = []
    for label, month in SEASON_SWEEP_MONTHS:
        md_row = list(BASE_MD)
        md_row[5] = _month_to_norm(month)
        arr = generate(anchor_prompt, md_row, seed=777)
        season_images.append((label, arr))

    # --- Save grids ---
    if accelerator.is_main_process:
        from PIL import Image, ImageDraw

        def save_grid(images, path, cols=2):
            size = 256
            n = len(images)
            rows = (n + cols - 1) // cols
            grid = Image.new(
                "RGB",
                (cols * (size + 10) + 10, rows * (size + 35) + 10),
                (255, 255, 255),
            )
            draw = ImageDraw.Draw(grid)
            for i, (caption, arr) in enumerate(images):
                r, c = i // cols, i % cols
                x = c * (size + 10) + 10
                y = r * (size + 35) + 30
                pil = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
                grid.paste(pil, (x, y))
                draw.text((x, y - 18), caption[:60], fill=(0, 0, 0))
            grid.save(path)

        out_dir = os.path.join(args.output_dir, f"validation-{step}")
        os.makedirs(out_dir, exist_ok=True)

        save_grid(normal_images, os.path.join(out_dir, "grid.png"))
        save_grid(geo_images, os.path.join(out_dir, "geo_sweep.png"))
        save_grid(season_images, os.path.join(out_dir, "season_sweep.png"))

        with open(os.path.join(out_dir, "prompts.txt"), "w") as f:
            f.write("=== normal ===\n")
            for p, _ in normal_images:
                f.write(p + "\n")
            f.write("\n=== geographic sweep ===\n")
            f.write(f"anchor prompt: {anchor_prompt}\n")
            for caption, _ in geo_images:
                f.write(caption + "\n")
            f.write("\n=== seasonal sweep ===\n")
            f.write(f"anchor prompt: {anchor_prompt}\n")
            for caption, _ in season_images:
                f.write(caption + "\n")

    model_unwrapped.train()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, args.logging_dir),
    )

    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[ddp_kwargs],
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"output_dir: {args.output_dir}")
        logger.info(f"pixart_id: {args.pixart_id}")
        logger.info(f"text encoder + VAE source: {args.pretrained_model_name_or_path}")
        logger.info(f"world_size: {accelerator.num_processes}")

    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------
    # Load Gemma-2 tokenizer + encoder + DC-AE VAE
    # ------------------------------------------------------------------
    logger.info("Loading Gemma-2 tokenizer + encoder + DC-AE VAE...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer",
    )
    text_encoder = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        #new add
        attn_implementation="eager",
    )
    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
    )
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # Gemma-2-2B exposes hidden_size; T5 used d_model
    gemma_hidden = getattr(text_encoder.config, 'hidden_size',
                           getattr(text_encoder.config, 'd_model', 2304))
    logger.info(f"Gemma-2 hidden_size: {gemma_hidden}")

    # DC-AE compresses 32x (vs SD VAE's 8x).
    # For 512px image: latent is 16x16x32 (1 token after patch=1)
    # vae.config.encoder_block_out_channels tells us the latent_channels (=32)
    dc_ae_channels = vae.config.latent_channels
    dc_ae_compression = 32  # DC-AE-f32c32: 32x spatial compression
    sana_latent_size = args.resolution // dc_ae_compression
    logger.info(f"DC-AE latent_channels: {dc_ae_channels}, "
                f"latent_size: {sana_latent_size}x{sana_latent_size}")

    # ------------------------------------------------------------------
    # Build SatSana model (loads pretrained Sana-0.6B weights)
    # ------------------------------------------------------------------
    logger.info(f"Building SatSana with pretrained {args.pixart_id}...")
    model = SatSana(
        latent_size=sana_latent_size,
        in_channels=dc_ae_channels,
        num_metadata=args.num_metadata,
        use_metadata=args.num_metadata > 0,
        cross_attention_dim=gemma_hidden,
        pretrained_sana_id=args.pixart_id,
        load_pretrained=True,
    )

    # ------------------------------------------------------------------
    # Optional: SatCLIP metadata encoder
    # ------------------------------------------------------------------
    metadata_encoder = None
    if args.use_satclip_encoder and args.num_metadata > 0:
        metadata_encoder = SatCLIPMetadataEncoder(
            embed_dim=model.embed_dim,  # Sana-0.6B hidden = 1152
            num_metadata=args.num_metadata,
        )
        model.set_metadata_encoder(metadata_encoder)
        n_sat = sum(p.numel() for p in metadata_encoder.parameters())
        logger.info(f"SatCLIP metadata encoder wired into SatSana ({n_sat:,} params)")
        logger.info("  -> SatCLIP replaces SatSana's per-field PerFieldMetadataEmbed")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"SatSana parameters: {total:,} total, {trainable:,} trainable")

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Optional: freeze inner transformer for warmup
    if args.freeze_transformer_steps > 0:
        logger.info(f"Freezing transformer blocks for first {args.freeze_transformer_steps} steps "
                    f"(only proj_out, AdaLN-single, patch_embed, metadata train).")
        for n, p in model.transformer.transformer_blocks.named_parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Flow matching loss
    # ------------------------------------------------------------------
    flow_loss_fn = FlowMatchingLoss(
        time_sampling=args.time_sampling,
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
        use_ot_cfm=args.use_ot_cfm,
    )

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            model_cls=SatSana,
            model_config=model.config,
        )
        logger.info(f"Using EMA with decay={args.ema_decay}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            OptimCls = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to torch.optim.AdamW")
            OptimCls = torch.optim.AdamW
    else:
        OptimCls = torch.optim.AdamW

    optimizer = OptimCls(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ------------------------------------------------------------------
    # Dataset: reuse FMoWDataset. It tokenizes with the tokenizer we pass,
    # which is now T5Tokenizer. T5 doesn't have model_max_length=77 like CLIP;
    # it defaults to 512. We override the tokenizer's model_max_length to keep
    # encoder sequences manageable.
    # ------------------------------------------------------------------
    tokenizer.model_max_length = args.t5_max_length

    train_transforms = T.Compose([
        T.ToTensor(),
        T.Resize(args.resolution, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(args.resolution) if args.center_crop else T.RandomCrop(args.resolution),
        T.RandomHorizontalFlip() if args.random_flip else T.Lambda(lambda x: x),
        T.Normalize([0.5], [0.5]),
    ])

    train_dataset = FMoWDataset(
        root_dir=args.fmow_root,
        tokenizer=tokenizer,
        resolution=args.resolution,
        transform=train_transforms,
        num_metadata=args.num_metadata,
        text_metadata=args.text_metadata,
        caption_drop_pct=args.caption_drop_prob,
        meta_csv_path=args.meta_csv_path,
        caption_dir=args.caption_dir,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=flowsat_collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # ------------------------------------------------------------------
    # Save/load hooks for EMA (same flat format as your sat_dit run)
    # ------------------------------------------------------------------
    def save_model_hook(models, weights, output_dir):
        for i, mdl in enumerate(models):
            torch.save(mdl.state_dict(), os.path.join(output_dir, f"model_{i}.pt"))
            weights.pop()
        if ema_model is not None:
            torch.save({
                "shadow_params": [p.detach().cpu().clone() for p in ema_model.shadow_params],
                "decay": ema_model.decay,
                "optimization_step": ema_model.optimization_step,
            }, os.path.join(output_dir, "ema.pt"))

    def load_model_hook(models, input_dir):
        if ema_model is not None:
            ema_path = os.path.join(input_dir, "ema.pt")
            if os.path.exists(ema_path):
                payload = torch.load(ema_path, map_location="cpu", weights_only=False)
                for tgt, src in zip(ema_model.shadow_params, payload["shadow_params"]):
                    tgt.data.copy_(src.to(tgt.device))
                ema_model.decay = payload.get("decay", ema_model.decay)
                ema_model.optimization_step = payload.get("optimization_step", 0)
                ema_model.to(accelerator.device)
        for i in range(len(models)):
            mdl = models.pop()
            sp = os.path.join(input_dir, f"model_{i}.pt")
            if os.path.exists(sp):
                sd = torch.load(sp, map_location="cpu", weights_only=True)
                mdl.load_state_dict(sd)
                del sd

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # ------------------------------------------------------------------
    # Prepare with accelerator
    # ------------------------------------------------------------------
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if ema_model is not None:
        ema_model.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    #text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=torch.float32)
    vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = {k: v for k, v in vars(args).items()}
        if args.wandb and HAS_WANDB:
            os.environ.setdefault("WANDB_PROJECT", args.wandb)
        accelerator.init_trackers(args.tracker_project_name, config=None)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        else:
            path = os.path.basename(args.resume_from_checkpoint)
        if path:
            logger.info(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** FlowSat-PixArt Training *****")
    logger.info(f"  samples       = {len(train_dataset)}")
    logger.info(f"  batch (total) = {total_batch_size}")
    logger.info(f"  per-device    = {args.train_batch_size}")
    logger.info(f"  accum         = {args.gradient_accumulation_steps}")
    logger.info(f"  max_steps     = {args.max_train_steps}")
    logger.info(f"  LR            = {args.learning_rate}")
    logger.info(f"  warmup        = {args.lr_warmup_steps}")
    logger.info(f"  t5_max_length = {args.t5_max_length}")

    progress = tqdm(range(global_step, args.max_train_steps),
                    disable=not accelerator.is_local_main_process,
                    desc="Training")

    last_v_pred = None
    last_v_target = None
    last_t = None
    last_grad_norm = 0.0

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Re-enable transformer training after warmup
            if args.freeze_transformer_steps > 0 and global_step == args.freeze_transformer_steps:
                logger.info(f"Step {global_step}: unfreezing transformer blocks.")
                unwrapped = accelerator.unwrap_model(model)
                for p in unwrapped.transformer.transformer_blocks.parameters():
                    p.requires_grad_(True)

            with accelerator.accumulate(model):
                # VAE encode
                with torch.no_grad():
                    # DC-AE returns AutoencoderDCOutput with .sample (no
                    # latent_dist — DC-AE is deterministic). The pretrained
                    # scaling_factor is in vae.config (typically 0.41407).
                    enc_out = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    )
                    # Some diffusers versions: enc_out.latent (dataclass)
                    # Others: enc_out has .sample
                    if hasattr(enc_out, 'latent'):
                        latents = enc_out.latent
                    elif hasattr(enc_out, 'sample'):
                        latents = enc_out.sample
                    else:
                        latents = enc_out  # fallback for older versions
                    latents = latents * vae.config.scaling_factor
                    latents = latents.clamp(-10.0, 10.0)  # safety bound

                # Gemma-2 encode (decoder LLM)
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(accelerator.device)
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()

                    #new add:
                    empty_rows = attention_mask.sum(dim=1) == 0
                    if empty_rows.any():
                        attention_mask[empty_rows, 0] = 1
                    #new add end
                    encoder_hidden_states = text_encoder(
                        input_ids,
                        attention_mask=attention_mask,
                    )[0]

                metadata = batch["metadata"].to(dtype=latents.dtype)
                bsz = latents.shape[0]

                # Per-field metadata dropout for CFG
                if args.num_metadata > 0:
                    keep_mask = torch.rand(bsz, args.num_metadata, device=metadata.device) > args.metadata_drop_prob
                    metadata = metadata * keep_mask.float()

                # Caption dropout: zero embeddings ONLY. Do NOT zero
                # attention_mask — that causes NaN in cross-attention softmax.
                caption_keep = torch.rand(bsz, device=latents.device) > args.caption_drop_prob
                caption_keep_emb = caption_keep[:, None, None].to(dtype=encoder_hidden_states.dtype)
                encoder_hidden_states = encoder_hidden_states * caption_keep_emb

                # Flow matching sample
                fm = flow_loss_fn.prepare_training_sample(latents)
                x_t, v_target, t = fm.x_t, fm.v_target, fm.t

                # Forward
                v_pred = model(
                    x_t, t,
                    encoder_hidden_states=encoder_hidden_states,
                    metadata=metadata if args.num_metadata > 0 else None,
                    attention_mask=attention_mask,
                ).sample

                loss = flow_loss_fn(v_pred, v_target, t)

                # Hard guard: never train through NaN/Inf
                if not torch.isfinite(loss):
                    logger.error(
                        f"[RANK {accelerator.process_index}] Non-finite loss at step "
                        f"{global_step}: loss={loss.item()}. "
                        f"v_pred finite: {torch.isfinite(v_pred).all().item()}, "
                        f"v_target finite: {torch.isfinite(v_target).all().item()}, "
                        f"ehs finite: {torch.isfinite(encoder_hidden_states).all().item()}, "
                        f"latents range: [{latents.min().item():.3e}, {latents.max().item():.3e}]"
                    )
                    raise RuntimeError("Training diverged with NaN/Inf loss.")

                last_v_pred = v_pred.detach()
                last_v_target = v_target.detach()
                last_t = t.detach()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                    last_grad_norm = float(grad_norm) if grad_norm is not None else 0.0

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                if ema_model is not None:
                    ema_model.step(model.parameters())

                if global_step % args.logging_steps == 0:
                    with torch.no_grad():
                        vp = last_v_pred.float().flatten(1)
                        vt = last_v_target.float().flatten(1)
                        cos = F.cosine_similarity(vp, vt, dim=1).mean().item()
                    logs = {
                        "loss": loss.item(),
                        "vel_cos": cos,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "grad_norm": last_grad_norm,
                    }
                    accelerator.log(logs, step=global_step)
                    progress.set_postfix(**{k: round(v, 4) for k, v in logs.items()})

                # Validation cadence: dense early, sparser later
                if (args.early_validation_steps is not None
                        and global_step <= args.early_validation_until):
                    val_freq = args.early_validation_steps
                else:
                    val_freq = args.validation_steps
                if global_step % val_freq == 0:
                    log_validation(model, vae, tokenizer, text_encoder, accelerator,
                                   args, global_step, weight_dtype)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint: {save_path}")
                        # Cleanup
                        if args.checkpoints_total_limit:
                            ckpts = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
                                key=lambda x: int(x.split("-")[1])
                            )
                            while len(ckpts) > args.checkpoints_total_limit:
                                old = ckpts.pop(0)
                                shutil.rmtree(os.path.join(args.output_dir, old))
                                logger.info(f"Cleaned: {old}")

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    progress.close()
    if accelerator.is_main_process:
        accelerator.end_training()

if __name__ == "__main__":
    main()