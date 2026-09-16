"""
checkpoint.py — resolve a checkpoint argument to a weight file on disk.

Everything that takes a `--checkpoint` accepts the same three forms, so a
command from the README works whether the weights are already downloaded or
not:

    checkpoints/flowsat-fmow-512          a local directory holding the weights
    checkpoints/flowsat-fmow-512/model_0.pt   a local file
    dsp81/flowsat-fmow-512                a Hugging Face repo id

The Hub form downloads once into the usual HF cache and is a no-op afterwards,
so it costs nothing to leave in a script. Two suffixes are understood:

    dsp81/flowsat-fmow-512@v1.0           a revision (branch, tag or commit)
    dsp81/flowsat-fmow-512:model_0.pt     a specific file in the repo

Weights are large. If a download is going to happen, this says so before it
starts rather than appearing to hang.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Tried in order inside a local directory or a Hub repo.
WEIGHT_NAMES = (
    "model_0.pt",            # what accelerate writes, and what we publish
    "model.safetensors",
    "model.pt",
    "pytorch_model.bin",
    "diffusion_pytorch_model.bin",
)


def _looks_like_hub_id(spec: str) -> bool:
    """True for 'org/name', false for anything that is really a path.

    A Hub id is exactly one slash, no path syntax, and nothing on disk by that
    name -- a local directory always wins, so a repo cloned to ./org/name does
    not silently become a download.
    """
    if os.path.sep in spec.replace("/", os.path.sep) and Path(spec).exists():
        return False
    if Path(spec).exists():
        return False
    if spec.startswith((".", "/", "~")) or "\\" in spec:
        return False
    # 'checkpoints/flowsat-fmow-512' has one slash and is a Hub id by shape, but
    # if ./checkpoints exists the user clearly meant a local path that is simply
    # missing -- and should be told that, not handed a Hub 401.
    if Path(spec).parent.exists():
        return False
    return spec.count("/") == 1 and all(spec.split("/"))


def resolve_checkpoint(spec: str, quiet: bool = False) -> Path:
    """Return a path to the weight file named by `spec`, downloading if needed."""
    raw = str(spec)
    repo_file: Optional[str] = None
    revision: Optional[str] = None

    if ":" in raw and not Path(raw).exists():
        raw, repo_file = raw.rsplit(":", 1)
    if "@" in raw and not Path(raw).exists():
        raw, revision = raw.rsplit("@", 1)

    path = Path(raw).expanduser()

    if path.is_file():
        return path

    if path.is_dir():
        found = next((path / n for n in WEIGHT_NAMES if (path / n).exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"no weight file in {path}. Looked for {', '.join(WEIGHT_NAMES)}; "
                f"found {sorted(p.name for p in path.iterdir())[:15]}")
        return found

    if not _looks_like_hub_id(raw):
        raise FileNotFoundError(
            f"checkpoint not found: {spec}\n"
            f"        Expected a local directory, a local file, or a Hugging "
            f"Face repo id such as 'dsp81/flowsat-fmow-512'.")

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        sys.exit("[error] huggingface_hub is needed to fetch weights from the "
                 "Hub. It ships with transformers: pip install huggingface_hub")

    candidates = [repo_file] if repo_file else list(WEIGHT_NAMES)
    if not quiet:
        print(f"[hub] resolving {raw}"
              f"{'@' + revision if revision else ''} — the first run downloads "
              f"~2.3 GB into the Hugging Face cache; later runs reuse it.")

    last: Optional[Exception] = None
    for name in candidates:
        try:
            return Path(hf_hub_download(repo_id=raw, filename=name,
                                        revision=revision))
        except EntryNotFoundError as e:       # wrong filename, try the next
            last = e
        except Exception as e:                # network, auth, missing repo
            hint = ("        If you meant a local path, it does not exist. If you "
                    "meant a Hub repo,\n        check the id; for a private one "
                    "run `huggingface-cli login`."
                    if "Repository Not Found" in str(e) or "401" in str(e)
                    else "        If the repo is private, run `huggingface-cli "
                         "login`. If you are offline, pass a local path instead.")
            raise SystemExit(
                f"[error] {raw} is neither a local checkpoint nor a readable "
                f"Hugging Face repo.\n        ({type(e).__name__}: "
                f"{str(e).splitlines()[0]})\n{hint}") from e

    raise SystemExit(
        f"[error] none of {', '.join(str(c) for c in candidates)} exist in {raw}. "
        f"Name the file explicitly as '{raw}:<filename>'. ({last})")
