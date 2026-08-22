#!/usr/bin/env python3
"""
export_for_web.py — publish generated images to the project website.

The site (docs/index.html) looks for images at fixed paths. This copies and
renames them out of a gen_gallery.py run so you never have to rename by hand:

    docs/static/teaser.jpg               hero strip
    docs/static/architecture.jpg         method figure (you supply this)
    docs/static/sweeps/month_07.jpg      interactive dial, season axis
    docs/static/sweeps/gsd_0.5.jpg       interactive dial, resolution axis
    docs/static/sweeps/cloud_25.jpg      interactive dial, cloud axis
    docs/static/compare/a.jpg  b.jpg     before/after divider
    docs/static/gallery/01.jpg ...       sample grid (lightbox)

Anything missing degrades to a labelled placeholder on the page rather than a
broken image, so it is safe to publish before every asset exists.

USAGE
    # from a gen_gallery.py run that wrote manifest.json
    python tools/export_for_web.py --gallery ./gallery_v1 --docs ./docs

    # or point at loose files
    python tools/export_for_web.py --teaser figs/teaser.png --docs ./docs
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

MAXW = 1600          # hero / architecture
MAXW_TILE = 768      # sweep frames and gallery tiles
QUALITY = 88


def put(src: Path, dst: Path, max_w: int) -> bool:
    """Copy src -> dst, downscaling and converting to JPEG when Pillow is around.
    Web assets should not be 4 MB PNGs; the page has to load on a phone."""
    if not src or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        shutil.copy2(src, dst)
        return True
    im = Image.open(src).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=QUALITY, optimize=True, progressive=True)
    return True


def from_manifest(manifest: Path):
    """Index a gen_gallery.py manifest by (task, axis-ish key)."""
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gallery", default=None,
                   help="a gen_gallery.py output dir containing manifest.json")
    p.add_argument("--docs", default="./docs")
    p.add_argument("--teaser", default=None, help="explicit hero strip image")
    p.add_argument("--architecture", default=None, help="explicit method figure")
    p.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"),
                   help="two images for the before/after divider")
    p.add_argument("--gallery-glob", default=None,
                   help="glob of sample images for the lightbox grid, "
                        "e.g. 'gallery_v1/rich/**/seed*.png'")
    a = p.parse_args()

    docs = Path(a.docs)
    static = docs / "static"
    n = 0

    if a.teaser and put(Path(a.teaser), static / "teaser.jpg", MAXW):
        print("teaser.jpg"); n += 1
    if a.architecture and put(Path(a.architecture), static / "architecture.jpg", MAXW):
        print("architecture.jpg"); n += 1
    if a.compare:
        for src, name in zip(a.compare, ("a.jpg", "b.jpg")):
            if put(Path(src), static / "compare" / name, MAXW_TILE):
                print(f"compare/{name}"); n += 1

    # ---- sweeps, straight out of the manifest -----------------------------
    if a.gallery:
        gal = Path(a.gallery)
        recs = from_manifest(gal / "manifest.json")
        if not recs:
            print(f"[warn] no manifest.json under {gal}")
        # axis= strings look like "month=7", "gsd=0.5", "cloud=25", "loc=egypt"
        for r in recs:
            axis = str(r.get("axis", ""))
            if "=" not in axis:
                continue
            k, v = axis.split("=", 1)
            k = k.strip().lower()
            src = Path(r["path"])
            if k == "month":
                key = f"{int(float(v)):02d}"
            elif k == "gsd":
                key = f"{float(str(v).rstrip('m')):.1f}"
            elif k == "cloud":
                key = f"{int(float(str(v).rstrip('%'))):02d}"
            else:
                continue
            if put(src, static / "sweeps" / f"{k}_{key}.jpg", MAXW_TILE):
                n += 1
        print(f"sweeps/ -> {len(list((static/'sweeps').glob('*.jpg')))} frames"
              if (static / "sweeps").exists() else "sweeps/ -> none")

    # ---- gallery tiles -----------------------------------------------------
    if a.gallery_glob:
        files = sorted(Path().glob(a.gallery_glob))[:16]
        for i, f in enumerate(files, 1):
            if put(f, static / "gallery" / f"{i:02d}.jpg", MAXW_TILE):
                n += 1
        print(f"gallery/ -> {len(files)} tiles")

    print(f"\n{n} files written under {static}")
    print("Preview locally:  python -m http.server -d docs 8000  ->  "
          "http://localhost:8000")
    print("Then commit and push; GitHub Pages serves /docs on the main branch.")


if __name__ == "__main__":
    main()
