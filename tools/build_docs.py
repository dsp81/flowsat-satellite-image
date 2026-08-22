#!/usr/bin/env python3
"""
build_docs.py — render the Markdown guides in docs/ as pages on the website.

The Markdown files stay the source of truth (they are what GitHub shows, and
what people read in a clone). This script wraps each one in the project page's
shell so the same text is readable on the site itself instead of only as an
outbound link:

    docs/METADATA_CONTROLLABILITY.md  ->  docs/controllability.html
    docs/NEW_DATASET.md               ->  docs/new-dataset.html
    docs/CAPTIONING.md                ->  docs/captioning.html

USAGE
    python tools/build_docs.py                 # rebuild all pages
    python tools/build_docs.py --check         # fail if a page is out of date

Re-run it after editing any of those Markdown files, and commit both the .md
and the regenerated .html. Styling lives in docs/static/doc.css — edit that
rather than the generated HTML, which is overwritten on every build.

Requires: markdown (pip install markdown)
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.exit("[error] pip install markdown")

REPO = "https://github.com/dsp81/flowsat-satellite-image"

# source .md -> (output .html, kicker, meta description)
PAGES = {
    "METADATA_CONTROLLABILITY.md": (
        "controllability.html", "Guide",
        "Why metadata conditioning succeeds or silently fails, and how to design "
        "a dataset, caption regime and metadata schema that leave room for control."),
    "NEW_DATASET.md": (
        "new-dataset.html", "Guide",
        "The adapter contract, metadata normalisation convention and commands for "
        "training FlowSat on a corpus of your own."),
    "CAPTIONING.md": (
        "captioning.html", "Guide",
        "Generating captions for a satellite corpus with a VLM, and how the "
        "captioning prompt decides what the model can control."),
}

SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — FlowSat</title>
<meta name="description" content="{desc}">
<meta property="og:title" content="{title} — FlowSat">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="https://dsp81.github.io/flowsat-satellite-image/{out}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="static/doc.css">
</head>
<body>

<nav><div class="wrap">
  <a class="brand" href="./">FlowSat</a>
  <span class="links" style="display:flex;gap:22px">
    <a href="./#explore">Explore</a><a href="./#method">Method</a>
    <a href="./#results">Results</a><a href="./#start">Get started</a>
  </span>
  <a href="{repo}" style="font-weight:600;color:var(--ink)">GitHub &#8599;</a>
</div></nav>

<header class="doc"><div class="wrap">
  <p class="kicker">{kicker}</p>
  <h1>{title}</h1>
  <p class="sub">{desc}</p>
</div></header>

<div class="wrap">
  <div class="layout">
    <article>
{body}
      <div class="backrow">
        <a href="./">&#8592; Project page</a>
        <a href="{repo}/blob/main/docs/{src}">View this page on GitHub</a>
      </div>
    </article>
    <aside class="toc">
      <p class="t">On this page</p>
{toc}
    </aside>
  </div>
</div>

<footer><div class="wrap">
  <p>Built on <a href="https://github.com/NVlabs/Sana">Sana</a>,
    <a href="https://github.com/mit-han-lab/efficientvit">DC-AE</a> and
    <a href="https://huggingface.co/google/gemma-2-2b-it">Gemma-2</a>.
    Location encoding follows <a href="https://github.com/microsoft/satclip">SatCLIP</a>.</p>
  <p>Sustainability Lab &#183; IIT Gandhinagar</p>
</div></footer>

</body>
</html>
"""


def slugify(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text)
    s = html.unescape(s).lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    return re.sub(r"\s+", "-", s.strip())


def render(src: Path) -> tuple[str, str, str]:
    """Return (title, article HTML, sidebar TOC HTML) for one Markdown file."""
    text = src.read_text()

    # The first H1 becomes the page header, not part of the article body.
    # Only that one line is dropped: "# ..." inside a fenced block is a shell
    # comment, not a heading, so fence state has to be tracked.
    lines = text.split("\n")
    title, fence, cut = src.stem, False, None
    for i, l in enumerate(lines):
        if l.lstrip().startswith("```"):
            fence = not fence
        elif not fence and cut is None and l.startswith("# "):
            title, cut = l[2:].strip(), i
    body_md = "\n".join(l for i, l in enumerate(lines) if i != cut)

    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists",
                                       "attr_list"])
    body = md.convert(body_md)

    # Anchor every H2 so the sidebar can link to it.
    entries = []

    def anchor(m):
        inner = m.group(1)
        sid = slugify(inner)
        entries.append((sid, re.sub(r"<[^>]+>", "", inner)))
        return f'<h2 id="{sid}">{inner}</h2>'

    body = re.sub(r"<h2>(.*?)</h2>", anchor, body, flags=re.S)

    # Relative links between the guides point at .md files, which Pages serves
    # as plain text. On the site they should reach the rendered sibling page.
    for md_name, (html_name, _, _) in PAGES.items():
        body = body.replace(f'href="{md_name}"', f'href="{html_name}"')

    toc = "\n".join(f'      <a href="#{sid}">{html.escape(t)}</a>'
                    for sid, t in entries)
    body = "\n".join("      " + l if l.strip() else l for l in body.split("\n"))
    return title, body, toc


def build(docs: Path, check: bool) -> int:
    stale = 0
    for name, (out, kicker, desc) in PAGES.items():
        src = docs / name
        if not src.exists():
            print(f"[skip] {name} not found")
            continue
        title, body, toc = render(src)
        page = SHELL.format(title=html.escape(title), desc=html.escape(desc),
                            kicker=kicker, body=body, toc=toc, out=out,
                            src=name, repo=REPO)
        dst = docs / out
        current = dst.read_text() if dst.exists() else None
        if current == page:
            print(f"  up to date  {out}")
            continue
        if check:
            print(f"  OUT OF DATE {out}  (run: python tools/build_docs.py)")
            stale += 1
            continue
        dst.write_text(page)
        print(f"  wrote       {out}  ({len(page) // 1024} KB, {len(toc.splitlines())} sections)")
    return stale


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="./docs")
    p.add_argument("--check", action="store_true",
                   help="report pages that need rebuilding, write nothing")
    a = p.parse_args()
    stale = build(Path(a.docs), a.check)
    if a.check and stale:
        sys.exit(f"\n{stale} page(s) out of date")


if __name__ == "__main__":
    main()
