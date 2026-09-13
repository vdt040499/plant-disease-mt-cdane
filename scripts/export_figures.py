"""Export the pipeline figures from their HTML source.

    python scripts/export_figures.py

`figures/pipeline-cdane-vs-mt.html` is the single source of truth for both
diagrams: edit that file, re-run this script, and the PNG and SVG files the
README points at are regenerated in place. The README references files, not
embedded markup, so it needs no edit when a figure changes.

Each `<svg>` in the source becomes two files named after its accessible-title
id (`cdane-title` -> `pipeline-cdane.*`):

    figures/pipeline-cdane.svg       standalone SVG, for LaTeX and vector tools
    figures/pipeline-cdane.png       2x raster, referenced by the README

PNG rendering needs a browser. Playwright is used when installed; otherwise a
local Chrome or Chromium is driven in headless mode. With neither available the
SVG files are still written and the script says so instead of failing.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import _bootstrap  # noqa: F401

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Instrument+Serif:ital@0;1&amp;family=Geist:wght@400;500;600"
    "&amp;family=Geist+Mono:wght@400;500;600"
    "&amp;family=Be+Vietnam+Pro:wght@400;500;600&amp;display=swap');"
)

SVG_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL)
TITLE_ID_RE = re.compile(r'aria-labelledby="([\w-]+?)-title')
VIEWBOX_RE = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')

CHROME_CANDIDATES = (
    os.environ.get("CHROME_PATH", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
)


def find_chrome() -> Optional[str]:
    for candidate in CHROME_CANDIDATES:
        if not candidate:
            continue
        if os.path.isfile(candidate):
            return candidate
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def extract_svgs(html: str) -> List[Tuple[str, str, int, int]]:
    """Return ``(slug, svg_markup, width, height)`` for every diagram found."""
    found = []
    for markup in SVG_RE.findall(html):
        slug_match = TITLE_ID_RE.search(markup)
        box_match = VIEWBOX_RE.search(markup)
        if not slug_match:
            raise ValueError("an <svg> has no aria-labelledby '<slug>-title' id")
        if not box_match:
            raise ValueError(f"<svg> '{slug_match.group(1)}' has no viewBox starting at 0 0")
        width, height = (int(float(v)) for v in box_match.groups())
        found.append((slug_match.group(1), markup, width, height))
    return found


def write_standalone_svg(markup: str, out_path: Path) -> None:
    """Write a diagram as a well-formed standalone SVG.

    The font @import is merged into the existing `<defs>` and its `&`
    separators are XML-escaped: a standalone .svg is parsed as strict XML,
    where a bare `&` opens an entity reference and breaks the whole file.
    """
    if "<defs>" in markup:
        svg = markup.replace("<defs>", f"<defs>\n<style>{FONT_IMPORT}</style>", 1)
    else:
        svg = re.sub(r"(<svg\b[^>]*>)", rf"\1\n<defs><style>{FONT_IMPORT}</style></defs>", markup, count=1)
    out_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg + "\n", encoding="utf-8")


def render_png_playwright(page_html: Path, out_path: Path, width: int, height: int, scale: int) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": width, "height": height}, device_scale_factor=scale
        )
        page.goto(page_html.as_uri())
        page.wait_for_load_state("networkidle")
        page.locator("svg").first.screenshot(path=str(out_path))
        browser.close()
    return True


def render_png_chrome(chrome: str, page_html: Path, out_path: Path, width: int, height: int, scale: int) -> bool:
    """Screenshot a page sized exactly to the diagram, so no cropping is needed."""
    cmd = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--force-device-scale-factor={scale}",
        f"--window-size={width},{height}",
        "--virtual-time-budget=8000",
        f"--screenshot={out_path}",
        page_html.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return out_path.exists() and proc.returncode == 0


def build_page(markup: str, width: int, height: int) -> str:
    """A minimal page holding one diagram at its exact design size."""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?"
        "family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600"
        "&family=Geist+Mono:wght@400;500;600"
        "&family=Be+Vietnam+Pro:wght@400;500;600&display=swap' rel='stylesheet'>"
        "<style>html,body{margin:0;padding:0;background:#f5f5f5}"
        f"svg{{display:block;width:{width}px;height:{height}px}}</style></head>"
        f"<body>{markup}</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="figures/pipeline-cdane-vs-mt.html")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--scale", type=int, default=2, help="PNG pixel density")
    parser.add_argument("--svg-only", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    diagrams = extract_svgs(source.read_text(encoding="utf-8"))
    print(f"{source.name}: {len(diagrams)} diagram(s)")

    chrome = None if args.svg_only else find_chrome()
    with tempfile.TemporaryDirectory() as tmp:
        for slug, markup, width, height in diagrams:
            stem = f"pipeline-{slug}"
            svg_path = out_dir / f"{stem}.svg"
            write_standalone_svg(markup, svg_path)
            print(f"  {svg_path.name}  ({width}x{height})")

            if args.svg_only:
                continue

            page = Path(tmp) / f"{stem}.html"
            page.write_text(build_page(markup, width, height), encoding="utf-8")
            png_path = out_dir / f"{stem}.png"

            if render_png_playwright(page, png_path, width, height, args.scale):
                print(f"  {png_path.name}  ({width * args.scale}x{height * args.scale}, playwright)")
            elif chrome and render_png_chrome(chrome, page, png_path, width, height, args.scale):
                print(f"  {png_path.name}  ({width * args.scale}x{height * args.scale}, chrome)")
            else:
                print(
                    f"  {png_path.name}  SKIPPED - no browser found.\n"
                    "    Install Playwright (pip install playwright && playwright install chromium)\n"
                    "    or set CHROME_PATH to a Chrome/Chromium binary."
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
