#!/usr/bin/env python3
"""Enlarge part of a photograph so a person can read small text on it.

    python3 training/transcribe/zoom.py PHOTO                      # 2x2 tiles at 2x
    python3 training/transcribe/zoom.py PHOTO --tiles 3            # 3x3 tiles at 2x
    python3 training/transcribe/zoom.py PHOTO --box x0 y0 x1 y1    # one region, ~1400 px wide
    python3 training/transcribe/zoom.py PHOTO --box x0 y0 x1 y1 --scale 4

Coordinates are pixels of the photograph as stored (EXIF orientation applied).
Prints the paths written; open them with an image viewer or the Read tool.
This only resizes pixels — it runs no OCR and reads no other file.
Output goes to --out, default data/ocr/real-photos/zooms/ (git ignores it).
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "ocr" / "real-photos" / "zooms"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--scale", type=float)
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    img = ImageOps.exif_transpose(Image.open(a.photo).convert("RGB"))
    w, h = img.size
    # The overlay of a photograph shares its basename in another directory, so
    # the parent directory is part of the name: a photo zoom must not overwrite
    # an overlay zoom of the same region (a counter lost one that way, 5 Sep).
    src = Path(a.photo).resolve()
    stem = f"{src.parent.name[:16]}__{src.stem[:40]}"
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{Path(a.photo).name}: {w}x{h}")
    if a.box:
        x0, y0, x1, y1 = a.box
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            raise SystemExit(f"empty box after clipping to {w}x{h}")
        crop = img.crop((x0, y0, x1, y1))
        s = a.scale or max(1.0, min(1400 / crop.width, 1400 / crop.height))
        crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
        p = a.out / f"{stem}_{x0}_{y0}_{x1}_{y1}.png"
        crop.save(p)
        print(p)
        return 0
    n, s = a.tiles, a.scale or 2.0
    for r in range(n):
        for c in range(n):
            x0, y0 = int(c * w / n), int(r * h / n)
            x1, y1 = int((c + 1) * w / n), int((r + 1) * h / n)
            crop = img.crop((x0, y0, x1, y1)).resize((int((x1 - x0) * s), int((y1 - y0) * s)), Image.LANCZOS)
            p = a.out / f"{stem}_tile{r}{c}_{x0}_{y0}_{x1}_{y1}.png"
            crop.save(p)
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
