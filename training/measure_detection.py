"""What the CRAFT detector finds on a real photograph, and what it misses.

Nothing in this project has ever measured finding text, only reading it. The
54.7% the reader scores per photograph is the product of two stages and the
number cannot say which one is the ceiling: a word the detector never boxed is
lost before the recogniser is asked, and a word boxed but misread is lost after.
Fine-tuning fixes the second and does nothing for the first, so choosing a
training recipe without this split is guessing.

The recogniser is fine-tuned; the detector is stock `craft_mlt_25k.pth`.

WHY THIS DOES NOT SCORE ITSELF. Deciding whether a box contains a particular
word from the answer key means reading the box, and the only reader available is
the thing under test. Every automatic version of that measures the recogniser
and prints the result under the detector's name. So this script does not decide
anything: it draws every box the detector produced onto the exact image the
detector saw and stops. A person then looks at the overlay beside the answer key
and counts, which is the same method that produced the answer key itself.

WHY THE BOXES LINE UP. `app.ocr.scan` shrinks anything over two megapixels
before reading it, so boxes come back in shrunk coordinates. Drawing them on the
original would put every box in the wrong place and the overlay would show a
detector far worse than the real one. This calls the same
`app.ocr.load_within_limits` and draws on what it returns.

WHY detail=1 AND paragraph=False. `scan` asks for `detail=0, paragraph=True`,
which throws the geometry away and then merges regions into paragraphs. The
merge happens *after* detection and changes nothing about what was found, so
ungrouped boxes are the detector's actual output. This is the same
`read_regions` call `scan` makes, with the two arguments that decide what is
returned rather than what is computed.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.guard import claim


def load_sample(path: Path) -> list:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def truth_words(truth: Path) -> dict:
    """filename -> the agreed words on that photograph, in reading order."""
    data = json.loads(truth.read_text(encoding="utf-8"))
    out = {}
    for name, entry in data["photos"].items():
        out[name] = [w for line in entry.get("lines", []) for w in line.split()]
    return out


def corners(box):
    """easyocr returns four points; overlays want a rectangle around them."""
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def draw(image, regions, out_path: Path, max_side: int = 1500) -> None:
    from PIL import Image, ImageDraw

    img = Image.fromarray(image).convert("RGB")
    canvas = ImageDraw.Draw(img)
    for i, (box, _text, _conf) in enumerate(regions):
        x0, y0, x1, y1 = corners(box)
        canvas.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        # The index, not the reading: the point of the overlay is to show where
        # the detector looked, and printing its guess next to the box invites
        # the eye to grade recognition instead.
        canvas.text((x0 + 3, max(y0 - 12, 0)), str(i), fill=(255, 0, 0))
    img.thumbnail((max_side, max_side))
    img.save(out_path, quality=90)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/scored")
    ap.add_argument("--sample", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/scored-sample.txt")
    ap.add_argument("--truth", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/truth.json")
    ap.add_argument("--overlays", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/detection-overlays")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/detection-boxes.json")
    args = ap.parse_args()

    # Claimed here and not at import, because a claim at import charges 1.4 GB
    # to merely importing the module: --help is refused, and so is any run that
    # loads no model at all. listen found the same defect in two of its scripts,
    # where importing cost 2 GB and a controls-only run could not start.
    claim(1.4, "detector measurement")
    from app.ocr import load_within_limits, read_regions

    names = load_sample(args.sample)
    truth = truth_words(args.truth)
    missing = [n for n in names if not (args.photos / n).exists()]
    if missing:
        # Refusing rather than scoring 37 of 40 and printing it as the number:
        # twenty-five of these photographs were deleted once already and the
        # measurement that followed would have quietly been of whatever was left.
        print(f"{len(missing)} of {len(names)} photographs are not in "
              f"{args.photos}: {missing[:5]}", file=sys.stderr)
        return 1

    args.overlays.mkdir(parents=True, exist_ok=True)
    record = {}
    for i, name in enumerate(names, 1):
        image = load_within_limits(str(args.photos / name))
        regions = read_regions(image, detail=1, paragraph=False)
        height, width = image.shape[:2]
        record[name] = {
            "width": width,
            "height": height,
            "truth_words": len(truth.get(name, [])),
            "boxes": [{"box": [[float(p[0]), float(p[1])] for p in box],
                       "text": text,
                       "conf": float(conf)}
                      for box, text, conf in regions],
        }
        draw(image, regions, args.overlays / f"{Path(name).stem}.jpg")
        args.out.write_text(json.dumps(record, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"  {i}/{len(names)} {name}  {width}x{height}  "
              f"{len(regions)} boxes  {len(truth.get(name, []))} truth words",
              flush=True)

    boxes = sum(len(v["boxes"]) for v in record.values())
    words = sum(v["truth_words"] for v in record.values())
    print(f"\n{len(record)} photographs, {boxes} detected regions, "
          f"{words} words in the answer key.")
    print(f"overlays: {args.overlays}")
    print(f"boxes:    {args.out}")
    print("\nDetection recall is not computed here on purpose. Read the "
          "overlays against the answer key and count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
