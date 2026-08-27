#!/usr/bin/env python3
"""Turn the cut-out crops into blind sheets a reader can transcribe, and collect
the transcriptions back.

    python3 data/scripts/label_crops.py sheets      # build the sheets
    python3 data/scripts/label_crops.py collect     # merge the answers back

Two things this script exists to enforce.

**The reader's own guess is never shown.** `crops/labels.tsv` carries Lilly's
reading in column 2. Showing that to whoever labels the crop turns the job from
transcription into proofreading, and a proofreader agrees with a confident wrong
answer far more often than a transcriber independently produces it. The training
set would then be shaped like the reader's existing errors, and training on it
would teach the reader to make them more firmly. So the sheets carry the picture
and an index, and nothing else.

**A crop with no readable text has to be sayable.** The region detector produces
false positives -- one sampled crop is a stretch of decorative moulding with no
text in it at all -- and it cuts some regions so tightly that a letter is sliced
off. An annotator with no way to say "there is nothing here" invents something,
and an invented label is worse than a missing one: it is noise the trainer
cannot tell from signal. `__NOTEXT__` and `__UNSURE__` are first-class answers.

Why sheets rather than one image at a time: 1,914 separate reads is 1,914 round
trips. Eight crops to a sheet is 240. The index is drawn into the picture beside
each crop rather than only listed in the manifest, so an answer can be checked
against the image it came from instead of against a position in a list.

Why upscaling: half these crops are under 20 pixels tall, and the median is 24.
At that size the text is legible to nothing. Scaled to 60px with LANCZOS the
same crops read cleanly -- "zvjezdarnice. Tada" out of a 13px strip. The
upscaling is for the annotator's eye only; the trainer still gets the original.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CROPS = REPO_ROOT / "data" / "ocr" / "crops"
SHEET_DIR = REPO_ROOT / "data" / "ocr" / "label-sheets"
ANSWER_DIR = REPO_ROOT / "data" / "ocr" / "label-answers"
TRUTH = CROPS / "labels-human.tsv"

PER_SHEET = 8
# What each crop is scaled to for the annotator. This started at 60px, which is
# legible but not comfortably so: the annotators who worked from it cropped
# almost every strip back out of the sheet and re-upscaled it themselves, one
# PIL call at a time, and spent most of their budget doing it. 110px is the
# same picture rendered once, properly, so that reading the sheet is usually
# enough. PER_SHEET is deliberately unchanged -- the sheet-and-index numbering
# is derived from crop order, so answers already collected stay valid.
TARGET_H = 110
MAX_W = 1100
GUTTER = 78            # room for the index number
PAD = 14


def build_sheets(per_sheet: int = PER_SHEET) -> int:
    from PIL import Image, ImageDraw, ImageFont

    names = [l.split("\t")[0] for l in
             (CROPS / "labels.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
    names = [n for n in names if (CROPS / n).exists()]
    if not names:
        print(f"no crops in {CROPS}", file=sys.stderr)
        return 0

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for stale in SHEET_DIR.glob("*"):
        stale.unlink()

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30)
    except OSError:
        font = ImageFont.load_default()

    sheets = 0
    manifest = []
    for start in range(0, len(names), per_sheet):
        chunk = names[start:start + per_sheet]
        tiles = []
        for name in chunk:
            im = Image.open(CROPS / name).convert("RGB")
            scale = TARGET_H / im.height
            w, h = max(int(im.width * scale), 1), TARGET_H
            if w > MAX_W:                      # very wide strips: fit the width instead
                h = max(int(h * MAX_W / w), 1)
                w = MAX_W
            tiles.append(im.resize((w, h), Image.LANCZOS))

        width = GUTTER + max(t.width for t in tiles) + PAD
        height = sum(t.height for t in tiles) + PAD * (len(tiles) + 1)
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)

        y = PAD
        for i, (name, tile) in enumerate(zip(chunk, tiles), start=1):
            draw.text((10, y + tile.height // 2 - 18), f"{i}", fill="#c00000", font=font)
            sheet.paste(tile, (GUTTER, y))
            # A hairline under every tile, so two crops of similar colour on a
            # pale background cannot read as one.
            draw.line([(GUTTER, y + tile.height + PAD // 2),
                       (width - PAD, y + tile.height + PAD // 2)], fill="#d0d0d0")
            manifest.append((sheets + 1, i, name))
            y += tile.height + PAD

        sheets += 1
        sheet.save(SHEET_DIR / f"sheet-{sheets:04d}.png")

    with open(SHEET_DIR / "manifest.tsv", "w", encoding="utf-8") as f:
        for sheet_no, idx, name in manifest:
            f.write(f"{sheet_no}\t{idx}\t{name}\n")

    print(f"{len(names):,} crops -> {sheets} sheets in {SHEET_DIR}")
    print(f"manifest: {SHEET_DIR / 'manifest.tsv'}  (sheet, index, crop filename)")
    print("the sheets carry no reading of any kind — that is the point")
    return sheets


def collect() -> int:
    """Merge per-agent answer files into one truth TSV, checking as it goes."""
    manifest = {}
    for line in (SHEET_DIR / "manifest.tsv").read_text(encoding="utf-8").splitlines():
        sheet_no, idx, name = line.split("\t")
        manifest[(int(sheet_no), int(idx))] = name

    answers, clashes = {}, 0
    files = sorted(ANSWER_DIR.glob("*.tsv"))
    if not files:
        print(f"no answer files in {ANSWER_DIR}", file=sys.stderr)
        return 1
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                print(f"  {path.name}:{lineno}: not three columns — skipped")
                continue
            try:
                key = (int(parts[0]), int(parts[1]))
            except ValueError:
                continue                      # a header line
            if key not in manifest:
                print(f"  {path.name}:{lineno}: sheet {key} is not in the manifest")
                continue
            text = "\t".join(parts[2:]).strip()
            if key in answers and answers[key] != text:
                clashes += 1
            answers[key] = text

    kept, notext, unsure = [], 0, 0
    for key, name in sorted(manifest.items()):
        if key not in answers:
            continue
        text = answers[key]
        if text == "__NOTEXT__":
            notext += 1
            continue
        if text == "__UNSURE__" or not text:
            unsure += 1
            continue
        kept.append((name, text))

    with open(TRUTH, "w", encoding="utf-8") as f:
        for name, text in kept:
            f.write(f"{name}\t{text}\n")

    total = len(manifest)
    print(f"answered : {len(answers):,} / {total:,} crops")
    print(f"usable   : {len(kept):,}")
    print(f"no text  : {notext:,}  (false detections — excluded, not guessed at)")
    print(f"unsure   : {unsure:,}  (excluded)")
    if clashes:
        print(f"clashes  : {clashes} (same crop answered twice, differently)")
    print(f"\nwrote {TRUTH}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("sheets", "collect"))
    ap.add_argument("--per-sheet", type=int, default=PER_SHEET)
    args = ap.parse_args()
    if args.action == "sheets":
        return 0 if build_sheets(args.per_sheet) else 1
    return collect()


if __name__ == "__main__":
    raise SystemExit(main())
