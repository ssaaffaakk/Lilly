#!/usr/bin/env python3
"""Build a training set for Lilly's photo reading.

Reading a photo is two models: one finds where the text is, one reads it. Only
the reader can be retrained — the finder's training code was never released by
its authors — and retraining the reader is what fixes Bosnian letters like
č ć đ š ž coming back wrong.

Two steps, because the labels have to come from a human:

    # 1. cut the text out of photos you took, with Lilly's guess filled in
    python3 training/prepare_ocr_data.py --photos data/ocr/photos

    # 2. open crops/labels.tsv, fix every wrong line by hand, then
    python3 training/prepare_ocr_data.py --labels data/ocr/crops/labels.tsv

Step 2 writes data/ocr/train/ and data/ocr/valid/, each a folder of images plus
a gt.txt of "image<TAB>text" — the layout the recogniser trainer expects. The
trainer itself lives upstream: see EasyOCR's trainer directory and its
custom_model.md, which also documents the .yaml and .py files a trained
recogniser needs before the app can load it.

Aim for a few thousand crops. A few dozen will not move anything.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.ocr_split import VALID_SHARE, is_valid_text, split_key  # noqa: E402

OCR_DIR = REPO_ROOT / "data" / "ocr"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
OURS = "syn"      # what this script names its files; anything else belongs to someone else
# the letters that separate Bosnian from plain Latin — worth watching coverage of
BOSNIAN_LETTERS = "čćđšžČĆĐŠŽ"


def cut_out_text(photo_dir: Path, out_dir: Path) -> int:
    """Crop every text region Lilly finds, with its reading as a first draft."""
    from PIL import Image
    from app.ocr import get_reader

    photos = sorted(p for p in photo_dir.rglob("*") if p.suffix.lower() in IMAGE_TYPES)
    if not photos:
        print(f"no images in {photo_dir}", file=sys.stderr)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = get_reader()

    rows = []
    for photo in photos:
        image = Image.open(photo).convert("RGB")
        for i, (box, text, confidence) in enumerate(reader.readtext(str(photo))):
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            crop = image.crop((max(min(xs), 0), max(min(ys), 0), max(xs), max(ys)))
            if crop.width < 8 or crop.height < 8:
                continue
            name = f"{photo.stem}_{i:03d}.png"
            crop.save(out_dir / name)
            rows.append((name, text.strip(), confidence))
        print(f"  {photo.name}: {sum(1 for r in rows if r[0].startswith(photo.stem))} crops")

    labels = out_dir / "labels.tsv"
    with open(labels, "w", encoding="utf-8") as f:
        for name, text, confidence in rows:
            f.write(f"{name}\t{text}\t{confidence:.2f}\n")
    print(f"\n{len(rows)} crops -> {out_dir}\nnow correct the second column of {labels} by hand")
    print("the third column is Lilly's confidence — the low ones are where it struggles")
    return len(rows)


def build_splits(labels_path: Path) -> int:
    """Turn corrected crops into the folder + gt.txt layout the trainer reads."""
    import shutil

    rows = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip():
            source = labels_path.parent / parts[0]
            if source.exists():
                rows.append((source, parts[1].strip()))
    if not rows:
        print(f"no labelled crops in {labels_path}", file=sys.stderr)
        return 1

    # Split on the label text, not on the row. One string becomes many images
    # — several variants here, separately photographed ones from
    # data/scripts/generate_ocr_photos.py — so cutting a shuffled row list
    # scatters pictures of the same word across both sides, and validation
    # starts scoring recall of the training set. Measured before this change:
    # 1,350 of 2,402 valid rows carried text the model had trained on.
    # is_valid_text() asks the string, so every rendering of it lands together,
    # whichever script drew it and whenever.
    chunks = {"valid": [], "train": []}
    for source, text in rows:
        chunks["valid" if is_valid_text(text) else "train"].append((source, text))

    for split, chunk in (("valid", chunks["valid"]), ("train", chunks["train"])):
        out = OCR_DIR / split
        # Clear only what this script put there. Someone else may be adding
        # crops to the same folder — photographs, hand-labelled examples — and
        # wiping the directory deletes their work between two runs of this one,
        # silently, with the folder looking freshly built either way.
        out.mkdir(parents=True, exist_ok=True)
        kept = []
        gt = out / "gt.txt"
        if gt.exists():
            for line in gt.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) == 2 and not parts[0].startswith(OURS) \
                        and (out / parts[0]).exists():
                    kept.append((parts[0], parts[1]))
        for ours in out.glob(f"{OURS}*"):
            ours.unlink()

        with open(gt, "w", encoding="utf-8") as f:
            for source, text in chunk:
                shutil.copy(source, out / source.name)
                f.write(f"{source.name}\t{text}\n")
            for name, text in kept:
                f.write(f"{name}\t{text}\n")
        note = f", kept {len(kept):,} from elsewhere" if kept else ""
        print(f"{split}: {len(chunk):,} crops -> {out}{note}")

    letters = Counter(c for _, text in rows for c in text)
    print("\nBosnian letter coverage — a letter the training set barely contains "
          "will stay wrong:")
    for letter in BOSNIAN_LETTERS:
        count = letters.get(letter, 0)
        print(f"  {letter}: {count:>5}{'   <- thin' if count < 20 else ''}")

    # A hash lands near the share, not on it, and the fewer distinct labels
    # there are the further it drifts — so print what was actually realised
    # rather than letting VALID_SHARE be assumed.
    print(f"\nvalid took {100 * len(chunks['valid']) / len(rows):.1f}% of the "
          f"crops (VALID_SHARE is {100 * VALID_SHARE:.0f}%)")
    if not chunks["valid"]:
        # Deciding per text has no equivalent of the old max(1, ...): with a
        # handful of distinct labels every one of them can hash to train, and an
        # empty valid folder fails much later, deep inside the trainer.
        print("nothing landed in valid — too few distinct labels for a split to "
              "hold any share; label more crops before training", file=sys.stderr)
        return 1
    return check_no_shared_text()


def check_no_shared_text() -> int:
    """Re-read both gt.txt files and refuse to sign off a leaking split.

    Reads what is on disk, not what this run just wrote: these folders also hold
    rows other generators appended straight into them, and a string shared
    between theirs and ours is exactly the leak the text-based split exists to
    prevent. Checking here means a leak announces itself in the run that caused
    it, instead of quietly inflating a validation score for months.
    """
    seen = {}
    for split in ("train", "valid"):
        gt = OCR_DIR / split / "gt.txt"
        texts = []
        if gt.exists():
            for line in gt.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) == 2:
                    # split_key, not the raw text: compare labels the same way
                    # the split assigned them, or this check disagrees with the
                    # rule it is checking.
                    texts.append(split_key(parts[1]))
        seen[split] = texts

    shared = set(seen["train"]) & set(seen["valid"])
    if not shared:
        print(f"no label text is on both sides: {len(set(seen['train'])):,} "
              f"distinct texts in train, {len(set(seen['valid'])):,} in valid")
        return 0

    rows = sum(1 for text in seen["valid"] if text in shared)
    print(f"\nLEAK: {len(shared):,} label texts are in train and valid both, "
          f"covering {rows:,} of {len(seen['valid']):,} valid rows — that much of "
          f"the validation score is the model reciting what it trained on.\n"
          f"Something wrote those rows without asking training/ocr_split.py's "
          f"is_valid_text(); e.g. "
          f"{', '.join(repr(t) for t in sorted(shared)[:3])}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, help="folder of photos to cut text out of")
    ap.add_argument("--labels", type=Path, help="corrected labels.tsv to build splits from")
    ap.add_argument("--crops", type=Path, default=OCR_DIR / "crops")
    args = ap.parse_args()

    if args.photos:
        return 0 if cut_out_text(args.photos, args.crops) else 1
    if args.labels:
        return build_splits(args.labels)
    ap.error("give either --photos or --labels")


if __name__ == "__main__":
    raise SystemExit(main())
