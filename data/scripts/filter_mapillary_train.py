#!/usr/bin/env python3
"""Keep the Mapillary crops that can enter a train list; quarantine the rest.

    python3 data/scripts/filter_mapillary_train.py \
        --crops data/ocr/crops-kaggle \
        --write-train-dir data/ocr/mapillary-train

EasyOCR labelled every region it found. A third of those labels are dashcam
on-screen display, many more are one-letter pul crops or low-confidence junk.
Those must not become ground truth. The keep rule is the one measured on
2026-09-01 (docs/mapillary-pseudo-labels.md): confidence ≥ 0.85, at least
three letters, no dashcam OSD.

Hats falling off (kuća → kuca) are kept on purpose. The product bar is that
the translator still gets the meaning (kuca → House), not that the crop gate
letter column rises.

Does not touch data/ocr/train or data/ocr/valid. Mixing these rows into the
shared split would make mly_* files look like "real" to the crop gate and
would score EasyOCR against itself. The train folder written here is loaded
with train_ocr.py --train-dir.

Writes labels.tsv in place (backup beside it) and moves dropped PNGs into
dropped-osd/, dropped-short/, dropped-lowconf/ rather than deleting them.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

MIN_CONF = 0.85
MIN_LETTERS = 3

# Pattern, not confidence: BlackVue overlays read at 0.99 and still are not
# signage. HIC OFF is MIC OFF with the M eaten. knh is km/h misread.
OSD_RE = re.compile(
    r"(?i)("
    r"DR750|BLACKVUE|BLACKUUE|BLACKUE|MIC OFF|HIC OFF|"
    r"\bHDR\b|\bFHD\b|\bUHD\b|"
    r"\d{2,3}\s*km/?h|kmh|knh|"
    r"\d{1,2}:\d{2}(:\d{2})?|"
    r"Auto-?\s*0|"
    r"CH/FHD|/NU\b|/NV\b"
    r")"
)
OSD_EXACT = re.compile(r"(?i)^(hdr|fhd|uhd|4k|mic|off|nv|hic)$")

DROP_DIRS = {
    "osd": "dropped-osd",
    "short": "dropped-short",
    "lowconf": "dropped-lowconf",
}


def letter_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def is_osd(text: str) -> bool:
    t = text.strip()
    return bool(OSD_RE.search(t) or OSD_EXACT.fullmatch(t))


def classify(text: str, conf: float,
             min_conf: float = MIN_CONF,
             min_letters: int = MIN_LETTERS) -> str | None:
    """Drop reason, or None to keep."""
    if is_osd(text):
        return "osd"
    if letter_count(text) < min_letters:
        return "short"
    if conf < min_conf:
        return "lowconf"
    return None


def parse_row(line: str) -> tuple[str, str, float] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None
    try:
        return parts[0], parts[1], float(parts[2])
    except ValueError:
        return None


def self_test() -> None:
    assert classify("BLACKVUE", 0.99) == "osd"
    assert classify("033kmh", 0.93) == "osd"
    assert classify("FHD", 1.00) == "osd"
    assert classify("MIC OFF", 0.76) == "osd"
    assert classify("P", 0.99) == "short"
    assert classify("7", 0.97) == "short"
    assert classify("mala kuhinja", 0.40) == "lowconf"
    assert classify("kuca", 0.85) is None
    assert classify("ŠKOLA", 1.00) is None
    assert classify("IZLAZ", 0.90) is None
    print("self-test ok")


def load_labels(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        sys.exit(f"{path} is empty")
    header = lines[0]
    if "file" not in header.split("\t")[0] and not header.startswith("file"):
        # some writers omit a header; treat the first line as a row
        return "file\ttext\tconfidence", lines
    return header, lines[1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True,
                    help="directory holding the PNGs and labels.tsv")
    ap.add_argument("--write-train-dir", type=Path, default=None,
                    help="copy kept crops + gt.txt here (does not touch "
                         "data/ocr/train)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    labels = args.crops / "labels.tsv"
    if not labels.exists():
        sys.exit(f"no labels.tsv in {args.crops}")

    header, rows = load_labels(labels)
    kept, dropped = [], Counter()
    drop_files: dict[str, list[str]] = {k: [] for k in DROP_DIRS}

    for row in rows:
        parsed = parse_row(row)
        if parsed is None:
            kept.append(row)
            continue
        name, text, conf = parsed
        reason = classify(text, conf)
        if reason is None:
            kept.append(row)
        else:
            dropped[reason] += 1
            drop_files[reason].append(name)

    print(f"{len(rows):,} latin rows in {labels}")
    for reason in ("osd", "short", "lowconf"):
        print(f"  drop {dropped[reason]:>5,}  {reason}")
    print(f"  keep {len(kept):>5,}  conf≥{MIN_CONF}, ≥{MIN_LETTERS} letters, no OSD")

    if args.dry_run:
        print("dry run, nothing written")
        return 0

    for reason, folder in DROP_DIRS.items():
        dest = args.crops / folder
        dest.mkdir(exist_ok=True)
        moved = 0
        for name in drop_files[reason]:
            src = args.crops / name
            if src.exists():
                shutil.move(str(src), str(dest / name))
                moved += 1
        print(f"moved {moved:,} PNGs to {dest}")

    backup = labels.with_name("labels.tsv.pre-train-filter")
    if not backup.exists():
        shutil.copy(labels, backup)
        print(f"backup {backup}")
    labels.write_text("\n".join([header] + kept) + "\n", encoding="utf-8")
    train_list = args.crops / "labels-train.tsv"
    train_list.write_text("\n".join([header] + kept) + "\n", encoding="utf-8")
    print(f"rewrote {labels} ({len(kept):,} rows)")
    print(f"wrote {train_list}")

    if args.write_train_dir is not None:
        out = args.write_train_dir
        out.mkdir(parents=True, exist_ok=True)
        copied = 0
        missing = 0
        with (out / "gt.txt").open("w", encoding="utf-8") as gt:
            for row in kept:
                parsed = parse_row(row)
                if parsed is None:
                    continue
                name, text, _conf = parsed
                src = args.crops / name
                if not src.exists():
                    missing += 1
                    continue
                shutil.copy2(src, out / name)
                gt.write(f"{name}\t{text.strip()}\n")
                copied += 1
        print(f"train dir {out}: {copied:,} crops"
              + (f", {missing:,} listed PNG missing" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
