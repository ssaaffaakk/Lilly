#!/usr/bin/env python3
"""Select and balance Mapillary pseudo-labels the way the self-training
literature says to, rather than the way that looked careful and failed twice.

    python3 data/scripts/build_mapillary_train.py \
        --crops data/ocr/crops-kaggle \
        --out data/ocr/mapillary-train-balanced

Pass-14 took conf >= 0.85 and got 915 crops; pass-15 took every Latin crop and
got 5,988. Both were refused, and the larger set did more damage. The reason
was not the amount of data — it was that a global confidence cut is a
class-unaware filter, and the class it cut hardest was the one the reader
most needs.

Three findings drive this script (docs/diacritic-gate-literature.md):

- Kahn et al., ICASSP 2020: "more aggressive filtering improves the label
  quality but results in worse model performance." A high threshold is not
  the safe choice it looks like.
- Zou et al., ECCV 2018: confidence filtering is "intrinsically biased
  towards the easy (i.e. more confident) classes". Words carrying č ć đ š ž
  are the hard class here, so 0.85 removed them preferentially — the training
  set carried a diacritic on 1.0% of rows where human labels carry them on
  10.6%.
- Noisy Student (Xie et al., CVPR 2020), the canonical success, used a
  confidence threshold of 0.3 and balanced the classes by duplicating
  under-represented ones. Not 0.85, and not unbalanced.

So: keep the low threshold, and rebalance the diacritic rate to match the
human-labelled reference instead of whatever the reader happened to emit.

Two filters are NOT confidence and stay in place. Dashcam on-screen display
(FHD, MIC OFF/NV, 047km/h) is out-of-domain, not low-quality — Mapillary
carries dashcam frames with burned-in overlays, and the reader garbles the
overlay itself at low confidence, so the pattern list has to match the
garbled forms too. Cyrillic-city crops are out because the recogniser's
character set is Latin-only and it invents Latin lookalikes over Cyrillic
signs; script routing before recognition is standard practice (ICDAR MLT),
and those signs belong to a Cyrillic reader, not to this training set.
"""
import argparse
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Noisy Student's threshold, not the 0.85 that filtered out the diacritics.
MIN_CONF = 0.30
MIN_LETTERS = 3

# The human-labelled reference rate: 180 of 1,701 rows in
# data/ocr/crops/labels-human.tsv carry one of č ć đ š ž.
TARGET_DIACRITIC_RATE = 0.106

DIACRITICS = re.compile("[čćđšžČĆĐŠŽ]")

# Dashcam OSD, including the forms the reader garbles it into. `053km/h` comes
# back as `053kih`, `MIC OFF/NV` as `NIC UFF /NU`, so matching only the clean
# spelling lets the low-confidence half of this junk straight through — which
# is exactly what happened when the threshold was lowered by hand.
OSD = re.compile(
    r"(?i)("
    r"\bfhd\b|\bhdr\b|\buhd\b|\b4k\b"
    r"|blackvue|blacku|dr\d{3}|viofo|\bgopro\b|dashcam"
    r"|\d{2,3}\s?k[a-z]{1,2}\b"          # 047km/h and 047kih alike
    r"|[mn]i?c\s*[uo]f?f"                # MIC OFF and NIC UFF
    r"|/n[uv]\b"
    r"|^auto[-\s]?\d{3,4}"
    r"|\d{4}[-/]\d{2}[-/]\d{2}|\d{2}:\d{2}:\d{2}"
    r"|^\d{4}$"
    r")"
)

CYRILLIC_CITIES = {"beograd", "novi_sad", "nis", "banjaluka"}


def letters(text: str) -> int:
    return len(re.sub(r"[^A-Za-zČčĆćĐđŠšŽž]", "", text))


def fold(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def load_cities(credits: Path) -> dict:
    with credits.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            ifile, icity = header.index("file"), header.index("city")
        except ValueError:
            sys.exit(f"{credits} has no file/city columns: {header}")
        return {Path(p[ifile]).stem: p[icity]
                for p in (r.rstrip("\n").split("\t") for r in fh)
                if len(p) > max(ifile, icity)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--credits", type=Path,
                    default=Path("data/ocr/real-photos/mapillary/CREDITS.tsv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-conf", type=float, default=MIN_CONF)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    labels = args.crops / "labels.tsv.bak"
    if not labels.exists():
        labels = args.crops / "labels.tsv"
    if not labels.exists():
        sys.exit(f"no labels in {args.crops}")

    city_of = load_cities(args.credits)
    rows, drops = [], Counter()

    with labels.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            name, text, raw = parts
            try:
                conf = float(raw)
            except ValueError:
                continue
            text = text.strip()

            if city_of.get(name.rsplit("_", 1)[0]) in CYRILLIC_CITIES:
                drops["cyrillic"] += 1
            elif OSD.search(text):
                drops["dashcam"] += 1
            elif letters(text) < MIN_LETTERS:
                drops["too-short"] += 1
            elif conf < args.min_conf:
                drops["below-threshold"] += 1
            elif not (args.crops / name).is_file():
                drops["png-missing"] += 1
            else:
                rows.append((name, text, conf))

    with_dia = [r for r in rows if DIACRITICS.search(r[1])]
    without = len(rows) - len(with_dia)

    print(f"threshold {args.min_conf} (Noisy Student used 0.3, not 0.85)")
    for reason in ("cyrillic", "dashcam", "too-short", "below-threshold",
                   "png-missing"):
        print(f"  drop {drops[reason]:>6,}  {reason}")
    print(f"  keep {len(rows):>6,}")
    rate = len(with_dia) / max(len(rows), 1)
    print(f"\ndiacritic rows: {len(with_dia):,} of {len(rows):,} = {100*rate:.1f}% "
          f"(human labels: {100*TARGET_DIACRITIC_RATE:.1f}%)")

    # Duplicate the under-represented rows until the rate matches the human
    # reference. Noisy Student balanced by duplication; the alternative,
    # throwing away the majority, would spend the corpus to fix a ratio.
    extra = 0
    if with_dia and rate < TARGET_DIACRITIC_RATE:
        # solve (d + x) / (n + x) = target
        need = (TARGET_DIACRITIC_RATE * len(rows) - len(with_dia)) / \
               (1 - TARGET_DIACRITIC_RATE)
        extra = max(0, round(need))
        print(f"upsampling diacritic rows x{1 + extra / max(len(with_dia),1):.1f} "
              f"({extra:,} duplicate entries) to reach "
              f"{100*TARGET_DIACRITIC_RATE:.1f}%")
    elif not with_dia:
        print("no diacritic rows at all — nothing to balance, and nothing to learn")

    total = len(rows) + extra
    print(f"\ntraining rows: {total:,}")

    if args.dry_run:
        print("dry run, nothing written")
        return 0

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    with (args.out / "gt.txt").open("w", encoding="utf-8") as gt:
        for name, text, _ in rows:
            shutil.copy2(args.crops / name, args.out / name)
            gt.write(f"{name}\t{text}\n")
        # Duplicates are extra gt.txt lines pointing at the same PNG, so the
        # sampler sees them more often without another copy on disk.
        for i in range(extra):
            name, text, _ = with_dia[i % len(with_dia)]
            gt.write(f"{name}\t{text}\n")

    written = sum(1 for _ in (args.out / "gt.txt").open(encoding="utf-8"))
    pngs = len(list(args.out.glob("*.png")))
    print(f"wrote {args.out}: {pngs:,} PNGs, {written:,} gt.txt rows")
    if written != total:
        sys.exit(f"expected {total} rows, wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
