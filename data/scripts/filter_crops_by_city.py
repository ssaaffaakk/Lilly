#!/usr/bin/env python3
"""Drop crops whose source photo came from a Cyrillic-signage city.

    python3 data/scripts/filter_crops_by_city.py \
        --crops data/ocr/crops-mapillary \
        --credits data/ocr/real-photos/mapillary/CREDITS.tsv

EasyOCR's latin_g2 recogniser has no Cyrillic in its character set. A Cyrillic
sign therefore does not come back unread — it comes back transcribed into
Latin lookalikes at ordinary confidence. Those rows would enter training as
ground truth, so they are removed rather than trusted.

v1 does not read Cyrillic, so this is a filter and not a loss.

Crop files are named `mly_<image_id>_<region>.png`; CREDITS.tsv maps
`mly_<image_id>.jpg` to the city it was harvested from.

Writes labels.tsv in place (keeping a .bak) and moves the dropped PNGs into a
`dropped-cyrillic/` subdirectory rather than deleting them, so the decision
stays reversible.
"""
import argparse
import shutil
import sys
from pathlib import Path

# Beograd, Novi Sad and Niš sign in Cyrillic. Banja Luka is officially
# Cyrillic and mixed in practice, so it is dropped too rather than sampled.
CYRILLIC_CITIES = {"beograd", "novi_sad", "nis", "banjaluka"}


def load_city_map(credits: Path) -> dict:
    with credits.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            ifile = header.index("file")
            icity = header.index("city")
        except ValueError:
            sys.exit(f"{credits} has no 'file'/'city' columns: {header}")

        city_of = {}
        for row in fh:
            parts = row.rstrip("\n").split("\t")
            if len(parts) > max(ifile, icity):
                city_of[Path(parts[ifile]).stem] = parts[icity]
    return city_of


def photo_stem(crop_name: str) -> str:
    """mly_1234_007.png -> mly_1234"""
    return crop_name.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True,
                    help="directory holding the PNGs and labels.tsv")
    ap.add_argument("--credits", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    labels = args.crops / "labels.tsv"
    if not labels.exists():
        sys.exit(f"no labels.tsv in {args.crops}")

    city_of = load_city_map(args.credits)
    print(f"{len(city_of):,} photos in CREDITS.tsv")

    lines = labels.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]

    kept, dropped = [], {}
    unknown = 0
    for row in rows:
        name = row.split("\t", 1)[0]
        city = city_of.get(photo_stem(name))
        if city is None:
            unknown += 1
            kept.append(row)
        elif city in CYRILLIC_CITIES:
            dropped.setdefault(city, []).append(name)
        else:
            kept.append(row)

    total_dropped = sum(len(v) for v in dropped.values())
    for city in sorted(dropped):
        print(f"  drop {len(dropped[city]):>6,} crops from {city}")
    if unknown:
        print(f"  keep {unknown:,} crops whose photo is not in CREDITS.tsv")
    print(f"\n{len(rows):,} crops -> {len(kept):,} kept, {total_dropped:,} dropped")

    if args.dry_run:
        print("dry run, nothing written")
        return

    if not total_dropped:
        print("nothing to drop")
        return

    quarantine = args.crops / "dropped-cyrillic"
    quarantine.mkdir(exist_ok=True)
    moved = 0
    for names in dropped.values():
        for name in names:
            src = args.crops / name
            if src.exists():
                shutil.move(str(src), str(quarantine / name))
                moved += 1

    shutil.copy(labels, labels.with_suffix(".tsv.bak"))
    labels.write_text("\n".join([header] + kept) + "\n", encoding="utf-8")

    print(f"moved {moved:,} PNGs to {quarantine}")
    print(f"rewrote {labels} ({len(kept):,} rows), backup at {labels}.bak")


if __name__ == "__main__":
    main()
