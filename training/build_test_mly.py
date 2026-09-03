#!/usr/bin/env python3
"""Draw a street-level test set from the Mapillary harvest — the product's domain.

docs/OCR-ROADMAP.md, step 2b. The owner's line (3 Sep 2026): Lilly reads small
signs and street names; long text is reported, not claimed. The Commons sets
(the 40, test-v2) lean toward plaques and boards. This is the other half: the
20,240 street photographs harvested from Mapillary, which have only ever been
used as a source of self-labelled crops and never as a test set.

    python3 training/build_test_mly.py             # on the Mac; the photos are there

Reads `data/ocr/real-photos/mapillary/CREDITS.tsv` — file, image_id, city,
near, licence, creator, bosnian — which is the only map from `mly_<id>.jpg` to
its city and its CC BY-SA attribution. Keeps the Latin-signage cities (Beograd
and Banja Luka are Cyrillic and the reader is Latin-only, `V2-BOUNDARIES.md`),
ranks by the same filename hash the 40 and test-v2 were drawn with, and takes
the first 240. Nothing is chosen by looking.

Two things a reader of the result must know:

- **The harvest was pre-screened by Lilly's own reader.** `harvest_mapillary.py`
  kept a photograph only when `app.ocr.read_regions` found two or more words
  in it. So this set is "street photographs the shipped reader can find text
  in", and its recall is an upper bound for the domain. It compares engines
  fairly only in the sense that both read the same photographs; the selection
  favoured EasyOCR's detector. A future harvest without that screen fixes it;
  this file does not pretend to.
- **The photographs are 1024px Mapillary thumbnails**, smaller than the 1280px
  Commons renderings. Report the size beside the number.

Transcription and scoring are the same as test-v2:
    python3 training/build_truth.py --result <pair.json> --out data/ocr/real-photos/test-mly/truth-mly.json
    python3 training/evaluate_ocr.py --truth data/ocr/real-photos/test-mly/truth-mly.json \\
        --photos data/ocr/real-photos/mapillary --sample data/ocr/real-photos/test-mly/sample.txt \\
        --cache data/ocr/real-photos/test-mly/reader-output.json --out training/RESULTS-ocr-test-mly.md

The remainder of the pool is what a future training set may be cut from —
with labels from a blind transcription, never from the reader (step 4).
`sample.txt` is frozen the day it is committed.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from training.build_test_v2 import rank  # noqa: E402

CREDITS = REPO_ROOT / "data" / "ocr" / "real-photos" / "mapillary" / "CREDITS.tsv"
OUT_DIR = REPO_ROOT / "data" / "ocr" / "real-photos" / "test-mly"
CYRILLIC_CITIES = ("beograd", "belgrade", "banjaluka")


def city_key(city: str) -> str:
    return re.sub(r"[^a-z]", "", city.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=240)
    ap.add_argument("--credits", type=Path, default=CREDITS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not args.credits.is_file():
        raise SystemExit(f"{args.credits} is missing. It lives on the Mac that ran the harvest "
                         "and should be committed (docs/OCR-ROADMAP.md, Mac checklist).")
    truth = args.out_dir / "truth-mly.json"
    sample = args.out_dir / "sample.txt"
    if sample.exists() and truth.exists():
        raise SystemExit(f"{sample.relative_to(REPO_ROOT)} is frozen: {truth.name} was transcribed "
                         "for it. A new draw is a new ruler.")

    with args.credits.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    for col in ("file", "city", "licence", "creator"):
        if col not in rows[0]:
            raise SystemExit(f"{args.credits} has no {col!r} column: {list(rows[0])}")
    seen = set()
    pool = []
    for r in rows:
        if r["file"] in seen:
            continue
        seen.add(r["file"])
        if city_key(r["city"]).startswith(CYRILLIC_CITIES):
            continue
        pool.append(r)
    drawn = sorted(pool, key=lambda r: rank(r["file"]))[:args.count]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample.write_text("".join(r["file"] + "\n" for r in drawn), encoding="utf-8")
    with (args.out_dir / "CREDITS.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t",
                           quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        for r in drawn:
            w.writerow(r)

    print(f"{len(rows)} harvested, {len(pool)} in Latin-signage cities, {len(drawn)} drawn")
    print("by city:", dict(Counter(r["city"] for r in drawn)))
    print(f"wrote {sample.relative_to(REPO_ROOT)} and CREDITS.tsv beside it — commit both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
