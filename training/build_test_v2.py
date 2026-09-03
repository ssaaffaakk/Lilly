#!/usr/bin/env python3
"""The pool for the reader's second test set, and the draw from it.

docs/OCR-ROADMAP.md, step 2. The 40 scored photographs cannot see a three-point
change: 373 agreed words, 25 with a diacritic. This builds the pool a larger set
is drawn from, and draws it, using only files already in git, so the draw is
reproducible on any clone and cannot be nudged.

    python3 training/build_test_v2.py             # writes pool.tsv and sample.txt
    python3 training/build_test_v2.py --count 300 # a different size, same order

The pool is every photograph the harvester ever screened
(`harvested/.state/screened.tsv`, plus the kept set in `harvested/CREDITS.tsv`)
that is a photograph with an allowed licence, minus two exclusions:

- the 40 scored photographs, by filename and by Commons title;
- every photograph a labelled crop was cut from (`crops/labels-human.tsv`,
  `crops2/labels-human.tsv`) — the shipped reader trained on those crops.

What is deliberately NOT excluded: photographs the harvester dropped for having
zero or one confident text region. The harvester was collecting crops and
wanted busy signs; a test set that took only the photographs the detector
liked would be selected by the detector under test. The 40 were drawn from the
unscreened staging area for the same reason, and this draw ranks the pool the
same way `training/sample_photos.py` did — a blake2b hash of the filename — so
the two sets are built by one method.

The draw is 280 by default. The 40 gave 373 agreed words from the 28 that
carried text, and 25 of the 40 were photographs the harvester dropped — so
about half of the "zero confident regions" drops still carry text the detector
missed, which is itself the detection question. Scaling from the 40, 280 should
give roughly 1,500 agreed words over 150 or more photographs with text, the
size at which a three-point change stands outside the interval. Blank
photographs are quick to dismiss and are part of the population.

Nothing here is a train set. `sample.txt` is frozen the day it is committed;
the rest of the pool is what a future training set may be built from.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL = REPO_ROOT / "data" / "ocr" / "real-photos"
HARVEST = REAL / "harvested"
OUT_DIR = REAL / "test-v2"

ALLOWED_LICENCES = {"CC0", "PD", "CC BY", "CC BY-SA"}
CROP = re.compile(r"_\d{3}\.png$")


def licence_family(label: str) -> str:
    """`CC BY-SA 4.0` -> `CC BY-SA`, `Public domain` -> `PD`, the harvester's short names."""
    label = (label or "").strip()
    if not label:
        return ""
    if label.lower().startswith(("public domain", "pd")):
        return "PD"
    if label.upper().startswith("CC0"):
        return "CC0"
    return re.sub(r"\s+\d+(\.\d+)?$", "", label)


def rank(name: str) -> int:
    """The same stable order sample_photos.py used for the 40."""
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")


def title_of(url_or_key: str) -> str:
    """`File:Mostar signs.JPG` from a page URL or a key, spaces and underscores alike."""
    m = re.search(r"File:(.+)$", url_or_key)
    if not m:
        return ""
    return urllib.parse.unquote(m.group(1)).replace("_", " ").strip().lower()


def read_tsv(path: Path) -> list:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def classify(row: dict) -> str:
    """What kind of row this is, from the harvester's verdict and reason."""
    verdict, reason = row["verdict"], row["reason"]
    if verdict == "keep":
        return "keep"
    if verdict == "drop":
        m = re.match(r"only (\d+) confident region", reason)
        if m:
            return f"drop-{m.group(1)}-region"
        if reason.startswith("no region with 3+ letters"):
            return "drop-short"
        if reason.startswith("flat colour"):
            return "not-a-photograph"
        return "drop-other"
    return verdict  # skip, refused


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=280)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    truth_v2 = args.out_dir / "truth-v2.json"
    sample_path = args.out_dir / "sample.txt"
    if sample_path.exists() and truth_v2.exists():
        raise SystemExit(f"{sample_path.relative_to(REPO_ROOT)} is frozen: {truth_v2.name} was "
                         "transcribed for it. A new draw is a new ruler — name it test-v3.")

    screened = read_tsv(HARVEST / ".state" / "screened.tsv")
    credits = {r["file"]: r for r in read_tsv(HARVEST / "CREDITS.tsv")}
    staged = json.loads((HARVEST / ".state" / "staged.json").read_text(encoding="utf-8"))

    # Exclusion 1: the 40, by filename and by Commons title.
    scored_files = {ln.strip() for ln in (REAL / "scored-sample.txt").read_text(encoding="utf-8").splitlines()
                    if ln.strip()}
    scored_titles = {title_of(r["page_url"]) for r in read_tsv(REAL / "scored-sources.tsv")}
    # Exclusion 2: source photographs of every labelled crop.
    crop_sources = set()
    for labels in (REPO_ROOT / "data/ocr/crops/labels-human.tsv",
                   REPO_ROOT / "data/ocr/crops2/labels-human.tsv"):
        for line in labels.read_text(encoding="utf-8").splitlines():
            name = line.split("\t")[0].strip()
            if CROP.search(name):
                crop_sources.add(CROP.sub("", name))

    rows, seen = [], set()
    for r in screened:
        if not r["file"] or r["file"] in seen:
            continue
        seen.add(r["file"])
        kind = classify(r)
        licence = licence_family(r["licence"] or credits.get(r["file"], {}).get("license", ""))
        why = ""
        if kind in {"skip", "refused", "not-a-photograph", "drop-other"}:
            why = kind
        elif licence not in ALLOWED_LICENCES:
            why = f"licence:{licence or 'none'}"
        elif r["file"] in scored_files or title_of(r["key"]) in scored_titles:
            why = "scored"
        elif Path(r["file"]).stem in crop_sources:
            why = "crop-source"
        meta = staged.get(r["file"], {}).get("meta", {})
        rows.append({"file": r["file"], "key": r["key"], "class": kind, "licence": licence,
                     "regions": r["regions"], "page_url": meta.get("page_url", ""),
                     "screen_url": meta.get("screen_url", ""),
                     "attribution": meta.get("attribution", credits.get(r["file"], {}).get("attribution", "")),
                     "excluded": why, "rank": rank(r["file"])})
    # Kept photographs that only appear in CREDITS (a later run than screened.tsv).
    for file, c in credits.items():
        if file in seen:
            continue
        seen.add(file)
        title = title_of(c["source_url"])
        licence = licence_family(c["license"])
        why = ("scored" if file in scored_files or title in scored_titles else
               "crop-source" if Path(file).stem in crop_sources else
               "" if licence in ALLOWED_LICENCES else f"licence:{licence or 'none'}")
        meta = staged.get(file, {}).get("meta", {})
        rows.append({"file": file, "key": "File:" + title, "class": "keep", "licence": licence,
                     "regions": c["text_regions"], "page_url": c["source_url"],
                     "screen_url": meta.get("screen_url", ""), "attribution": c["attribution"],
                     "excluded": why, "rank": rank(file)})

    matched = sum(1 for r in rows if r["excluded"] == "crop-source")
    if matched < 0.8 * len(crop_sources):
        raise SystemExit(f"only {matched} of {len(crop_sources)} crop-source photographs matched a "
                         "pool filename — the naming differs and the exclusion would be a hole")

    eligible = sorted((r for r in rows if not r["excluded"]), key=lambda r: r["rank"])
    drawn = eligible[:args.count]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["file", "key", "class", "licence", "regions", "excluded", "page_url", "screen_url",
              "attribution"]
    with (args.out_dir / "pool.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (bool(r["excluded"]), r["rank"])):
            w.writerow(r)
    sample_path.write_text("".join(r["file"] + "\n" for r in drawn), encoding="utf-8")

    from collections import Counter
    print(f"pool: {len(rows)} photographs screened; {len(eligible)} eligible")
    print("excluded:", dict(Counter(r["excluded"] for r in rows if r["excluded"])))
    print("eligible by class:", dict(Counter(r["class"] for r in eligible)))
    print(f"drawn {len(drawn)}: {dict(Counter(r['class'] for r in drawn))}")
    print(f"  with a stored screen_url: {sum(bool(r['screen_url']) for r in drawn)}; the rest are "
          "fetched by title (data/scripts/fetch_test_v2.py)")
    print(f"wrote {(args.out_dir / 'pool.tsv').relative_to(REPO_ROOT)} and "
          f"{sample_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
