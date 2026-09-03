#!/usr/bin/env python3
"""Rebuild the 40 photographs the reader is scored on, into a directory nothing prunes.

    python3 data/scripts/restore_scored_photos.py           # restore what is missing
    python3 data/scripts/restore_scored_photos.py --check   # report only

Why this script has to exist.

The scored photographs lived in `harvested/.state/staging`, which is the
harvester's scratch area, and the harvester treats it as scratch. Two lines do
it: a photograph judged `drop` or `skip` has its staged rendering unlinked the
moment the verdict is written, and at the end of a run the whole staging
directory is emptied and removed. Both are right for a scratch area. Neither is
survivable for a test set living inside one.

Twenty-five of the forty were already gone when this was found, deleted by a
re-screening pass that judged them `drop` -- and the run still going would have
taken the other fifteen at its cleanup. The reader's 36% would then have been a
number with nothing left to re-measure it against, which is the worst way to
lose a ruler: quietly, with the score still written down.

So the scored set moves to `data/ocr/real-photos/scored/`, which no harvest
touches, and this script rebuilds it from `staged.json` when a file is missing.

One detail that decides whether the recovery is honest: the photographs are
re-fetched from `screen_url`, not `keep_url`. `screen_url` is the 1280px
rendering the harvester stages and the rendering the reader actually read; the
`keep_url` original is a different, larger picture and would silently re-scale
the whole measurement. The point of restoring is to get the same ruler back,
not a similar one.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PHOTOS = REPO_ROOT / "data" / "ocr" / "real-photos"
SAMPLE = PHOTOS / "scored-sample.txt"
SCORED = PHOTOS / "scored"
STAGED = PHOTOS / "harvested" / ".state" / "staged.json"
STAGING = PHOTOS / "harvested" / ".state" / "staging"
# The tracked manifest, and the reason a fresh clone can rebuild the ruler at
# all. staged.json is the harvester's own working state: it is gitignored, it is
# rewritten every run, and it keeps only what that run decided to keep -- so the
# source of a photograph the next run judges `drop` disappears from it. The
# manifest is written once, committed, and holds all forty regardless.
SOURCES = PHOTOS / "scored-sources.tsv"


def read_sources() -> dict:
    """file -> screen_url, from the committed manifest, falling back to local state."""
    if SOURCES.exists():
        rows = SOURCES.read_text(encoding="utf-8").splitlines()
        out = {}
        for line in rows[1:]:                     # skip the header
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0].strip():
                out[parts[0]] = parts[1]
        if out:
            return out
    if STAGED.exists():
        print(f"(no {SOURCES.name}; falling back to the harvester's local state)")
        return {n: i["meta"]["screen_url"]
                for n, i in json.loads(STAGED.read_text(encoding="utf-8")).items()}
    return {}


def wanted() -> list:
    return [l.strip() for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report, restore nothing")
    args = ap.parse_args()

    names = wanted()
    SCORED.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (SCORED / n).exists()]
    print(f"scored set: {len(names)} photographs, {len(names) - len(missing)} present")
    if not missing:
        print("nothing to restore")
        return 0

    # Anything still sitting in staging is the exact file the reader read, so
    # take that in preference to re-fetching it.
    import shutil
    rescued = 0
    for name in list(missing):
        if (STAGING / name).exists():
            shutil.copy2(STAGING / name, SCORED / name)
            missing.remove(name)
            rescued += 1
    if rescued:
        print(f"copied {rescued} straight out of staging (byte-identical)")

    if not missing:
        return 0
    print(f"{len(missing)} to re-fetch from Commons")
    if args.check:
        for n in missing:
            print(f"  missing: {n}")
        return 1

    sources = read_sources()
    unknown = [n for n in missing if n not in sources]
    if unknown:
        print(f"\n{len(unknown)} photograph(s) have no recorded source and cannot be "
              f"restored:", file=sys.stderr)
        for n in unknown:
            print(f"  {n}", file=sys.stderr)

    from harvest_sign_photos import Fetcher
    fetcher = Fetcher()
    # Thumbnails first: upload.wikimedia.org limits requests for originals per
    # address (Fetcher.download waits the window out); the 1280px renderings
    # are exempt, so they come down before any original can hold them up.
    missing.sort(key=lambda n: "/thumb/" not in sources.get(n, ""))
    got, failed = 0, []
    for i, name in enumerate(missing, 1):
        url = sources.get(name)                   # the rendering that was read
        if not url:
            continue
        if fetcher.download(url, SCORED / name):
            got += 1
            print(f"  {i}/{len(missing)} {name[:60]}", flush=True)
        else:
            failed.append(name)
            print(f"  {i}/{len(missing)} {name[:60]}  FAILED", flush=True)

    print(f"\nrestored {got}, failed {len(failed)}")
    for n in failed:
        print(f"  still missing: {n}")
    present = sum(1 for n in names if (SCORED / n).exists())
    print(f"scored set now {present}/{len(names)} in {SCORED}")
    return 0 if present == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
