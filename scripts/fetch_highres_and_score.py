#!/usr/bin/env python3
"""Prepare the shipped-reader score on Commons originals — Mac only when approved.

DEFERRED: do not run this until the amendment drafted in
training/FEASIBILITY-ocr-open-levers-2026-09-16.md is copied into
training/PREREGISTRATION.md and the owner approves the one held-out look. The
committed test-v2 photos are downscales (e.g.
20130606_Mostar_034.jpg is 1280 px on disk but 3968x2976 on Commons). The app
reads at up to 2 MP (~1633 px), so on a real high-resolution upload the reader
gets more detail than the 1280 px benchmark. This fetches each test-v2 photo's
Commons original, reads it through the shipped reader (app.ocr.scan, PP-OCRv6 at
floor 0.9 — unchanged), and scores against the same truth-v2.json (truth is by
word, resolution-independent), so the only difference from the committed
test-v2 score is the input resolution.

WHY THE MAC: bulk-fetching Commons originals is rate-limited from the cloud
(HTTP 429, 0/6). A residential IP is not. Run this on the Mac, then commit the
result JSON and push; the cloud can read the number from there.

    export LILLY_READER=paddle LILLY_PADDLE_REC_THRESH=0.9
    .venv/bin/python3 scripts/fetch_highres_and_score.py

Disk-safe: each original is deleted right after it is read, so at most one full
photo sits on disk at a time. Resumable: the reading cache is saved per photo,
so a re-run skips what it already read. Polite: 1.5 s between fetches, backoff
on 429.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "data" / "ocr" / "real-photos" / "test-v2"
OUT = REPO / "training" / "highres"
CACHE = OUT / "reader-output-fullres.json"        # evaluate_ocr cache schema 2
ORIG = OUT / "_orig.tmp"                            # one photo at a time, deleted after read
UA = {"User-Agent": "Lilly-OCR-research/1.0 (github.com/ssaaffaakk/Lilly)"}
BASELINE = REPO / "training" / "paddle-floor" / "test-v2-floor0.9.json"  # the downscaled score


def fetch(name: str, dest: Path) -> bool:
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(name.replace("_", " "))
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            return False
        except Exception:
            time.sleep(2 ** attempt)
    return False


def main() -> int:
    os.environ.setdefault("LILLY_READER", "paddle")
    os.environ.setdefault("LILLY_PADDLE_VERSION", "PP-OCRv6")
    os.environ.setdefault("LILLY_PADDLE_REC_THRESH", "0.9")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "training"))
    import cv2
    from PIL import Image
    from app.ocr import MAX_WORKING_PIXELS, reader_identity, scan
    from evaluate_ocr import (
        file_sha256,
        load_reading_cache,
        reader_cache_context,
        write_reading_cache,
    )

    expected = "paddle:PP-OCRv6_medium_det+PP-OCRv6_medium_rec:3.7.0:rec>=0.9"
    identity = reader_identity()
    if identity != expected:
        raise SystemExit(f"reader is {identity!r}, not shipped {expected!r}")
    if cv2.__version__ != "4.10.0":
        raise SystemExit(
            f"cv2 is {cv2.__version__}, not the shipped 4.10.0; do not read"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    truth = json.loads((V2 / "truth-v2.json").read_text(encoding="utf-8"))["photos"]
    names = [n for n in sorted(truth) if truth[n].get("lines")]  # photos with text

    context = reader_cache_context(full=False)  # originals still enter shipped scan's 2 MP cap
    have = load_reading_cache(CACHE, context)
    for name, entry in have.items():
        if not isinstance(entry, dict) or not all(
            field in entry
            for field in ("text", "source_sha256", "original_size", "working_size")
        ):
            raise SystemExit(f"{CACHE} entry {name!r} has incomplete provenance")

    missed = []
    for i, name in enumerate(names, 1):
        if name in have:
            continue
        if not fetch(name, ORIG):
            missed.append(name)
            print(f"  {i}/{len(names)} {name[:45]}: fetch failed", flush=True)
            time.sleep(1.5)
            continue
        try:
            source_hash = file_sha256(ORIG)
            with Image.open(ORIG) as image:
                width, height = image.size
            scale = min(1.0, (MAX_WORKING_PIXELS / (width * height)) ** 0.5)
            working = [max(int(width * scale), 1), max(int(height * scale), 1)]
            reading = scan(str(ORIG))
            have[name] = {
                "source_sha256": source_hash,
                "text": reading,
                "source_url": (
                    "https://commons.wikimedia.org/wiki/Special:FilePath/"
                    + urllib.parse.quote(name.replace("_", " "))
                ),
                "original_size": [width, height],
                "working_size": working,
            }
            write_reading_cache(CACHE, context, have)
        except Exception as exc:
            missed.append(name)
            print(f"  {i}/{len(names)} {name[:45]}: read failed {exc}", flush=True)
        finally:
            if ORIG.exists():
                ORIG.unlink()
        words = have.get(name, {}).get("text", "").split()
        print(f"  {i}/{len(names)} {name[:45]}: {len(words)} words", flush=True)
        time.sleep(1.5)

    read = sum(name in have for name in names)
    print(f"\nread {read}/{len(names)} photos at full resolution; {len(missed)} could not be fetched/read")
    missing = [name for name in names if name not in have]
    if missed or missing or read != len(names):
        print("the held-out gate requires 132/132 originals; no local downscale may fill a hole")
        (OUT / "missed.json").write_text(json.dumps(missed, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1

    # Score the full-res cache against the same truth, through evaluate_ocr (no re-read: all cached).
    import subprocess
    out_json = OUT / "test-v2-fullres.json"
    subprocess.run([sys.executable, "training/evaluate_ocr.py",
                    "--json", str(out_json), "--out", str(OUT / "test-v2-fullres.md"),
                    "--cache", str(CACHE), "--truth", str(V2 / "truth-v2.json"),
                    "--photos", str(V2 / "photos"), "--sample", str(V2 / "sample.txt"),
                    "--sealed-cache"],
                   cwd=REPO, check=True)
    d = json.loads(out_json.read_text(encoding="utf-8"))
    b = json.loads(BASELINE.read_text(encoding="utf-8"))
    print("\n=== full-resolution vs the 1280 px benchmark (shipped reader, floor 0.9) ===")
    print(f"  words/photo : {b['per_photo']:.1f}%  (downscaled)  ->  {d['per_photo']:.1f}%  (full-res)")
    print(f"  pooled      : {b['pooled']:.1f}%  ->  {d['pooled']:.1f}%")
    print(f"  invented    : {b['invented']}  ->  {d['invented']}")
    print(f"  diacritic   : {b['diacritic']:.1f}%  ->  {d['diacritic']:.1f}%")
    print(f"\nwrote {out_json.relative_to(REPO)} — commit training/highres/ and push; the cloud reads the number.")
    print("Then a paired per-photo interval (rescue_report.py --shipped BASELINE --rescue this) decides if the rise is real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
