#!/usr/bin/env python3
"""Re-score the shipped reader on FULL-RESOLUTION test-v2 photos — run on the Mac.

Pre-registered in training/PREREGISTRATION.md, "the reader on full-resolution
inputs". The committed test-v2 photos are downscales (e.g.
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
CACHE = OUT / "reader-output-fullres.json"        # {"reader": fp, "readings": {name: text}}
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
    from app.ocr import scan
    from evaluate_ocr import reader_fingerprint  # same stamp evaluate_ocr expects

    OUT.mkdir(parents=True, exist_ok=True)
    truth = json.loads((V2 / "truth-v2.json").read_text(encoding="utf-8"))["photos"]
    names = [n for n in sorted(truth) if truth[n].get("lines")]  # photos with text

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {"reader": None, "readings": {}}
    have = cache["readings"]
    fp = reader_fingerprint()
    if cache.get("reader") not in (None, fp) and have:
        print(f"cache was written by a different reader ({cache['reader']} != {fp}); starting fresh")
        have = {}
    cache["reader"] = fp

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
            have[name] = scan(str(ORIG))
        except Exception as exc:
            missed.append(name)
            print(f"  {i}/{len(names)} {name[:45]}: read failed {exc}", flush=True)
        finally:
            if ORIG.exists():
                ORIG.unlink()
        cache["readings"] = have
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {i}/{len(names)} {name[:45]}: {len(have.get(name,'').split())} words", flush=True)
        time.sleep(1.5)

    read = len(have)
    print(f"\nread {read}/{len(names)} photos at full resolution; {len(missed)} could not be fetched/read")
    if len(missed) > len(names) * 0.15:
        print("more than 15% missing — do not score a holey set; re-run to retry the misses, then score")
        (OUT / "missed.json").write_text(json.dumps(missed, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1

    # Score the full-res cache against the same truth, through evaluate_ocr (no re-read: all cached).
    import subprocess
    out_json = OUT / "test-v2-fullres.json"
    subprocess.run([sys.executable, "training/evaluate_ocr.py",
                    "--json", str(out_json), "--out", str(OUT / "test-v2-fullres.md"),
                    "--cache", str(CACHE), "--truth", str(V2 / "truth-v2.json"),
                    "--photos", str(V2 / "photos"), "--sample", str(V2 / "sample.txt")],
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
