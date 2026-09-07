#!/usr/bin/env python3
"""Draw test-v2b: the next 160 photographs of the same hash-ranked pool.

training/RUBRIC.md refuses to score the reader below 200 real, held-out,
eye-transcribed photographs -- "void, not low". test-v2 drew 280 and 132 of them
carry text. At that rate 160 more photographs should carry about 75, which
clears 200 with margin.

The draw is not a new method. build_test_v2.py ranks every eligible photograph
by a blake2b hash of its filename and takes a prefix; sample.txt is the first
280 of that order and this file takes the next 160, from the committed pool.tsv
and nothing else. So the extension is reproducible on any clone, cannot be
nudged toward photographs the reader likes, and is disjoint from test-v2 by
construction. It refuses to run if test-v2/sample.txt is not exactly the first
280 eligible rows of pool.tsv, because then "the next 160" would mean nothing.

    python3 training/extend_test_v2.py            # writes test-v2b/sample.txt + pool.tsv
    python3 data/scripts/fetch_test_v2.py --set test-v2b
    python3 training/transcription_pass.py sheet --set test-v2b --pass a
    python3 training/transcription_pass.py sheet --set test-v2b --pass b
    python3 training/transcription_pass.py check --set test-v2b
    python3 training/transcription_pass.py pair  --set test-v2b
    python3 training/build_truth.py --result data/ocr/real-photos/test-v2b/pair.json \\
        --out data/ocr/real-photos/test-v2b/truth-v2b.json

Two blind passes, the same as the 280. The reader is not run on these
photographs until the key exists.
"""
import argparse
import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL = REPO_ROOT / "data" / "ocr" / "real-photos"
SRC = REAL / "test-v2"
DEST = REAL / "test-v2b"
FIRST = 280          # the frozen draw
COUNT = 160          # the extension


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=COUNT)
    args = ap.parse_args()

    with (SRC / "pool.tsv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    eligible = [r["file"] for r in rows if not r["excluded"]]
    sample = [ln.strip() for ln in (SRC / "sample.txt").read_text(encoding="utf-8").splitlines()
              if ln.strip()]
    if eligible[:FIRST] != sample:
        raise SystemExit("test-v2/sample.txt is not the first 280 eligible rows of pool.tsv — "
                         "the pool has changed under the draw; not extending it")
    nxt = eligible[FIRST:FIRST + args.count]
    if len(nxt) < args.count:
        raise SystemExit(f"pool has only {len(nxt)} eligible photographs past the first {FIRST}")
    if set(nxt) & set(sample):
        raise SystemExit("extension overlaps the frozen draw — refusing")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "sample.txt").write_text("\n".join(nxt) + "\n", encoding="utf-8")
    shutil.copyfile(SRC / "pool.tsv", DEST / "pool.tsv")
    print(f"test-v2b: {len(nxt)} photographs, ranks {FIRST + 1}-{FIRST + len(nxt)} of the pool")
    print(f"wrote {DEST.relative_to(REPO_ROOT)}/sample.txt and pool.tsv")
    print("next: python3 data/scripts/fetch_test_v2.py --set test-v2b   (on the Mac)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
