#!/usr/bin/env python3
"""What the "invented words" column counts on a set, photograph by photograph.

    python3 training/invented_words.py --set test-v2 --arms stock paddle-v6 lilly

evaluate_ocr.py counts, on every photograph whose key holds at least one agreed
word, the reader's words beyond the ones that matched the key -- and reports the
sum as "words not on any sign". On the 40 that was invention: the transcribers
were told to enlarge and read everything. On test-v2 it is not only that. The
key is the words BOTH passes marked clear; a transcriber who wrote "the body
paragraphs are below resolution and were not guessed" left a board's text out
of the key on purpose, and a reader that reads it is charged for every word.

So this prints, per arm, the same count three ways, and where it sits:

  vs agreed key        exactly evaluate_ocr.py's number (checked: it reproduces)
  vs either clear      words either pass marked clear, agreed or not
  vs anything written  words either pass wrote at all, clear or unclear

and the photographs that carry most of it. A column whose mass is on three
dense boards the transcribers declined is measuring where the transcription
stopped, not where the reader invented -- and must not decide a bake-off row
until its definition is pre-registered for the set (docs/OCR-ROADMAP.md).

Reads the per-arm reading caches (<set>/reader-output-<arm>.json), which are
not in git: run it where the readings were made.
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL = REPO_ROOT / "data" / "ocr" / "real-photos"
# Same tokeniser as evaluate_ocr.py (not imported: that module claims memory at
# import time). Letters only, lowercased, diacritics kept.
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def words(text: str) -> list:
    return [w.lower() for w in WORD.findall(unicodedata.normalize("NFC", text))]


def hits(got: Counter, want: Counter) -> int:
    return sum((want & got).values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test-v2")
    ap.add_argument("--arms", nargs="+", default=["lilly", "stock", "paddle-v6", "paddle-v5"])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    d = REAL / args.set
    truth = json.loads((d / f"truth-{args.set.replace('test-', '')}.json").read_text(encoding="utf-8"))["photos"]
    passes = {p: {e["filename"]: e for e in json.loads((d / f"pass-{p}.json").read_text(encoding="utf-8"))}
              for p in ("a", "b")}

    def pass_words(p, name, clear_only=False):
        return Counter(w for ln in passes[p][name]["lines"]
                       if not clear_only or ln["legibility"] == "clear" for w in words(ln["text"]))

    keys = {n: Counter(w for ln in e["lines"] for w in words(ln)) for n, e in truth.items()}
    print(f"{'arm':10s} {'vs agreed key':>13} {'vs either clear':>15} {'vs anything written':>20}  photographs")
    tables = {}
    for arm in args.arms:
        cache = d / (f"reader-output-{arm}.json")
        if not cache.is_file():
            print(f"{arm:10s} no cache at {cache.relative_to(REPO_ROOT)}")
            continue
        readings = json.loads(cache.read_text(encoding="utf-8")).get("readings", {})
        totals = [0, 0, 0]
        per = []
        for name, text in readings.items():
            key = keys.get(name, Counter())
            if not key:                              # evaluate_ocr.py skips these
                continue
            got = Counter(words(text))
            n = sum(got.values())
            clear = pass_words("a", name, True) | pass_words("b", name, True)
            anything = pass_words("a", name) | pass_words("b", name)
            row = [max(0, n - hits(got, key)), max(0, n - hits(got, clear)), max(0, n - hits(got, anything))]
            totals = [t + r for t, r in zip(totals, row)]
            unclear = sum(1 for p in ("a", "b") for ln in passes[p][name]["lines"] if ln["legibility"] == "unclear")
            per.append((row[0], name, sum(key.values()), unclear))
        print(f"{arm:10s} {totals[0]:13d} {totals[1]:15d} {totals[2]:20d}  {len(per)}")
        tables[arm] = (per, totals[0])

    for arm, (per, total) in tables.items():
        per.sort(reverse=True)
        top = per[:args.top]
        share = sum(p[0] for p in top)
        few = sum(1 for p in per if p[0] <= 2)
        print(f"\n{arm}: top {len(top)} photographs carry {share} of {total} ({100 * share / max(1, total):.0f}%); "
              f"{few} of {len(per)} photographs have two or fewer")
        print(f"  {'invented':>8} {'key words':>9} {'unclear lines (a+b)':>20}  photograph")
        for inv, name, kw, unclear in top:
            print(f"  {inv:8d} {kw:9d} {unclear:20d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
