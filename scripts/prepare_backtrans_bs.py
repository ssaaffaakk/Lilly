#!/usr/bin/env python3
"""Run B data prep — sample, clean, de-duplicate and HOLD OUT MaCoCu-bs.

Pre-registered in `training/PREREGISTRATION.md`, "Run B — reply (English →
Bosnian) — back-translation from MaCoCu-bs". This is step 1-2 of that method and
nothing else: it takes monolingual Bosnian (MaCoCu-bs 1.0, CC0) and produces the
sentences that will be back-translated bs→en on Kaggle. The back-translation and
the training are the GPU steps and live in the notebook; this is CPU and testable.

The one guarantee this file exists to make is **no leakage**: not one FLORES dev/
devtest/test Bosnian sentence, and not one `bench/cases.tsv` target, may survive
into the sample — otherwise the eval would score its own training data. The
holdout is by exact match AND by a normalised near-match (case, punctuation and
whitespace folded), and every drop is counted so the report can be read as a
filter and not a selection.

    python3 scripts/prepare_backtrans_bs.py \
        --input macocu-bs.txt \
        --holdout flores.bs-dev flores.bs-devtest flores.bs-test \
        --bench bench/cases.tsv \
        --parallel data/extra/train-mix-en-bs.tsv \
        --n 1000000 --out backtrans-bs.txt --report backtrans-report.json

`--holdout` files are one sentence per line; `--bench` is the tsv whose `bs`
column carries the reference targets; `--parallel` (optional) is the existing
en→bs training tsv whose Bosnian side is de-duplicated against too. Sampling is
deterministic given `--seed`.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

_WS = re.compile(r"\s+")
_STRIP = re.compile(r"[^\w\s]", re.UNICODE)


def ordered_hash(lines) -> str:
    """Content-and-order fingerprint used to prove shard union identity."""
    digest = hashlib.blake2b(digest_size=32)
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize(s: str) -> str:
    """Fold case, punctuation and whitespace for near-match dedup/holdout.

    NFKC first so că/ć composed vs decomposed collapse to one key; the point is
    that a sentence cannot slip past the holdout on a stray comma or a capital.
    """
    s = unicodedata.normalize("NFKC", s).casefold()
    s = _STRIP.sub(" ", s)
    return _WS.sub(" ", s).strip()


def load_lines(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def bench_targets(path: Path):
    """The Bosnian reference side of bench/cases.tsv (the `bs` column)."""
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            bs = (r.get("bs") or "").strip()
            if bs:
                yield bs


def parallel_bs(path: Path):
    """The Bosnian side of an en→bs training tsv. Tries a `bs` header, else col 2."""
    with open(path, encoding="utf-8") as f:
        sample = f.readline()
        f.seek(0)
        if "\t" in sample:
            has_header = "bs" in sample.lower().split("\t")
            reader = csv.reader(f, delimiter="\t")
            if has_header:
                header = next(reader)
                idx = [c.lower() for c in header].index("bs")
            else:
                idx = 1
            for row in reader:
                if len(row) > idx and row[idx].strip():
                    yield row[idx].strip()


def build_holdout(holdout_files, bench_file, parallel_file) -> set:
    keys = set()
    for p in holdout_files or []:
        for s in load_lines(Path(p)):
            keys.add(normalize(s))
    if bench_file:
        for s in bench_targets(Path(bench_file)):
            keys.add(normalize(s))
    if parallel_file:
        for s in parallel_bs(Path(parallel_file)):
            keys.add(normalize(s))
    keys.discard("")
    return keys


def shard_bounds(total: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    """Return a balanced, gap-free slice of ``range(total)`` for one shard."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} is outside [0, {shard_count})")
    width, remainder = divmod(total, shard_count)
    start = shard_index * width + min(shard_index, remainder)
    stop = start + width + (1 if shard_index < remainder else 0)
    return start, stop


def prepare(lines, holdout: set, n: int, min_tok: int, max_tok: int, seed: int,
            shard_index: int | None = None, shard_count: int | None = None):
    """Clean, de-duplicate, sample, then optionally return one exact shard."""
    report = {"read": 0, "kept": 0, "sample_n": n, "seed": seed,
              "dropped": {"too short": 0, "too long": 0, "url/boilerplate": 0,
                          "duplicate": 0, "holdout": 0}}
    seen = set()
    kept = []
    for line in lines:
        report["read"] += 1
        toks = line.split()
        if len(toks) < min_tok:
            report["dropped"]["too short"] += 1
            continue
        if len(toks) > max_tok:
            report["dropped"]["too long"] += 1
            continue
        if "http://" in line or "https://" in line or line.count("|") > 2:
            report["dropped"]["url/boilerplate"] += 1
            continue
        key = normalize(line)
        if key in holdout:
            report["dropped"]["holdout"] += 1
            continue
        if key in seen:
            report["dropped"]["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(line)
    # Deterministic sample down to n.
    if n and len(kept) > n:
        random.Random(seed).shuffle(kept)
        kept = kept[:n]
    sample_kept = len(kept)
    report["sample_order_hash"] = ordered_hash(kept)
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be provided together")
    if shard_index is not None and shard_count is not None:
        start, stop = shard_bounds(sample_kept, shard_index, shard_count)
        kept = kept[start:stop]
        report.update({"sample_kept": sample_kept, "shard_index": shard_index,
                       "shard_count": shard_count, "shard_start": start,
                       "shard_stop": stop})
    report["kept"] = len(kept)
    report["holdout_size"] = len(holdout)
    return kept, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="MaCoCu-bs text, one sentence/line")
    ap.add_argument("--holdout", nargs="*", default=[], help="FLORES bs files to exclude")
    ap.add_argument("--bench", default=None, help="bench/cases.tsv (its bs column excluded)")
    ap.add_argument("--parallel", default=None, help="existing en-bs tsv (its bs side excluded)")
    ap.add_argument("--n", type=int, default=1_000_000, help="max kept sentences (0 = all)")
    ap.add_argument("--min-tokens", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--shard-index", type=int, default=None,
                    help="zero-based deterministic shard to emit")
    ap.add_argument("--shard-count", type=int, default=None,
                    help="number of balanced shards in the sampled corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    holdout = build_holdout(args.holdout, args.bench, args.parallel)
    if not holdout:
        print("refusing to run with an EMPTY holdout — pass --holdout/--bench so "
              "FLORES and the bench cannot leak into training", file=sys.stderr)
        return 1
    kept, report = prepare(load_lines(Path(args.input)), holdout, args.n,
                           args.min_tokens, args.max_tokens, args.seed,
                           args.shard_index, args.shard_count)
    Path(args.out).write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
