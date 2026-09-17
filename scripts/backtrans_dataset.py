#!/usr/bin/env python3
"""Validate Run B producer shards and write the consumer dataset manifest."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

try:
    from .prepare_backtrans_bs import normalize, shard_bounds
except ImportError:  # Direct execution/import with scripts/ on sys.path.
    from prepare_backtrans_bs import normalize, shard_bounds

SAMPLE_N = 1_000_000
SHARD_COUNT = 3
SEED = 20260914
FORWARD_FINGERPRINT = "1aedcc11231cdf50817ff12f99ff0d1e"
FORMAT_VERSION = 1


def pair_hash(rows) -> str:
    digest = hashlib.blake2b(digest_size=32)
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _digest_key(text: str) -> bytes:
    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).digest()


def validate_union(root: Path, *, expected_n: int = SAMPLE_N,
                   shard_count: int = SHARD_COUNT,
                   forward_fingerprint: str = FORWARD_FINGERPRINT,
                   seed: int = SEED) -> dict:
    """Validate all logical rows and return a content-bound union manifest."""
    root = Path(root)
    reports = sorted(root.rglob("backtrans-shard-*.json"))
    if len(reports) != shard_count:
        raise SystemExit(f"expected {shard_count} producer reports, found {len(reports)} under {root}")

    union_source_digest = hashlib.blake2b(digest_size=32)
    union_pair_digest = hashlib.blake2b(digest_size=32)
    seen, duplicate_sources, shard_records = set(), 0, []
    forward_normalized = 0
    sample_hash = producer_git = None
    total = 0

    for expected_index, report_path in enumerate(reports):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for key, want in (("format_version", FORMAT_VERSION),
                          ("shard_index", expected_index), ("shard_count", shard_count),
                          ("sample_n", expected_n), ("sample_kept", expected_n),
                          ("seed", seed), ("forward_fingerprint", forward_fingerprint),
                          ("status", "complete")):
            if report.get(key) != want:
                raise SystemExit(f"{report_path.name}: {key}={report.get(key)!r}, expected {want!r}")
        start, stop = shard_bounds(expected_n, expected_index, shard_count)
        if report.get("shard_start") != start or report.get("shard_stop") != stop:
            raise SystemExit(f"{report_path.name}: range is not registered slice {start}:{stop}")
        want_rows = stop - start
        if report.get("rows") != want_rows or report.get("kept") != want_rows:
            raise SystemExit(f"{report_path.name}: expected {want_rows} rows")
        normalized = report.get("forward_normalized")
        if (not isinstance(normalized, int) or isinstance(normalized, bool)
                or normalized < 0 or normalized > want_rows):
            raise SystemExit(f"{report_path.name}: invalid forward_normalized={normalized!r}")
        forward_normalized += normalized

        this_sample_hash = report.get("sample_order_hash")
        if not this_sample_hash:
            raise SystemExit(f"{report_path.name}: missing sample_order_hash")
        if sample_hash is None:
            sample_hash = this_sample_hash
        elif this_sample_hash != sample_hash:
            raise SystemExit(f"{report_path.name}: producer sampled a different source order")
        this_git = report.get("git")
        if not this_git:
            raise SystemExit(f"{report_path.name}: missing producer git SHA")
        if producer_git is None:
            producer_git = this_git
        elif this_git != producer_git:
            raise SystemExit(f"{report_path.name}: producer git SHA differs")

        data_path = report_path.parent / str(report.get("data_file"))
        if not data_path.is_file():
            raise SystemExit(f"{report_path.name}: missing {report.get('data_file')}")
        shard_source_digest = hashlib.blake2b(digest_size=32)
        shard_pair_digest = hashlib.blake2b(digest_size=32)
        shard_rows = 0
        with gzip.open(data_path, "rt", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                row = line.rstrip("\n")
                fields = row.split("\t")
                if len(fields) != 3 or fields[0] != "backtrans-macocu":
                    raise SystemExit(f"{data_path.name}:{line_no}: expected backtrans-macocu\\tbs\\ten")
                bs, en = fields[1], fields[2]
                if not bs.strip() or not en.strip():
                    raise SystemExit(f"{data_path.name}:{line_no}: empty source or target")
                key = _digest_key(bs)
                if key in seen:
                    duplicate_sources += 1
                seen.add(key)
                for digest, value in ((shard_source_digest, bs), (union_source_digest, bs),
                                      (shard_pair_digest, row), (union_pair_digest, row)):
                    digest.update(value.encode("utf-8")); digest.update(b"\n")
                shard_rows += 1; total += 1
        if shard_rows != want_rows:
            raise SystemExit(f"{data_path.name}: has {shard_rows} rows, expected {want_rows}")
        got_source_hash, got_pair_hash = shard_source_digest.hexdigest(), shard_pair_digest.hexdigest()
        if got_source_hash != report.get("source_order_hash"):
            raise SystemExit(f"{data_path.name}: source_order_hash mismatch")
        if got_pair_hash != report.get("pair_order_hash"):
            raise SystemExit(f"{data_path.name}: pair_order_hash mismatch")
        shard_records.append({"index": expected_index, "rows": shard_rows,
                              "range": [start, stop], "data_file": data_path.name,
                              "report_file": report_path.name,
                              "forward_normalized": normalized,
                              "source_order_hash": got_source_hash,
                              "pair_order_hash": got_pair_hash})

    missing = expected_n - total
    if total != expected_n or missing != 0 or duplicate_sources != 0:
        raise SystemExit(f"refusing Run B union: rows={total:,}, missing={missing:,}, duplicate_sources={duplicate_sources:,}")
    union_source_hash = union_source_digest.hexdigest()
    if union_source_hash != sample_hash:
        raise SystemExit("producer shards do not reconstruct the registered source order")
    return {"format_version": FORMAT_VERSION, "status": "complete", "sample_n": expected_n,
            "rows": total, "missing": missing, "duplicate_sources": duplicate_sources,
            "shard_count": shard_count, "seed": seed,
            "forward_normalized": forward_normalized,
            "forward_fingerprint": forward_fingerprint, "producer_git": producer_git,
            "source_order_hash": union_source_hash,
            "pair_order_hash": union_pair_digest.hexdigest(), "shards": shard_records}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path); ap.add_argument("--write", type=Path, default=None)
    args = ap.parse_args(); manifest = validate_union(args.root)
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.write: args.write.write_text(text, encoding="utf-8")
    print(text, end=""); return 0


if __name__ == "__main__":
    raise SystemExit(main())
