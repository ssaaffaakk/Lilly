#!/usr/bin/env python3
"""Validate Run B producer shards and write the consumer dataset manifest."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    from .prepare_backtrans_bs import normalize, shard_bounds
except ImportError:  # Direct execution/import with scripts/ on sys.path.
    from prepare_backtrans_bs import normalize, shard_bounds

SAMPLE_N = 1_000_000
SHARD_COUNT = 3
SEED = 20260914
FORWARD_FINGERPRINT = "1aedcc11231cdf50817ff12f99ff0d1e"
FORMAT_VERSION = 3

# Every repo file whose bytes decide a producer shard's output. The producer
# notebook (Lilly_Backtrans_Producer_Kaggle.ipynb) clones `main` at run time, so
# the `git` SHA a shard records is only whichever tip that clone happened to pull
# — unrelated commits (docs, eval jobs) move it between shards without touching a
# byte of producer output. Provenance, not integrity: the source/pair/sample
# hashes below are what actually pin the bytes. This is the set we compare across
# commits to tell a spurious SHA gap from a real producer-code change.
PRODUCER_PATHS = (
    "training/Lilly_Backtrans_Producer_Kaggle.ipynb",
    "scripts/prepare_backtrans_bs.py",
    "scripts/backtrans_dataset.py",
    "app",
    "training/kaggle_offload.py",
    "data/scripts",
    "requirements.txt",
)


def pair_hash(rows) -> str:
    digest = hashlib.blake2b(digest_size=32)
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _digest_key(text: str) -> bytes:
    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).digest()


def validate_source_filter_report(report: dict, label: str = "producer report") -> dict:
    """Prove that named pre-sample source rejections add up exactly."""
    dropped = report.get("dropped")
    reasons = report.get("pathological_reasons")
    if not isinstance(dropped, dict) or not isinstance(reasons, dict):
        raise SystemExit(f"{label}: missing source-filter accounting")
    pathological = dropped.get("pathological source")
    valid_values = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                       for v in reasons.values())
    if (not isinstance(pathological, int) or isinstance(pathological, bool)
            or pathological < 0 or not valid_values
            or sum(reasons.values()) != pathological):
        raise SystemExit(
            f"{label}: inconsistent source-filter accounting: "
            f"dropped={pathological!r}, reasons={reasons!r}")
    return reasons


def validate_model_vocabulary_gate(report: dict, label: str = "producer report") -> dict:
    """Require a complete, deterministic pre-sample tokenizer gate."""
    gate = report.get("model_vocabulary_gate")
    if not isinstance(gate, dict) or gate.get("enabled") is not True:
        raise SystemExit(f"{label}: model-vocabulary gate was not enabled")
    checked = gate.get("checked")
    rejected = gate.get("rejected_unknown")
    accepted = gate.get("accepted")
    values = (checked, rejected, accepted)
    if (any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values)
            or checked != rejected + accepted
            or accepted != report.get("sample_kept")):
        raise SystemExit(f"{label}: inconsistent model-vocabulary gate: {gate!r}")
    return gate


def validate_decode_fallbacks(report: dict, label: str = "producer report") -> dict:
    """Validate accounting for non-empty alternative/greedy decoder recovery."""
    fallbacks = report.get("forward_decode_fallbacks")
    if not isinstance(fallbacks, dict) or set(fallbacks) != {"alternative", "greedy"}:
        raise SystemExit(f"{label}: invalid forward_decode_fallbacks={fallbacks!r}")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
           for v in fallbacks.values()):
        raise SystemExit(f"{label}: invalid forward_decode_fallbacks={fallbacks!r}")
    return fallbacks


def _producer_paths_tree_hash(sha: str) -> str | None:
    """Content hash of exactly PRODUCER_PATHS at `sha`, or None if git cannot
    resolve that commit in this checkout.

    Uses `git ls-tree`, so it reads the object store directly — no working-tree
    checkout, no network. Two commits whose producer files are byte-identical
    hash the same; adding, removing, or editing any producer file changes the
    hash. None means "cannot verify" (git missing, or the commit is not in this
    clone's history), which the caller treats as a refusal, never a pass."""
    repo = Path(__file__).resolve().parent.parent
    try:
        listing = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "-r", "--full-tree", sha, "--",
             *PRODUCER_PATHS],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = sorted(line for line in listing.splitlines() if line.strip())
    if not rows:
        return None
    digest = hashlib.blake2b(digest_size=32)
    for line in rows:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_producer_git(producer_gits) -> tuple:
    """One provenance for the union, or a git-verified proof that several commits
    carry byte-identical producer code.

    The old gate required every shard's `git` string to match, which refuses a
    perfectly good union the moment an unrelated commit (an eval job, a doc)
    advances `main` between two shards. This checks the invariant the SHA string
    was only a proxy for: it hashes exactly the files that define producer output
    at each distinct commit and requires them equal. It fails closed — a commit
    it cannot resolve, or producer code that actually differs, refuses the union.
    Owner-sanctioned relaxation of git *string* equality; the fingerprint, sample
    hash, and row-count gates in validate_union still stand."""
    distinct = sorted(set(producer_gits))
    if len(distinct) == 1:
        return distinct[0], None
    hashes = {sha: _producer_paths_tree_hash(sha) for sha in distinct}
    unresolved = sorted(sha for sha, digest in hashes.items() if digest is None)
    if unresolved:
        raise SystemExit(
            "producer shards span commits " + ", ".join(distinct) + " but producer-code "
            "equivalence cannot be checked from git for " + ", ".join(unresolved) + ". "
            "Run validate_union inside a full clone that contains those commits, or "
            "re-produce every shard at one producer commit.")
    if len(set(hashes.values())) != 1:
        raise SystemExit(
            "producer shards span commits whose PRODUCER CODE DIFFERS: "
            + json.dumps(hashes) + ". Refusing the union — re-produce every shard at "
            "one producer commit.")
    paths_hash = next(iter(hashes.values()))
    print("producer spans " + str(len(distinct)) + " commits (" + ", ".join(distinct)
          + "); PRODUCER_PATHS byte-identical at " + paths_hash[:16]
          + " — union permitted.", file=sys.stderr, flush=True)
    return distinct, paths_hash


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
    forward_decode_fallbacks = {"alternative": 0, "greedy": 0}
    sample_hash = source_filter_accounting = vocabulary_gate = None
    producer_gits = []
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
        this_decode_fallbacks = validate_decode_fallbacks(report, report_path.name)
        for kind, count in this_decode_fallbacks.items():
            forward_decode_fallbacks[kind] += count
        this_accounting = {
            "dropped": report.get("dropped"),
            "pathological_reasons": validate_source_filter_report(report, report_path.name),
        }
        if source_filter_accounting is None:
            source_filter_accounting = this_accounting
        elif this_accounting != source_filter_accounting:
            raise SystemExit(f"{report_path.name}: producer source-filter accounting differs")
        this_vocabulary_gate = validate_model_vocabulary_gate(report, report_path.name)
        if vocabulary_gate is None:
            vocabulary_gate = this_vocabulary_gate
        elif this_vocabulary_gate != vocabulary_gate:
            raise SystemExit(f"{report_path.name}: producer model-vocabulary gate differs")

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
        producer_gits.append(this_git)

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
                              "forward_decode_fallbacks": this_decode_fallbacks,
                              "source_order_hash": got_source_hash,
                              "pair_order_hash": got_pair_hash})

    missing = expected_n - total
    if total != expected_n or missing != 0 or duplicate_sources != 0:
        raise SystemExit(f"refusing Run B union: rows={total:,}, missing={missing:,}, duplicate_sources={duplicate_sources:,}")
    union_source_hash = union_source_digest.hexdigest()
    if union_source_hash != sample_hash:
        raise SystemExit("producer shards do not reconstruct the registered source order")
    producer_git, producer_paths_hash = resolve_producer_git(producer_gits)
    return {"format_version": FORMAT_VERSION, "status": "complete", "sample_n": expected_n,
            "rows": total, "missing": missing, "duplicate_sources": duplicate_sources,
            "shard_count": shard_count, "seed": seed,
            "forward_normalized": forward_normalized,
            "forward_decode_fallbacks": forward_decode_fallbacks,
            "source_filter_accounting": source_filter_accounting,
            "model_vocabulary_gate": vocabulary_gate,
            "forward_fingerprint": forward_fingerprint, "producer_git": producer_git,
            "producer_paths_hash": producer_paths_hash,
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
