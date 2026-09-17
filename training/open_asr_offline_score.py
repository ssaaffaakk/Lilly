#!/usr/bin/env python3
"""Score cached Lilly predictions with a pinned Open ASR normalizer offline.

This is supplementary.  The current Open ASR multilingual leaderboard does
not register Bosnian in its public dataset matrix, so this result is not called
an official submission and nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.evaluate_speech import edits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--harness", type=Path, required=True)
    ap.add_argument("--harness-commit", required=True)
    ap.add_argument("--listener", default="listen")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.harness))
    from normalizer import data_utils
    from normalizer.eval_utils import normalize_compound_pairs

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    listener = payload["listeners"].get(args.listener)
    if not listener:
        raise SystemExit(f"no listener {args.listener!r} in predictions")
    rows = listener["predictions"]
    if len(rows) != 925:
        raise SystemExit(f"Open ASR offline score needs 925 predictions, found {len(rows)}")
    refs = [data_utils.ml_normalizer(r["reference"], lang="bs") for r in rows]
    hyps = [data_utils.ml_normalizer(r["hypothesis"], lang="bs") for r in rows]
    refs, hyps = normalize_compound_pairs(refs, hyps)
    wrong = sum(edits(ref.split(), hyp.split()) for ref, hyp in zip(refs, hyps))
    n_words = sum(len(ref.split()) for ref in refs)
    result = {
        "schema": 1, "harness": "huggingface/open_asr_leaderboard",
        "harness_commit": args.harness_commit, "normalizer": "ml_normalizer(lang=bs)",
        "listener": args.listener, "listener_fingerprint": listener["fingerprint"],
        "clips": len(rows), "wrong": wrong, "words": n_words,
        "wer": 100 * wrong / n_words,
        "official_submission": False, "uploaded": False,
        "caveat": ("The current Open ASR multilingual dataset matrix does not register "
                   "Bosnian. This is its pinned normalizer/scoring logic applied offline "
                   "to Lilly's hash-pinned FLEURS bs predictions, not a leaderboard entry."),
    }
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
