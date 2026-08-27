#!/usr/bin/env python3
"""Put Lilly's models on this machine.

The weights are about 900 MB, so they are not in git. They live in the project's
model repository instead, and this pulls them down into models/lilly/ where the
app looks for them. A fresh clone needs this once before anything will run.

    python3 scripts/fetch_models.py

Point it somewhere else with LILLY_MODELS_REPO, and pass a token in HF_TOKEN if
the repository is private.
"""
import os
import sys
from pathlib import Path

REPO = os.environ.get("LILLY_MODELS_REPO", "Safak11/lilly")
DEST = Path(__file__).resolve().parents[1] / "models" / "lilly"
# What the published bundle actually contains, and what the app needs to run.
# Not "translate": that is the untuned float32 base, which only training reads,
# and it is deliberately left out of the release. Expecting it here made a fresh
# clone download the whole bundle and then fail with "still missing: translate".
PARTS = ("translator", "listen", "speak", "read")


def main() -> int:
    have = [p for p in PARTS if (DEST / p).is_dir()]
    if len(have) == len(PARTS):
        print(f"already here: {DEST}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub first", file=sys.stderr)
        return 1

    print(f"fetching {REPO} -> {DEST} (about 900 MB, once)")
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO, repo_type="model", local_dir=str(DEST),
                      token=os.environ.get("HF_TOKEN") or None)

    missing = [p for p in PARTS if not (DEST / p).is_dir()]
    if missing:
        print(f"still missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    size = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"ready: {size / 1073741824:.2f} GB in {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
