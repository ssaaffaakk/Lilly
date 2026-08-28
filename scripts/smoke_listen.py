#!/usr/bin/env python3
"""One-clip smoke test after installing listen weights."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.guard import claim  # noqa: E402
from training.train_speech import read_tsv  # noqa: E402


def main() -> int:
    claim(2.0, "smoke listen")
    rows = read_tsv(REPO / "data" / "speech" / "test.tsv")[:1]
    if not rows:
        print("no test clips — skipped")
        return 0
    clip, reference = rows[0]
    if not Path(clip).exists():
        print(f"clip missing: {clip} — skipped")
        return 0
    from app.speech import transcribe

    heard = transcribe(str(clip), language="bs")
    print(f"clip: {clip.name}")
    print(f"ref:  {reference}")
    print(f"heard: {heard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
