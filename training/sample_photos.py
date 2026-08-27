#!/usr/bin/env python3
"""Pick the photographs the reader will be scored on — before it reads them.

The selection has to be independent of the model, or the measurement is worthless.
Choosing photographs because our own reader found text in them would select for
the cases it already handles and report that as accuracy; the same shape of error
as scoring a model on sentences it was trained on.

So the sample is a hash of the filename. It is reproducible, it cannot be nudged,
and re-running it after adding photographs keeps the ones already transcribed.

    python3 training/sample_photos.py --count 40
"""
import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / "data/ocr/real-photos/harvested/.state/staging"
OUT = REPO_ROOT / "data/ocr/real-photos/scored-sample.txt"


def rank(name: str) -> int:
    """A stable pseudo-random order over filenames."""
    return int.from_bytes(
        hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--photos", type=Path, default=STAGING)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    photos = sorted(p for p in args.photos.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not photos:
        raise SystemExit(f"no photographs in {args.photos}")

    chosen = sorted(photos, key=lambda p: rank(p.name))[:args.count]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(p.name for p in chosen) + "\n", encoding="utf-8")

    print(f"{len(chosen)} of {len(photos)} photographs chosen by filename hash")
    print(f"  -> {args.out.relative_to(REPO_ROOT)}")
    for p in chosen[:5]:
        print(f"     {p.name}")
    print(f"     ... and {len(chosen) - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
