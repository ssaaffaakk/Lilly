#!/usr/bin/env python3
"""The human crops, scored through the app's door, with the split named.

`data/ocr/crops/labels-human-latin.tsv` is the 737 of 1,701 hand-transcribed
crops a Latin-only reader can fairly be asked to read. It is not one set. The
shipped reader trained on the crops whose label text fell on the training side
of `training/ocr_split.is_valid_text` — 666 of the 737 — and held out 71. Pass
18 and pass 19 were compared against the shipped reader on all 737, which is a
comparison on the shipped reader's own training data (docs/OCR-ROADMAP.md,
mistake 1). This script reports the two halves separately and never adds them.

    LILLY_READER=paddle python3 training/score_crops.py --json out.json

Every crop goes through `app.ocr.read_regions`, the same door the photograph
score uses, so whichever engine LILLY_READER selects is the one measured. A crop
that cannot be read is a failed score, not a skipped row.
"""
import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from training.ocr_split import is_valid_text  # noqa: E402

CROPS = REPO_ROOT / "data" / "ocr" / "crops"
LABELS = CROPS / "labels-human-latin.tsv"


def fold(text: str) -> str:
    """Diacritics folded away, as training/cells/clean_crop_eval.py folds them."""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


def wilson(hit: int, n: int, z: float = 1.96) -> tuple:
    """95% interval for a proportion, in percent. (0, 0) when n is 0."""
    if not n:
        return (0.0, 0.0)
    p = hit / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, 100 * (centre - half)), min(100.0, 100 * (centre + half)))


def load_rows(labels: Path) -> list:
    rows = []
    for line in labels.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--crops", type=Path, default=CROPS)
    ap.add_argument("--json", type=Path, help="write counts, intervals and per-crop outcomes")
    ap.add_argument("--limit", type=int, help="first N rows of each half — a smoke test, not a score")
    ap.add_argument("--time", type=Path, metavar="PHOTO",
                    help="also time app.ocr.scan on this photograph (second of two runs)")
    args = ap.parse_args()

    from scripts.guard import claim
    claim(1.4, "crop scoring")

    rows = load_rows(args.labels)
    halves = {"held_out": [r for r in rows if is_valid_text(r[1])],
              "training_side": [r for r in rows if not is_valid_text(r[1])]}
    if args.limit:
        halves = {k: v[:args.limit] for k, v in halves.items()}
    missing = [n for half in halves.values() for n, _t in half if not (args.crops / n).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} of the crops are not on disk (first: {missing[0]}). "
                         "A hole is a failed score. Restore the crops or fix --crops.")

    import numpy as np
    from PIL import Image
    from app.ocr import read_regions, reader_identity

    identity = reader_identity()
    print(f"reader: {identity}")
    print(f"{args.labels.name}: {len(rows)} rows — "
          f"{len(halves['held_out'])} held out, {len(halves['training_side'])} training-side")

    out = {"reader": identity, "labels": str(args.labels.relative_to(REPO_ROOT)), "halves": {},
           "outcomes": {}}
    started = time.time()
    for name, half in halves.items():
        exact = folded = 0
        for i, (crop, truth) in enumerate(half, 1):
            array = np.asarray(Image.open(args.crops / crop).convert("RGB"))
            try:
                got = " ".join(read_regions(array, detail=0, paragraph=False)).strip()
            except Exception as exc:
                raise SystemExit(f"{crop}: read failed ({exc}) — a hole is a failed score")
            e = got == truth
            f = fold(got) == fold(truth)
            exact += e
            folded += f
            out["outcomes"][crop] = [int(e), int(f)]
            if i % 100 == 0:
                print(f"  {name}: {i}/{len(half)}", flush=True)
        n = len(half)
        out["halves"][name] = {"n": n, "exact": exact, "folded": folded,
                               "exact_ci": wilson(exact, n), "folded_ci": wilson(folded, n)}
    print(f"read {len(rows)} crops in {time.time() - started:.0f}s\n")

    print(f"{'':<16}{'exact':>14}{'ignoring diacritics':>26}   95% interval (folded)")
    for name, h in out["halves"].items():
        n = h["n"] or 1
        lo, hi = h["folded_ci"]
        print(f"{name:<16}{h['exact']:>6}/{h['n']:<4}{100 * h['exact'] / n:>5.1f}%"
              f"{h['folded']:>12}/{h['n']:<4}{100 * h['folded'] / n:>5.1f}%   {lo:.1f}–{hi:.1f}")
    print("\nOnly the held-out row compares readers. The shipped reader trained on "
          "the training-side crops.")

    if args.time:
        from app.ocr import scan
        scan(str(args.time))                       # warm: model load, first-call costs
        t0 = time.time()
        scan(str(args.time))
        seconds = time.time() - t0
        with Image.open(args.time) as img:
            w, h = img.size
        out["timing"] = {"photo": args.time.name, "pixels": w * h, "seconds": seconds}
        print(f"\n{args.time.name} ({w}x{h}): {seconds:.1f}s per read, second of two")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
