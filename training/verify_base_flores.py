#!/usr/bin/env python3
"""Reproduce a direction's committed FLORES numbers on the untouched base.

The English -> Bosnian pre-registration fixes its bar against two numbers that
were measured months ago and carried forward in prose: 29.57 BLEU and 58.96
chrF2 on 2,009 FLORES-200 pairs (`training/RESULTS-en-bs.md`). A candidate is
about to be compared against them. A bar nobody has re-measured is a bar nobody
has checked, so this re-measures it before the candidate exists.

It is deliberately NOT a second instrument. Everything that touches a number
here is imported from `training/evaluate.py` -- the same loader, the same
length-sorted batching (batching in file order costs 8.7 BLEU, measured), the
same tag stripping, the same sacrebleu call. The only thing this file adds is
the ability to score FLORES without the in-house split, which `evaluate.py`
loads unconditionally and which is not on every machine.

    .venv/bin/python3 training/verify_base_flores.py --direction en-bs

Prints the numbers and, if the direction has committed ones, the difference.
Nothing here decides anything: it either confirms the bar or says the bar has
moved, and a bar that has moved is a finding for the write-up, not a licence to
pick whichever number is kinder.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# `training` goes in FRONT of the repo root, and both go in front of
# site-packages, ON PURPOSE. `evaluate` is also a PyPI package -- HuggingFace's
# -- and Kaggle ships it. Reorder these two lines and `from evaluate import
# score` silently binds to that package instead of the repo's evaluate.py, so
# the run would report numbers from a different scorer with no error anywhere.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

# The committed numbers this run is checking, quoted from the results files so
# the comparison is against what the project actually published.
COMMITTED = {
    "en-bs": {"bleu": 29.57, "chrf2": 58.96, "pairs": 2009,
              "source": "training/RESULTS-en-bs.md, base (untuned), FLORES-200"},
    "bs-en": {"bleu": 41.60, "chrf2": 67.58, "pairs": 2009,
              "source": "training/RESULTS.md, Base (untuned), FLORES-200"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="en-bs", choices=("bs-en", "en-bs"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from evaluate import load_flores, translate_all, strip_tags, score
    from train_translation import DIRECTIONS, base_model

    rows = load_flores(args.limit, args.direction)
    if not rows:
        print("no FLORES on disk — run data/scripts/download_flores.py first",
              file=sys.stderr)
        return 1
    base = base_model(args.direction)
    tag = DIRECTIONS[args.direction]["tag"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"direction {args.direction} | base {base} | tag {tag} | "
          f"{len(rows)} FLORES pairs | {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base).to(device)
    sources = [f"{tag} {s}" if tag else s for _, s, _ in rows]
    t0 = time.time()
    raw = translate_all(model, tokenizer, sources, device)
    hyps, leaked = strip_tags(raw)
    bleu, chrf2 = score(hyps, [r for _, _, r in rows])
    print(f"\n=== base on FLORES, {args.direction} ===")
    print(f"  pairs  : {len(rows)}")
    print(f"  BLEU   : {bleu:.2f}")
    print(f"  chrF2  : {chrf2:.2f}")
    print(f"  language tag left in the output: {leaked}/{len(rows)}")
    print(f"  ({time.time() - t0:.0f}s)")

    want = COMMITTED.get(args.direction)
    if want and not args.limit:
        print(f"\ncommitted ({want['source']}):")
        print(f"  BLEU  {want['bleu']:.2f}  ->  {bleu:.2f}   ({bleu - want['bleu']:+.2f})")
        print(f"  chrF2 {want['chrf2']:.2f}  ->  {chrf2:.2f}   ({chrf2 - want['chrf2']:+.2f})")
        if len(rows) != want["pairs"]:
            print(f"  NOTE: {len(rows)} pairs scored against a figure published "
                  f"on {want['pairs']} — not the same measurement")

    out = REPO / "training" / "form-rate" / f"base-flores-{args.direction}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"direction": args.direction, "base": str(base), "tag": tag,
         "pairs": len(rows), "bleu": bleu, "chrf2": chrf2, "tag_leak": leaked,
         "committed": want}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
