#!/usr/bin/env python3
"""Two sets of served-path translations of the same build, paired.

training/evaluate_app.py compares two BUILDS through one code path. This
compares one build through two versions of the code path: the translations
it saved before a change to app.translate and the translations it saved
after. That is the measurement a change to the splitter needs, and evaluate_app
cannot produce it, because both of its columns go through the splitter as it
is today.

    python3 training/compare_hypotheses.py OLD.json NEW.json --side lilly
    python3 training/compare_hypotheses.py OLD.json NEW.json --side base

OLD.json and NEW.json are evaluate_app.py's saved files (`--saved`). Both must
hold every FLORES pair, and both must name the same build for the side being
compared -- otherwise this would be comparing builds, which is evaluate_app's
job, and the fingerprint check refuses.

Scores are BLEU and chrF2 with the language tag stripped, on all pairs and on
the devtest half alone, with sacrebleu's paired bootstrap between the two
files. Pre-registered use: training/PREREGISTRATION.md, "v4 -- translate --
ordinals".
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training import evaluate_app  # noqa: E402


def load(path: Path, side: str, n: int) -> tuple:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get(side) or []
    if data.get("n") != n or len(rows) != n:
        raise SystemExit(f"{path}: holds {len(rows)} {side} translations for n={data.get('n')}, "
                         f"but {n} FLORES pairs are on disk. Both files must cover the whole set.")
    return rows, data.get(f"{side}_build") or ""


def scored(hyps: list, refs: list) -> tuple:
    clean = [evaluate_app.LANGUAGE_TAG.sub("", h).strip() for h in hyps]
    return clean, evaluate_app.score(clean, refs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--side", choices=("lilly", "base"), default="lilly")
    ap.add_argument("--flores", type=Path, default=evaluate_app.FLORES,
                    help="the FLORES directory the references are read from")
    ap.add_argument("--allow-different-builds", action="store_true",
                    help="compare even if the two files name different builds")
    ap.add_argument("--json", type=Path, default=None, help="write the numbers here too")
    args = ap.parse_args()

    evaluate_app.FLORES = args.flores
    src, refs = evaluate_app.pairs()
    old, old_build = load(args.old, args.side, len(src))
    new, new_build = load(args.new, args.side, len(src))
    if old_build != new_build and not args.allow_different_builds:
        raise SystemExit(f"{args.old.name} is build {old_build[:16] or '(unrecorded)'} and "
                         f"{args.new.name} is {new_build[:16] or '(unrecorded)'}: two builds, "
                         f"not two code paths. That comparison is evaluate_app.py's; "
                         f"pass --allow-different-builds if you mean it.")
    print(f"{len(src):,} pairs, side {args.side}, build {new_build[:16] or '(unrecorded)'}")

    out = {"side": args.side, "build": new_build, "n": len(src), "splits": {}}
    for name in ("all", "devtest"):
        cut = evaluate_app.split_range(name, len(src))
        r = refs[cut]
        o, (ob, oc) = scored(old[cut], r)
        n, (nb, nc) = scored(new[cut], r)
        p = evaluate_app.significance(o, n, r)
        changed = sum(1 for a, b in zip(o, n) if a != b)
        print(f"\n{name}: {len(r):,} pairs, {changed:,} translations differ")
        print(f"  {'':<6} {'BLEU':>7} {'chrF2':>7}")
        print(f"  {'old':<6} {ob:>7.2f} {oc:>7.2f}")
        print(f"  {'new':<6} {nb:>7.2f} {nc:>7.2f}")
        print(f"  {'gap':<6} {nb - ob:>+7.2f} {nc - oc:>+7.2f}")
        for metric, pv in p.items():
            print(f"    {metric} p = {pv:.4f}" + ("" if pv < 0.05 else "   (does not clear 0.05)"))
        out["splits"][name] = {"pairs": len(r), "changed": changed,
                               "old": {"bleu": ob, "chrf": oc}, "new": {"bleu": nb, "chrf": nc},
                               "p": p}
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
