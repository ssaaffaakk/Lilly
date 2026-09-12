#!/usr/bin/env python3
"""Recompute the published English→Bosnian numbers from the stored outputs.

`training/verify_base_flores.py` re-decodes the untouched base to check a
committed figure. This file needs no model at all: `training/evaluate.py` saved
every translation it scored into `training/hypotheses-en-bs.json`, so the four
headline scores in `training/RESULTS-en-bs.md` and the interval beside them can
be checked by arithmetic alone, in seconds, on any machine.

That matters because the two documents disagreed. Until 12 September 2026
README.md published the BLEU gap's 95% interval as [+0.70, +1.62] where
training/RESULTS-en-bs.md:31 published [+0.67, +1.65]. A bootstrap is a random
procedure, so two runs of it differ — but only one of the two could be
reproduced from the outputs on disk, and the owner ruled that the reproducible
one is what gets published. Both documents now say [+0.67, +1.65], and this file
is what holds them to it: the intervals below are quoted from the documents, so
a change to either has to come past a failing run rather than past nobody.

    python3 training/verify_published_en_bs.py
    python3 training/verify_published_en_bs.py --served training/app-hypotheses-en-bs.json

With --served it does the same for the int8 build the app answers with, whose
translations `training/evaluate_app.py --direction en-bs` leaves in that file.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STORED = REPO_ROOT / "training" / "hypotheses-en-bs.json"

# Quoted from the documents this file exists to check, so that a change to
# either has to come past a failing run rather than past nobody.
PUBLISHED = {
    "FLORES-200 2,009, base": ("base_fl", "flores_refs", 29.57, 58.96),
    "FLORES-200 2,009, Lilly": ("tuned_fl", "flores_refs", 30.73, 60.00),
    "in-house 1,500, base": ("base_in", "in_house_refs", 31.94, 58.74),
    "in-house 1,500, Lilly": ("tuned_in", "in_house_refs", 34.06, 60.11),
}
PUBLISHED_INTERVALS = {
    "README.md:284": {"bleu": [0.67, 1.65], "chrf2": [0.69, 1.35]},
    "training/RESULTS-en-bs.md:31": {"bleu": [0.67, 1.65]},
}


def paired_bootstrap(bleu, chrf, base, tuned, refs, n, seed=11):
    """Percentile interval on the tuned-minus-base gap, resampling sentences.

    BLEU and chrF are both computed from per-sentence sufficient statistics that
    add up, so the statistics come out once and each resample only re-adds them.
    Re-scoring 2,009 strings a thousand times over takes an hour for the
    identical answer.
    """
    stats = {name: {"bleu": bleu._extract_corpus_statistics(hyps, [refs]),
                    "chrf": chrf._extract_corpus_statistics(hyps, [refs])}
             for name, hyps in (("base", base), ("tuned", tuned))}
    rng, m = random.Random(seed), len(refs)
    gaps_b, gaps_c = [], []
    started = time.time()
    for i in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        gaps_b.append(bleu._aggregate_and_compute(
            [stats["tuned"]["bleu"][k] for k in idx]).score
            - bleu._aggregate_and_compute(
                [stats["base"]["bleu"][k] for k in idx]).score)
        gaps_c.append(chrf._aggregate_and_compute(
            [stats["tuned"]["chrf"][k] for k in idx]).score
            - chrf._aggregate_and_compute(
                [stats["base"]["chrf"][k] for k in idx]).score)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n}  ({time.time() - started:.0f}s)", flush=True)
    gaps_b.sort()
    gaps_c.sort()
    lo, hi = int(0.025 * n), int(0.975 * n) - 1
    return {"bleu_ci": [round(gaps_b[lo], 2), round(gaps_b[hi], 2)],
            "chrf2_ci": [round(gaps_c[lo], 2), round(gaps_c[hi], 2)],
            "bleu_at_or_below_zero": sum(1 for g in gaps_b if g <= 0),
            "chrf2_at_or_below_zero": sum(1 for g in gaps_c if g <= 0),
            "resamples": n, "seed": seed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--stored", type=Path, default=STORED,
                    help="outputs training/evaluate.py saved")
    ap.add_argument("--served", type=Path, default=None,
                    help="outputs training/evaluate_app.py --direction en-bs saved")
    ap.add_argument("--json", type=Path, help="also write the findings as json")
    args = ap.parse_args()

    from sacrebleu.metrics import BLEU, CHRF
    bleu, chrf = BLEU(), CHRF(word_order=0)

    def score(hyps, refs):
        return (bleu.corpus_score(hyps, [refs]).score,
                chrf.corpus_score(hyps, [refs]).score)

    if not args.stored.exists():
        raise SystemExit(f"no stored outputs at {args.stored}")
    h = json.loads(args.stored.read_text(encoding="utf-8"))
    found = {"stored": str(args.stored), "checks": {}, "disagreements": []}

    print(f"published scores, recomputed from {args.stored.name}")
    worst = 0.0
    for label, (key, ref_key, want_bleu, want_chrf) in PUBLISHED.items():
        b, c = score(h[key], h[ref_key])
        ok = round(b, 2) == want_bleu and round(c, 2) == want_chrf
        worst = max(worst, abs(b - want_bleu), abs(c - want_chrf))
        print(f"  {label:26s} {b:6.2f} / {c:6.2f}   published {want_bleu:6.2f} /"
              f" {want_chrf:6.2f}   {'MATCH' if ok else 'MISMATCH'}")
        found["checks"][label] = {"recomputed": [round(b, 4), round(c, 4)],
                                  "published": [want_bleu, want_chrf], "match": ok}

    print(f"\npaired bootstrap, {args.resamples} resamples:")
    boot = paired_bootstrap(bleu, chrf, h["base_fl"], h["tuned_fl"],
                            h["flores_refs"], args.resamples)
    print(f"  BLEU  gap interval {boot['bleu_ci']}   "
          f"{boot['bleu_at_or_below_zero']} of {args.resamples} at or below zero")
    print(f"  chrF2 gap interval {boot['chrf2_ci']}   "
          f"{boot['chrf2_at_or_below_zero']} of {args.resamples} at or below zero")
    found["bootstrap"] = boot

    for where, claims in PUBLISHED_INTERVALS.items():
        for metric, published in claims.items():
            mine = boot[f"{metric}_ci"]
            agrees = all(abs(a - b) <= 0.02 for a, b in zip(published, mine))
            print(f"  {where:30s} {metric:5s} {published} "
                  f"{'reproduces' if agrees else 'DOES NOT reproduce'}")
            if not agrees:
                found["disagreements"].append(
                    {"where": where, "metric": metric, "published": published,
                     "reproduced": mine})

    if args.served:
        if not args.served.exists():
            print(f"\nno served outputs at {args.served} — run "
                  f"training/evaluate_app.py --direction en-bs first")
        else:
            a = json.loads(args.served.read_text(encoding="utf-8"))
            refs = h["flores_refs"]
            if len(a["base"]) != len(refs):
                raise SystemExit(f"{len(a['base'])} served rows against "
                                 f"{len(refs)} references")
            bb, bc = score(a["base"], refs)
            tb, tc = score(a["lilly"], refs)
            print(f"\nthe int8 build the app serves, same {len(refs):,} pairs")
            print(f"  base  {bb:6.2f} / {bc:6.2f}")
            print(f"  Lilly {tb:6.2f} / {tc:6.2f}   gap {tb - bb:+.2f} / {tc - bc:+.2f}")
            served_boot = paired_bootstrap(bleu, chrf, a["base"], a["lilly"], refs,
                                           args.resamples)
            print(f"  BLEU  gap interval {served_boot['bleu_ci']}")
            print(f"  chrF2 gap interval {served_boot['chrf2_ci']}")
            found["served"] = {"base": [round(bb, 4), round(bc, 4)],
                               "lilly": [round(tb, 4), round(tc, 4)],
                               "bootstrap": served_boot}

    if args.json:
        args.json.write_text(json.dumps(found, indent=1) + "\n", encoding="utf-8")
        print(f"\nwritten to {args.json}")

    # A published score that no longer comes out of the stored outputs is a
    # failure, not a note: exit non-zero so a checker cannot pass it by.
    if any(not c["match"] for c in found["checks"].values()):
        print(f"\nFAILED: a published score does not reproduce "
              f"(worst difference {worst:.2f})", file=sys.stderr)
        return 1
    # An interval is only held to the documents at the settings the documents
    # were written from. A bootstrap is random: run it with another resample
    # count and the bounds move by more than the tolerance for no reason worth
    # failing over, so that case warns instead of failing.
    if found["disagreements"]:
        canonical = args.resamples == 1000 and found["bootstrap"]["seed"] == 11
        print(f"\n{'FAILED' if canonical else 'NOTE'}: a published interval does "
              f"not reproduce", file=sys.stderr)
        for d in found["disagreements"]:
            print(f"  {d['where']} {d['metric']}: published {d['published']}, "
                  f"recomputed {d['reproduced']}", file=sys.stderr)
        if canonical:
            return 1
        print(f"  (not failing: {args.resamples} resamples rather than the "
              f"1,000 the documents were written from)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
