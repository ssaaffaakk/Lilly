#!/usr/bin/env python3
"""Recompute the published Bosnian→English numbers from the stored outputs.

The reply direction has had `training/verify_published_en_bs.py` since
12 September. The forward direction — the product's MAIN direction, the one
README.md leads with at 43.25 BLEU — had nothing. No script anywhere in the
repository asserted 43.25, and `grep -rn "43\\.25" --include="*.py"` returned
only a print statement in a launcher. That asymmetry was found on 13 September
while auditing the repository against itself: the secondary direction was
better guarded than the primary one.

It needs no model and no GPU. `training/evaluate_app.py --direction bs-en`
saved every translation it scored on the Kaggle T4 into
`training/app-hypotheses-ordinals-kaggle.json` (committed), and the English
references it scored them against are committed beside it in
`training/flores-refs-en.json` — because `data/flores/` is NOT tracked, so a
clone that had only the translations could still not check anything. Both files
together are the whole proof, and this runs in seconds on any machine.

    python3 training/verify_published_bs_en.py
    python3 training/verify_published_bs_en.py --resamples 1000

THE TAG MATTERS HERE IN A WAY IT DOES NOT FOR THE REPLY DIRECTION. The base
model writes its own language tag into 576 of 2,009 translations; Lilly writes
it into none. The published base figures are the tag-stripped ones, because
otherwise the fine-tuning is credited with translation quality it did not gain
— it fixed a defect in what the model emits. So the strip is applied to both
sides before scoring, exactly as `evaluate_app.py` reports it, and the leak
count is checked as a published number in its own right.

Exit code 1 if any published score or interval stops reproducing.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HYPS = REPO_ROOT / "training" / "app-hypotheses-ordinals-kaggle.json"
REFS = REPO_ROOT / "training" / "flores-refs-en.json"

# The same pattern evaluate_app.py strips with, kept here rather than imported
# so this file needs neither ctranslate2 nor the app package to run.
LANGUAGE_TAG = re.compile(r"^\s*(>>[a-zA-Z_]+<<\s*)+")

# Quoted from the documents this file exists to check, so a change to any of
# them has to come past a failing run rather than past nobody. Every figure is
# tag-stripped on both sides, which is how all three documents publish them.
PUBLISHED = {
    "FLORES devtest 1,012, base": ("devtest", "base", 42.08, 67.85),
    "FLORES devtest 1,012, Lilly": ("devtest", "lilly", 43.25, 68.10),
    "FLORES dev+devtest 2,009, base": ("all", "base", 41.77, 67.66),
    "FLORES dev+devtest 2,009, Lilly": ("all", "lilly", 43.03, 67.81),
}
# README.md:229 and training/RESULTS-product.md:42 both publish this count.
PUBLISHED_LEAKS = {"base": 576, "lilly": 0}
# training/RESULTS-product.md:61-64, quoted there as the fine-tune's own gain
# on all 2,009 pairs. p is reported, not an interval, so the check is on the
# point estimates and on the share of resamples at or below zero.
PUBLISHED_GAP = {"bleu": 1.26, "chrf2": 0.15}

# The build digests the Kaggle run recorded. If someone regenerates the cache
# from different weights, these stop matching and the scores below mean nothing.
PUBLISHED_BUILDS = {"base_build": "348a984c324510cee218dfce8a7228e8",
                    "lilly_build": "1aedcc11231cdf50817ff12f99ff0d1e"}


def strip_tags(rows):
    return [LANGUAGE_TAG.sub("", r) for r in rows]


def paired_bootstrap(bleu, chrf, base, tuned, refs, n, seed=11):
    """Percentile interval on the tuned-minus-base gap, resampling sentences.

    Lifted in shape from verify_published_en_bs.py: BLEU and chrF are computed
    from per-sentence sufficient statistics that add up, so the statistics come
    out once and each resample only re-adds them. Re-scoring 2,009 strings a
    thousand times over takes an hour for the identical answer.
    """
    stats = {name: {"bleu": bleu._extract_corpus_statistics(hyps, [refs]),
                    "chrf": chrf._extract_corpus_statistics(hyps, [refs])}
             for name, hyps in (("base", base), ("tuned", tuned))}
    rng, m = random.Random(seed), len(refs)
    gaps = {"bleu": [], "chrf2": []}
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        for key, metric in (("bleu", bleu), ("chrf2", chrf)):
            stat = "bleu" if key == "bleu" else "chrf"
            scores = {}
            for name in ("base", "tuned"):
                rows = stats[name][stat]
                total = [sum(r[j] for r in (rows[i] for i in idx))
                         for j in range(len(rows[0]))]
                scores[name] = metric._compute_score_from_stats(total).score
            gaps[key].append(scores["tuned"] - scores["base"])
    return gaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resamples", type=int, default=1000,
                    help="bootstrap resamples for the gap (default 1000)")
    args = ap.parse_args()

    for path in (HYPS, REFS):
        if not path.exists():
            print(f"missing {path.relative_to(REPO_ROOT)} — this file is the "
                  f"proof, not a cache; it should be committed", file=sys.stderr)
            return 1

    h = json.loads(HYPS.read_text(encoding="utf-8"))
    r = json.loads(REFS.read_text(encoding="utf-8"))
    refs, bounds = r["refs"], r["bounds"]

    failures = []
    for key, want in PUBLISHED_BUILDS.items():
        got = h.get(key)
        if got != want:
            failures.append(f"{key}: recorded {want}, cache says {got}")
    if len(refs) != h["n"] or len(h["base"]) != h["n"] or len(h["lilly"]) != h["n"]:
        print(f"length mismatch: {h['n']} hypotheses against {len(refs)} references",
              file=sys.stderr)
        return 1

    print(f"builds: base {h['base_build']}  Lilly {h['lilly_build']}")
    leaks = {who: sum(1 for x in h[who] if LANGUAGE_TAG.match(x))
             for who in ("base", "lilly")}
    for who, want in PUBLISHED_LEAKS.items():
        mark = "MATCH" if leaks[who] == want else "MISMATCH"
        pct = 100 * leaks[who] / h["n"]
        print(f"  language tag written into the translation, {who:5} "
              f"{leaks[who]:4} of {h['n']} ({pct:4.1f}%)   published {want:4}   {mark}")
        if leaks[who] != want:
            failures.append(f"tag leaks, {who}: published {want}, recomputed {leaks[who]}")

    from sacrebleu.metrics import BLEU, CHRF
    # word_order=0 is what evaluate_app.py and the en-bs verifier both use; the
    # documents call it chrF2, which is the beta=2 default, not chrF++.
    bleu, chrf = BLEU(), CHRF(word_order=0)
    spans = {"all": slice(0, h["n"]), "devtest": slice(*bounds["devtest"])}

    print("\npublished scores, recomputed from the stored outputs (tag stripped "
          "from both sides)")
    for label, (span, who, pb, pc) in PUBLISHED.items():
        sl = spans[span]
        hyps, rf = strip_tags(h[who][sl]), refs[sl]
        b = bleu.corpus_score(hyps, [rf]).score
        c = chrf.corpus_score(hyps, [rf]).score
        ok = abs(b - pb) < 0.005 and abs(c - pc) < 0.005
        print(f"  {label:34} {b:6.2f} / {c:6.2f}   published {pb:6.2f} / {pc:6.2f}"
              f"   {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{label}: published {pb}/{pc}, recomputed {b:.2f}/{c:.2f}")

    sl = spans["all"]
    base, tuned, rf = strip_tags(h["base"][sl]), strip_tags(h["lilly"][sl]), refs[sl]
    print(f"\npaired bootstrap over the 2,009 pairs, {args.resamples} resamples:")
    gaps = paired_bootstrap(bleu, chrf, base, tuned, rf, args.resamples)
    for key, want in PUBLISHED_GAP.items():
        g = sorted(gaps[key])
        lo, hi = g[int(0.025 * len(g))], g[int(0.975 * len(g)) - 1]
        at_or_below = sum(1 for x in g if x <= 0)
        point = (bleu if key == "bleu" else chrf)
        got = (point.corpus_score(tuned, [rf]).score
               - point.corpus_score(base, [rf]).score)
        ok = abs(got - want) < 0.005
        print(f"  {key:5} gap {got:+.2f}  published {want:+.2f}  "
              f"{'MATCH' if ok else 'MISMATCH'}   95% [{lo:+.2f}, {hi:+.2f}]   "
              f"{at_or_below} of {len(g)} at or below zero")
        if not ok:
            failures.append(f"{key} gap: published {want:+.2f}, recomputed {got:+.2f}")

    if failures:
        print("\nFAILED — a published number does not reproduce:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("\nevery published Bosnian -> English number reproduces from the "
          "committed outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
