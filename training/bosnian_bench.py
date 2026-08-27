#!/usr/bin/env python3
"""BosnianBench — does Lilly read Bosnian, or generic Serbo-Croatian?

BLEU cannot answer that. One wrong term in a twenty-word sentence costs it a
point or two and the signal disappears into the average, which is the whole
reason docs/BOSNIAN_METRIC.md exists. So this scores the decision instead of
the sentence: on each case there is a term the Bosnian sources use and the
Serbian or Croatian ones do not, and the model either carried its meaning into
English or it did not. One target, one bit, no averaging it away.

## What is scored, and why it is scored this way

The direction is bs->en, so nothing Bosnian survives into the output — you
cannot look at English and ask whether it is ijekavica. What you can ask is
whether the model *understood* the Bosnian-only word. Every case in
bench/cases.tsv carries a target term and the English a professional translator
wrote for it, so:

    term recall = targets whose accepted English appears in the output
                  ------------------------------------------------------
                                  targets attempted

Accepted English is not written here and not written in bench/build.py either.
It is recovered by aligning the term against the English side of the same
professional corpora — P(english word | the term is in the sentence), filtered
by lift so merely common words drop out — then narrowed to what this sentence's
own translator chose. Writing the accepted answers by hand would make the exam
and the answer key the same document.

Term recall on its own is not enough: a good Serbo-Croatian model scores well on
it simply by being a good translator. So each case is run twice — once as the
professional wrote it, once with the target swapped for its Serbian or Croatian
counterpart, one word changed and nothing else:

    variant gap = term recall on the Bosnian form - term recall on the other form

A model that has learned Bosnian specifically reads both, so its gap sits near
zero. A model that leans on generic Serbo-Croatian does better on the swapped
sentence, and the gap goes negative. The gap is the part BLEU cannot see, and
it is a within-sentence control: the two sides differ by one word, so anything
else about the sentence cancels.

## The checks on the measurement itself

Three run every time, because a metric nobody measured is not evidence:

  reference control  the professional English through the scorer. Near 100% or
                     the matching code is broken (case folding, word boundaries).
                     Note what it does *not* prove: the accepted answers were
                     intersected with this sentence's reference when the case was
                     built, so they are inside it by construction. This control
                     tests the matcher, never the answer key.
  copy control       the untranslated Bosnian sentence through the scorer.
                     Near 0% or the metric is rewarding something other than
                     translation.
  variant control    the paired swap above. If a model scores the two sides
                     identically term by term, the swap is not landing on
                     anything the model notices, and the gap column is empty of
                     meaning rather than evidence of parity.

## What it cannot see — measured on the 2026-08-27 run, not guessed at

A hit needs the reference's own English word. The model that writes "entire"
where the professional wrote "whole", or "aircraft" for "plane", is marked wrong
having understood the term perfectly. Of the fine-tune's 30 missed targets, 7
are a plain suffix inflection of the wanted word — time/times, visit/visits,
song/songs — and reading the rest, most are either the same lemma in another
shape (change/changing, intent/intention, believes/believed), a spelling
variant (neighbouring/neighboring), or a synonym (whole/entire, children/kids,
victory/win). So the residual few points are mostly lexical choice, not
comprehension, and the recall figure understates both models by an unknown but
similar amount.

The swap is also weaker than it looks here: the base is opus-mt-tc-big-zls-en,
trained on the whole South Slavic family, so it reads ijekavica and ekavica
equally well and the gap has little room to open. Only 3.6% of swapped targets
scored differently from their Bosnian twin. The gap column is near zero because
the swap does not land, which is not the same as evidence of parity.

**There are no Turkisms in here, and that is now a measured fact rather than a
missing corpus.** kahva, čaršija, avlija, sokak and mahala are the words that
most obviously mark Bosnian, and every one of them is rejected. The first
explanation was that news and Wikipedia simply do not contain them, which was
true — all five read zero. So OpenSubtitles v2024 bs-en, 18.5M lines of the
spoken register, was counted: they are there, kahva 162 times, mahala 175,
komšija 3,380. They still do not pass, and cutting the corpus to the part that
is Bosnian by its own orthography does not rescue them either — komšija tops
out at a 0.78 share, kahva at 0.035.

The reason is the admission rule, not the data. A term is admitted when Bosnian
writers pick it and not its Serbian or Croatian counterpart, at least 98% of the
time. Yat pairs meet that because they are in complementary distribution: a
Bosnian text writes svijet and never svet. Turkisms are not in complementary
distribution — kafa, ulica, most, prozor, susjed and vrt are all ordinary
Bosnian too, so both words live in the same Bosnian sentence and no share test
can separate them. Measuring Turkisms needs a different instrument, and this is
not it. What this scores remains ijekavian forms and lexical doublets. The
counts are in bench/terms-rejected.tsv; bench/build.py prints the working.

    python3 training/bosnian_bench.py
    python3 training/bosnian_bench.py --limit 40      # a quick look
    python3 training/bosnian_bench.py --model models/lilly/translator

About twenty minutes on a laptop for the full set on both models.
"""
import argparse
import collections
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CASES = REPO / "bench" / "cases.tsv"
# Translating 1,500 sentences twice takes an hour; the cache means a change to
# the scoring does not cost that hour again. Dotted so it stays out of the way
# of the two TSVs, which are the artefacts worth reading.
CACHE = REPO / "bench" / ".outputs.json"
BASE_BUILD = REPO / "models" / "lilly" / "translator-base"
TUNED_BUILD = REPO / "models" / "lilly" / "translator"

# The base model prints its language tag into the translation. It cannot create
# a false hit, but it is stripped anyway so the two models are read the same way
# — training/evaluate_app.py strips it for the same reason.
LANGUAGE_TAG = re.compile(r"^\s*(>>[a-zA-Z_]+<<\s*)+")


def load_cases(limit=None):
    with open(CASES, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["terms"] = r["terms"].split("|")
        r["categories"] = r["categories"].split("|")
        r["accept"] = [a.split(",") for a in r["accept_en"].split("|")]
    return rows[:limit] if limit else rows


def hit(output: str, accepted: list) -> bool:
    """Did any English a professional used for this term reach the output?"""
    low = LANGUAGE_TAG.sub("", output).lower()
    return any(re.search(rf"\b{re.escape(a)}\b", low) for a in accepted)


def translate_all(build: Path, texts: list, label: str) -> list:
    from app.translate import Engine

    if not (build / "model.bin").exists():
        raise SystemExit(f"no build at {build} — scripts/build_translator.py makes one")
    engine = Engine(directory=build)
    out, started = [], time.time()
    for i, text in enumerate(texts):
        out.append(engine.translate(text, truncate=True))
        if (i + 1) % 100 == 0:
            print(f"  {label} {i + 1}/{len(texts)}  ({time.time() - started:.0f}s)",
                  flush=True)
    print(f"  {label} done in {time.time() - started:.0f}s", flush=True)
    return out


def per_target(cases, outputs, key="bs"):
    """One record per target term, so a case with two targets carries two votes."""
    marks = []
    for case, out in zip(cases, outputs):
        if key == "variant_bs" and not case["variant_bs"]:
            continue
        for term, cat, accept in zip(case["terms"], case["categories"], case["accept"]):
            marks.append((case["case_id"], term, cat, hit(out, accept)))
    return marks


def rate(marks):
    return (sum(1 for m in marks if m[3]) / len(marks)) if marks else 0.0


def wilson(k, n, z=1.96):
    """Interval for a proportion. At n in the hundreds the normal approximation
    is already misleading near the ceiling, which is exactly where these land."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(marks_a, marks_b, n_samples=2000, seed=11):
    """p for "the two models differ", resampling cases rather than targets.

    Targets from one sentence are not independent — a model that drops the whole
    clause misses both — so the resample takes whole cases.
    """
    by_case = {}
    for cid, term, cat, ok in marks_a:
        by_case.setdefault(cid, ([], []))[0].append(ok)
    for cid, term, cat, ok in marks_b:
        by_case.setdefault(cid, ([], []))[1].append(ok)
    ids = [c for c, (a, b) in by_case.items() if a and b]
    observed = rate(marks_b) - rate(marks_a)
    rng, worse = random.Random(seed), 0
    for _ in range(n_samples):
        pick = [ids[rng.randrange(len(ids))] for _ in ids]
        a = [x for c in pick for x in by_case[c][0]]
        b = [x for c in pick for x in by_case[c][1]]
        delta = sum(b) / len(b) - sum(a) / len(a)
        if (delta <= 0) if observed > 0 else (delta >= 0):
            worse += 1
    return observed, worse / n_samples


def by_category(marks):
    out = {}
    for cid, term, cat, ok in marks:
        k, n = out.get(cat, (0, 0))
        out[cat] = (k + ok, n + 1)
    return out


def sentence_scores(hyps, refs):
    """chrF2 over the same cases — the number the term test is meant to beat.

    Printed alongside so the claim "BLEU cannot see this" is shown rather than
    asserted: if the two columns move together the term test adds nothing.
    """
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError:
        return None
    return (BLEU().corpus_score(hyps, [refs]).score,
            CHRF(word_order=0).corpus_score(hyps, [refs]).score)


def report(cases, results):
    refs = [c["en"] for c in cases]
    labels = list(results)
    total = len(results[labels[0]]["marks_bs"])

    print(f"\n{'':<22}{'term recall':>13}{'95% interval':>18}{'same sentence,':>17}"
          f"{'gap':>7}{'BLEU':>8}{'chrF2':>8}")
    print(f"{'':<22}{'(Bosnian)':>13}{'':>18}{'swapped':>17}")
    for label, r in results.items():
        left, right = marks_paired(r["marks_bs"], r["marks_variant"])
        k, n = sum(1 for m in r["marks_bs"] if m[3]), total
        lo, hi = wilson(k, n)
        s = sentence_scores(r["out_bs"], refs)
        print(f"  {label:<20}{100 * k / n:>11.1f}%  [{100 * lo:>5.1f}, {100 * hi:>5.1f}]"
              f"{100 * rate(right):>16.1f}%{100 * (rate(left) - rate(right)):>+7.1f}"
              + (f"{s[0]:>8.1f}{s[1]:>8.1f}" if s else f"{'—':>8}{'—':>8}"))
    print(f"  {'targets':<20}{total:>11}"
          f"{len(marks_paired(results[labels[0]]['marks_bs'], results[labels[0]]['marks_variant'])[0]):>34}")

    print("\nby category — term recall on the Bosnian form, and the swapped gap")
    cats = sorted({c for r in results.values() for _, _, c, _ in r["marks_bs"]})
    print(f"  {'':<20}" + "".join(f"{c:>16}" for c in cats))
    for label, r in results.items():
        row = by_category(r["marks_bs"])
        left, right = marks_paired(r["marks_bs"], r["marks_variant"])
        gl, gr = by_category(left), by_category(right)
        cells = []
        for c in cats:
            if c not in row:
                cells.append(f"{'—':>16}")
                continue
            g = ""
            if c in gl and c in gr and gl[c][1]:
                g = f" {100 * (gl[c][0] / gl[c][1] - gr[c][0] / gr[c][1]):+.1f}"
            cells.append(f"{100 * row[c][0] / row[c][1]:>10.1f}%{g:>6}")
        print(f"  {label:<20}" + "".join(cells))
    base_cat = by_category(results[labels[0]]["marks_bs"])
    print(f"  {'targets':<20}" + "".join(f"{base_cat[c][1]:>16}" for c in cats))

    if len(labels) == 2:
        a, b = results[labels[0]]["marks_bs"], results[labels[1]]["marks_bs"]
        delta, p = paired_bootstrap(a, b)
        print(f"\n{labels[1]} - {labels[0]}: {100 * delta:+.1f} points of term recall, "
              f"paired bootstrap p = {p:.4f}"
              + ("" if p < 0.05 else "  (does not clear 0.05)"))

    # The swapped gap is the more direct look at this project's claim than term
    # recall is — a model leaning on generic Serbo-Croatian should do relatively
    # better when the Bosnian term is replaced by its Croatian or Serbian form,
    # so a wider gap is the shape "it really is Bosnian" would take. It is
    # reported second and tested, not promoted, because it was not the measure
    # named in training/PREREGISTRATION.md and it was looked at after the
    # headline was already known. A number noticed afterwards that happens to
    # favour us is exactly what pre-registration exists to keep in its place.
    if len(labels) == 2:
        per_model = []
        for label in labels:
            r = results[label]
            left, right = marks_paired(r["marks_bs"], r["marks_variant"])
            swapped = {(c, t): ok for c, t, _, ok in right}
            per_model.append({(c, t): ok - swapped[(c, t)]
                              for c, t, _, ok in left if (c, t) in swapped})
        shared = sorted(set(per_model[0]) & set(per_model[1]))
        if shared:
            diffs = [per_model[1][k] - per_model[0][k] for k in shared]
            observed = 100 * sum(diffs) / len(diffs)
            rng = random.Random(41)
            centre = sum(diffs) / len(diffs)
            extreme = 0
            for _ in range(10000):
                draw = sum(diffs[rng.randrange(len(diffs))] - centre
                           for _ in range(len(diffs))) / len(diffs) * 100
                extreme += abs(draw) >= abs(observed)
            p_gap = extreme / 10000
            print(f"\nswapped gap, {labels[1]} against {labels[0]}: "
                  f"{observed:+.1f} points on {len(shared)} shared targets, "
                  f"paired bootstrap p = {p_gap:.4f}"
                  + ("" if p_gap < 0.05 else "  (does not clear 0.05)"))
            print("  secondary and post-hoc: this measure was not the one "
                  "pre-registered, and it was tested after the headline was known")

    print("\nchecks on the measurement")
    for name, marks in results[labels[0]]["controls"].items():
        print(f"  {name:<20}{100 * rate(marks):>11.1f}%   ({len(marks)} targets)")
    agree = disagree = 0
    for label, r in results.items():
        left, right = marks_paired(r["marks_bs"], r["marks_variant"])
        for x, y in zip(left, right):
            agree += x[3] == y[3]
            disagree += x[3] != y[3]
    print(f"  {'variant control':<20}{100 * disagree / (agree + disagree):>11.1f}%   "
          f"of swapped targets scored differently from the Bosnian one")

    # Which terms are missed matters more than how many: a metric that cannot
    # name its own failures is not much use for fixing anything.
    for label, r in results.items():
        missed = collections.Counter(term for _, term, _, ok in r["marks_bs"] if not ok)
        if missed:
            print(f"\nmissed most often by {label}: "
                  + ", ".join(f"{t} ({k})" for t, k in missed.most_common(12)))
    if len(labels) == 2:
        a = {(c, t): ok for c, t, _, ok in results[labels[0]]["marks_bs"]}
        b = {(c, t): ok for c, t, _, ok in results[labels[1]]["marks_bs"]}
        broke = sorted(t for k, t in ((k, k[1]) for k in a) if a[k] and not b.get(k, True))
        fixed = sorted(t for k, t in ((k, k[1]) for k in a) if not a[k] and b.get(k, False))
        print(f"\n{labels[1]} fixed {len(fixed)} targets the base model missed "
              f"and broke {len(broke)} it had right")
        print(f"  broke: {', '.join(broke[:15]) or 'none'}")


def marks_paired(marks_bs, marks_variant):
    """The two sides restricted to the targets that exist on both, so the gap is
    a difference on the same terms rather than on two different populations."""
    keys = {(c, t) for c, t, _, _ in marks_variant}
    left = [m for m in marks_bs if (m[0], m[1]) in keys]
    keys_left = {(c, t) for c, t, _, _ in left}
    right = [m for m in marks_variant if (m[0], m[1]) in keys_left]
    return left, right


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="score only the first N cases")
    ap.add_argument("--fresh", action="store_true", help="re-translate, ignore the cache")
    ap.add_argument("--model", action="append", type=Path,
                    help="a build to score; repeat for more. Default: base and Lilly")
    args = ap.parse_args()

    builds = args.model or [BASE_BUILD, TUNED_BUILD]
    cases = load_cases(args.limit)
    variants = [c for c in cases if c["variant_bs"]]
    print(f"{len(cases)} cases, {sum(len(c['terms']) for c in cases)} target terms, "
          f"{len(variants)} of them with a swapped variant")

    cached = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    results, dirty = {}, False
    for build in builds:
        label = build.name
        key = f"{label}:{len(cases)}"
        if not args.fresh and key in cached:
            print(f"  reusing saved translations for {label} — --fresh to redo them")
            out_bs, out_var = cached[key]["bs"], cached[key]["variant"]
        else:
            out_bs = translate_all(build, [c["bs"] for c in cases], label)
            out_var = translate_all(build, [c["variant_bs"] for c in variants],
                                    f"{label} variant")
            cached[key], dirty = {"bs": out_bs, "variant": out_var}, True
        var_by_id = dict(zip([c["case_id"] for c in variants], out_var))
        aligned = [var_by_id.get(c["case_id"], "") for c in cases]
        results[label] = {
            "out_bs": out_bs,
            "marks_bs": per_target(cases, out_bs),
            "marks_variant": per_target(cases, aligned, key="variant_bs"),
            "controls": {
                "reference control": per_target(cases, [c["en"] for c in cases]),
                "copy control": per_target(cases, [c["bs"] for c in cases]),
            },
        }
    if dirty:
        CACHE.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")

    report(cases, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
