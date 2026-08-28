#!/usr/bin/env python3
"""SpeechBench — is Lilly's listener hearing *Bosnian*, or generic Serbo-Croatian?

training/evaluate_speech.py answers one question, word error rate, and it cannot
answer this one. That matters right now: the listener is about to be retrained
with Croatian and Serbian audio, because Bosnian speech data is scarce. The
predictable result is a WER that falls while the thing we are selling gets
worse — a model that hears *sedmicu* and writes *tjedan*, hears *mjesto* and
writes *mesto*. Each of those is one substitution in a twenty-word sentence:
half a point of WER, invisible in the average, and the whole product.

So this scores the decision instead of the sentence. Unlike the translator,
where nothing Bosnian survives into English, here the output is Bosnian text —
the variety the model chose is written down in front of you. On each target
the model did one of three things:

    bosnian   it wrote a Bosnian form of the term        (sedmica, mjesto, duhan)
    alt       it wrote the Croatian or Serbian form      (tjedan, mesto, duvan)
    neither   it wrote something else entirely, or nothing

and the two numbers that follow are

    term recall         = bosnian / all targets
    variety substitution = alt / (bosnian + alt)

Term recall moves when general transcription accuracy moves, so it is the one
to read next to WER. The substitution rate is the number this file exists for:
it is computed only over the targets where the model committed to one variety
or the other, so a model that simply gets worse at hearing does not move it.
A model drifting toward Croatian/Serbian does.

## Where the terms come from

Nothing here is a word list anybody typed. bench/terms.tsv is built by
bench/build.py by counting professional Bosnian corpora — a pair is admitted
only where professionals chose the Bosnian form and effectively never the
alternative — and this file reuses that file rather than writing its own.

There is one thing it has to fix. All 925 FLEURS test transcripts are FLORES
sentences, and FLORES is one of the three corpora whose counts admitted those
terms, so a term could in principle be admitted on the strength of the very
sentences it is about to be scored on. So admission is recomputed here with
every test transcript removed from the evidence, and a term that no longer
clears bench/build.py's own thresholds (n >= 20, Bosnian share >= 0.98) is
dropped and reported. This is not a leak of *model* information — the listener
never trained on the test split, checked below — but the term set would
otherwise be chosen partly on the text it grades.

## The checks on the measurement

Three, mirroring training/bosnian_bench.py, run on every invocation and needing
no model at all:

  reference control  the human transcript through the scorer. 100% recall and
                     0% substitution, or the matcher is broken. Its second job
                     is to catch a term whose "alternative" form is a real word
                     that turns up in Bosnian references for other reasons —
                     that shows up here as substitution above zero.
  shuffled control   the *neighbouring clip's* transcript through the scorer.
                     This is the audio analogue of bosnian_bench's copy control:
                     there is no input text to copy, so instead the scorer is
                     handed fluent, professional Bosnian that is not what was
                     said. Near 0%, or the metric is rewarding plausible
                     Bosnian rather than hearing.
  variant control    the transcript with each target swapped for its
                     Croatian/Serbian counterpart. Recall must collapse and
                     substitution must go to ~100%, or the swap is not landing
                     and the substitution column is empty of meaning.

The variant control is weaker here than in the translator's bench, and the
report says so: there it perturbs the model's *input* and measures the model;
here the input is a recording of a person, which cannot be edited to say
*mesto* instead of *mjesto*. So it tests the scorer, not the model.

    python3 training/speech_bench.py                      # both listeners, 200 clips
    python3 training/speech_bench.py --limit 0            # the whole test split
    python3 training/speech_bench.py --controls-only      # no model, seconds
    python3 training/speech_bench.py --model models/lilly/listen

The default 200 clips is the same prefix training/evaluate_speech.py --limit 200
scored for the 38.5% -> 35.5% figure in models/lilly/listen/built.json, so the
WER column is directly comparable to it. Transcriptions are cached in
bench/speech/.outputs.json and written every 20 clips, so an interrupted run
resumes rather than restarting.
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

# The same word error rate as the existing evaluator, by importing it rather
# than reimplementing it — two copies of a metric drift apart.
from training.evaluate_speech import edits, normalise  # noqa: E402
from training.train_speech import read_tsv  # noqa: E402

OUT = REPO / "bench" / "speech"
TERMS_IN = REPO / "bench" / "terms.tsv"
TEST_TSV = REPO / "data" / "speech" / "test.tsv"
TRAIN_TSV = REPO / "data" / "speech" / "train.tsv"
CACHE = OUT / ".outputs.json"
TUNED = REPO / "models" / "lilly" / "listen"
BEFORE = REPO / "models" / "lilly" / "listen.before-training"

WORD = re.compile(r"[a-zA-ZčćžšđČĆŽŠĐ]+", re.UNICODE)

# bench/build.py's admission thresholds, reused unchanged. Loosening them here
# to keep more terms would be tuning the exam to the answers.
MIN_BOS = 20
MIN_SHARE = 0.98


# ---------------------------------------------------------------- corpora


def _read_col(path: Path, want: str) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.rstrip("\n").split("\t")
        if len(p) == 3 and p[0] == want and p[1].strip():
            rows.append(p[1].strip())
    return rows


def evidence_sentences(exclude: set) -> dict:
    """The professional Bosnian bench/build.py counts, minus the graded text.

    Same three corpora, same roles; the only change is that any sentence that is
    also a test transcript is dropped, so no term can be admitted on the
    strength of a sentence it is about to be scored on.
    """
    flores = []
    for split in ("dev", "devtest"):
        flores += [l.strip() for l in
                   (REPO / "data/flores" / f"{split}.bs").read_text(encoding="utf-8").splitlines()
                   if l.strip()]
    ntrex = _read_col(REPO / "data/extra/extra-train.tsv", "ntrex")
    setimes = _read_col(REPO / "data/clean/train.tsv", "SETIMES")
    dropped = sum(1 for s in flores + ntrex + setimes if s in exclude)
    kept = {"flores": [s for s in flores if s not in exclude],
            "ntrex": [s for s in ntrex if s not in exclude],
            "setimes": [s for s in setimes if s not in exclude]}
    return kept, dropped


def count_forms(corpora: dict, wanted: set) -> collections.Counter:
    """Token counts, but only for the forms the term list asks about."""
    freq = collections.Counter()
    for sentences in corpora.values():
        for s in sentences:
            for w in WORD.findall(s):
                low = w.lower()
                if low in wanted:
                    freq[low] += 1
    return freq


# ---------------------------------------------------------------- terms


def load_terms() -> list:
    if not TERMS_IN.exists():
        raise SystemExit(f"{TERMS_IN} is missing — bench/build.py writes it")
    with open(TERMS_IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["bos"] = [w for w in r["bosnian_forms"].split("|") if w]
        r["alt"] = [w for w in r["alt_forms"].split("|") if w]
    return rows


def reverify(terms: list, corpora: dict) -> tuple:
    """Re-run bench/build.py's admission with the graded sentences removed."""
    wanted = {w for r in terms for w in r["bos"] + r["alt"]}
    freq = count_forms(corpora, wanted)
    admitted, dropped = [], []
    for r in terms:
        nb = sum(freq[w] for w in r["bos"])
        na = sum(freq[w] for w in r["alt"])
        share = nb / (nb + na) if nb + na else 0.0
        nm = int(r["n_alt_in_mixed_bcs"])   # mixed BCS holds no test transcripts
        r["heldout_n_bosnian"] = nb
        r["heldout_n_alt"] = na
        r["heldout_share"] = round(share, 4)
        (admitted if (nb >= MIN_BOS and share >= MIN_SHARE and nm >= 1)
         else dropped).append(r)
    return admitted, dropped


# ---------------------------------------------------------------- cases


def build_cases(rows: list, terms: list) -> list:
    """One target per (clip, term) whose Bosnian form the human transcript uses.

    A clip is not a case for a term whose alternative *also* appears in the
    reference: there the two varieties cannot be told apart in the output, so
    the target would be unscoreable rather than merely hard.
    """
    by_form = {}
    for r in terms:
        for form in r["bos"]:
            by_form.setdefault(form, r)
    alt_of = {}
    for r in terms:
        for form in r["alt"]:
            alt_of.setdefault(form, r)

    cases = []
    for clip, reference in rows:
        toks = [w.lower() for w in WORD.findall(reference)]
        present = set(toks)
        targets, seen = [], set()
        for tok in toks:
            r = by_form.get(tok)
            if r is None or r["term_id"] in seen:
                continue
            if any(a in present for a in r["alt"]):
                continue                      # both varieties in the reference
            seen.add(r["term_id"])
            targets.append({"term_id": r["term_id"], "category": r["category"],
                            "alt_variety": r["alt_variety"], "surface": tok,
                            "bos": r["bos"], "alt": r["alt"],
                            "variant": nearest(tok, r["alt"])})
        cases.append({"clip": clip, "reference": reference, "targets": targets})
    return cases


def char_edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def nearest(surface: str, forms: list) -> str:
    """The alternative form closest to the form actually spoken.

    Only the variant control uses this. For a yat pair or an h-pair it recovers
    the exact counterpart (mjesto->mesto, duhana->duvana); where the two words
    share no stem it returns some form of the right word, which is all a check
    on the matcher needs.
    """
    return min(forms, key=lambda f: (char_edits(surface, f), f)) if forms else ""


# ---------------------------------------------------------------- scoring


def outcome(hyp_tokens: set, target: dict) -> str:
    bos = any(f in hyp_tokens for f in target["bos"])
    alt = any(f in hyp_tokens for f in target["alt"])
    if bos and not alt:
        return "bosnian"
    if alt and not bos:
        return "alt"
    if bos and alt:
        # Both varieties in one transcript. Rare, and not a decision, so it is
        # not counted as either — reported separately rather than buried.
        return "both"
    return "neither"


def score(cases: list, hyps: list) -> list:
    """One mark per target: (clip, term_id, category, alt_variety, outcome)."""
    marks = []
    for case, hyp in zip(cases, hyps):
        toks = set(normalise(hyp))
        for t in case["targets"]:
            marks.append((case["clip"].name, t["term_id"], t["category"],
                          t["alt_variety"], outcome(toks, t)))
    return marks


def recall(marks: list) -> float:
    return sum(1 for m in marks if m[4] == "bosnian") / len(marks) if marks else 0.0


def substitution(marks: list) -> tuple:
    """alt / (bosnian + alt) — the varieties the model actually committed to."""
    b = sum(1 for m in marks if m[4] == "bosnian")
    a = sum(1 for m in marks if m[4] == "alt")
    return (a / (a + b) if a + b else 0.0), a, a + b


def word_error_rate(cases: list, hyps: list) -> tuple:
    total_edits = total_words = 0
    for case, hyp in zip(cases, hyps):
        ref_words, hyp_words = normalise(case["reference"]), normalise(hyp)
        total_edits += edits(hyp_words, ref_words)
        total_words += len(ref_words)
    return 100 * total_edits / max(total_words, 1), total_edits, total_words


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(marks_a: list, marks_b: list, stat, n_samples=2000, seed=11) -> tuple:
    """p for "the two listeners differ", resampling clips rather than targets.

    Two targets in one clip are not independent — a model that loses the clip
    loses both — so whole clips are drawn.
    """
    by_clip = {}
    for m in marks_a:
        by_clip.setdefault(m[0], ([], []))[0].append(m)
    for m in marks_b:
        by_clip.setdefault(m[0], ([], []))[1].append(m)
    ids = [c for c, (a, b) in by_clip.items() if a and b]
    if not ids:
        return 0.0, 1.0
    observed = stat(marks_b) - stat(marks_a)
    rng, worse = random.Random(seed), 0
    for _ in range(n_samples):
        pick = [ids[rng.randrange(len(ids))] for _ in ids]
        a = [x for c in pick for x in by_clip[c][0]]
        b = [x for c in pick for x in by_clip[c][1]]
        delta = stat(b) - stat(a)
        if (delta <= 0) if observed > 0 else (delta >= 0):
            worse += 1
    return observed, worse / n_samples


# ---------------------------------------------------------------- controls


def control_hypotheses(cases: list) -> dict:
    """The three checks, all built from the human transcripts and no model."""
    refs = [c["reference"] for c in cases]
    shuffled = refs[1:] + refs[:1]          # every clip gets its neighbour's words
    variant = []
    for c in cases:
        text = c["reference"]
        for t in c["targets"]:
            if not t["variant"]:
                continue
            text = re.sub(rf"(?i)\b{re.escape(t['surface'])}\b", t["variant"], text)
        variant.append(text)
    return {"reference control": refs,
            "shuffled control": shuffled,
            "variant control": variant}


# ---------------------------------------------------------------- transcribing


def transcribe(build: Path, cases: list, language: str, cache: dict, key: str) -> list:
    have = cache.get(key, {})
    todo = [c for c in cases if c["clip"].name not in have]
    if todo:
        if not (build / "model.bin").exists():
            raise SystemExit(f"no listener at {build}")
        # The app's own listener, and its own decode settings. This file used to
        # construct a WhisperModel of its own; two copies of a decode path are
        # two things that can drift, and this project has twice published a
        # number produced by the wrong one.
        from scripts.guard import claim
        from app.speech import release, transcribe
        claim(1.2, f"speech bench ({build.name})")
        started = time.time()
        for i, case in enumerate(todo, 1):
            have[case["clip"].name] = transcribe(str(case["clip"]),
                                                 language=language, build=build)
            if i % 20 == 0 or i == len(todo):
                cache[key] = have
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"  {build.name} {i}/{len(todo)}  "
                      f"({time.time() - started:.0f}s)", flush=True)
        cache[key] = have
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        # Hand the memory back before the next listener is loaded. Two whisper
        # builds resident at once on this 8 GB machine is how the guard's own
        # cautionary tale started.
        release(build)
    else:
        print(f"  reusing saved transcriptions for {build.name} — --fresh to redo")
    return [have[c["clip"].name] for c in cases]


# ---------------------------------------------------------------- report


def blindspots(cases: list, marks: list, hyps: list) -> None:
    """What the metric misses, counted on this run rather than asserted."""
    by_target = {}
    for case, hyp in zip(cases, hyps):
        toks = set(normalise(hyp))
        for t in case["targets"]:
            by_target[(case["clip"].name, t["term_id"])] = (t, toks)

    neither = [m for m in marks if m[4] == "neither"]
    near = 0
    for m in neither:
        t, toks = by_target[(m[0], m[1])]
        if any(char_edits(t["surface"], w) <= 2 for w in toks):
            near += 1
    print(f"  targets the model resolved to neither variety: {len(neither)}"
          f" of {len(marks)} ({100 * len(neither) / max(len(marks), 1):.1f}%)")
    print(f"    of those, {near} ({100 * near / max(len(neither), 1):.0f}%) have a word "
          f"within two characters of the spoken form in the transcript —"
          f" heard, mis-spelled, and scored as neither rather than as drift")
    both = sum(1 for m in marks if m[4] == "both")
    if both:
        print(f"  transcripts containing both varieties of one term: {both}")


def report(cases: list, results: dict, controls: dict, terms: list) -> None:
    labels = list(results)
    total = len(results[labels[0]]["marks"]) if labels else 0

    if labels:
        print(f"\n{'':<26}{'WER':>8}{'term recall':>13}{'95% interval':>18}"
              f"{'variety substitution':>23}{'95% interval':>18}")
        for label, r in results.items():
            marks = r["marks"]
            k = sum(1 for m in marks if m[4] == "bosnian")
            lo, hi = wilson(k, len(marks))
            s, a, decided = substitution(marks)
            slo, shi = wilson(a, decided)
            print(f"  {label:<24}{r['wer']:>7.1f}%{100 * k / max(len(marks), 1):>12.1f}%"
                  f"  [{100 * lo:>5.1f}, {100 * hi:>5.1f}]{100 * s:>21.1f}%"
                  f"  [{100 * slo:>5.1f}, {100 * shi:>5.1f}]")
        print(f"  {'targets':<24}{'':>8}{total:>12}{'':>18}"
              f"{substitution(results[labels[0]]['marks'])[2]:>22} decided")

        print("\nby category — term recall, then variety substitution")
        cats = sorted({m[2] for r in results.values() for m in r["marks"]})
        print(f"  {'':<24}" + "".join(f"{c:>22}" for c in cats))
        for label, r in results.items():
            cells = []
            for c in cats:
                sub = [m for m in r["marks"] if m[2] == c]
                s, _, dec = substitution(sub)
                cells.append(f"{100 * recall(sub):>12.1f}% {100 * s:>7.1f}%")
            print(f"  {label:<24}" + "".join(cells))
        base = results[labels[0]]["marks"]
        print(f"  {'targets':<24}"
              + "".join(f"{sum(1 for m in base if m[2] == c):>22}" for c in cats))

        print("\nby the variety the alternative belongs to")
        varieties = sorted({m[3] for m in base})
        print(f"  {'':<24}" + "".join(f"{v:>22}" for v in varieties))
        for label, r in results.items():
            cells = []
            for v in varieties:
                sub = [m for m in r["marks"] if m[3] == v]
                s, _, dec = substitution(sub)
                cells.append(f"{100 * recall(sub):>12.1f}% {100 * s:>7.1f}%")
            print(f"  {label:<24}" + "".join(cells))
        print(f"  {'targets':<24}"
              + "".join(f"{sum(1 for m in base if m[3] == v):>22}" for v in varieties))

        if len(labels) == 2:
            a, b = results[labels[0]]["marks"], results[labels[1]]["marks"]
            d, p = paired_bootstrap(a, b, recall)
            print(f"\n{labels[1]} - {labels[0]}: {100 * d:+.1f} points of term recall, "
                  f"paired bootstrap p = {p:.4f}"
                  + ("" if p < 0.05 else "  (does not clear 0.05)"))
            d, p = paired_bootstrap(a, b, lambda m: substitution(m)[0])
            print(f"{labels[1]} - {labels[0]}: {100 * d:+.1f} points of variety "
                  f"substitution, paired bootstrap p = {p:.4f}"
                  + ("" if p < 0.05 else "  (does not clear 0.05)"))

    print("\nchecks on the measurement — no model involved, the transcripts themselves")
    for name, marks in controls.items():
        s, a, decided = substitution(marks)
        print(f"  {name:<22}recall {100 * recall(marks):>6.1f}%   "
              f"substitution {100 * s:>6.1f}%   ({decided} of {len(marks)} decided)")
    print("  variant control is a check on the scorer, not on a model: audio of a "
          "person saying\n    'mjesto' cannot be edited to say 'mesto', so there is no "
          "audio-side counterfactual.")

    if labels:
        print("\nwhat this metric does not see")
        blindspots(cases, results[labels[0]]["marks"], results[labels[0]]["hyps"])
        base = results[labels[0]]["marks"]
        yat = sum(1 for m in base if m[2] == "yat")
        sr = sum(1 for m in base if m[3] == "sr")
        print(f"  {yat} of {len(base)} targets ({100 * yat / max(len(base), 1):.0f}%) are "
              f"yat pairs, whose alternative is Serbian only, and {sr} of {len(base)} "
              f"({100 * sr / max(len(base), 1):.0f}%)\n    contrast with Serbian at all — "
              f"drift toward Croatian has far less to land on here.")
        print("  drift into a form that is not in the term's alternative list — a "
              "different\n    inflection, a paraphrase — lands in 'neither', which "
              "lowers recall without\n    raising substitution.")
        for label, r in results.items():
            missed = collections.Counter(m[1] for m in r["marks"] if m[4] == "alt")
            if missed:
                print(f"\n  written in the other variety by {label}: "
                      + ", ".join(f"{t} ({k})" for t, k in missed.most_common(15)))


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=TEST_TSV)
    ap.add_argument("--limit", type=int, default=200,
                    help="clips to score, from the top of the TSV; 0 for all. The "
                         "default matches the prefix evaluate_speech.py scored")
    ap.add_argument("--model", action="append", type=Path,
                    help="a listener to score; repeat for more. "
                         "Default: before-training and the current one")
    ap.add_argument("--language", default="bs")
    ap.add_argument("--controls-only", action="store_true",
                    help="build the term set and run the three checks, no model")
    ap.add_argument("--fresh", action="store_true", help="ignore cached transcriptions")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(args.data)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no usable rows in {args.data}")

    graded = {r[1] for r in read_tsv(args.data)}
    trained = {r[1] for r in read_tsv(TRAIN_TSV)} if TRAIN_TSV.exists() else set()
    overlap = len(graded & trained)
    print(f"clips: {len(rows)} of {len(graded)} in {args.data}")
    print(f"transcripts shared with data/speech/train.tsv: {overlap}"
          + ("" if not overlap else "   *** the listener trained on graded audio ***"))

    corpora, dropped = evidence_sentences(graded)
    print(f"term evidence: flores {len(corpora['flores'])}, ntrex {len(corpora['ntrex'])}, "
          f"setimes {len(corpora['setimes'])} sentences "
          f"({dropped} removed for also being a graded transcript)")

    terms = load_terms()
    admitted, refused = reverify(terms, corpora)
    print(f"terms in bench/terms.tsv: {len(terms)};  still admitted on held-out "
          f"evidence: {len(admitted)};  dropped: {len(refused)}")

    cases = build_cases(rows, admitted)
    targets = sum(len(c["targets"]) for c in cases)
    carrying = sum(1 for c in cases if c["targets"])
    used = sorted({t["term_id"] for c in cases for t in c["targets"]})
    print(f"targets: {targets} across {carrying} of {len(rows)} clips, "
          f"{len(used)} distinct terms")
    if targets < 100:
        print("  small sample: the intervals below are wide, and a difference of a "
              "few points\n  between two listeners will not clear them")

    write_artifacts(admitted, refused, cases)

    controls = {name: score(cases, hyps)
                for name, hyps in control_hypotheses(cases).items()}

    results = {}
    if not args.controls_only:
        builds = args.model or [BEFORE, TUNED]
        cache = {} if args.fresh else (
            json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {})
        for build in builds:
            hyps = transcribe(build, cases, args.language, cache,
                              f"{build.name}:{args.language}")
            wer, wrong, words = word_error_rate(cases, hyps)
            results[build.name] = {"hyps": hyps, "marks": score(cases, hyps),
                                   "wer": wer, "wrong": wrong, "words": words}
            print(f"  {build.name}: WER {wer:.1f}%  ({wrong} wrong of {words} words)")

    report(cases, results, controls, admitted)
    return 0


TERM_COLUMNS = ["term_id", "category", "alt_variety", "bosnian_forms", "alt_forms",
                "n_bosnian_sources", "n_alt_in_bosnian_sources",
                "heldout_n_bosnian", "heldout_n_alt", "heldout_share"]


def write_artifacts(admitted: list, refused: list, cases: list) -> None:
    used = collections.Counter(t["term_id"] for c in cases for t in c["targets"])
    with open(OUT / "terms.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, TERM_COLUMNS + ["n_clips_in_run"], delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        for r in sorted(admitted, key=lambda r: (r["category"], -used[r["term_id"]])):
            w.writerow({**{c: r[c] for c in TERM_COLUMNS},
                        "n_clips_in_run": used[r["term_id"]]})
    with open(OUT / "terms-dropped.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, TERM_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in sorted(refused, key=lambda r: -int(r["n_bosnian_sources"])):
            w.writerow({c: r[c] for c in TERM_COLUMNS})
    with open(OUT / "cases.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["clip", "reference", "terms", "categories", "alt_varieties",
                    "spoken_forms", "variant_forms", "variant_reference"])
        for c in cases:
            if not c["targets"]:
                continue
            variant = c["reference"]
            for t in c["targets"]:
                if t["variant"]:
                    variant = re.sub(rf"(?i)\b{re.escape(t['surface'])}\b",
                                     t["variant"], variant)
            w.writerow([c["clip"].name, c["reference"],
                        "|".join(t["term_id"] for t in c["targets"]),
                        "|".join(t["category"] for t in c["targets"]),
                        "|".join(t["alt_variety"] for t in c["targets"]),
                        "|".join(t["surface"] for t in c["targets"]),
                        "|".join(t["variant"] for t in c["targets"]), variant])


if __name__ == "__main__":
    raise SystemExit(main())
