#!/usr/bin/env python3
"""SpeechBench — is Lilly's listener hearing *Bosnian*, or generic Serbo-Croatian?

training/evaluate_speech.py answers one question, word error rate, and it cannot
answer this one. The listener has been retrained with Croatian audio, because
Bosnian speech data is scarce. The predictable result is a WER that falls while
the thing we are selling gets worse — a model that hears *sedmicu* and writes
*tjedan*, hears *mjesto* and writes *mesto*. Each of those is one substitution in
a twenty-word sentence: half a point of WER, invisible in the average, and the
whole product.

So this scores the decision instead of the sentence. Unlike the translator, where
nothing Bosnian survives into English, here the output is Bosnian text — the
variety the model chose is written down in front of you. On each target the model
did one of these things:

    bosnian   it wrote a Bosnian form of the term        (sedmica, mjesto, duhan)
    croatian  it wrote the Croatian form                 (tjedan, sustav, tisuca)
    serbian   it wrote the Serbian form                  (mesto, duvan, istorija)
    neither   it wrote something else entirely, or nothing

and the numbers that follow are

    term recall             = bosnian / all targets
    variety substitution    = alt / (bosnian + alt)
    CROATIAN substitution   = croatian / (bosnian + croatian), over the targets
                              that have a Croatian contrast at all
    SERBIAN substitution    = serbian / (bosnian + serbian), likewise

The last two are the reason this file was rewritten, and the story is worth
keeping. The first version reported one substitution column. It scored the run
that added 3,430 clips of *Croatian* and passed it — on 85 targets of which 73
were yat pairs whose alternative form is Serbian. Croatian is ijekavian like
Bosnian, so those 73 targets cannot see Croatian drift at all: the instrument
had 12 targets for the failure it was pointed at, and one number that mixed them
in with the rest. A rate that lumps two drifts together cannot say which one
moved, which is the only question this lane has.

## Where the terms come from

Nothing here is a word list anybody typed. data/scripts/build_speech_terms.py
mines them from 135,075 sentence-aligned Bosnian/Croatian/Serbian triples —
FLORES-200, NTREX-128 and SETIMES v2 — and admits a pair only on measured rate,
measured exclusivity in both varieties, and measured alignment. Its docstring
carries the gates and the reasons. This file reads bench/speech/contrast-terms.tsv
and does not second-guess it.

Two things this file does on top:

  * inflections of one word are collapsed into a family, so a sentence yields one
    decision per word rather than three. Two forms of one lemma in one sentence
    are not two independent chances to drift, and counting them as two narrows
    every interval below.
  * the alternative forms of a family are pooled across it, so a listener that
    drifts to *duljini* when the mined pair happened to record *duljine* is still
    scored as drift instead of falling into "neither".

`--terms legacy` scores against bench/terms.tsv instead, through the same code,
which is how the 65.9% -> 68.2% figures already published can still be
reproduced. Legacy admission is recomputed with every graded transcript removed
from the evidence, because FLEURS transcripts are FLORES sentences and FLORES is
one of the corpora that admitted those terms.

## Which clips

The FLEURS Bosnian test split is 925 clips of only **349 distinct sentences** —
245 sentences read by three speakers, 86 by two, 18 by one. That matters twice.
It is why `--clips distinct` is the default: one clip per sentence reaches every
target the split contains for a third of the transcription. And it is why the
bootstrap below resamples *sentences* rather than clips. Two recordings of one
sentence carry the same words and the same difficulty; drawing them as if they
were independent understates the variance and prints a p-value smaller than the
evidence supports.

`--clips first200` is the exact prefix behind the published 35.5% -> 34.9%, kept
so that comparison stays reproducible. Its 200 clips are 167 distinct sentences.

## The checks on the measurement

Six, run on every invocation and needing no model at all:

  reference control  the human transcript through the scorer. 100% recall and
                     0% substitution, or the matcher is broken. Its second job
                     is to catch a term whose "alternative" form is a real word
                     that turns up in Bosnian references for other reasons —
                     that shows up here as substitution above zero.
  shuffled control   the *neighbouring* sentence through the scorer. There is no
                     input text to copy, as there is in the translator's bench,
                     so instead the scorer is handed fluent professional Bosnian
                     that is not what was said. Near 0%, or the metric is
                     rewarding plausible Bosnian rather than hearing.
  croatian variant   the transcript with every Croatian-contrasting target
                     swapped for its Croatian counterpart. Croatian substitution
                     must go to ~100% and Serbian substitution must NOT move, or
                     the two columns are not separating what they claim to.
  serbian variant    the mirror of it, and the same requirement in reverse.
  croatian rendering the professional CROATIAN translation of the same FLORES
                     sentence, scored as if a listener had written it. Not a
                     word swapped in by a regular expression — Croatian written
                     by a Croatian translator who never heard of this bench.
                     Every one of the 349 graded sentences has one, and all 349
                     were held out of the term mining, so no pair can have been
                     admitted on the sentence that then checks it.
  serbian rendering  the same in Serbian, transliterated from Cyrillic.

The variant controls are weaker here than in the translator's bench, and the
report says so: there they perturb the model's *input* and measure the model;
here the input is a recording of a person, which cannot be edited to say *mesto*
instead of *mjesto*. So they test the scorer, not the model. What they establish
is that the Croatian column answers to Croatian and the Serbian column to
Serbian — which the single-column version could not even be asked.

The rendering controls are the strong form of that, and they are the evidence
that the mined pairs are real rather than plausible. Measured, on 349 sentences
no part of the mining ever saw:

    croatian rendering   croatian substitution 84.1%   serbian 0.0%
    serbian rendering    croatian  0.0%               serbian 98.2%

Recall falls a long way on both, to 41.9% and 21.8%, because an independent
translation rewords far more than one term — but of what it decides, it lands in
its own column and never in the other's.

    python3 training/speech_bench.py                    # both listeners, 349 sentences
    python3 training/speech_bench.py --controls-only    # no model, seconds
    python3 training/speech_bench.py --clips first200   # the published prefix
    python3 training/speech_bench.py --model models/lilly/listen

Transcriptions are cached in bench/speech/.outputs.json and written every 20
clips, so an interrupted run resumes rather than restarting.
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
MINED = OUT / "contrast-terms.tsv"
LEGACY_IN = REPO / "bench" / "terms.tsv"
TEST_TSV = REPO / "data" / "speech" / "test.tsv"
TRAIN_TSV = REPO / "data" / "speech" / "train.tsv"
CACHE = OUT / ".outputs.json"
TUNED = REPO / "models" / "lilly" / "listen"
BEFORE = REPO / "models" / "lilly" / "listen.before-training"

WORD = re.compile(r"[a-zA-ZčćžšđČĆŽŠĐ]+", re.UNICODE)

# bench/build.py's admission thresholds, reused unchanged for --terms legacy.
# Loosening them here to keep more terms would be tuning the exam to the answers.
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


def evidence_sentences(exclude: set) -> tuple:
    """The professional Bosnian bench/build.py counts, minus the graded text.

    Only --terms legacy needs this; the mined term set does its own holding out,
    over its own corpora, inside data/scripts/build_speech_terms.py.
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


def yat_pair(bos: str, alt: str) -> bool:
    """Is this a yat reflex — mjesto/mesto, vrijeme/vreme, dvije/dve?

    Used only to label a term in the report, never to admit or refuse one.
    """
    for long, short in (("ije", "e"), ("je", "e")):
        if long in bos and bos.replace(long, short, 1) == alt:
            return True
    return False


def load_mined_terms(path: Path) -> list:
    """One term per family, with the Croatian and Serbian forms pooled over it."""
    if not path.exists():
        raise SystemExit(f"{path} is missing — "
                         "run data/scripts/build_speech_terms.py")
    fams = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            fam = fams.setdefault(r["family"], {"term_id": r["family"],
                                                "bos": set(), "hr": set(),
                                                "sr": set(), "yat": False})
            fam["bos"].add(r["bosnian_form"])
            if r["hr_form"]:
                fam["hr"].add(r["hr_form"])
            if r["sr_form"]:
                fam["sr"].add(r["sr_form"])
                fam["yat"] = fam["yat"] or yat_pair(r["bosnian_form"], r["sr_form"])
    terms = []
    for fam in fams.values():
        # A form that is spelled the same in Bosnian is not an alternative: it
        # would make every correct transcription look like drift.
        fam["hr"] -= fam["bos"]
        fam["sr"] -= fam["bos"]
        if not (fam["hr"] or fam["sr"]):
            continue
        terms.append({"term_id": fam["term_id"],
                      "category": "yat" if fam["yat"] else "lex",
                      "bos": sorted(fam["bos"]), "hr": sorted(fam["hr"]),
                      "sr": sorted(fam["sr"])})
    return terms


def load_legacy_terms(exclude: set) -> tuple:
    """bench/terms.tsv, re-admitted on evidence with the graded text removed."""
    if not LEGACY_IN.exists():
        raise SystemExit(f"{LEGACY_IN} is missing — bench/build.py writes it")
    with open(LEGACY_IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["bos"] = [w for w in r["bosnian_forms"].split("|") if w]
        r["alt"] = [w for w in r["alt_forms"].split("|") if w]

    corpora, dropped = evidence_sentences(exclude)
    print(f"term evidence: flores {len(corpora['flores'])}, ntrex "
          f"{len(corpora['ntrex'])}, setimes {len(corpora['setimes'])} sentences "
          f"({dropped} removed for also being a graded transcript)")
    wanted = {w for r in rows for w in r["bos"] + r["alt"]}
    freq = count_forms(corpora, wanted)

    terms, refused = [], []
    for r in rows:
        nb = sum(freq[w] for w in r["bos"])
        na = sum(freq[w] for w in r["alt"])
        share = nb / (nb + na) if nb + na else 0.0
        nm = int(r["n_alt_in_mixed_bcs"])   # mixed BCS holds no test transcripts
        if not (nb >= MIN_BOS and share >= MIN_SHARE and nm >= 1):
            refused.append(r)
            continue
        # The legacy file records one alt_variety string per term. Splitting it
        # is what lets legacy terms be scored by the same variety-aware code —
        # and it is also the measurement that made the case for this rewrite:
        # 223 of its 241 terms are "sr" alone.
        variety = r["alt_variety"]
        terms.append({"term_id": r["term_id"], "category": r["category"],
                      "bos": r["bos"],
                      "hr": r["alt"] if "hr" in variety else [],
                      "sr": r["alt"] if "sr" in variety else []})
    print(f"terms in {LEGACY_IN.name}: {len(rows)};  still admitted on held-out "
          f"evidence: {len(terms)};  dropped: {len(refused)}")
    return terms, refused


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

    cases = []
    for clip, reference in rows:
        toks = [w.lower() for w in WORD.findall(reference)]
        present = set(toks)
        targets, seen = [], set()
        for tok in toks:
            r = by_form.get(tok)
            if r is None or r["term_id"] in seen:
                continue
            hr = [a for a in r["hr"] if a not in present]
            sr = [a for a in r["sr"] if a not in present]
            if len(hr) != len(r["hr"]) or len(sr) != len(r["sr"]):
                continue                      # both varieties in the reference
            if not (hr or sr):
                continue
            seen.add(r["term_id"])
            targets.append({"term_id": r["term_id"], "category": r["category"],
                            "surface": tok, "bos": r["bos"], "hr": hr, "sr": sr,
                            "variant_hr": nearest(tok, hr),
                            "variant_sr": nearest(tok, sr)})
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

    Only the variant controls use this. For a yat pair or an h-pair it recovers
    the exact counterpart (mjesto->mesto, duhana->duvana); where the two words
    share no stem it returns some form of the right word, which is all a check
    on the matcher needs.
    """
    return min(forms, key=lambda f: (char_edits(surface, f), f)) if forms else ""


# ---------------------------------------------------------------- scoring


def outcome(hyp_tokens: set, target: dict) -> str:
    """Which variety the listener committed to on this target, if any."""
    bos = any(f in hyp_tokens for f in target["bos"])
    hr = [f for f in target["hr"] if f in hyp_tokens]
    sr = [f for f in target["sr"] if f in hyp_tokens]
    if bos and not (hr or sr):
        return "bosnian"
    if bos:
        # Both a Bosnian form and an alternative in one transcript. Rare, and
        # not a decision, so it is not counted as either.
        return "both"
    if hr and sr:
        # The same word is Croatian and Serbian here — historija/povijest and
        # historija/istorija share nothing, but marama answers both. It is drift,
        # and it is not attributable, so it is counted as drift and left out of
        # the two per-variety columns.
        return "alt" if set(hr) & set(sr) else "both-alt"
    if hr:
        return "croatian"
    if sr:
        return "serbian"
    return "neither"


ALT = ("croatian", "serbian", "alt", "both-alt")


def score(cases: list, hyps: list) -> list:
    """One mark per target.

    The first field is the *sentence*, not the clip. Everything downstream that
    resamples uses it, because the test split reads most sentences three times
    and two readings of one sentence are one piece of evidence about vocabulary.
    """
    marks = []
    for case, hyp in zip(cases, hyps):
        toks = set(normalise(hyp))
        for t in case["targets"]:
            marks.append({"sentence": case["reference"], "clip": case["clip"].name,
                          "term": t["term_id"], "category": t["category"],
                          "has_hr": bool(t["hr"]), "has_sr": bool(t["sr"]),
                          "outcome": outcome(toks, t)})
    return marks


def recall(marks: list) -> float:
    n = len(marks)
    return sum(1 for m in marks if m["outcome"] == "bosnian") / n if n else 0.0


def substitution(marks: list) -> tuple:
    """alt / (bosnian + alt) — the varieties the model actually committed to."""
    b = sum(1 for m in marks if m["outcome"] == "bosnian")
    a = sum(1 for m in marks if m["outcome"] in ALT)
    return (a / (a + b) if a + b else 0.0), a, a + b


def variety_substitution(marks: list, variety: str) -> tuple:
    """Drift into one named variety, over the targets that can show it.

    A target whose Croatian form is the same word as its Bosnian one cannot
    record Croatian drift, and including it in the denominator would dilute the
    rate with cases where drift was impossible. That dilution is exactly what
    made the single-column version unable to see Croatian.
    """
    field, name = ("has_hr", "croatian") if variety == "hr" else ("has_sr", "serbian")
    sub = [m for m in marks if m[field]]
    b = sum(1 for m in sub if m["outcome"] == "bosnian")
    a = sum(1 for m in sub if m["outcome"] == name)
    return (a / (a + b) if a + b else 0.0), a, a + b, len(sub)


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
    """p for "the two listeners differ", resampling SENTENCES.

    Not clips and not targets. Two targets in one sentence are not independent —
    a model that loses the sentence loses both — and neither are two recordings
    of the same sentence by different speakers, which is what the FLEURS test
    split mostly consists of. Resampling the smallest unit that is actually
    independent is what keeps this p honest; the earlier version drew clips, and
    on a split of 925 clips over 349 sentences that counts each sentence up to
    three times and prints a p smaller than the evidence supports.
    """
    by_sentence = {}
    for m in marks_a:
        by_sentence.setdefault(m["sentence"], ([], []))[0].append(m)
    for m in marks_b:
        by_sentence.setdefault(m["sentence"], ([], []))[1].append(m)
    ids = [s for s, (a, b) in by_sentence.items() if a and b]
    if not ids:
        return 0.0, 1.0
    observed = stat(marks_b) - stat(marks_a)
    rng, worse = random.Random(seed), 0
    for _ in range(n_samples):
        pick = [ids[rng.randrange(len(ids))] for _ in ids]
        a = [x for s in pick for x in by_sentence[s][0]]
        b = [x for s in pick for x in by_sentence[s][1]]
        delta = stat(b) - stat(a)
        if (delta <= 0) if observed > 0 else (delta >= 0):
            worse += 1
    return observed, worse / n_samples


# ---------------------------------------------------------------- controls


FLORES_TEXT = REPO / "data" / "speech-extra" / "text"


def real_renderings() -> dict:
    """Bosnian FLORES sentence -> the professional Croatian and Serbian ones.

    Every one of the 349 graded transcripts is a FLORES devtest sentence, exactly,
    and FLORES ships the same sentence translated into Croatian and into Serbian
    by professionals who were not thinking about this bench. Those are the best
    control this measurement can have: not a word swapped by a regular expression
    into a Bosnian sentence, but Croatian written by a Croatian translator.

    They are also *held out* of the term mining — build_speech_terms.py drops all
    349 triples before it counts anything — so a term cannot have been admitted
    on the sentence it is then checked against here.
    """
    out = {}
    for split in ("dev", "devtest"):
        try:
            bs = (FLORES_TEXT / f"flores.{split}.bs").read_text(encoding="utf-8")
            hr = (FLORES_TEXT / f"flores.{split}.hr").read_text(encoding="utf-8")
            sr = (FLORES_TEXT / f"flores.{split}.sr-cyrl").read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        from data.scripts.build_speech_terms import to_latin
        for b, h, s in zip(bs.splitlines(), hr.splitlines(), sr.splitlines()):
            if b.strip():
                out[b.strip()] = (h.strip(), to_latin(s.strip()))
    return out


def control_hypotheses(cases: list) -> tuple:
    """The checks, all built from human text and no model."""
    refs = [c["reference"] for c in cases]
    shuffled = refs[1:] + refs[:1]          # every clip gets its neighbour's words

    def swap(field):
        out = []
        for c in cases:
            text = c["reference"]
            for t in c["targets"]:
                if t[field]:
                    text = re.sub(rf"(?i)\b{re.escape(t['surface'])}\b",
                                  t[field], text)
            out.append(text)
        return out

    controls = {"reference control": refs,
                "shuffled control": shuffled,
                "croatian variant": swap("variant_hr"),
                "serbian variant": swap("variant_sr")}

    real, matched = real_renderings(), 0
    if real:
        hr_real, sr_real = [], []
        for c in cases:
            pair = real.get(c["reference"])
            matched += 1 if pair else 0
            hr_real.append(pair[0] if pair else "")
            sr_real.append(pair[1] if pair else "")
        if matched:
            controls["croatian rendering"] = hr_real
            controls["serbian rendering"] = sr_real
    return controls, matched


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
        from app.speech import release, transcribe as listen
        claim(1.2, f"speech bench ({build.name})")
        started = time.time()
        for i, case in enumerate(todo, 1):
            have[case["clip"].name] = listen(str(case["clip"]), language=language,
                                             build=build)
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

    neither = [m for m in marks if m["outcome"] == "neither"]
    near = 0
    for m in neither:
        t, toks = by_target[(m["clip"], m["term"])]
        if any(char_edits(t["surface"], w) <= 2 for w in toks):
            near += 1
    print(f"  targets the model resolved to neither variety: {len(neither)}"
          f" of {len(marks)} ({100 * len(neither) / max(len(marks), 1):.1f}%)")
    print(f"    of those, {near} ({100 * near / max(len(neither), 1):.0f}%) have a word "
          f"within two characters of the spoken form in the transcript —"
          f" heard, mis-spelled, and scored as neither rather than as drift")
    for label in ("both", "both-alt", "alt"):
        n = sum(1 for m in marks if m["outcome"] == label)
        if n:
            print(f"  targets scored '{label}': {n}")


def coverage(marks: list) -> None:
    """How much of this instrument can see each drift. The headline diagnostic."""
    n = len(marks)
    hr = sum(1 for m in marks if m["has_hr"])
    sr = sum(1 for m in marks if m["has_sr"])
    both = sum(1 for m in marks if m["has_hr"] and m["has_sr"])
    yat = sum(1 for m in marks if m["category"] == "yat")
    print(f"  targets: {n};  can show Croatian drift: {hr} ({100 * hr / max(n, 1):.0f}%);"
          f"  Serbian drift: {sr} ({100 * sr / max(n, 1):.0f}%);  both: {both}")
    print(f"  yat pairs, which are Serbian-only by construction: {yat} "
          f"({100 * yat / max(n, 1):.0f}%)")
    if hr < 30:
        print("  *** fewer than 30 Croatian-contrasting targets: this run cannot "
              "judge a Croatian-trained model ***")


def report(cases: list, results: dict, controls: dict) -> None:
    labels = list(results)

    if labels:
        base = results[labels[0]]["marks"]
        print(f"\n{'':<26}{'WER':>8}{'term recall':>13}{'95% interval':>18}"
              f"{'substitution':>15}{'CROATIAN':>11}{'SERBIAN':>10}")
        for label, r in results.items():
            marks = r["marks"]
            k = sum(1 for m in marks if m["outcome"] == "bosnian")
            lo, hi = wilson(k, len(marks))
            s, _, _ = substitution(marks)
            shr, _, _, _ = variety_substitution(marks, "hr")
            ssr, _, _, _ = variety_substitution(marks, "sr")
            print(f"  {label:<24}{r['wer']:>7.1f}%{100 * k / max(len(marks), 1):>12.1f}%"
                  f"  [{100 * lo:>5.1f}, {100 * hi:>5.1f}]{100 * s:>14.1f}%"
                  f"{100 * shr:>10.1f}%{100 * ssr:>9.1f}%")
        _, _, dec_hr, n_hr = variety_substitution(base, "hr")
        _, _, dec_sr, n_sr = variety_substitution(base, "sr")
        print(f"  {'targets':<24}{'':>8}{len(base):>12}{'':>18}"
              f"{substitution(base)[2]:>13} dec{dec_hr:>10}{dec_sr:>9}")
        print(f"  {'':<24}{'':>8}{'':>12}{'':>18}{'':>17}"
              f"{n_hr:>8} tgt{n_sr:>6} tgt")

        print("\nby category — term recall, then Croatian and Serbian substitution")
        cats = sorted({m["category"] for m in base})
        print(f"  {'':<24}" + "".join(f"{c:>26}" for c in cats))
        for label, r in results.items():
            cells = []
            for c in cats:
                sub = [m for m in r["marks"] if m["category"] == c]
                cells.append(f"{100 * recall(sub):>12.1f}%"
                             f"{100 * variety_substitution(sub, 'hr')[0]:>6.1f}%"
                             f"{100 * variety_substitution(sub, 'sr')[0]:>6.1f}%")
            print(f"  {label:<24}" + "".join(cells))
        print(f"  {'targets':<24}"
              + "".join(f"{sum(1 for m in base if m['category'] == c):>26}"
                        for c in cats))

        if len(labels) == 2:
            a, b = results[labels[0]]["marks"], results[labels[1]]["marks"]
            print()
            for name, stat in (
                    ("term recall", recall),
                    ("variety substitution", lambda m: substitution(m)[0]),
                    ("CROATIAN substitution", lambda m: variety_substitution(m, "hr")[0]),
                    ("SERBIAN substitution", lambda m: variety_substitution(m, "sr")[0])):
                d, p = paired_bootstrap(a, b, stat)
                print(f"{labels[1]} - {labels[0]}: {100 * d:+.1f} points of {name}, "
                      f"paired bootstrap over sentences p = {p:.4f}"
                      + ("" if p < 0.05 else "  (does not clear 0.05)"))

    print("\nchecks on the measurement — no model involved, the transcripts themselves")
    for name, marks in controls.items():
        s, _, decided = substitution(marks)
        shr = variety_substitution(marks, "hr")[0]
        ssr = variety_substitution(marks, "sr")[0]
        print(f"  {name:<20}recall {100 * recall(marks):>6.1f}%   "
              f"substitution {100 * s:>6.1f}%   croatian {100 * shr:>6.1f}%   "
              f"serbian {100 * ssr:>6.1f}%   ({decided} decided)")
    print("  the variant controls check the scorer, not a model: audio of a person "
          "saying\n    'mjesto' cannot be edited to say 'mesto', so there is no "
          "audio-side counterfactual.\n    What they establish is that the Croatian "
          "column answers to Croatian and the\n    Serbian column to Serbian.")
    print("  the two RENDERING rows are the strong version of that check: the "
          "professional\n    Croatian and Serbian translations of the same FLORES "
          "sentences, held out of\n    the term mining, scored as if a listener had "
          "produced them. Recall falls a\n    long way because an independent "
          "translation rewords much more than a term,\n    but of the targets it "
          "does decide, nearly all should land in its own column.")

    if labels:
        print("\nwhat this instrument can and cannot see")
        coverage(results[labels[0]]["marks"])
        blindspots(cases, results[labels[0]]["marks"], results[labels[0]]["hyps"])
        print("  drift into a form that is not in the term's alternative list — a "
              "different\n    inflection, a paraphrase — lands in 'neither', which "
              "lowers recall without\n    raising substitution.")
        for label, r in results.items():
            for variety, name in (("croatian", "Croatian"), ("serbian", "Serbian")):
                missed = collections.Counter(m["term"] for m in r["marks"]
                                             if m["outcome"] == variety)
                if missed:
                    print(f"\n  written in {name} by {label}: "
                          + ", ".join(f"{t} ({k})" for t, k in missed.most_common(15)))


# ---------------------------------------------------------------- main


def choose_clips(rows: list, mode: str, limit: int) -> list:
    """Which recordings to score, and why the default is one per sentence."""
    if mode == "first200":
        return rows[:200]
    if mode == "all":
        picked = rows
    else:
        seen, picked = set(), []
        for clip, text in rows:
            if text not in seen:
                seen.add(text)
                picked.append((clip, text))
    return picked[:limit] if limit else picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=TEST_TSV)
    ap.add_argument("--clips", choices=("distinct", "first200", "all"),
                    default="distinct",
                    help="distinct: one clip per distinct transcript (349). "
                         "first200: the prefix behind the published numbers. "
                         "all: every clip, most sentences three times over")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the chosen clips; 0 for all of them")
    ap.add_argument("--terms", default="mined", choices=("mined", "legacy"))
    ap.add_argument("--model", action="append", type=Path,
                    help="a listener to score; repeat for more. "
                         "Default: before-training and the current one")
    ap.add_argument("--language", default="bs")
    ap.add_argument("--controls-only", action="store_true",
                    help="build the term set and run the checks, no model")
    ap.add_argument("--fresh", action="store_true", help="ignore cached transcriptions")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = read_tsv(args.data)
    rows = choose_clips(all_rows, args.clips, args.limit)
    if not rows:
        raise SystemExit(f"no usable rows in {args.data}")

    graded = {r[1] for r in all_rows}
    trained = {r[1] for r in read_tsv(TRAIN_TSV)} if TRAIN_TSV.exists() else set()
    overlap = len(graded & trained)
    print(f"clips: {len(rows)} ({args.clips}) of {len(all_rows)} in {args.data}, "
          f"{len(graded)} distinct transcripts")
    print(f"transcripts shared with data/speech/train.tsv: {overlap}"
          + ("" if not overlap else "   *** the listener trained on graded audio ***"))

    if args.terms == "legacy":
        terms, refused = load_legacy_terms(graded)
    else:
        terms, refused = load_mined_terms(MINED), []
        print(f"terms: {len(terms)} families from {MINED.name} "
              f"({sum(1 for t in terms if t['hr'])} with a Croatian contrast, "
              f"{sum(1 for t in terms if t['sr'])} with a Serbian one)")

    cases = build_cases(rows, terms)
    targets = sum(len(c["targets"]) for c in cases)
    carrying = sum(1 for c in cases if c["targets"])
    used = sorted({t["term_id"] for c in cases for t in c["targets"]})
    print(f"targets: {targets} across {carrying} of {len(rows)} clips, "
          f"{len(used)} distinct terms")
    if targets < 100:
        print("  small sample: the intervals below are wide, and a difference of a "
              "few points\n  between two listeners will not clear them")

    write_artifacts(terms, refused, cases)

    hypotheses, matched = control_hypotheses(cases)
    controls = {name: score(cases, hyps) for name, hyps in hypotheses.items()}
    if matched:
        print(f"professional Croatian/Serbian renderings found for {matched} of "
              f"{len(cases)} graded sentences")

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
    else:
        print("\nterm-set coverage of the chosen clips")
        coverage(score(cases, [c["reference"] for c in cases]))

    report(cases, results, controls)
    return 0


def write_artifacts(terms: list, refused: list, cases: list) -> None:
    used = collections.Counter(t["term_id"] for c in cases for t in c["targets"])
    with open(OUT / "terms.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["term_id", "category", "bosnian_forms", "hr_forms", "sr_forms",
                    "n_clips_in_run"])
        for r in sorted(terms, key=lambda r: (r["category"], -used[r["term_id"]])):
            w.writerow([r["term_id"], r["category"], "|".join(r["bos"]),
                        "|".join(r["hr"]), "|".join(r["sr"]), used[r["term_id"]]])
    if refused:
        with open(OUT / "terms-dropped.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["term_id", "category", "bosnian_forms", "alt_forms"])
            for r in sorted(refused, key=lambda r: -int(r["n_bosnian_sources"])):
                w.writerow([r["term_id"], r["category"], r["bosnian_forms"],
                            r["alt_forms"]])
    with open(OUT / "cases.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["clip", "reference", "terms", "categories", "spoken_forms",
                    "hr_variants", "sr_variants", "croatian_reference",
                    "serbian_reference"])
        for c in cases:
            if not c["targets"]:
                continue
            variants = {}
            for field in ("variant_hr", "variant_sr"):
                text = c["reference"]
                for t in c["targets"]:
                    if t[field]:
                        text = re.sub(rf"(?i)\b{re.escape(t['surface'])}\b",
                                      t[field], text)
                variants[field] = text
            w.writerow([c["clip"].name, c["reference"],
                        "|".join(t["term_id"] for t in c["targets"]),
                        "|".join(t["category"] for t in c["targets"]),
                        "|".join(t["surface"] for t in c["targets"]),
                        "|".join(t["variant_hr"] for t in c["targets"]),
                        "|".join(t["variant_sr"] for t in c["targets"]),
                        variants["variant_hr"], variants["variant_sr"]])


if __name__ == "__main__":
    raise SystemExit(main())
