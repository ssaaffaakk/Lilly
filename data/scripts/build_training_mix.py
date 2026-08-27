#!/usr/bin/env python3
"""Build data/clean/train-mix.tsv: the training mix that stops the fine-tune
from shortening its output.

data/clean/valid.tsv, data/clean/test.tsv and data/flores/ are held out and are
never opened here. The only inputs are data/clean/train.tsv and
data/extra/extra-train.tsv. Every FLORES figure quoted below is a constant
copied from the measurement that produced this recipe, not something this script
reads.

WHY THIS EXISTS. Fine-tuning raises BLEU and drops chrF2. The whole of the chrF2
loss is the model shortening its output: on FLORES the 1,694 lines Lilly does not
shorten gain +0.08 chrF2, the 315 lines it shortens by more than 8% lose -2.00,
and the weighted sum -0.24 reproduces the measured -0.22. The cause is not the
training data's length *mean* — that already matches FLORES to within 0.3% — it
is the *variance*: WikiMatrix sd(log char ratio) is 0.213 against FLORES's 0.129.
A model trained on pairs whose length ratio swings that widely learns that
dropping a clause is an acceptable translation.

So this file does not filter for "quality" in general. Every step below either
removes length-ratio noise or removes a reason the model would learn to be terse.

THE EIGHT STEPS, in this order, with what each one actually removes:

  1. strip [1]-style wiki citation markers   9,560 rows  (pure ratio noise)
  2. split to one sentence per example       4,809 rows dropped, 359,309 pairs out
  3. asymmetric band on log(en/bs chars)    25,309 pairs  (the variance cut)
  4. drop micro-pairs under 3 characters        46 pairs
  5. deduplicate                             9,664 pairs
  6. hard 128-SentencePiece-token cap          753 pairs  (MAX_LEN=128 lossless)
  7. hold back 462 ntrex pairs               -> data/clean/ntrex-holdout.tsv
  8. integer corpus weights, one shuffle    361,621 training examples

Every count is asserted. A silent upstream change — a re-download that returns
fewer rows, a corpus tag that got renamed — must kill this run, not quietly train
on less data than the recipe was measured on. The three numbers at the end are
asserted with a tolerance instead, because they are the property being
engineered rather than a bookkeeping total.

WHERE THIS DIFFERS FROM THE RECIPE. Steps 1, 3, 4, 5, 6, 7 and 8 land within 1%
of the recipe's counts and several land on them exactly. Step 2 does not: the
recipe drops 6,942 rows there and this splitter drops 4,809. That gap is the
splitter, not the data — the recipe's own control reproduces here to two decimal
places (app/translate.py's naive splitter disagrees on 12.94% of SETIMES and
16.87% of WikiMatrix, against the recipe's 12.9% and 16.9%), so the inputs are
the same rows and this exception list is simply wider. It keeps 2,133 more rows.
Everything downstream is 0.3-0.5% larger as a result, and the blend still lands
on the recipe's target: sd(log char ratio) 0.1302 against a target of 0.130.
The full per-step comparison is in the comment above each check() below.

Usage:
    python3 build_training_mix.py              # write train-mix.tsv + ntrex-holdout.tsv
    python3 build_training_mix.py --dry-run    # count and measure only, write nothing
    python3 build_training_mix.py --review 30  # print pairs to read by hand
    python3 build_training_mix.py --review 30 --seed 99      # a different sample
"""
import argparse
import collections
import math
import os
import random
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = DATA_DIR / "clean"
TRAIN = CLEAN_DIR / "train.tsv"
EXTRA = DATA_DIR / "extra" / "extra-train.tsv"
OUT_MIX = CLEAN_DIR / "train-mix.tsv"
OUT_HOLDOUT = CLEAN_DIR / "ntrex-holdout.tsv"

# The tokenizer in step 6 has to be the one the model actually trains with,
# otherwise the cap is a guess. Same resolution order as training/train_translation.py.
BASE_MODEL = os.environ.get("LILLY_BASE") or str(DATA_DIR.parent / "models" / "lilly" / "translate")
MAX_TOKENS = 128            # training/train_translation.py: MAX_LEN = 128

SEED = 41                   # same seed the trainer shuffles with

# ---------------------------------------------------------------- step 1
CITATION = re.compile(r"\[\d+\]")
WHITESPACE = re.compile(r"\s+")

# ---------------------------------------------------------------- step 2
# A period is a sentence boundary unless one of these says otherwise. The list is
# the reason this splitter exists at all: app/translate.py breaks on
# `(?<=[.!?])\s+` with no exceptions, and that naive rule disagrees with the
# other side of the pair on 12.9% of SETIMES and 16.9% of WikiMatrix rows —
# roughly two thirds of which is Bosnian ordinal notation ("31. maja 1992."),
# not real misalignment. Protecting ordinals turns those false breaks back into
# usable rows.
ABBREV = set("""
dr mr mrs ms prof g gđa gdja tzv npr itd tj br sv st no vs etc jr sr inc ltd co corp
sen gen col lt maj capt rev ave blvd vol pp fig al approx dept univ assoc ed eds
god str op cit ing br o.š mn hrsg vidi usp
""".split())
PUNCT_RUN = re.compile(r"[.!?]+")
CLOSERS = "\"'»”’)]}"          # a boundary may sit behind a closing quote or bracket
OPENERS = "\"'«„“‘([{-–—"      # ...and the next sentence may open with one
TRIM = "()[]{}\"'«»„“”‘’.,;:!?-–—"
LETTER = r"[^\W\d_]"
INITIALS = re.compile(rf"(?:{LETTER}\.)+{LETTER}", re.UNICODE)   # "U.S", "e.g"
ROMAN = re.compile(r"[IVXLCDM]+")   # "XIX. stoljeće" is an ordinal too

# ---------------------------------------------------------------- step 3
# The band is one-sided on purpose, and the direction is the direction of the
# measured error. Floor = FLORES's ~2nd percentile of log(en/bs chars),
# ceiling = its ~99.8th. Relative to the test distribution the cut removes ~2%
# of FLORES-shaped pairs on the short-English side and ~0.2% on the long side,
# because short-English is what teaches clause-dropping.
# Tail mass below log r = -0.30: FLORES 1.14%, SETIMES 1.44%, WikiMatrix 6.83%.
# Above +0.35:                   FLORES 0.40%, SETIMES 0.36%, WikiMatrix 5.55%,
#                                wikimedia 5.68%.
# Both of WikiMatrix's tails run 6-14x the test distribution. The left one
# teaches dropping clauses, the right one teaches inventing them.
BAND_LO = -0.28
BAND_HI = +0.40

MIN_CHARS = 3               # step 4

# ---------------------------------------------------------------- step 5
# Letters only, digits dropped — the same normalisation filter_train_data.py
# uses. Keeping digits in the key looks more correct and is not: it leaves 4,899
# pairs in that differ from a pair already kept in nothing but a number, and read
# by hand those are almost entirely template lines, not sentences. Twelve drawn
# at random came back eleven boilerplate — SETIMES bylines ("Igor Jovanović za
# Southeast European Times iz Beograda -- 08/06/07", the same line at 4,000
# different dates), "Arhivirano s originala, 16 April 2021.", bare ISBN lines —
# against one genuinely distinct sentence. Thousands of copies of one byline
# teach the model a byline.
TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

# ---------------------------------------------------------------- step 7
# valid.tsv is drawn from the same WikiMatrix+SETIMES mixture as training, so
# eval_loss on it rewards exactly the corpus-style fitting that costs chrF2 on
# FLORES. NTREX is professionally translated, unseen by both models, and has zero
# FLORES overlap; it is also a held-out *domain* (news, where FLORES is Wikipedia
# prose), so it tells us whether a FLORES gain is real or a FLORES artifact.
HOLDOUT_N = 462             # 24% of the 1,924 NTREX-128 rows we hold

# ---------------------------------------------------------------- step 8
# WikiMatrix stays at full weight because it is the best domain match measured
#   (unigram cross-entropy on FLORES English 10.976, OOV 4.83%, at an equal 350k
#   token budget; SETIMES 11.140 / 8.84%, wikimedia 11.064 / 5.22%). What is
#   wrong with it is per-pair alignment, which step 3 fixes — not domain.
# wikimedia x2 buys the second-best domain match (also Wikipedia-derived, like
#   FLORES) without WikiMatrix's noise. x2 is 4 exposures over 2 epochs; x3 would
#   put 35k pairs at 27% of the mix on the strength of a corpus we have never
#   trained on once.
# ntrex x3 because 1,349 professionally translated sentences are the only
#   human-professional signal in the pool and at x1 they are invisible. 6
#   exposures is the ceiling before 1.3k sentences start being memorised.
# SETIMES is NOT downsampled. Its share falls 43.5% -> 36.0% purely because the
#   wiki-domain half grew. It is the only large corpus whose length distribution
#   already matches the test set (sd 0.110 vs FLORES 0.129) — it is the
#   low-variance anchor the whole recipe rests on.
# Tatoeba stays at x1 despite being 0.1% and unlike anything in FLORES: the app
#   sends segments under 6 words 8.4% of the time (212 of the 2,526 segments
#   FLORES actually produces through app/translate.py), and 484 well-formed short
#   pairs are the only faithful short material in the pool.
WEIGHTS = {
    "WikiMatrix": 1,
    "SETIMES": 1,
    "wikimedia": 2,
    "TED2020": 1,
    "ntrex": 3,
    "Tatoeba": 1,
}

# FLORES targets. Constants from the measurement that produced this recipe —
# this script does not open data/flores/.
FLORES_SD_LOG_CHAR = 0.129
FLORES_CHAR_RATIO = 0.9938
FLORES_WORD_RATIO = 1.0772

# What the mix must come out at. These are the point of the exercise, so unlike
# the row counts they are checked with a tolerance: they are the property being
# engineered, and a rebuild that lands at 0.131 has still done the job, while one
# at 0.15 has not. Tolerances are ~1.5% of the value, tight enough that any real
# change in the blend trips them.
TARGET_SD_LOG_CHAR = 0.130
TARGET_CHAR_RATIO = 1.0184      # +2.5% against FLORES's 0.9938 — the deliberate
TARGET_WORD_RATIO = 1.0891      # long tilt, sized against the 2.7% shortfall the
TARGET_TOL = 0.002              # current model measured (0.992 vs base 1.019)


# ===================================================================
# checks
# ===================================================================
class Drift(SystemExit):
    pass


_checks = []


def check(label: str, actual, expected, note: str = "") -> None:
    """Assert a number the recipe was measured on.

    Loud on purpose. Every one of these guards a step that would otherwise fail
    silently and hand the trainer a smaller, differently-shaped corpus than the
    one the recipe's chrF2 arithmetic was done on.
    """
    ok = actual == expected
    _checks.append((ok, label, actual, expected))
    if not ok:
        raise Drift(
            f"\nDRIFT: {label}\n"
            f"  expected {expected:,}\n"
            f"  measured {actual:,}\n"
            f"  {note or 'The inputs are not the ones this recipe was measured on.'}\n"
            f"  Re-measure with --dry-run and update the constant, or find what "
            f"changed upstream. Do not train past this.\n")


# ===================================================================
# step 1 — citation markers
# ===================================================================
def strip_citations(text: str) -> str:
    return WHITESPACE.sub(" ", CITATION.sub("", text)).strip()


# ===================================================================
# step 2 — sentence splitter
# ===================================================================
def _suppressed(before: str, after: str) -> bool:
    """True when the period between `before` and `after` is not a sentence end.

    `before` is the whole pending sentence; only its last whitespace-delimited
    token decides — the period belongs to that token, not to the sentence.
    """
    tail = before.split()
    core = tail[-1].strip(TRIM) if tail else ""
    if core:
        if core.isdigit():
            return True                             # "2016.", "31.", "60."
        if len(core) == 1 and core.isalpha():
            return True                             # "George W."
        if core.lower() in ABBREV:
            return True                             # "dr.", "npr.", "itd."
        if "." in core and INITIALS.fullmatch(core):
            return True                             # "U.S.", "e.g."
        if ROMAN.fullmatch(core):
            return True                             # "XIX. stoljeće"
    nxt = after.lstrip().lstrip(OPENERS)
    if nxt and nxt[0].isalpha() and nxt[0].islower():
        return True                                 # a sentence does not start lowercase
    return False


def split_sentences(text: str) -> list:
    """One sentence per element.

    Breaks on [.!?]+ followed by whitespace, and on newlines. Newlines are an
    unconditional break for the same reason app/translate.py treats them as one:
    text off a photo or a subtitle arrives with the punctuation missing and the
    line break *is* the boundary.
    """
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        start = 0
        for m in PUNCT_RUN.finditer(line):
            end = m.end()
            while end < len(line) and line[end] in CLOSERS:
                end += 1
            if end >= len(line) or not line[end].isspace():
                continue                            # not a boundary candidate
            if _suppressed(line[start:m.start()], line[end:]):
                continue
            piece = line[start:end].strip()
            if piece:
                out.append(piece)
            start = end
        tail = line[start:].strip()
        if tail:
            out.append(tail)
    return out


# ===================================================================
# measurement
# ===================================================================
def describe(pairs) -> dict:
    """Length shape of a pair list, in the terms the recipe is written in.

    The two mean ratios are corpus aggregates — sum(en) / sum(bs) — not the mean
    of the per-pair ratios. That is how the FLORES constants below were computed,
    and the two are not interchangeable: on this mix the aggregate char ratio is
    1.018 and the mean of per-pair ratios is 1.029, because a short sentence's
    ratio is noisy and counts as much as a long one's. Comparing a mean-of-ratios
    against an aggregate FLORES number would invent a 1-point tilt that is not
    there.

    The variance is per-pair and has to be: the whole diagnosis is about what
    individual training examples teach.
    """
    logs = [math.log(len(en) / len(bs)) for _, bs, en in pairs if bs and en]
    mean = sum(logs) / len(logs)
    return {
        "n": len(pairs),
        "sd_log_char": math.sqrt(sum((x - mean) ** 2 for x in logs) / len(logs)),
        "char_ratio": (sum(len(en) for _, _, en in pairs)
                       / sum(len(bs) for _, bs, _ in pairs)),
        "word_ratio": (sum(len(en.split()) for _, _, en in pairs)
                       / sum(len(bs.split()) for _, bs, _ in pairs)),
    }


# ===================================================================
# pipeline
# ===================================================================
def read_rows(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                rows.append(tuple(parts))
    return rows


def build(verbose=True):
    log = print if verbose else (lambda *a, **k: None)

    for path in (TRAIN, EXTRA):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    rows = read_rows(TRAIN) + read_rows(EXTRA)
    check("input rows", len(rows), 351_889,
          "data/clean/train.tsv (313,612) + data/extra/extra-train.tsv (38,277).")
    log(f"in: {len(rows):,} rows")

    # ---- step 1 ------------------------------------------------------
    # 26.28% of the wikimedia rows carry [1]-style markers: 8,833 on both sides,
    # 721 on only one. That 721 is pure length-ratio noise — the same sentence
    # with three extra characters on one side and not the other. FLORES contains
    # zero of them. Worth 7,648 extra usable wikimedia pairs downstream, because
    # "Prijedor.[1]" was blocking the sentence split in step 2.
    cit_both = collections.Counter()
    cit_one = collections.Counter()
    stripped = []
    for corpus, bs, en in rows:
        hb, he = bool(CITATION.search(bs)), bool(CITATION.search(en))
        if hb and he:
            cit_both[corpus] += 1
        elif hb or he:
            cit_one[corpus] += 1
        stripped.append((corpus, strip_citations(bs), strip_citations(en)))
    check("step 1 wikimedia citations both sides", cit_both["wikimedia"], 8_833)
    check("step 1 wikimedia citations one side", cit_one["wikimedia"], 721)
    # The recipe's 8,833/721 is the wikimedia count. Measured here, SETIMES also
    # carries 6 such rows (3 both-side, 3 one-side) and every other corpus zero,
    # so the global figures are 8,836/724. Asserted separately rather than folded
    # in, so a change in either corpus is attributable.
    check("step 1 citations both sides, all corpora", sum(cit_both.values()), 8_836)
    check("step 1 citations one side, all corpora", sum(cit_one.values()), 724)
    log(f"1. citation markers stripped: {sum(cit_both.values()):,} both-side, "
        f"{sum(cit_one.values()):,} one-side "
        f"({cit_both['wikimedia']:,}/{cit_one['wikimedia']:,} of it wikimedia)")

    # ---- step 2 ------------------------------------------------------
    # Same n on both sides -> emit n aligned pairs. Different n -> drop the row.
    # That disagreement is a genuine sentence-count misalignment: one side
    # literally says something the other does not. It is cheap to drop because
    # with ordinals protected the two sides agree on 98-99% of rows.
    #
    # MEASURED, AND IT DOES NOT MATCH THE RECIPE. The recipe says this step drops
    # 6,942 rows; this splitter drops 4,809. The gap is the splitter, not the
    # inputs — the recipe's own control reproduces here exactly: running
    # app/translate.py's naive `(?<=[.!?])\s+` over these same rows disagrees on
    # 12.94% of SETIMES and 16.87% of WikiMatrix, against the 12.9% and 16.9% the
    # recipe quotes. Same data, different splitter, and this one keeps 2,133 more
    # rows. The per-corpus split says where: SETIMES 785 (recipe 770), TED2020 745
    # (763) and ntrex 16 (36) are all within noise of it, while WikiMatrix comes
    # out 964 against 2,389 and wikimedia 2,297 against 2,982. Those two are the
    # corpora dense in the constructions the exception list protects — years,
    # "U.S.", "pp.", "st." — and the exception list here is evidently wider than
    # the one the recipe was measured with. Suppressions actually fired, by rule:
    # all-digit token 67,363, single letter 8,627, next word lowercase 8,394,
    # dotted initials 2,474, abbreviation 8,190, roman numeral 116. The digit rule
    # alone is two thirds of it, which is the recipe's own claim about where the
    # false breaks are.
    split_pairs, split_drop = [], collections.Counter()
    provenance = {}          # pair index -> index of the row it was split out of
    multi = 0
    for ridx, (corpus, bs, en) in enumerate(stripped):
        bs_s, en_s = split_sentences(bs), split_sentences(en)
        if len(bs_s) != len(en_s) or not bs_s:
            split_drop[corpus] += 1
            continue
        if len(bs_s) > 1:
            multi += 1
        for b, e in zip(bs_s, en_s):
            provenance[len(split_pairs)] = ridx
            split_pairs.append((corpus, b, e))
    check("step 2 rows dropped", sum(split_drop.values()), 4_809,
          f"per corpus: {dict(split_drop)}. Recipe said 6,942 — see the note above.")
    for corpus, expected in (("SETIMES", 785), ("WikiMatrix", 964),
                             ("wikimedia", 2_297), ("TED2020", 745),
                             ("ntrex", 16), ("Tatoeba", 2)):
        check(f"step 2 rows dropped [{corpus}]", split_drop[corpus], expected)
    check("step 2 pairs out", len(split_pairs), 359_309)
    log(f"2. split: {len(split_pairs):,} one-sentence pairs "
        f"({multi:,} rows were multi-sentence), {sum(split_drop.values()):,} rows dropped")

    # ---- step 3 ------------------------------------------------------
    banded, band_drop = [], collections.Counter()
    kept_prov = {}
    for i, (corpus, bs, en) in enumerate(split_pairs):
        if not bs or not en:
            band_drop[corpus] += 1
            continue
        r = math.log(len(en) / len(bs))
        if BAND_LO <= r <= BAND_HI:
            kept_prov[len(banded)] = provenance[i]
            banded.append((corpus, bs, en))
        else:
            band_drop[corpus] += 1
    # 25,309 against the recipe's 25,120. The +189 is arithmetic, not disagreement:
    # step 2 handed this step 2,133 more rows than the recipe's step 2 did, and
    # they carry the same tail mass as the rest. ntrex (98) and Tatoeba (54) land
    # on the recipe's numbers exactly.
    check("step 3 pairs dropped", sum(band_drop.values()), 25_309,
          f"per corpus: {dict(band_drop)}")
    for corpus, expected in (("WikiMatrix", 18_884), ("wikimedia", 2_373),
                             ("SETIMES", 3_064), ("TED2020", 836),
                             ("ntrex", 98), ("Tatoeba", 54)):
        check(f"step 3 pairs dropped [{corpus}]", band_drop[corpus], expected)
    log(f"3. length band [{BAND_LO}, {BAND_HI}]: {sum(band_drop.values()):,} dropped, "
        f"{len(banded):,} left")

    # ---- step 4 ------------------------------------------------------
    micro, kept = 0, []
    prov2 = {}
    for i, (corpus, bs, en) in enumerate(banded):
        if len(bs) < MIN_CHARS or len(en) < MIN_CHARS:
            micro += 1
            continue
        prov2[len(kept)] = kept_prov[i]
        kept.append((corpus, bs, en))
    # 46, recipe 64. Same cause as everywhere else in this file: a different set
    # of pairs reached here.
    check("step 4 micro-pairs dropped", micro, 46)
    log(f"4. micro-pairs (<{MIN_CHARS} chars): {micro:,} dropped")

    # ---- step 5 ------------------------------------------------------
    # A minor lever, kept for the record rather than for the effect: exact
    # duplicates are only 1.4% of the input and cross-corpus overlap is ~150 rows.
    seen, deduped, dup = set(), [], 0
    prov3 = {}
    for i, (corpus, bs, en) in enumerate(kept):
        key = (tuple(t.lower() for t in TOKEN.findall(bs)),
               tuple(t.lower() for t in TOKEN.findall(en)))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        prov3[len(deduped)] = prov2[i]
        deduped.append((corpus, bs, en))
    check("step 5 duplicates dropped", dup, 9_664)   # recipe 9,767
    log(f"5. duplicates: {dup:,} dropped, {len(deduped):,} left")

    # ---- step 6 ------------------------------------------------------
    # MAX_LEN=128 in train_translation.py truncates, and a truncated target is a
    # target cut mid-sentence and then handed a </s>. That is a small but free
    # contribution to terseness — the model is shown, 1,912 times, that stopping
    # early is correct. Dropping those pairs makes the cap lossless instead.
    lengths = token_lengths(deduped)
    final, over = [], 0
    prov4 = {}
    for i, (corpus, bs, en) in enumerate(deduped):
        if max(lengths[i]) > MAX_TOKENS:
            over += 1
            continue
        prov4[len(final)] = prov3[i]
        final.append((corpus, bs, en))
    check("step 6 over-length pairs dropped", over, 753,          # recipe 713
          f"cap is {MAX_TOKENS} SentencePiece tokens under {BASE_MODEL}")
    check("total unique one-sentence pairs", len(final), 323_537)   # recipe 322,271
    log(f"6. over {MAX_TOKENS} tokens: {over:,} dropped")
    log(f"   -> {len(final):,} unique one-sentence pairs")

    # ---- step 7 ------------------------------------------------------
    rng = random.Random(SEED)
    ntrex_idx = [i for i, (c, _, _) in enumerate(final) if c == "ntrex"]
    rng.shuffle(ntrex_idx)
    holdout_idx = set(ntrex_idx[:HOLDOUT_N])
    # A row that split into two sentences must not put one half in the holdout
    # and the other half in training. Those siblings are dropped outright rather
    # than added to the holdout, so the holdout stays exactly the size the recipe
    # names and no ntrex sentence appears on both sides of the split.
    holdout_rows = {prov4[i] for i in holdout_idx}
    leaked = {i for i, (c, _, _) in enumerate(final)
              if c == "ntrex" and i not in holdout_idx and prov4[i] in holdout_rows}
    holdout = [final[i] for i in sorted(holdout_idx)]
    pool = [p for i, p in enumerate(final)
            if i not in holdout_idx and i not in leaked]
    check("step 7 holdout size", len(holdout), HOLDOUT_N)
    check("step 7 training pool", len(pool), 323_074)   # recipe 321,809
    log(f"7. ntrex holdout: {len(holdout):,} pairs "
        f"({len(leaked):,} sibling sentences dropped to keep the split clean)")

    # ---- step 8 ------------------------------------------------------
    by_corpus = collections.Counter(c for c, _, _ in pool)
    # Recipe: WikiMatrix 144,102, SETIMES 130,487, wikimedia 35,377,
    # TED2020 10,026, ntrex 1,333, Tatoeba 484 — 321,809. Every one of these is
    # within 1% of it, and Tatoeba is identical. The shares the recipe reasons
    # about are unchanged to the tenth of a percent.
    for corpus, expected in (("WikiMatrix", 145_057), ("SETIMES", 130_313),
                             ("wikimedia", 35_849), ("TED2020", 10_022),
                             ("ntrex", 1_349), ("Tatoeba", 484)):
        check(f"step 8 pool [{corpus}]", by_corpus[corpus], expected)
    unknown = set(by_corpus) - set(WEIGHTS)
    if unknown:
        raise Drift(f"\nDRIFT: unweighted corpus tag(s) {sorted(unknown)} in the pool.\n"
                    f"  A new corpus must be given a weight deliberately, not "
                    f"dropped in at x1 by accident.\n")

    mix = []
    for corpus, bs, en in pool:
        mix.extend([(corpus, bs, en)] * WEIGHTS[corpus])
    random.Random(SEED).shuffle(mix)
    check("step 8 training examples", len(mix), 361_621)   # recipe 359,852
    log("8. weights:")
    for corpus, w in sorted(WEIGHTS.items(), key=lambda kv: -by_corpus[kv[0]] * kv[1]):
        n = by_corpus[corpus] * w
        log(f"   {corpus:<12} {by_corpus[corpus]:>7,} x{w} = {n:>7,}  ({n / len(mix):5.1%})")
    log(f"   {'TOTAL':<12} {len(pool):>7,}      {len(mix):>7,}")

    return mix, holdout, final, split_drop, band_drop, prov4, stripped


def token_lengths(pairs) -> list:
    """Per-side SentencePiece lengths, counted the way the trainer counts them.

    Marian carries two sentencepiece models, source.spm and target.spm, and the
    English side has to be counted with the target one. Counting English with the
    source spm roughly doubles it — one 78-character English sentence comes out
    28 tokens against the source spm and 13 against the target — and would drop
    ~2,000 pairs that in fact fit.

    So the call is the same one training/train_translation.py's PairDataset
    makes: tok(bs, text_target=en). Lengths include the </s> the tokenizer
    appends, because MAX_LEN=128 truncation counts it too.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("step 6 needs transformers: pip install -r requirements.txt")
    if not Path(BASE_MODEL).exists():
        raise SystemExit(f"step 6 needs the model's tokenizer at {BASE_MODEL} "
                         f"(or set LILLY_BASE)")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    out = []
    for i in range(0, len(pairs), 4096):
        chunk = pairs[i:i + 4096]
        enc = tok([bs for _, bs, _ in chunk], text_target=[en for _, _, en in chunk])
        out.extend((len(a), len(b)) for a, b in zip(enc["input_ids"], enc["labels"]))
        print(f"   tokenising {min(i + 4096, len(pairs)):,}/{len(pairs):,}",
              end="\r", file=sys.stderr)
    print(" " * 40, end="\r", file=sys.stderr)
    return out


# ===================================================================
# review
# ===================================================================
def review(final, prov4, stripped, n: int, seed: int) -> None:
    """Print pairs to read by hand — the only check that decides whether the
    splitter is right. Two things are being read for: did a split row's two
    sides stay aligned, and did an ordinal survive intact."""
    rng = random.Random(seed)

    rows_of = collections.defaultdict(list)
    for i, pair in enumerate(final):
        rows_of[prov4[i]].append(pair)
    multi = [(r, ps) for r, ps in rows_of.items() if len(ps) > 1]

    n_split = n // 2
    print(f"\n{'=' * 70}\nSPLIT ROWS — are the two sides still saying the same thing,\n"
          f"in the same order, sentence for sentence?\n{'=' * 70}")
    shown = 0
    for ridx, pairs in rng.sample(multi, min(len(multi), n_split)):
        if shown >= n_split:
            break
        corpus, bs, en = stripped[ridx]
        print(f"\n--- {corpus}, row split into {len(pairs)} ---")
        print(f"  SOURCE BS  {bs}")
        print(f"  SOURCE EN  {en}")
        for k, (_, b, e) in enumerate(pairs, 1):
            print(f"    {k}. BS  {b}")
            print(f"    {k}. EN  {e}")
            shown += 1

    print(f"\n{'=' * 70}\nORDINALS — is every '2016.' / '31.' still attached to its\n"
          f"sentence rather than ending one?\n{'=' * 70}")
    ordinal = re.compile(r"\b\d{1,4}\.")
    have = [p for p in final if ordinal.search(p[1])]
    for corpus, bs, en in rng.sample(have, min(len(have), n - shown)):
        print(f"\n--- {corpus} ---")
        print(f"  BS  {bs}")
        print(f"  EN  {en}")

    print(f"\n{'=' * 70}\n{len(multi):,} of {len(rows_of):,} surviving rows were split "
          f"into more than one sentence.\n{len(have):,} kept pairs carry a "
          f"digit-plus-period ordinal.\n{'=' * 70}")


# ===================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="count and measure only, write nothing")
    ap.add_argument("--review", type=int, metavar="N",
                    help="print N pairs to read by hand")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="sample seed for --review")
    args = ap.parse_args()

    mix, holdout, final, _, _, prov4, stripped = build()

    # The point of the whole recipe, measured on what was actually produced.
    # Everything above is bookkeeping; these three numbers are the deliverable.
    shape = describe(mix)
    print(f"\nblend against the target (FLORES numbers are constants from the "
          f"measurement, not read here):")
    print(f"  sd(log char ratio)  {shape['sd_log_char']:.4f}  "
          f"target {TARGET_SD_LOG_CHAR:.3f}, FLORES {FLORES_SD_LOG_CHAR:.3f}")
    print(f"  en/bs chars         {shape['char_ratio']:.4f}  "
          f"target {TARGET_CHAR_RATIO:.4f}, FLORES {FLORES_CHAR_RATIO:.4f} "
          f"({shape['char_ratio'] / FLORES_CHAR_RATIO - 1:+.1%} tilt)")
    print(f"  en/bs words         {shape['word_ratio']:.4f}  "
          f"target {TARGET_WORD_RATIO:.4f}, FLORES {FLORES_WORD_RATIO:.4f}")

    for label, got, want in (("sd(log char ratio)", shape["sd_log_char"], TARGET_SD_LOG_CHAR),
                             ("en/bs char ratio", shape["char_ratio"], TARGET_CHAR_RATIO),
                             ("en/bs word ratio", shape["word_ratio"], TARGET_WORD_RATIO)):
        if abs(got - want) > TARGET_TOL:
            raise Drift(f"\nDRIFT: {label} is {got:.4f}, target {want:.4f} "
                        f"(+-{TARGET_TOL}).\n"
                        f"  This is the property the mix exists to have. Do not "
                        f"train on it until this is understood.\n")

    if args.review:
        review(final, prov4, stripped, args.review, args.seed)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    with open(OUT_MIX, "w", encoding="utf-8") as f:
        for corpus, bs, en in mix:
            f.write(f"{corpus}\t{bs}\t{en}\n")
    with open(OUT_HOLDOUT, "w", encoding="utf-8") as f:
        for corpus, bs, en in holdout:
            f.write(f"{corpus}\t{bs}\t{en}\n")
    print(f"\nwrote {OUT_MIX} ({len(mix):,})")
    print(f"wrote {OUT_HOLDOUT} ({len(holdout):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
