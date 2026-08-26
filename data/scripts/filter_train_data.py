#!/usr/bin/env python3
"""Drop the WikiMatrix pairs in train.tsv that are not translations of each other.

WikiMatrix is mined out of Wikipedia by embedding similarity, not by document
alignment, so a "pair" only has to be *about* the same thing. A sizeable part of
it is two sentences on one topic that state different facts, an English citation
line copied to both sides, or two sentences with nothing in common at all.
Training on those teaches the model that inventing content is acceptable.

The OPUS moses distribution we download in download_data.py ships WikiMatrix as
bare .bs/.en text. The margin score that the original LASER release attaches to
every mined pair — the one signal that would make all of this unnecessary — is
not in it. So the evidence has to be reconstructed from the sentences.

A pair is dropped only when something in it is positive evidence of misalignment.
Each rule below was calibrated on SETIMES (document-aligned news) and TED2020
(human-made subtitles) as controls: whatever a rule fires on there is its false
positive rate, because those two corpora are aligned by construction.

  numbers disagree ......... both sides carry a number the other side does not.
                             Fires on 0.14% of SETIMES, 0.39% of TED2020,
                             4.8% of WikiMatrix.
  bosnian side is english .. mined citation/title lines copied through
                             untranslated. 0.14% of SETIMES, 0.16% of TED2020,
                             6.3% of WikiMatrix.
  words do not correspond .. the english side's words have no counterpart among
                             the bosnian side's words, under a table learned by
                             IBM Model 1 from the corpora we trust. Set by hand
                             review at the only point where it is trustworthy:
                             0.00% of SETIMES, 0.04% of TED2020, 0.2% of
                             WikiMatrix.

What this cannot catch is said at the bottom of the file.

test.tsv and valid.tsv are held out and are never read or written here.

Usage:
    python3 filter_train_data.py               # rewrite data/clean/train.tsv
    python3 filter_train_data.py --dry-run     # count only, touch nothing
    python3 filter_train_data.py --review 40   # print pairs to read by hand
    python3 filter_train_data.py --review 40 --seed 99      # a different sample
"""
import argparse
import collections
import pickle
import random
import re
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = DATA_DIR / "clean"
TRAIN = CLEAN_DIR / "train.tsv"
BACKUP = CLEAN_DIR / "train.tsv.bak"
TABLE = CLEAN_DIR / ".lexical-table.pkl"

TARGET_CORPUS = "WikiMatrix"
# SETIMES is document-aligned news, TED2020 and Tatoeba are human-made. Their
# word co-occurrence is trustworthy, so the translation table learns from them.
# WikiMatrix is the corpus under suspicion and must not teach the table that its
# own bad pairs are normal.
TRUSTED = ("SETIMES", "TED2020", "Tatoeba")

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
STEM = 6            # bosnian inflects heavily; 6 characters collapses most of it
NUM = re.compile(r"\d[\d.,]*\d|\d")
DIACRITIC = re.compile(r"[čćžšđČĆŽŠĐ]")

# an english word counts as covered when some bosnian word in the same sentence
# translates to it with at least this probability
COVER_P = 0.02
# only english words this common in the trusted corpora are scored. A rare
# technical term missing its bosnian counterpart usually means the table never
# learned the bosnian word, not that the pair is broken — reading a hand sample
# at a stricter setting, half of what it dropped were correct translations of
# military, chemical and legal vocabulary that SETIMES news simply never uses.
COVER_MIN_EN_FREQ = 50
# below this fraction of covered english words the two sides are not saying the
# same thing. Deliberately low. Two rounds of hand review put good and bad pairs
# on top of each other everywhere between 0.15 and 0.35 — a freely translated
# sentence and an unrelated one look the same to a bag-of-words score — so the
# rule only speaks where almost nothing on the english side has a counterpart.
COVER_MIN = 0.15
# under five scoreable english words the fraction is too coarse to trust, so the
# rule abstains rather than guesses
COVER_MIN_WORDS = 5

# metric on the bosnian side against imperial on the english side is a correct
# translation, not a disagreement — value ratios en/bs, matched within 2%
UNIT_RATIOS = (2.20462,    # kg -> lb
               0.621371,   # km -> mi
               3.28084,    # m -> ft
               0.393701,   # cm -> in
               0.386102,   # km2 -> sq mi
               2.47105,    # ha -> acres
               0.264172)   # l -> gal

BOSNIAN_MARKERS = set("""
je su i u na se da za od koji koja koje sa bio bila bili nije kao ali ili te po do iz prema
godine godina njegov njegova ova ovaj ovo taj ta to nakon prije poslije kroz među sve svi
jedan jedna jedno bilo biti smo ste tako također takođe zbog pod nad oko ka ih im mu joj
gdje kada kako što šta nego samo već još uvijek ne li ni njih nam vam jer pa niti kod
""".split())
ENGLISH_MARKERS = set("""
the of and in to a is was for on with as at by an be from that it this these those were are
has have had not but or which who when where while their its his her they he she we you been
being into about after before during through between over under more most other such can may
""".split())


def stems(text: str) -> list:
    return [w[:STEM] for w in WORD.findall(text.lower()) if len(w) > 1]


def numbers(text: str) -> collections.Counter:
    """Numbers as values, not as strings.

    Bosnian writes 1,8 and 100.000 where English writes 1.8 and 100,000, and
    comparing the raw strings makes every one of those look like a disagreement.
    A separator followed by exactly three digits is a thousands group and goes
    away; any other comma is a decimal point.
    """
    out = collections.Counter()
    for match in NUM.findall(text):
        s = re.sub(r"[.,](?=\d{3}(?!\d))", "", match).replace(",", ".").rstrip(".")
        try:
            out[float(s)] += 1
        except ValueError:
            pass
    return out


def _same_quantity(bs_value: float, en_value: float) -> bool:
    """Two spellings of one quantity: unit conversion, Fahrenheit, or a decade.

    "90-ih godina" against "the 1990s" is a translation, not a disagreement, and
    without the decade case the rule throws out every abbreviated decade in the
    corpus.
    """
    for a, b in ((bs_value, en_value), (en_value, bs_value)):
        if 0 <= a <= 99 and 1000 <= b <= 2100 and b % 100 == a:
            return True
        for ratio in UNIT_RATIOS:
            if b and abs(a * ratio - b) <= 0.02 * abs(b):
                return True
    return abs(bs_value * 1.8 + 32 - en_value) <= 1.0


def numbers_disagree(bs: str, en: str) -> bool:
    """True when each side carries a number the other one does not.

    Requiring the mismatch to run *both* ways is what makes this safe. One side
    holding a number the other spells out ("petnaestog decembra" / "December
    15th") or omits is normal translation; a 30.000 facing a 60,000 is two
    different claims. The one-way version of this rule fires four times as often
    on SETIMES for exactly that reason.

    Unit conversions have to be forgiven explicitly: "127 i 158 kilograma" and
    "280 and 350 pounds" are the same sentence, and without this the rule throws
    away every converted measurement in the corpus.
    """
    left = list((numbers(bs) - numbers(en)).elements())
    right = list((numbers(en) - numbers(bs)).elements())
    for x in list(left):
        for y in list(right):
            if _same_quantity(x, y):
                left.remove(x)
                right.remove(y)
                break
    return bool(left) and bool(right)


def bosnian_side_is_english(bs: str, en: str) -> bool:
    """True when the 'Bosnian' side was never translated.

    WikiMatrix mines reference lists, so thousands of pairs are an English book
    title or journal citation sitting on both sides. Those teach the model to
    copy its input through, which is the one failure a translator must not have.
    Two independent ways to see it: the Bosnian side carries English function
    words and no Bosnian ones, or the two sides are nearly the same bag of words.
    """
    words = [w.lower() for w in WORD.findall(bs)]
    if not words:
        return False
    bosnian_hits = sum(w in BOSNIAN_MARKERS for w in words) + bool(DIACRITIC.search(bs))
    if bosnian_hits == 0 and any(w in ENGLISH_MARKERS for w in words):
        return True
    a = set(words)
    b = {w.lower() for w in WORD.findall(en)}
    return bool(a) and bool(b) and len(a & b) / len(a | b) >= 0.80


def build_table(rows: dict) -> dict:
    """IBM Model 1 by EM over the trusted corpora: t[(bs_stem, en_stem)] = p(en|bs).

    Model 1 ignores word order, which is what we want — the question is whether
    the English words have Bosnian counterparts present at all, not whether they
    arrived in the same order.
    """
    sents = []
    for name in TRUSTED:
        for bs, en in rows.get(name, ()):
            b, e = stems(bs), stems(en)
            if 1 <= len(b) <= 45 and 1 <= len(e) <= 45:
                sents.append((b, e))
    print(f"  learning a translation table from {len(sents):,} trusted pairs")

    cooc = collections.Counter()
    for b, e in sents:
        for x in set(b):
            for y in set(e):
                cooc[(x, y)] += 1
    # a stem pair seen together once is an accident, and keeping those triples
    # the table for no gain
    t = {k: 1.0 for k, v in cooc.items() if v >= 2}
    print(f"  {len(cooc):,} co-occurring stem pairs, {len(t):,} seen more than once")

    for it in range(6):
        started = time.time()
        count = collections.defaultdict(float)
        total = collections.defaultdict(float)
        for b, e in sents:
            bset = set(b)
            for y in e:
                ps = [(x, t.get((x, y), 0.0)) for x in bset]
                z = sum(p for _, p in ps)
                if z <= 0:
                    continue
                for x, p in ps:
                    if p > 0:
                        count[(x, y)] += p / z
                        total[x] += p / z
        t = {k: v / total[k[0]] for k, v in count.items() if total[k[0]] > 0}
        # a stem pair this unlikely can never win an alignment; dropping it keeps
        # every later iteration cheap
        t = {k: v for k, v in t.items() if v >= 1e-4}
        print(f"  EM pass {it + 1}/6: {len(t):,} entries ({time.time() - started:.0f}s)")
    return t


def index_table(t: dict, rows: dict):
    """(en_stem -> {bs_stem: p}, english stems common enough to score).

    A Bosnian stem seen twice in training collects garbage probability mass in
    Model 1 and will cheerfully "explain" any English word it meets, which would
    let bad pairs through. Five occurrences is where that stops.
    """
    bs_freq, en_freq = collections.Counter(), collections.Counter()
    for name in TRUSTED:
        for bs, en in rows.get(name, ()):
            bs_freq.update(stems(bs))
            en_freq.update(stems(en))
    solid = {x for x, n in bs_freq.items() if n >= 5}
    by_en = collections.defaultdict(dict)
    for (x, y), p in t.items():
        if x in solid:
            by_en[y][x] = p
    scoreable = {y for y in by_en if en_freq[y] >= COVER_MIN_EN_FREQ}
    return by_en, scoreable


def coverage(bs: str, en: str, by_en: dict, scoreable: set):
    """Fraction of the English words that have a counterpart on the Bosnian side.

    Words outside the trusted corpora's everyday vocabulary are skipped instead
    of counted as failures — WikiMatrix is encyclopedic and the trusted corpora
    are news, so unknown vocabulary is expected and is not evidence of anything.
    A word that appears verbatim on both sides counts as covered: WikiMatrix is
    dense with proper nouns, and "Activision Blizzard" facing itself is the
    strongest alignment anchor a pair can have. Returns None when too little is
    left to judge.
    """
    source = set(stems(bs))
    targets = [y for y in stems(en) if y in scoreable]
    if len(targets) < COVER_MIN_WORDS:
        return None
    covered = 0
    for y in targets:
        if y in source or any(by_en[y].get(x, 0.0) >= COVER_P for x in source):
            covered += 1
    return covered / len(targets)


def load_rows(path: Path) -> list:
    rows = []
    for line in path.open(encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3:
            rows.append(tuple(parts))
    return rows


def verdict(bs: str, en: str, by_en: dict, scoreable: set):
    """(reason to drop, coverage score) — reason is None when the pair is kept."""
    score = coverage(bs, en, by_en, scoreable)
    if numbers_disagree(bs, en):
        return "numbers_disagree", score
    if bosnian_side_is_english(bs, en):
        return "not_translated", score
    if score is not None and score < COVER_MIN:
        return "words_do_not_correspond", score
    return None, score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="count what would go, write nothing")
    ap.add_argument("--review", type=int, default=0, metavar="N",
                    help="print N pairs across the score range to read by hand")
    ap.add_argument("--seed", type=int, default=17,
                    help="which pairs --review draws, so a second reader gets a fresh sample")
    ap.add_argument("--retrain", action="store_true",
                    help="rebuild the translation table instead of using the cache")
    args = ap.parse_args()

    all_rows = load_rows(TRAIN)
    by_corpus = collections.defaultdict(list)
    for corpus, bs, en in all_rows:
        by_corpus[corpus].append((bs, en))
    print(f"{TRAIN}: {len(all_rows):,} rows")
    for name, pairs in sorted(by_corpus.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name}: {len(pairs):,} ({len(pairs) / len(all_rows):.1%})")

    if TABLE.exists() and not args.retrain:
        t = pickle.loads(TABLE.read_bytes())
        print(f"  translation table from cache: {len(t):,} entries ({TABLE})")
    else:
        t = build_table(by_corpus)
        TABLE.write_bytes(pickle.dumps(t))
    by_en, scoreable = index_table(t, by_corpus)

    kept, dropped = [], []
    drops = collections.Counter()
    scores = {}
    for corpus, bs, en in all_rows:
        if corpus != TARGET_CORPUS:
            kept.append((corpus, bs, en))
            continue
        reason, score = verdict(bs, en, by_en, scoreable)
        scores[(bs, en)] = score
        if reason:
            drops[reason] += 1
            dropped.append((reason, score, bs, en))
        else:
            kept.append((corpus, bs, en))

    n_target = len(by_corpus[TARGET_CORPUS])
    print(f"\n{TARGET_CORPUS}: {n_target:,} pairs -> {n_target - len(dropped):,} kept, "
          f"{len(dropped):,} dropped ({len(dropped) / n_target:.1%})")
    for reason, n in drops.most_common():
        print(f"  {reason}: {n:,} ({n / n_target:.1%})")

    if args.review:
        review(dropped, kept, scores, args.review, args.seed)

    print(f"\ntrain.tsv: {len(all_rows):,} rows -> {len(kept):,} rows "
          f"({(len(all_rows) - len(kept)) / len(all_rows):.1%} removed)")
    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    if not BACKUP.exists():
        BACKUP.write_bytes(TRAIN.read_bytes())
        print(f"backed up original to {BACKUP}")
    with TRAIN.open("w", encoding="utf-8") as f:
        for corpus, bs, en in kept:
            f.write(f"{corpus}\t{bs}\t{en}\n")
    print(f"wrote {TRAIN}")
    return 0


def review(dropped, kept, scores, n: int, seed: int) -> None:
    """Print pairs to be read by a human — the only check that actually decides
    whether the thresholds above are right."""
    rng = random.Random(seed)
    per_reason = collections.defaultdict(list)
    for reason, score, bs, en in dropped:
        per_reason[reason].append((score, bs, en))
    n_drop = n // 2
    print(f"\n{'=' * 70}\nDROPPED — is each of these genuinely broken?\n{'=' * 70}")
    share = max(1, n_drop // max(1, len(per_reason)))
    for reason, items in sorted(per_reason.items()):
        print(f"\n--- {reason} ({len(items):,} pairs) ---")
        for score, bs, en in rng.sample(items, min(share, len(items))):
            tag = f"{score:.2f}" if score is not None else "n/a"
            print(f"[cover {tag}]\n  BS  {bs}\n  EN  {en}")

    print(f"\n{'=' * 70}\nKEPT — is each of these genuinely fine?\n{'=' * 70}")
    wiki_kept = [(scores.get((bs, en)), bs, en)
                 for corpus, bs, en in kept if corpus == TARGET_CORPUS]
    bands = [(COVER_MIN, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01), (None, None)]
    per_band = max(1, (n - n_drop) // len(bands))
    for lo, hi in bands:
        if lo is None:
            items = [(s, b, e) for s, b, e in wiki_kept if s is None]
            label = "coverage not scoreable (too few known words)"
        else:
            items = [(s, b, e) for s, b, e in wiki_kept if s is not None and lo <= s < hi]
            label = f"coverage {lo:.2f}–{hi:.2f}"
        print(f"\n--- {label} ({len(items):,} pairs) ---")
        for score, bs, en in rng.sample(items, min(per_band, len(items))):
            tag = f"{score:.2f}" if score is not None else "n/a"
            print(f"[cover {tag}]\n  BS  {bs}\n  EN  {en}")


# What this does NOT catch, measured rather than guessed. Twenty surviving
# WikiMatrix pairs drawn uniformly at random and read by hand came back 13 clearly
# correct, 2 borderline, 5 still broken — so this removes roughly a third of the
# misalignment, not all of it:
#   - Two sentences with the same shape and one swapped entity. "Ronaldo, koji
#     igra za Real Madrid" against "Ronaldo, who plays for Juventus" scores 0.91
#     coverage, because every word but the club name does correspond. Nothing
#     short of a real semantic model sees this.
#   - Same-topic pairs from different articles: a Qur'anic verse against a
#     Biblical one, an Islamic prayer against a Hindu one. Shared religious
#     vocabulary carries the coverage score straight past the threshold.
#   - Freely translated sentences and unrelated sentences produce the same
#     coverage. Between 0.15 and 0.50 the two are indistinguishable — that band
#     still holds ~9,300 pairs, enriched in noise, and no threshold separates it.
#   - Negation and reversed agents are invisible to a bag-of-words score.
#   - The real fix is upstream. The LASER release of WikiMatrix carries a margin
#     score for every mined pair; the OPUS moses repackaging we download drops it.
#     Fetching WikiMatrix.bs-en.tsv.gz from the LASER release and joining the
#     score back on would beat every rule in this file. That is a new download
#     and a change to download_data.py, so it is left as a decision to take, not
#     one to take quietly here.

if __name__ == "__main__":
    raise SystemExit(main())
