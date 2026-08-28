#!/usr/bin/env python3
"""Is any held-out row present in anything the translator trains on?

Written because the answer "FLORES leakage 0" has already been wrong once on
this project. An independent re-check of the same corpus found 61 overlaps
against our own held-out sets — 59 with train (harmless fragments, median 20
characters), 1 with valid and 1 with test. The two real ones were deleted. The
first check was not dishonest; it checked FLORES and stopped, and FLORES was
genuinely clean. So this script checks every held-out set we have, and reports
per set rather than as one total that a single zero can hide inside.

WHAT IS HELD OUT, and what each one decides:

    data/flores/{dev,devtest}.{bs,en}   2,009 pairs. The pre-registered
                                        decision is made on these. A leak here
                                        invalidates every published number.
    data/clean/test.tsv                 1,500 pairs.
    data/clean/valid.tsv                1,500 pairs. Early stopping reads this,
                                        so a leak here is a leak into the
                                        training loop, not only into a report.

None of the three is ever opened for writing here, and this script imports
nothing that could.

WHAT IS CHECKED AGAINST THEM:

    data/clean/train.tsv        313,612 rows. The filter's output.
    data/extra/extra-train.tsv   38,277 rows. wikimedia + NTREX, the "unseen"
                                 corpus.
    data/clean/train-mix.tsv    361,621 rows. THE ONE THAT MATTERS, and the one
                                 no previous check looked at.

That last line is the point of this file. build_training_mix.py splits every
row into one sentence per example (step 2), so a held-out sentence sitting
inside a longer paragraph is invisible to a whole-row comparison against the
two inputs and lands as an exact, standalone, sentence-level match in the file
the model is actually fed. Checking the inputs and not the output is checking
the wrong artifact, and it is the only one of the three that the trainer reads.

THREE MATCH STRENGTHS, because one number cannot separate a leak from a coincidence:

    pair    both sides of a held-out row appear as both sides of a training row.
            Unambiguous. Any count above 0 is a leak.
    bs      the Bosnian side matches, the English does not. Either a leak with a
            different reference translation, or boilerplate.
    en      the English side matches, the Bosnian does not. Same reading.

A one-side match on "web." is not a leak; a one-side match on a 180-character
sentence is. So every one-side hit is reported with its character length and the
distribution is printed, rather than being counted and waved through. The floor
below which a match is treated as coincidence is stated as a flag and defaults
to 0 — nothing is silently discarded.

NORMALISATION. Comparison is on NFC, whitespace-collapsed, case-folded text with
the wiki citation markers build_training_mix.py strips in its own step 1 removed,
because a leak that differs from its held-out twin by a "[1]" or a capital is
still a leak. `--exact` turns all of that off and compares raw bytes.

Usage:
    python3 data/scripts/audit_translation_leakage.py
    python3 data/scripts/audit_translation_leakage.py --min-chars 25
    python3 data/scripts/audit_translation_leakage.py --show 20
    python3 data/scripts/audit_translation_leakage.py --exact
"""
import argparse
import collections
import re
import sys
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_DIR.parent
CLEAN_DIR = DATA_DIR / "clean"
FLORES_DIR = DATA_DIR / "flores"

CITATION = re.compile(r"\[\d+\]")
WHITESPACE = re.compile(r"\s+")


def norm(text: str, exact: bool = False) -> str:
    if exact:
        return text
    text = unicodedata.normalize("NFC", text)
    text = CITATION.sub("", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text.casefold()


def read_tsv(path: Path):
    """(bs, en) from a source/bs/en TSV. Malformed rows are reported, not skipped."""
    rows, bad = [], 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                bad += 1
                continue
            rows.append((parts[1], parts[2]))
    if bad:
        print(f"  ! {path.name}: {bad} rows without exactly 3 fields", file=sys.stderr)
    return rows


def read_flores():
    rows = []
    for split in ("dev", "devtest"):
        bs = (FLORES_DIR / f"{split}.bs").read_text(encoding="utf-8").splitlines()
        en = (FLORES_DIR / f"{split}.en").read_text(encoding="utf-8").splitlines()
        if len(bs) != len(en):
            raise SystemExit(f"flores {split}: {len(bs)} bs against {len(en)} en")
        rows += list(zip(bs, en))
    return rows


def audit(held_name, held_rows, train_name, train_rows, exact, min_chars):
    """Every held-out row that appears in the training rows, by match strength."""
    bs_index = collections.defaultdict(list)
    en_index = collections.defaultdict(list)
    for i, (bs, en) in enumerate(train_rows):
        nb, ne = norm(bs, exact), norm(en, exact)
        if nb:
            bs_index[nb].append(i)
        if ne:
            en_index[ne].append(i)

    hits = {"pair": [], "bs": [], "en": []}
    for bs, en in held_rows:
        nb, ne = norm(bs, exact), norm(en, exact)
        in_bs = nb in bs_index if nb else False
        in_en = ne in en_index if ne else False
        if not (in_bs or in_en):
            continue
        if in_bs and in_en:
            # Both sides present. A pair leak only if some ONE training row
            # carries both; two unrelated rows each sharing one side is not.
            shared = set(bs_index[nb]) & set(en_index[ne])
            kind = "pair" if shared else "bs"
            hits[kind].append((bs, en, len(nb) if kind != "en" else len(ne)))
            if not shared:
                hits["en"].append((bs, en, len(ne)))
        elif in_bs:
            hits["bs"].append((bs, en, len(nb)))
        else:
            hits["en"].append((bs, en, len(ne)))

    kept = {k: [h for h in v if h[2] >= min_chars] for k, v in hits.items()}
    return hits, kept


def near_duplicates(held_name, held_rows, train_name, train_rows, top=15):
    """Held-out rows that are not identical to a training row but close to one.

    Exact matching answers "was this row copied". It does not answer "was this
    row copied and then edited", and on this corpus that second question is the
    live one: two SETIMES rows below share an identical English reference and
    differ only in the Bosnian, which no exact check on pairs would report.
    FLORES matters more still — it is built from Wikinews, Wikijunior and
    Wikivoyage, and our largest new corpus is Wikimedia, so the two are drawn
    from overlapping ground and a paraphrase leak is the plausible one.

    ONE LIMITATION, STATED BECAUSE IT CHANGES HOW THE OUTPUT READS. TOKEN below
    is letters-only, so digits are invisible to this comparison. That is
    deliberate for paraphrase detection — "500 people" and "350 people" are the
    same sentence for this purpose — but it means SETIMES bylines that differ
    only in their date ("... in Skopje -- 05/05/11" against "... -- 12/01/11")
    and citation lines that differ only in a page number score a perfect 1.000.
    Those are template collisions, not leaked content, and they are most of what
    a high score against test.tsv and valid.tsv turns out to be. Read the
    examples, not the count.

    Jaccard over word types, on the English side, with candidates drawn from an
    inverted index over each row's rarest tokens so this stays a few seconds
    rather than 2,009 x 361,621 comparisons. Reported as a distribution, because
    the number that matters is the maximum: if the closest training row to any
    held-out row is far away, there is no near-leak to argue about.
    """
    df = collections.Counter()
    toks = []
    for _, en in train_rows:
        t = set(TOKEN.findall(norm(en)))
        toks.append(t)
        df.update(t)
    inv = collections.defaultdict(list)
    for i, t in enumerate(toks):
        for w in sorted(t, key=lambda w: df[w])[:4]:
            if df[w] <= 4000:
                inv[w].append(i)

    best = []
    for bs, en in held_rows:
        t = set(TOKEN.findall(norm(en)))
        if not t:
            continue
        cand = collections.Counter()
        for w in sorted(t, key=lambda w: df[w])[:8]:
            for i in inv.get(w, ()):
                cand[i] += 1
        hi, hj = 0.0, -1
        for i, _ in cand.most_common(400):
            u = toks[i]
            if not u:
                continue
            j = len(t & u) / len(t | u)
            if j > hi:
                hi, hj = j, i
        best.append((hi, en, hj))
    best.sort(reverse=True)
    vals = [b[0] for b in best]
    if not vals:
        return
    vals_sorted = sorted(vals)
    print(f"  {held_name} vs {train_name}: max Jaccard {max(vals):.3f}, "
          f"p99 {vals_sorted[int(0.99 * len(vals))]:.3f}, "
          f"median {vals_sorted[len(vals) // 2]:.3f}")
    for cut in (0.9, 0.7, 0.5):
        print(f"      >= {cut}: {sum(1 for v in vals if v >= cut)}")
    for jac, en, i in best[:top]:
        if jac < 0.7:
            break
        print(f"      [{jac:.3f}] held: {en[:100]}")
        print(f"              train: {train_rows[i][1][:100]}")


TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", action="store_true",
                    help="also report nearest training row per held-out row (Jaccard)")
    ap.add_argument("--min-chars", type=int, default=0,
                    help="length floor below which a one-side match is reported "
                         "separately as coincidence. Default 0: nothing is discarded.")
    ap.add_argument("--show", type=int, default=8, help="example hits to print per bucket")
    ap.add_argument("--exact", action="store_true", help="compare raw text, no normalisation")
    args = ap.parse_args()

    held = [
        ("flores (2,009)", read_flores()),
        ("test.tsv", read_tsv(CLEAN_DIR / "test.tsv")),
        ("valid.tsv", read_tsv(CLEAN_DIR / "valid.tsv")),
    ]
    train = [
        ("train.tsv", read_tsv(CLEAN_DIR / "train.tsv")),
        ("extra-train.tsv", read_tsv(DATA_DIR / "extra" / "extra-train.tsv")),
        ("train-mix.tsv", read_tsv(CLEAN_DIR / "train-mix.tsv")),
    ]

    print(f"normalisation: {'OFF (raw bytes)' if args.exact else 'NFC + citations + space + case'}")
    print(f"length floor for one-side matches: {args.min_chars} chars\n")
    for hn, hr in held:
        print(f"{hn}: {len(hr)} held-out rows")
    print()

    worst = 0
    for tn, tr in train:
        print(f"=== {tn} ({len(tr):,} rows) " + "=" * (44 - len(tn)))
        for hn, hr in held:
            hits, kept = audit(hn, hr, tn, tr, args.exact, args.min_chars)
            line = (f"  {hn:<16} pair {len(hits['pair']):>4}   "
                    f"bs-only {len(hits['bs']):>4}   en-only {len(hits['en']):>4}")
            if args.min_chars:
                line += (f"   | >= {args.min_chars}ch: "
                         f"{len(kept['bs'])} / {len(kept['en'])}")
            print(line)
            worst = max(worst, len(hits["pair"]))
            for kind in ("pair", "bs", "en"):
                shown = kept[kind] if args.min_chars else hits[kind]
                if not shown:
                    continue
                lengths = sorted(h[2] for h in shown)
                med = lengths[len(lengths) // 2]
                print(f"      {kind}: {len(shown)} hits, median {med} chars, "
                      f"min {lengths[0]}, max {lengths[-1]}")
                for bs, en, n in shown[: args.show]:
                    print(f"        [{n:>4}ch] bs={bs[:90]!r}")
                    print(f"                 en={en[:90]!r}")
        print()

    if args.near:
        print("=== nearest-neighbour (paraphrase) check, English side " + "=" * 20)
        for tn, tr in train:
            for hn, hr in held:
                near_duplicates(hn, hr, tn, tr)
        print()

    print(f"whole-pair leaks, worst over all combinations: {worst}")
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())
