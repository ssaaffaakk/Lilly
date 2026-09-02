#!/usr/bin/env python3
"""Keep the pseudo-labels a Bosnian lexicon recognises, and restore their
diacritics from it.

    python3 data/scripts/lexicon_filter.py \
        --in data/ocr/mapillary-train-balanced \
        --out data/ocr/mapillary-train-lex

Two findings from docs/diacritic-gate-literature.md, applied together:

- Rijhwani et al., TACL 2021, got 15-29% relative error reduction from
  self-training on OCR **only** when it was combined with lexically aware
  decoding. Pass-16 lowered the confidence threshold to 0.30 on the strength
  of Noisy Student and skipped this half, and the training set filled with
  things that are not words: `dska`, `hesp ,`, `Imax]_`, `BZ4knh`, `UFT`.
  A confidence score cannot tell a misread sign from a read one; a lexicon
  can.

- Krstev et al., 2018: 95% of de-diacriticised Serbian word types have
  exactly one candidate. So when a lexicon holds exactly one accented form of
  a folded label, restoring it is safe. That fixes the diacritic rate at
  source — the labels become correct — instead of duplicating the few rows
  that happened to keep their accents, which is what pass-16 did.

Sources for the lexicon, all already in the repo:
  data/clean/train.tsv                   313k Bosnian sentences
  data/corpus-signs/toponyms-and-names.txt   40k place and family names
  data/signs/sign-text.tsv               9.5k real OSM sign strings
"""
import argparse
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# EasyOCR resizes every crop to 64px tall, so a 16px crop is upscaled 4x and
# an 8px one is mush. The human test set was cut at the same floor after
# Coca-Cola came back empty from a 22x18 crop; whatever is unfair to score on
# is not worth training on either.
MIN_HEIGHT = 16
MIN_WIDTH = 32

WORD = re.compile(r"[A-Za-zÀ-ÿČčĆćĐđŠšŽž]+")
DIACRITICS = re.compile("[čćđšžČĆĐŠŽ]")


def fold(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def build_vocabulary(repo: Path) -> dict:
    """folded lowercase word -> Counter of the accented forms seen."""
    vocab = defaultdict(Counter)

    def add(text: str) -> None:
        for word in WORD.findall(text):
            if len(word) >= 2:
                vocab[fold(word).lower()][word] += 1

    corpus = repo / "data" / "clean" / "train.tsv"
    if corpus.is_file():
        with corpus.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split("\t")
                if len(parts) >= 2:
                    add(parts[1])          # the Bosnian side
        print(f"  {corpus.name}: vocabulary now {len(vocab):,} forms")

    toponyms = repo / "data" / "corpus-signs" / "toponyms-and-names.txt"
    if toponyms.is_file():
        add(toponyms.read_text(encoding="utf-8", errors="replace"))
        print(f"  {toponyms.name}: vocabulary now {len(vocab):,} forms")

    signs = repo / "data" / "signs" / "sign-text.tsv"
    if signs.is_file():
        with signs.open(encoding="utf-8", errors="replace") as fh:
            next(fh, None)
            for line in fh:
                add(line.split("\t")[0])
        print(f"  {signs.name}: vocabulary now {len(vocab):,} forms")

    if not vocab:
        sys.exit("no lexicon sources found — cannot filter")
    return vocab


# A word seen once in 313k sentences is a typo or a fragment, not evidence
# that the reader got it right. MIC, BZ, knh and AtM all "exist" in a corpus
# that large; kuca, bankomat and Popravci are common. The threshold is what
# separates a lexicon check from a substring check.
MIN_CORPUS_COUNT = 5
MIN_WORD_LEN = 3

# Signs do not carry these. Their presence means the crop was cut through a
# glyph or the reader invented punctuation: nova], Imax]_, Auto €008b.
JUNK = re.compile(r"[\[\]_{}|\\<>~^`@#$%€£+*=]")


def looks_like_sign_text(text: str) -> bool:
    if JUNK.search(text):
        return False
    letters = sum(c.isalpha() for c in text)
    # BZ4knh, 053kih: mostly digits or a digit wedged inside a word.
    return letters >= MIN_WORD_LEN and letters >= 0.6 * len(text.strip())


def restore(text: str, vocab: dict) -> tuple:
    """Return (restored_text, every_word_known, n_restored).

    A word is restored only when the lexicon holds exactly one accented form
    for it, or one form that is overwhelmingly the most common. Ambiguous
    words are left as the reader wrote them.
    """
    out = []
    known = True
    restored = 0
    pos = 0
    for m in WORD.finditer(text):
        out.append(text[pos:m.start()])
        word = m.group()
        forms = vocab.get(fold(word).lower())
        if len(word) < MIN_WORD_LEN or not forms or sum(forms.values()) < MIN_CORPUS_COUNT:
            known = False
            out.append(word)
        else:
            best, count = forms.most_common(1)[0]
            total = sum(forms.values())
            # Unambiguous, or dominant enough that Krstev's 95% applies.
            if len(forms) == 1 or count / total >= 0.9:
                shaped = (best.upper() if word.isupper()
                          else best.capitalize() if word[:1].isupper()
                          else best.lower())
                if shaped != word:
                    restored += 1
                out.append(shaped)
            else:
                out.append(word)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), known, restored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--min-height", type=int, default=MIN_HEIGHT)
    ap.add_argument("--min-width", type=int, default=MIN_WIDTH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gt = args.src / "gt.txt"
    if not gt.is_file():
        sys.exit(f"no gt.txt in {args.src}")

    print("building lexicon:")
    vocab = build_vocabulary(args.repo)

    rows = []
    for line in gt.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].strip():
            rows.append((parts[0], parts[1].strip()))

    # gt.txt may list a PNG more than once (pass-16 duplicated rows to balance
    # the diacritic rate). Restoration makes that unnecessary, so collapse
    # back to one row per image and let the corrected labels carry the rate.
    seen = set()
    unique = []
    for name, text in rows:
        if name not in seen:
            seen.add(name)
            unique.append((name, text))

    kept, dropped, total_restored = [], Counter(), 0
    for name, text in unique:
        if not looks_like_sign_text(text):
            dropped["not-sign-text"] += 1
            continue
        fixed, known, n = restore(text, vocab)
        if not known:
            dropped["not-in-lexicon"] += 1
            continue
        png = args.src / name
        if not png.is_file():
            dropped["png-missing"] += 1
            continue
        w, h = Image.open(png).size
        if h < args.min_height or w < args.min_width:
            dropped["too-small"] += 1
            continue
        total_restored += n
        kept.append((name, fixed))

    def rate(rs):
        if not rs:
            return 0.0
        return 100 * sum(1 for _, t in rs if DIACRITICS.search(t)) / len(rs)

    print(f"\n{len(rows):,} gt rows -> {len(unique):,} unique images")
    for reason, n in dropped.most_common():
        print(f"  drop {n:>6,}  {reason}")
    print(f"  keep {len(kept):>6,}")
    print(f"\ndiacritics restored on {total_restored:,} words")
    print(f"diacritic rows: {rate(unique):.1f}% before -> {rate(kept):.1f}% after "
          f"(human labels: 10.6%)")

    if args.dry_run:
        print("\ndry run, nothing written")
        for name, text in kept[:15]:
            print(f"  {text}")
        return 0

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    with (args.out / "gt.txt").open("w", encoding="utf-8") as fh:
        for name, text in kept:
            shutil.copy2(args.src / name, args.out / name)
            fh.write(f"{name}\t{text}\n")
    print(f"\nwrote {args.out}: {len(kept):,} crops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
