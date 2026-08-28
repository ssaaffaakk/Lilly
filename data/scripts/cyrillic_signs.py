"""Cyrillic for the photo reader: what we have, and what a transliteration adds.

Two jobs, both read-only by default. Nothing here writes to the corpus, runs the
generator, or trains anything.

  census        what the hand-transcribed crops actually contain, per letter
  transliterate what the existing toponym list would yield in Cyrillic

WHY THIS EXISTS. The identified next move for this lane is fine-tuning
`cyrillic_g2` on the Cyrillic crops already transcribed. Two claims were being
carried into that plan unverified, and both were wrong in ways that would have
mattered:

  - "276 Cyrillic crops". It is 272. Small, but it is the training set size.
  - "all twelve of Ђ Ј Љ Њ Ћ Џ verified present". True of `cyrillic_g2`'s
    OUTPUT CLASSES and not of our data, where Џ has zero examples and Ђ and Љ
    have three each. Read together those two sentences plan a run that trains a
    class with nothing in it.

The generator cannot help as things stand: every one of the 40,727 lines in
`data/corpus-signs/toponyms-and-names.txt` is Latin, so it renders no Cyrillic
at all. Six of the seven faces in `data/fonts/` carry the full Serbian set, so
THE BLOCKER IS THE WORD LIST, NOT THE TYPEFACES — which is the cheap kind of
blocker. `transliterate` measures what closing it would buy.

Run:
    python3 data/scripts/cyrillic_signs.py census
    python3 data/scripts/cyrillic_signs.py transliterate
    python3 data/scripts/cyrillic_signs.py transliterate --write data/corpus-signs/toponyms-cyrillic.txt
"""

from __future__ import annotations   # Python 3.9 here: `Path | None` in a
                                     # signature is evaluated at def time
                                     # without it, and raises TypeError.

import argparse
import collections
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LABELS = REPO_ROOT / "data/ocr/crops/labels-human.tsv"
TOPONYMS = REPO_ROOT / "data/corpus-signs/toponyms-and-names.txt"

DIACRITICS = "čćđšžČĆĐŠŽ"

# Bosnian/Serbian Latin -> Cyrillic. The digraphs must be replaced before the
# single letters, or `lj` becomes `лј` instead of `љ`; and the two-capital forms
# must come before the title-case ones, or `LJ` becomes `Љj`. Hence the order.
DIGRAPHS = [("DŽ", "Џ"), ("Dž", "Џ"), ("dž", "џ"),
            ("LJ", "Љ"), ("Lj", "Љ"), ("lj", "љ"),
            ("NJ", "Њ"), ("Nj", "Њ"), ("nj", "њ")]

SINGLE = {
    "A": "А", "B": "Б", "V": "В", "G": "Г", "D": "Д", "Đ": "Ђ", "E": "Е",
    "Ž": "Ж", "Z": "З", "I": "И", "J": "Ј", "K": "К", "L": "Л", "M": "М",
    "N": "Н", "O": "О", "P": "П", "R": "Р", "S": "С", "T": "Т", "Ć": "Ћ",
    "U": "У", "F": "Ф", "H": "Х", "C": "Ц", "Č": "Ч", "Š": "Ш",
    "a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "đ": "ђ", "e": "е",
    "ž": "ж", "z": "з", "i": "и", "j": "ј", "k": "к", "l": "л", "m": "м",
    "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т", "ć": "ћ",
    "u": "у", "f": "ф", "h": "х", "c": "ц", "č": "ч", "š": "ш",
}

# THE TWO PREDICATES THAT DECIDE WHAT IS DROPPED. Stated as constants because
# the counts they produce go into a plan, and a count nobody can re-derive is a
# claim rather than a measurement.
#
# ANGLICISED is the trap. The toponym list carries English-transliterated
# duplicates beside the native spellings -- `Abdulichi` next to `Abdići`,
# `Akhmichi` next to `Ahmići`. Transliterating those character by character
# yields `Абдулицхи`: `цх` is not a Bosnian letter pair and appears on no sign
# in the country. Feeding them to the generator spends real training capacity
# teaching sequences the language does not contain, which is a self-inflicted
# version of the exact problem the run exists to fix.
#
# `yu|ya|ye` catch Cyrillic-through-English spellings (Yugoslavia-era forms),
# and `ii\b` catches Russian-style endings. They are in the pattern because
# they are in the data; dropping them from the pattern changes the count.
ANGLICISED = re.compile(r"ch|sh|zh|kh|ts|yu|ya|ye|ii\b", re.IGNORECASE)
NATIVE = re.compile(f"[{DIACRITICS}]")

# Two entries are wrapped in double parentheses -- `(( Vranovic ))`. Editorial
# marks, not names. Matched on the doubled bracket rather than on any bracket,
# because single parentheses appear inside legitimate names.
BRACKETED = "(("


def is_cyrillic(text: str) -> bool:
    return any("CYRILLIC" in unicodedata.name(ch, "") for ch in text)


def transliterate(text: str) -> str:
    for latin, cyrillic in DIGRAPHS:
        text = text.replace(latin, cyrillic)
    return "".join(SINGLE.get(ch, ch) for ch in text)


def is_anglicised(entry: str) -> bool:
    """An English-transliterated duplicate, not a native spelling.

    The diacritic test is the discriminator: `Ahmići` contains `ch` inside
    nothing and carries `ć`, so it stays; `Akhmichi` carries no native letter at
    all, so it goes. Without that condition the filter eats correct names.
    """
    return bool(ANGLICISED.search(entry)) and not NATIVE.search(entry)


def load_labels() -> list:
    rows = [line.rstrip("\n").split("\t") for line in
            LABELS.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if len(r) == 2]


def census() -> int:
    """What the hand-transcribed crops contain. Reads labels-human.tsv alone."""
    rows = load_labels()
    print(f"usable hand-transcribed labels: {len(rows)}")

    cyrillic = [r for r in rows if is_cyrillic(r[1])]
    letters_only_cyrillic = [
        r for r in rows
        if [c for c in r[1] if c.isalpha()]
        and all("CYRILLIC" in unicodedata.name(c, "")
                for c in r[1] if c.isalpha())]
    non_latin = [
        r for r in rows
        if any(c.isalpha() and "LATIN" not in unicodedata.name(c, "")
               for c in r[1])]

    print(f"\n  containing any Cyrillic letter : {len(cyrillic)} "
          f"({100 * len(cyrillic) / len(rows):.1f}%)")
    print(f"  entirely Cyrillic (has letters) : {len(letters_only_cyrillic)}")
    print(f"  containing any NON-LATIN letter : {len(non_latin)}  "
          f"<- what a Latin-only model cannot represent")
    print(f"  non-Latin but not Cyrillic      : "
          f"{len([r for r in non_latin if not is_cyrillic(r[1])])}")

    # Concentration decides how the fine-tune must be split. 97 crops off one
    # information board share font, lighting, camera and repeated words, so a
    # split by crop trains and tests on the same board.
    sources = collections.Counter(
        re.sub(r"_\d+\.png$", "", r[0]) for r in cyrillic)
    counts = sorted(sources.values(), reverse=True)
    print(f"\n  from {len(sources)} distinct source photographs")
    for name, n in sources.most_common(5):
        print(f"    {n:4d}  {100 * n / len(cyrillic):5.1f}%  {name[:60]}")
    print(f"    top five combined: {100 * sum(counts[:5]) / len(cyrillic):.1f}%")
    print(f"    median {counts[len(counts) // 2]} crops per source, "
          f"{sum(1 for v in counts if v == 1)} singletons")
    print("    -> SPLIT BY SOURCE PHOTOGRAPH, NOT BY CROP")

    seen = collections.Counter()
    for _, text in cyrillic:
        for ch in text:
            if "CYRILLIC" in unicodedata.name(ch, ""):
                seen[ch] += 1
    print("\n  the letters that make this alphabet Bosnian rather than Russian:")
    for upper, lower in zip("ЂЈЉЊЋЏ", "ђјљњћџ"):
        print(f"    {upper} {seen[upper]:4d}   {lower} {seen[lower]:4d}")

    print("\n  diacritics, on the Latin side (upper case matters: a recogniser")
    print("  treats Đ and đ as different classes):")
    marks = collections.Counter()
    for _, text in rows:
        for ch in unicodedata.normalize("NFC", text):
            if ch in DIACRITICS:
                marks[ch] += 1
    with_any = sum(1 for _, t in rows
                   if any(c in DIACRITICS for c in unicodedata.normalize("NFC", t)))
    print(f"    labels carrying any of č ć đ š ž: {with_any} "
          f"({100 * with_any / len(rows):.1f}%)")
    for lower in "čćđšž":
        upper = lower.upper()
        print(f"    {lower} {marks[lower]:4d}   {upper} {marks[upper]:4d}   "
              f"total {marks[lower] + marks[upper]:4d}")
    return 0


def transliterate_report(write: Path | None) -> int:
    lines = [l.strip() for l in
             TOPONYMS.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{TOPONYMS.relative_to(REPO_ROOT)}: {len(lines)} entries, "
          f"{sum(1 for l in lines if is_cyrillic(l))} containing Cyrillic")

    anglicised = [l for l in lines if is_anglicised(l)]
    bracketed = [l for l in lines if BRACKETED in l]
    dropped = set(anglicised) | set(bracketed)
    kept = [l for l in lines if l not in dropped]

    print(f"\n  dropped, anglicised duplicates : {len(anglicised)} "
          f"({100 * len(anglicised) / len(lines):.1f}%)")
    print(f"    predicate: re.search(r'{ANGLICISED.pattern}', entry, re.I) "
          f"and not re.search(r'[{DIACRITICS}]', entry)")
    print(f"    examples: {anglicised[:6]}")
    print(f"  dropped, '((' editorial marks  : {len(bracketed)} {bracketed}")
    print(f"  kept                           : {len(kept)} "
          f"({100 * len(kept) / len(lines):.1f}%)")

    out = [transliterate(l) for l in kept]
    rare = "ЂЈЉЊЋЏђјљњћџ"
    occurrences = collections.Counter()
    containing = collections.Counter()
    for s in out:
        for ch in set(s) & set(rare):
            containing[ch] += 1
        for ch in s:
            if ch in rare:
                occurrences[ch] += 1

    print("\n  what the transliteration yields, against what the real crops have:")
    print("    letter   occurrences   entries   in real crops")
    have = {"Ђ": 3, "Ј": 39, "Љ": 3, "Њ": 14, "Ћ": 6, "Џ": 0,
            "ђ": 3, "ј": 43, "љ": 8, "њ": 12, "ћ": 13, "џ": 2}
    for ch in rare:
        print(f"      {ch}      {occurrences[ch]:8d}  {containing[ch]:8d}"
              f"        {have[ch]:4d}")
    five = sum(1 for s in out if any(c in "ЂЉЊЋЏђљњћџ" for c in s))
    print(f"\n  entries containing at least one of Ђ Љ Њ Ћ Џ: {five}")
    print(f"  Џ goes from ZERO real examples to {containing['Џ'] + containing['џ']} entries.")

    if write:
        # Only ever a NEW file. The Latin corpus is what the shipped generator
        # reads and overwriting it would silently change every future synthetic
        # run, including reruns of ones already reported.
        if write.exists():
            print(f"\nrefusing to overwrite {write}", file=sys.stderr)
            return 1
        write.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\nwrote {len(out)} transliterated entries to "
              f"{write.relative_to(REPO_ROOT)}")
    else:
        print("\nNothing written. Pass --write PATH to emit, to a new file only.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("census", help="what the transcribed crops contain")
    tr = sub.add_parser("transliterate", help="what the toponym list would yield")
    tr.add_argument("--write", type=Path, help="emit to a new file (never overwrites)")
    args = ap.parse_args()
    if args.cmd == "census":
        return census()
    return transliterate_report(args.write)


if __name__ == "__main__":
    raise SystemExit(main())
