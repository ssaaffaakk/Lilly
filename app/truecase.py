#!/usr/bin/env python3
"""Put shouted source text back into sentence case before it is translated.

Signs are written in capitals -- "ZABRANJEN ULAZ", "DANGER HIGH VOLTAGE KEEP
OUT" -- and this translator was trained on ordinary case, so an all-caps word
splits into sub-word pieces it barely knows and the answer comes out wrong.
Measured on the served build, ALL-CAPS source costs it 25.5 chrF2 against the
same sentence in ordinary case, and a crude restorer gives 23.1 of it back; it
is the largest single win in
docs/REPORT-what-would-raise-the-numbers-2026-09-13.md (section 1) and needs no
GPU. This module is that restorer.

It is staged, not shipped. The flag LILLY_TRUECASE turns it on (default off),
because the report's 25.5 chrF2 is a diagnostic measured on uppercased FLORES
sentences: real signs are shorter, and the honest bar for the product is a score
on photographs. That needs its own pre-registration and a photograph-bar
measurement first. Until then app.translate.Engine behaves exactly as it does
today unless the flag is set.
"""
import os
import re

# How loud is loud enough to act on. Two guards, both needed:
#
#   * at least MIN_LETTERS letters. A four-letter all-caps token is as likely to
#     be an acronym ("NASA", "USA") as a sign, and lowercasing it would corrupt
#     ordinary text; a sign worth recasing is usually longer than eight letters.
#   * at least UPPER_RATIO of the letters uppercase. A title, a name, an
#     ordinary sentence all sit far below this, so they pass through
#     byte-for-byte. The 0.8 rather than 1.0 tolerates the stray lowercase
#     letter an OCR pass leaves in an otherwise shouted line, which is exactly
#     the input this runs on.
MIN_LETTERS = 8
UPPER_RATIO = 0.8

# Names are the one thing a blanket lowercase destroys, and the report measured
# the residual loss after a crude restorer as "mostly proper nouns", so keeping
# them is what closes the last of the 25.5. Bosnian inflects a name heavily --
# Bosna, Bosne, Bosni, bosanski -- and each stem is listed with only the endings
# that keep it a *name*. That distinction is the whole point: -a/-e/-i/-u/-om
# are the noun cases, while -ski/-ska in the middle of a sentence is the derived
# adjective, which Bosnian writes lowercase ("govorim srpski", "bosanski jezik")
# even though a name is not. Matching the bare stem would capitalise those too;
# matching stem-plus-name-ending does not. Deliberately short and expandable; a
# name that is not here is lowercased like any other word. A middle dot stands
# for the bare stem ("mostar"), because an empty string is easy to miss.
#
# Days and months are deliberately absent: standard Bosnian/Serbian/Croatian
# writes them lowercase ("ponedjeljak", "januar"), so capitalising them would be
# the mistake, not the fix.
PROPER_FORMS = {
    "sarajev": ("o", "a", "u", "om"),
    "bosn": ("a", "e", "i", "u", "om"),
    "hercegov": ("ina", "ine", "ini", "inu", "inom"),
    "beograd": ("", "a", "u", "om"),
    "mostar": ("", "a", "u", "om"),
    "tuzl": ("a", "e", "i", "u", "om"),
    "zenic": ("a", "e", "i", "u", "om"),
    # Hrvatska/Srpska are the country names; hrvatski/srpski are the languages.
    "hrvat": ("", "a", "e", "i", "skoj", "sku", "skom", "skoga"),
    "srbij": ("a", "e", "i", "u", "om"),
    "srpsk": ("a", "e", "oj", "u", "om", "oga"),
    "jugoslavij": ("a", "e", "i", "u", "om"),
    "evrop": ("a", "e", "i", "u", "om"),
    "europ": ("a", "e", "i", "u", "om"),
}

# Words (letters only, so a diacritic stays inside its word), numbers, runs of
# punctuation, and whitespace. Splitting this way is what lets the restorer
# lowercase word by word instead of running str.lower() over the string, which
# would also flatten the capital that marks a sentence start.
_TOKEN = re.compile(r"[^\W\d_]+|\d+|[^\w\s]+|\s+", re.UNICODE)


def enabled() -> bool:
    """Whether LILLY_TRUECASE turns the recaser on, on app/ocr.py's pattern.

    Unset or "0" means off, which is the shipped default and the only value any
    serving path gets unless somebody sets it. An unrecognised value raises
    rather than guessing, the same way LILLY_READER does.
    """
    raw = os.environ.get("LILLY_TRUECASE", "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw not in ("1", "true", "yes", "on"):
        raise RuntimeError(f"LILLY_TRUECASE={raw!r}; want 1 or 0")
    return True


def is_predominantly_upper(text: str) -> bool:
    """True when the text is shouted (a sign) rather than merely capitalised.

    Only letters count: digits, punctuation and spacing carry no case and would
    otherwise make a blank form look calm. Text with too few letters, or text
    whose letters are not mostly uppercase, is left alone -- that is what keeps
    ordinary and mixed-case input byte-for-byte unchanged.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < MIN_LETTERS:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= UPPER_RATIO


def _capitalise(word: str) -> str:
    """First letter up, the rest as given -- never str.capitalize().

    str.capitalize() lowercases everything but the first character, which is
    fine until a word is a proper noun we meant to keep. Uppercasing one
    character leaves č ć đ š ž untouched wherever they are not first.
    """
    return word[:1].upper() + word[1:]


def _is_proper(lower: str) -> bool:
    """True when the lowercased word is a listed name in one of its own cases."""
    for stem, endings in PROPER_FORMS.items():
        if lower.startswith(stem) and lower[len(stem):] in endings:
            return True
    return False


def restore_sentence_case(text: str) -> tuple:
    """Lowercase shouted text and capitalise each sentence. -> (text, changed).

    Returns the input untouched -- and changed=False -- whenever the text is not
    predominantly uppercase, so a caller can apply this unconditionally and only
    an actually shouted string is rewritten. Pure: no model, no I/O.
    """
    if not is_predominantly_upper(text):
        return text, False

    out = []
    at_start = True      # a sentence begins at the next word
    prev_number = False  # the token just before this one was a number
    for token in _TOKEN.findall(text):
        if token.isspace():
            out.append(token)
            if "\n" in token:
                # A line break is a sentence boundary here, the same as in the
                # translator's own splitter: a sign's lines are separate notices.
                at_start = True
            prev_number = False
        elif token[0].isdigit():
            out.append(token)
            at_start = False
            prev_number = True
        elif token[0].isalpha():
            lower = token.lower()
            if at_start or _is_proper(lower):
                out.append(_capitalise(lower))
            else:
                out.append(lower)
            at_start = False
            prev_number = False
        else:
            out.append(token)
            # "5. maja 1990. godine" -- a period right after a digit is an
            # ordinal, not a full stop, so the word after it must stay lowercase.
            # In all-caps text there is no lowercase word to read the boundary
            # from, so the safe reading is the ordinal one; a real sentence that
            # ends in a number keeps its following word lowercase, which is the
            # small residual this restorer does not attempt (the report's
            # measurement has the same floor).
            if token[-1] in ".!?" and not (token == "." and prev_number):
                at_start = True
            prev_number = False

    restored = "".join(out)
    return restored, restored != text
