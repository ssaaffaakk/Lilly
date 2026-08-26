#!/usr/bin/env python3
"""Which split a label text belongs to — the one answer every generator uses.

An OCR label gets rendered many times over: several synthetic variants from
data/scripts/generate_ocr_data.py, several photographed ones from
data/scripts/generate_ocr_photos.py, sometimes a hand-labelled crop as well.
Split the *rows* and those renderings scatter across both sides, so the
validation score partly measures memorisation. Deciding from the text itself
makes that impossible: the same string always answers the same way, whatever
drew it, in whatever order, however many times.

Both scripts import from here rather than each keeping a copy — a second copy
of this rule is a second rule, and the two only have to drift by one edit for
the leak to come back.
"""
import hashlib
import unicodedata

VALID_SHARE = 0.1


def split_key(text: str) -> str:
    """What counts as "the same label".

    NFC first: c-with-caron is one code point in some sources and a plain c
    followed by a combining caron in others, depending on which corpus file,
    editor or filesystem the text came through. The two look identical in
    gt.txt and hash differently, so without this the same visible word can
    still end up on both sides.
    """
    return unicodedata.normalize("NFC", text.strip())


def is_valid_text(text: str, valid_share: float = VALID_SHARE) -> bool:
    """True if this label belongs in valid — the same answer every time.

    A digest, not the builtin hash(): Python salts string hashing per process,
    so hash() would redraw the split on every run and re-running any generator
    would swap words between the sides. blake2b depends on nothing but the
    string — same answer across runs, machines and Python versions, with no
    shared state, no seed and no ordering for the scripts to agree on. That is
    what lets a generator write straight into train/ and valid/ without
    coordinating with anyone.

    Reading the first 8 bytes as a fraction of 2**64 spreads texts evenly over
    [0, 1), so the share below is a share of *distinct texts*, and the row
    share follows it as long as texts are rendered a similar number of times.
    """
    digest = hashlib.blake2b(split_key(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2 ** 64 < valid_share
