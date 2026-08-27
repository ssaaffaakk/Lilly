#!/usr/bin/env python3
"""Photo scan: image of Bosnian text -> text.

Bosnian ("bs") is supported directly and phone photos are handled well
(neural text detector). Weights are read from models/lilly/read/.

Usage:
    python3 app/ocr.py photo.jpg
"""
import os
import sys
import threading
from pathlib import Path

from app.lilly import READ_DIR, BadInput

# A photo's file size says nothing about what it costs to read: PNG compresses
# flat colour to almost nothing, so a 172 KB file can decode to a gigabyte. The
# limit that matters is pixels, and it has to be checked before decoding.
MAX_DECLARED_PIXELS = 50_000_000   # beyond this, refuse to decode at all
MAX_WORKING_PIXELS = 2_000_000     # above this, shrink first
# Measured on a 12 MP photo of a sign: reading it at 8 MP takes 22 s, at 4 MP
# 18 s, at 2 MP 9 s — and all three misread the same diacritic, so the detail
# was not buying accuracy. Half the wait is worth more than pixels nobody reads.

# easyocr's Bosnian character list is missing six of the language's own letters:
# c, c and d with their diacritics, plus the capitals, are absent from lang_char
# while s and z are present. That is not a cosmetic gap. easyocr turns the
# language list into an ignore set and subtracts it from the logits before
# decoding, so those six letters cannot come out at all -- not off a cleaner
# photo, not off a better-trained reader. Measured on a plain white image
# reading "ccdsz CCDSZ": the six came back as different letters every time.
# It means "Carsija" could never have read as "Čaršija" however good the photo,
# which is exactly the failure we were about to blame on the reader's training.
# The allowlist puts them back while keeping the rest of the language
# restriction, which is what stops Cyrillic and Vietnamese lookalikes.
BOSNIAN_LETTERS = "čćđšžČĆĐŠŽ"

_reader = None
_allowlist = None
_reader_lock = threading.Lock()
_read_lock = threading.Lock()


class ImageTooLarge(BadInput):
    """Raised when an image is too big to be worth decoding."""


class UnreadableImage(BadInput):
    """Raised when the upload is not an image we can open at all."""


def language_characters(languages) -> set:
    """The letters easyocr says these languages are written with."""
    from easyocr.config import BASE_PATH

    found = set()
    for language in languages:
        path = Path(BASE_PATH) / "character" / f"{language}_char.txt"
        if path.exists():
            found |= set(path.read_text(encoding="utf-8-sig").split())
    return found


def get_reader():
    global _reader, _allowlist
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr

                # Our fine-tuned reader ships as a network of its own rather
                # than as a replacement for latin_g2.pth. easyocr checks that
                # file's MD5 against the weights it published, so writing our
                # own there makes it refuse to load and report a corrupt
                # download — which is what happened the first time the trained
                # reader was installed: the photo endpoint returned 500 for
                # every image, and the training that had just improved word
                # accuracy from 67.1% to 86.8% had made the feature unusable.
                # LILLY_READER=stock loads easyocr's own Latin weights
                # instead, which is the reader this one was trained from. The
                # project keeps a "before" build of the translator and of the
                # listener and compares against them; the reader had no such
                # thing, so there was no way to ask whether its training helped
                # on anything but the synthetic set it was trained on. This is
                # that build. It is a measurement path, and it is here rather
                # than in the evaluation script because a comparison is only
                # worth anything if both sides go through the same code.
                trained = READ_DIR / "lilly.pth"
                network = READ_DIR / "user_network"
                stock = os.environ.get("LILLY_READER", "").lower() == "stock"
                extra = {} if stock else (
                    {"recog_network": "lilly",
                     "user_network_directory": str(network)}
                    if trained.exists() and (network / "lilly.yaml").exists() else {})
                reader = easyocr.Reader(["bs", "en"], gpu=False,
                                        model_storage_directory=str(READ_DIR),
                                        download_enabled=False, **extra)
                # Stop loudly rather than read Bosnian without its own letters.
                # If the weights are ever swapped for a set that cannot produce
                # these, silence would look like a reader that reads badly.
                cannot = [c for c in BOSNIAN_LETTERS if c not in reader.character]
                if cannot:
                    raise RuntimeError(
                        f"the recogniser cannot produce {''.join(cannot)} — these "
                        f"weights are not the Latin set Lilly reads Bosnian with")
                # Built from the language files rather than from
                # reader.lang_char, because the two are not the same thing once
                # a custom network is registered: easyocr then sets lang_char to
                # the network's whole character set, and the restriction that
                # keeps Cyrillic and Vietnamese lookalikes out of the guesses
                # quietly disappears. Reading the files keeps the restriction
                # identical whichever reader is loaded.
                # Letters are restricted to the two languages; everything that
                # is not a letter is not. The language files hold only letters —
                # easyocr keeps digits, punctuation and space in a separate
                # symbol set — so a list built from them alone silently forbids
                # every number on every sign. Measured: "Radovi 24. jula, 1995.
                # godine!" came back as "RadoviZAjulaIIIDgodinel", with both
                # dates destroyed.
                symbols = {c for c in reader.character if not c.isalpha()}
                _allowlist = "".join(sorted(language_characters(["bs", "en"])
                                            | set(BOSNIAN_LETTERS) | symbols))
                _reader = reader
    return _reader


def load_within_limits(image_path: str):
    """Open a photo at a size worth working on, or refuse it."""
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_DECLARED_PIXELS
    try:
        img = Image.open(image_path)         # reads the header, not the pixels
    except Exception as exc:
        raise UnreadableImage("that file is not an image we can read") from exc
    with img:
        width, height = img.size
        if width * height > MAX_DECLARED_PIXELS:
            raise ImageTooLarge(f"that image is {width}x{height}; too large to read")
        img = img.convert("RGB")
        if width * height > MAX_WORKING_PIXELS:
            scale = (MAX_WORKING_PIXELS / (width * height)) ** 0.5
            img = img.resize((max(int(width * scale), 1), max(int(height * scale), 1)))
        return np.asarray(img)


def read_regions(image, **kwargs):
    """Every text region the reader finds — the only place readtext is called.

    The allowlist is not optional and not the caller's business. Leaving it to
    each call site is how the label-making path in training/prepare_ocr_data.py
    spent its life producing first-draft labels that could not contain c, c or d
    with their diacritics: it built its reader from this module and then called
    readtext itself, so it inherited the check on the weights and none of the
    fix. One door, and forgetting stops being possible.
    """
    get_reader()                        # builds the reader and the allowlist
    with _read_lock:
        return _reader.readtext(image, allowlist=_allowlist, **kwargs)


def scan(image_path: str) -> str:
    image = load_within_limits(image_path)
    results = read_regions(image, detail=0, paragraph=True)
    return "\n".join(results).strip()


def main() -> int:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        print("usage: python3 app/ocr.py <image-file>", file=sys.stderr)
        return 1
    print(scan(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
