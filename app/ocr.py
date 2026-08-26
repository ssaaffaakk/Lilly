#!/usr/bin/env python3
"""Photo scan: image of Bosnian text -> text.

Bosnian ("bs") is supported directly and phone photos are handled well
(neural text detector). Weights are read from models/lilly/read/.

Usage:
    python3 app/ocr.py photo.jpg
"""
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


def get_reader():
    global _reader, _allowlist
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                reader = easyocr.Reader(["bs", "en"], gpu=False,
                                        model_storage_directory=str(READ_DIR),
                                        download_enabled=False)
                # Stop loudly rather than read Bosnian without its own letters.
                # If the weights are ever swapped for a set that cannot produce
                # these, silence would look like a reader that reads badly.
                cannot = [c for c in BOSNIAN_LETTERS if c not in reader.character]
                if cannot:
                    raise RuntimeError(
                        f"the recogniser cannot produce {''.join(cannot)} — these "
                        f"weights are not the Latin set Lilly reads Bosnian with")
                _allowlist = "".join(sorted(set(reader.lang_char)
                                            | set(BOSNIAN_LETTERS)))
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


def scan(image_path: str) -> str:
    image = load_within_limits(image_path)
    with _read_lock:
        results = get_reader().readtext(image, detail=0, paragraph=True,
                                       allowlist=_allowlist)
    return "\n".join(results).strip()


def main() -> int:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        print("usage: python3 app/ocr.py <image-file>", file=sys.stderr)
        return 1
    print(scan(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
