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

_reader = None
_reader_lock = threading.Lock()
_read_lock = threading.Lock()


class ImageTooLarge(BadInput):
    """Raised when an image is too big to be worth decoding."""


class UnreadableImage(BadInput):
    """Raised when the upload is not an image we can open at all."""


def get_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                _reader = easyocr.Reader(["bs", "en"], gpu=False,
                                         model_storage_directory=str(READ_DIR),
                                         download_enabled=False)
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
        results = get_reader().readtext(image, detail=0, paragraph=True)
    return "\n".join(results).strip()


def main() -> int:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        print("usage: python3 app/ocr.py <image-file>", file=sys.stderr)
        return 1
    print(scan(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
