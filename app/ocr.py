#!/usr/bin/env python3
"""Photo scan: image of Bosnian text -> text.

Bosnian ("bs") is supported directly and phone photos are handled well
(neural text detector). Weights are read from models/lilly/read/.

Usage:
    python3 app/ocr.py photo.jpg
"""
import sys
from pathlib import Path

from app.lilly import READ_DIR

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["bs", "en"], gpu=False,
                                 model_storage_directory=str(READ_DIR),
                                 download_enabled=False)
    return _reader


def scan(image_path: str) -> str:
    results = get_reader().readtext(image_path, detail=0, paragraph=True)
    return "\n".join(results).strip()


def main() -> int:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        print("usage: python3 app/ocr.py <image-file>", file=sys.stderr)
        return 1
    print(scan(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
