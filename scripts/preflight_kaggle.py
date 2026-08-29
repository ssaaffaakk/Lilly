#!/usr/bin/env python3
"""Preflight checks before pushing Kaggle notebooks. Run via scripts/state.py or directly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEECH = REPO / "training" / "Lilly_Speech_Kaggle.ipynb"
OCR = REPO / "training" / "Lilly_OCR_Kaggle.ipynb"


def read_nb(path: Path) -> str:
    return "\n".join("".join(c.get("source", [])) for c in json.loads(path.read_text())["cells"])


def fail(msg: str) -> None:
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_speech(text: str) -> None:
    if 'model.safetensors").is_file(), "nothing was trained"' in text:
        fail("speech notebook still asserts single model.safetensors")
    if "model*.safetensors" not in text and "glob(" not in text:
        fail("speech notebook missing sharded-weight check")
    if "lilly-listen-trained.zip" not in text:
        fail("speech notebook missing early zip of listen-trained")


def check_ocr(text: str) -> None:
    if "from data.scripts import" in text:
        fail("OCR notebook imports data.scripts as package — will break on Kaggle")
    if "--extra-index-url" in text or 'pip install"*, "torch"' in text:
        fail("OCR notebook must not pip-install CPU torch")
    if "read-trained.zip" not in text:
        fail("OCR notebook missing early zip of read-trained.pth")
    if "check=False" not in text and "returncode" not in text:
        fail("OCR notebook must not treat train_ocr exit 1 as failure when keep-trained exists")
    if "latin_g2.pth" not in text:
        fail("OCR notebook missing base weight check")
    if "user_network" not in text:
        fail("OCR notebook missing user_network/lilly.yaml paths")
    if "/kaggle/temp" not in text:
        fail("OCR notebook must clone to scratch — a clone in /kaggle/working "
             "floods Output and the weights zip never downloads")
    if "--quick-test" not in text:
        fail("OCR notebook missing GPU training smoke (--quick-test) before long run")
    if '"--epochs", "7"' not in text and '"--epochs", "10"' not in text:
        fail("OCR notebook should train 7+ epochs for heavy pass-2")
    if "--weights" not in text:
        fail("OCR notebook missing --weights for pass-2 continuation")
    if "rglob(\"lilly.pth\")" not in text and "rglob('lilly.pth')" not in text:
        fail("OCR notebook must search /kaggle/input for lilly.pth (zip or nested)")


def check_train_ocr() -> None:
    text = (REPO / "training" / "train_ocr.py").read_text(encoding="utf-8")
    if "shutil.copy(weights, app_weights)" in text:
        fail("train_ocr installs the starting checkpoint as lilly.pth — "
             "Kaggle pass-1 shipped stock latin_g2 (md5 46986913)")


def main() -> int:
    for path, fn in ((SPEECH, check_speech), (OCR, check_ocr)):
        if not path.is_file():
            fail(f"missing {path}")
        fn(read_nb(path))
    check_train_ocr()
    print("preflight ok: speech + OCR notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
