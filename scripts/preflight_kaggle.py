#!/usr/bin/env python3
"""Preflight checks before pushing Kaggle notebooks. Run via scripts/state.py or directly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEECH = REPO / "training" / "Lilly_Speech_Kaggle.ipynb"
SPEECH2 = REPO / "training" / "Lilly_Speech_Kaggle_Half2.ipynb"
OCR = REPO / "training" / "Lilly_OCR_Kaggle.ipynb"


def read_nb(path: Path) -> str:
    return "\n".join("".join(c.get("source", [])) for c in json.loads(path.read_text())["cells"])


def fail(msg: str) -> None:
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_speech(text: str) -> None:
    if "/kaggle/working/Lilly" in text or 'os.chdir("/kaggle/working")' in text:
        fail("speech half-1 must clone to /kaggle/temp — a clone in /kaggle/working "
             "floods Output and the zip never downloads")
    if "/kaggle/temp" not in text:
        fail("speech half-1 must clone to /kaggle/temp")
    if "--keep-adapter" not in text:
        fail("speech half-1 must --keep-adapter (Trainer checkpoint, not merged 3 GB)")
    if "lilly-listen-half1.zip" not in text:
        fail("speech half-1 must zip lilly-listen-half1.zip into /kaggle/working")
    if "evaluate_speech.py" in text:
        fail("speech half-1 must skip BEFORE/AFTER WER — that is what missed the 12h wall")
    if "listen-trained" in text:
        fail("speech half-1 must not merge/zip listen-trained (3 GB merged large-v3)")
    if "voxpopuli_hr" not in text:
        fail("speech pass-2 must pull voxpopuli_hr, not FLEURS hr alone")
    if "BOSNIAN_SHARE = 0.47" not in text:
        fail("speech mix share must be 0.47 once extra Croatian is in")
    if "SPEECH_EPOCHS = 1" not in text:
        fail("speech half-1 must set SPEECH_EPOCHS = 1 (two epochs miss the 12h wall)")


def check_speech_half2(text: str) -> None:
    if "/kaggle/working/Lilly" in text or 'os.chdir("/kaggle/working")' in text:
        fail("speech half-2 must clone to /kaggle/temp — a clone in /kaggle/working "
             "floods Output and the zip never downloads")
    if "/kaggle/temp" not in text:
        fail("speech half-2 must clone to /kaggle/temp")
    if "--resume" not in text:
        fail("speech half-2 must --resume the Trainer checkpoint, not --base merged weights")
    if "SPEECH_EPOCHS = 2" not in text:
        fail("speech half-2 must set SPEECH_EPOCHS = 2 so Trainer continues epoch 2")
    if "lilly-listen-half1.zip" not in text:
        fail("speech half-2 must read lilly-listen-half1.zip from /kaggle/input")
    if "lilly-listen.zip" not in text:
        fail("speech half-2 must zip lilly-listen.zip for the app")
    if "evaluate_speech.py" not in text:
        fail("speech half-2 must run AFTER WER — skipping it was how we shipped unmeasured weights")
    eval_at = text.find("evaluate_speech.py")
    zip_at = text.find("/kaggle/working/lilly-listen.zip")
    if eval_at < 0 or zip_at < 0 or zip_at < eval_at:
        fail("speech half-2 must score WER before zipping lilly-listen.zip")
    if "voxpopuli_hr" not in text:
        fail("speech half-2 must pull voxpopuli_hr, same mix as half-1")
    if "BOSNIAN_SHARE = 0.47" not in text:
        fail("speech mix share must be 0.47 once extra Croatian is in")


def check_ocr(text: str) -> None:
    if "from data.scripts import" in text:
        fail("OCR notebook imports data.scripts as package — will break on Kaggle")
    if "--extra-index-url" in text or 'pip install"*, "torch"' in text:
        fail("OCR notebook must not pip-install CPU torch")
    if "read-trained.zip" not in text:
        fail("OCR notebook missing early zip of read-trained.pth")
    if "NOT SHIPPABLE" in text:
        fail("OCR must not swallow a gate refusal as a complete kernel")
    if "check=False" in text:
        fail("OCR must not check=False — a failed train_ocr must stop the kernel, "
             "not zip weights and continue")
    if "latin_g2.pth" not in text:
        fail("OCR notebook missing base weight check")
    if "user_network" not in text:
        fail("OCR notebook missing user_network/lilly.yaml paths")
    if "/kaggle/temp" not in text:
        fail("OCR notebook must clone to scratch — a clone in /kaggle/working "
             "floods Output and the weights zip never downloads")
    if "--quick-test" not in text:
        fail("OCR notebook missing GPU training smoke (--quick-test) before long run")
    if '"--epochs", "5"' not in text:
        fail("OCR notebook should train 5 epochs for pass-7b")
    if '"--count", "50000"' not in text:
        fail("OCR notebook should generate 50k synthetic crops")
    if "generate_ocr_photos" not in text:
        fail("OCR pass-7 must generate photo-style synthetic on Kaggle")
    if "REAL_REPEAT = 2" not in text:
        fail("OCR pass-7 must repeat real train crops ×2, not ×6")
    if "heavy-pass7c" not in text:
        fail("OCR notebook still labelled an older pass")
    # Substring-only checks passed pass-7c even though the names were never
    # assigned — Kaggle then NameError'd on AUTO_CAP_MULT in cell 5d.
    if "AUTO_MIN_CONF =" not in text:
        fail("OCR harvest must assign AUTO_MIN_CONF (mentioning the name is not enough)")
    if "AUTO_CAP_MULT =" not in text:
        fail("OCR harvest must assign AUTO_CAP_MULT (mentioning the name is not enough)")
    if '"-u"' not in text and "'-u'" not in text:
        fail("OCR train_ocr must run unbuffered (python -u) so Kaggle logs show loss")
    if "--weights" not in text:
        fail("OCR notebook missing --weights to continue from the installed reader")
    if "rglob(\"lilly.pth\")" not in text and "rglob('lilly.pth')" not in text:
        fail("OCR notebook must search /kaggle/input for lilly.pth (zip or nested)")
    if "lilly-ocr-crops" not in text:
        fail("OCR notebook must attach/copy real crops (lilly-ocr-crops)")
    if "no real crop images in clone — synthetic only" in text:
        fail("OCR notebook must not silently fall back to synthetic-only on Kaggle")
    if 'HARVEST_EXT' not in text or ".jpg" not in text:
        fail("OCR harvest copy must include .jpg (Commons photos are JPEG, not PNG)")
    if "ocr-harvest is attached but only" not in text:
        fail("OCR notebook must refuse to train if harvest is attached but unused")
    if "training on human crops + synthetic only" in text:
        fail("OCR notebook silently falls back to synthetic-only — "
             "harvest is required for this pass; raise SystemExit instead")
    if "raise SystemExit" not in text or "Harvest photos are required" not in text:
        fail("OCR notebook harvest else-branch must raise SystemExit when no harvest "
             "photos are present, not print and continue")


def check_train_ocr() -> None:
    text = (REPO / "training" / "train_ocr.py").read_text(encoding="utf-8")
    if "shutil.copy(weights, app_weights)" in text:
        fail("train_ocr installs the starting checkpoint as lilly.pth — "
             "Kaggle pass-1 shipped stock latin_g2 (md5 46986913)")
    if "valid_real" not in text or "after_syn" not in text:
        fail("train_ocr must split real vs synthetic valid (pass-5 pooled gate refused a photo run)")
    min_idx = text.find("if steps < MIN_STEPS:")
    if min_idx < 0 or "return 0" in text[min_idx:min_idx + 220]:
        fail("a too-short OCR run must return 1, not 0 (exit 0 would package the old reader)")


def main() -> int:
    for path, fn in ((SPEECH, check_speech), (SPEECH2, check_speech_half2),
                     (OCR, check_ocr)):
        if not path.is_file():
            fail(f"missing {path}")
        fn(read_nb(path))
    check_train_ocr()
    kaggle_train = (REPO / "scripts" / "kaggle_train.py").read_text(encoding="utf-8")
    if "hv is None" not in kaggle_train:
        fail("kaggle_train.py must exit 1 when push_ocr_harvest returns None — "
             "currently it can skip harvest and still launch the notebook")
    speech_train = (REPO / "training" / "train_speech.py").read_text(encoding="utf-8")
    if "--keep-adapter" not in speech_train or "resume_from_checkpoint" not in speech_train:
        fail("train_speech.py must support --keep-adapter and resume_from_checkpoint")
    enc = speech_train.find("the encoder is not learning")
    if enc < 0 or "raise SystemExit" not in speech_train[max(0, enc - 80):enc + 80]:
        fail("train_speech must stop when the encoder gets no gradient, not warn and continue")
    if "FiniteLossCheck" not in speech_train or "logging_nan_inf_filter=False" not in speech_train:
        fail("train_speech must stop on NaN/Inf loss, not filter it out of the log")
    print("preflight ok: speech half-1 + half-2 + OCR notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
