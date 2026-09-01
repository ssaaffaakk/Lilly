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


def check_offload(text: str, name: str) -> None:
    """agentic-kaggle-skill logs + trainproof tee. NVIDIA submit is not this job."""
    if "competitions submit" in text or "kaggle kernels submit" in text:
        fail(f"{name}: competition submit is NVIDIA/kaggle-skill work, not a train notebook")
    if 'Path("/kaggle/working/stdout.txt")' not in text:
        fail(f"{name}: run() must tee child stdout to /kaggle/working/stdout.txt")
    if "subprocess.Popen" not in text:
        fail(f"{name}: run() must Popen+tee — subprocess.run hides train logs "
             "(OCR pass-7c / speech half-1 hole)")
    if "from training.kaggle_offload import Offload" not in text:
        fail(f"{name}: must import Offload after clone")
    if "experiment_log.json" not in text:
        fail(f"{name}: must write experiment_log.json")
    if "check_trainproof" not in text:
        fail(f"{name}: must scan the tee for NaN/0-grad before packaging")


def check_speech(text: str) -> None:
    check_offload(text, "speech half-1")
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
    check_offload(text, "speech half-2")
    if "require_wer" not in text or "speech-wer.json" not in text:
        fail("speech half-2 must write speech-wer.json and require_wer before "
             "lilly-listen.zip (trainsafe analog: scored nothing is not shippable)")
    wer_at = text.find("require_wer")
    zip_at = text.find("/kaggle/working/lilly-listen.zip")
    if wer_at < 0 or zip_at < 0 or zip_at < wer_at:
        fail("speech half-2 must require_wer before zipping lilly-listen.zip")
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
    check_offload(text, "OCR")
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
    if '"--epochs", "5"' in text:
        fail("OCR must not 5-epoch human-only (pass-10 overfit 41.7%→39.4%)")
    if '"--epochs", "2"' not in text or '"--epochs", "3"' not in text:
        fail("OCR pass-11 must train plates 2 epochs then human 3 epochs")
    if text.count("training/train_ocr.py") < 3:
        fail("OCR pass-11 needs quick-test + plates stage + human stage")
    if "--no-install" not in text or "--checkpoint" not in text:
        fail("OCR pass-11 stage 1 must --no-install --checkpoint (plates are not an install)")
    if "read-stage1.pth" not in text:
        fail("OCR pass-11 must write a stage-1 checkpoint for the human half to load")
    if "plate_lines" not in text or "human_lines" not in text:
        fail("OCR pass-11 must split plates and human into two train lists")
    if "two stages, not mixed" not in text:
        fail("OCR pass-11 must not mix plates and human in one train_ocr")
    if "REAL_REPEAT = 4" in text:
        fail("OCR still repeats human ×4 — pass-10 overfit that way")
    if "share >= 0.45" in text:
        fail("OCR still uses pass-9 45% human + letter-dense plates — that spent letters")
    if "share >= 0.99" in text:
        fail("OCR still uses pass-10 single-stage human-only")
    if "letter_score" in text:
        fail("OCR still sorts plates by diacritic count into train — pass-9 spent letters that way")
    if "mixed_lines = human * REAL_REPEAT + other" in text:
        fail("OCR still adds plates to train — pass-11 trains one part then the other")
    if "heavy-pass12" not in text:
        fail("OCR notebook still labelled an older pass (want heavy-pass12)")
    if "pass-10 already refused on this 1,294-crop set" in text:
        fail("OCR notebook still SystemExits pass-10 — pass-11 is two sequential trains")
    if 'LILLY_RUN_ID"] = "heavy-pass10"' in text:
        fail("OCR notebook still launches pass-10 — human-only already overfit")
    if "heavy-pass9" in text:
        fail("OCR notebook still launches pass-9 — crop gate already refused that mix")
    if '"--count", "50000"' in text:
        fail("OCR pass-8 must not regenerate 50k synthetic on Kaggle — "
             "attach lilly-ocr-sign-letters (pass-7c mix already refused)")
    if "data/scripts/generate_ocr_photos.py" in text:
        fail("OCR pass-8 must not run generate_ocr_photos.py "
             "(pass-7c photo-style + harvest already refused on photographs)")
    if 'LILLY_RUN_ID"] = "heavy-pass8"' in text or "heavy-pass8" in text:
        fail("OCR notebook still launches pass-8 — crop gate already refused that mix")
    if "heavy-pass7c" in text:
        fail("OCR notebook still launches pass-7c — that recipe already lost on photographs")
    if "AUTO_MIN_CONF =" in text or "AUTO_CAP_MULT =" in text:
        fail("OCR pass-8 must not EasyOCR-crop harvest (AUTO_* knobs are pass-7c)")
    if '"-u"' not in text and "'-u'" not in text:
        fail("OCR train_ocr must run unbuffered (python -u) so Kaggle logs show loss")
    if "--weights" not in text:
        fail("OCR notebook missing --weights to continue from the installed reader")
    if "rglob(\"lilly.pth\")" not in text and "rglob('lilly.pth')" not in text:
        fail("OCR notebook must search /kaggle/input for lilly.pth (zip or nested)")
    if "lilly-ocr-crops" not in text:
        fail("OCR notebook must attach/copy real crops (lilly-ocr-crops)")
    if "lilly-ocr-sign-letters" not in text:
        fail("OCR pass-8 must attach sign-letter plates (lilly-ocr-sign-letters)")
    if "Sign-letter plates are required" not in text:
        fail("OCR pass-8 must SystemExit when sign-letters are missing")
    if "no real crop images in clone — synthetic only" in text:
        fail("OCR notebook must not silently fall back to synthetic-only on Kaggle")
    if "training on human crops + synthetic only" in text:
        fail("OCR notebook still has the pass-7c synthetic-only fallback string")
    if "Harvest photos are required" in text:
        fail("OCR pass-8 must not require harvest — pass-7c already refused on photographs")
    if '"diacritic"' not in text or "invented" not in text:
        fail("OCR photograph gate must check diacritic % and invented count, not only pooled")
    if "png.name.startswith(\"syn\")" not in text and "png.name.startswith('syn')" not in text:
        fail("OCR crop copy must skip syn* so sign-letter plates do not land in crops/")


def check_train_ocr() -> None:
    text = (REPO / "training" / "train_ocr.py").read_text(encoding="utf-8")
    if "shutil.copy(weights, app_weights)" in text:
        fail("train_ocr installs the starting checkpoint as lilly.pth — "
             "Kaggle pass-1 shipped stock latin_g2 (md5 46986913)")
    if "valid_real" not in text or "after_syn" not in text:
        fail("train_ocr must split real vs synthetic valid (pass-5 pooled gate refused a photo run)")
    if "diacritic crops" not in text:
        fail("train_ocr must print the ~23 real crops that carry the 25 letters "
             "(64%→60% is 16/25→15/25; the aggregate hid which word moved)")
    if "--no-install" not in text or "--checkpoint" not in text:
        fail("train_ocr must support a non-install stage checkpoint "
             "(pass-11 plates half is not an install)")
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
    if '"needs_ocr_harvest": True' in kaggle_train:
        fail("OCR job must not require harvest on pass-8 (pass-7c already refused)")
    if "needs_ocr_sign_letters" not in kaggle_train or "push_ocr_sign_letters" not in kaggle_train:
        fail("kaggle_train.py must push lilly-ocr-sign-letters for pass-8")
    ocr_nb = read_nb(OCR)
    if "heavy-pass11" in ocr_nb and "pass-11 already refused" not in kaggle_train:
        fail("kaggle_train.py must refuse to launch pass-11 "
             "(crop gate already refused; RESULTS-ocr-pass11.md)")
    poller = (REPO / "scripts" / "kaggle_poll.py").read_text(encoding="utf-8")
    ocr_job = poller.split('"ocr":', 1)[1][:400]
    if "lilly-read.zip" not in ocr_job:
        fail("OCR poller must treat lilly-read.zip as the done artefact")
    if "lilly-read-trained.zip" in ocr_job:
        fail("OCR poller must not treat lilly-read-trained.zip as done — "
             "that zip lands before the photograph gate")
    speech_train = (REPO / "training" / "train_speech.py").read_text(encoding="utf-8")
    if "--keep-adapter" not in speech_train or "resume_from_checkpoint" not in speech_train:
        fail("train_speech.py must support --keep-adapter and resume_from_checkpoint")
    enc = speech_train.find("the encoder is not learning")
    if enc < 0 or "raise SystemExit" not in speech_train[max(0, enc - 80):enc + 80]:
        fail("train_speech must stop when the encoder gets no gradient, not warn and continue")
    if "FiniteLossCheck" not in speech_train or "logging_nan_inf_filter=False" not in speech_train:
        fail("train_speech must stop on NaN/Inf loss, not filter it out of the log")
    eval_speech = (REPO / "training" / "evaluate_speech.py").read_text(encoding="utf-8")
    if '"--json"' not in eval_speech and "'--json'" not in eval_speech:
        fail("evaluate_speech.py must write --json so half-2 can gate on WER")
    offload = (REPO / "training" / "kaggle_offload.py").read_text(encoding="utf-8")
    if "experiment_log.json" not in offload or "scan_trainproof" not in offload:
        fail("training/kaggle_offload.py must write experiment_log.json and scan the tee")
    print("preflight ok: speech half-1 + half-2 + OCR notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
