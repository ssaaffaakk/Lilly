#!/usr/bin/env python3
"""Generate the eval-only Listen and Read Kaggle notebooks."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
COMMIT_MARKER = "__LAUNCHER_GIT_COMMIT__"


SETUP = '''\
import hashlib, json, os, shutil, subprocess, sys, urllib.error, urllib.request, zipfile
from pathlib import Path
import torch
assert torch.cuda.is_available(), "Kaggle GPU is required for this heavy evaluation"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU = torch.cuda.get_device_name(0)
print(torch.cuda.device_count(), "GPU(s), using", GPU)

def reachable(url):
    try:
        urllib.request.urlopen(url, timeout=20).close()
    except urllib.error.HTTPError:
        pass
    except Exception as exc:
        raise SystemExit(f"network unavailable for {url}: {exc}")
for host in ("https://github.com", "https://pypi.org", "https://huggingface.co"):
    reachable(host)

TEE = Path("/kaggle/working/stdout.txt")
# Offload writes /kaggle/working/experiment_log.json beside the evidence zip.
TEE.parent.mkdir(parents=True, exist_ok=True)
def run(*cmd, quiet=False, env=None):
    line = "$ " + " ".join(str(x) for x in cmd)
    print(line, flush=True)
    with TEE.open("a", encoding="utf-8") as sink:
        sink.write(line + "\\n")
        child = subprocess.Popen([str(x) for x in cmd], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 env={**os.environ, **(env or {})})
        for output in child.stdout:
            if not quiet:
                print(output, end="", flush=True)
            sink.write(output)
        code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)
'''

CLONE = f'''\
EXPECTED_GIT_COMMIT = "{COMMIT_MARKER}"
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
CLONE = SCRATCH / "Lilly"
if CLONE.exists():
    shutil.rmtree(CLONE)
run("git", "clone", "-q", "https://github.com/ssaaffaakk/Lilly.git", str(CLONE))
os.chdir(CLONE)
run("git", "checkout", "-q", EXPECTED_GIT_COMMIT)
got_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if got_commit != EXPECTED_GIT_COMMIT:
    raise SystemExit(f"clone is {{got_commit}}, expected {{EXPECTED_GIT_COMMIT}}")
sys.path.insert(0, str(CLONE))
from training.kaggle_offload import Offload
OFF = Offload(JOB, os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "manual"))
OFF.hardware(GPU)
print("exact code commit", got_commit)
'''


SPEECH_PIP = '''\
packages = ["faster-whisper==1.2.1", "ctranslate2==4.8.1", "soundfile==0.14.0",
            "pyarrow==25.0.1", "huggingface-hub==1.31.0", "transformers==4.49.0"]
run(sys.executable, "-m", "pip", "install", "-q", *packages)
import ctranslate2, faster_whisper, soundfile
print("runtime", faster_whisper.__version__, ctranslate2.__version__, soundfile.__version__)
print("CUDA compute types", ctranslate2.get_supported_compute_types("cuda"))
'''

SPEECH_DATA = '''\
SOURCE = CLONE / "training/clean-eval/speech-source.json"
MANIFEST = CLONE / "training/clean-eval/speech-fleurs-bs-test.tsv"
source = json.loads(SOURCE.read_text(encoding="utf-8"))
if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != source["manifest_sha256"]:
    raise SystemExit("committed speech manifest does not match source record")
run(sys.executable, "training/fetch_pinned_fleurs_test.py", "--source", str(SOURCE),
    "--manifest", str(MANIFEST), "--out", str(CLONE / "data/speech"))
rows = (CLONE / "data/speech/test.tsv").read_text(encoding="utf-8").splitlines()
if len(rows) != 925:
    raise SystemExit(f"FLEURS test has {len(rows)} rows, not 925")
run(sys.executable, "training/probe_speech_clips.py", "--data", "data/speech/test.tsv",
    "--out", "/kaggle/working/speech-waveform-diagnostics.json")
OFF.metric("speech_clips", 925, stage="data")
OFF.metric("manifest_sha256", source["manifest_sha256"], stage="data")
'''

SPEECH_MODELS = '''\
PINS = {"listen-previous": "a76342f6ab59b382", "listen": "e6bb58483586b06c"}
BASES = {"listen-previous": "openai/whisper-small", "listen": "openai/whisper-large-v3"}
def listener_fingerprint(build):
    h = hashlib.md5()
    for name in sorted(p.name for p in build.iterdir()
                       if p.is_file() and p.name != "dataset-metadata.json"):
        h.update(name.encode()); h.update((build / name).read_bytes())
    return h.hexdigest()[:16]
attached = [p.parent for p in Path("/kaggle/input").rglob("built.json")
            if (p.parent / "model.bin").is_file()]
for label in ("listen-previous", "listen"):
    matches = [p for p in attached
               if json.loads((p / "built.json").read_text()).get("base") == BASES[label]]
    if len(matches) != 1:
        raise SystemExit(f"{label}: need exactly one attached {BASES[label]}, found {matches}")
    destination = CLONE / "models/lilly" / label
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(matches[0], destination,
                    ignore=shutil.ignore_patterns("dataset-metadata.json"))
    got = listener_fingerprint(destination)
    if got != PINS[label]:
        raise SystemExit(f"{label} fingerprint {got}, expected {PINS[label]}")
    print(label, json.loads((destination / "built.json").read_text()), got,
          (destination / "model.bin").stat().st_size)
'''

SPEECH_SCORE = '''\
GPU_ENV = {"LILLY_SPEECH_DEVICE": "cuda", "CT2_CUDA_ALLOW_FP16": "1"}
PREV = CLONE / "models/lilly/listen-previous"
SHIPPED = CLONE / "models/lilly/listen"
run(sys.executable, "training/speech_bench.py", "--clips", "all", "--limit", "8",
    "--model", str(PREV), "--model", str(SHIPPED),
    "--json", "/kaggle/working/speech-smoke.json", env=GPU_ENV)
GATE = Path("/kaggle/working/speech-gate.json")
run(sys.executable, "training/speech_bench.py", "--clips", "all",
    "--model", str(PREV), "--model", str(SHIPPED), "--json", str(GATE), env=GPU_ENV)
gate = json.loads(GATE.read_text())
if gate["n_clips"] != 925 or gate["n_clips_in_split"] != 925:
    raise SystemExit(f"speech gate scored partial data: {gate['n_clips']}/{gate['n_clips_in_split']}")
WER = Path("/kaggle/working/speech-legibility.json")
run(sys.executable, "training/speech_legibility_report.py", "--data", "data/speech/test.tsv",
    "--manifest", str(MANIFEST), "--cache", "bench/speech/.outputs.json",
    "--model", str(PREV), "--model", str(SHIPPED), "--json", str(WER),
    "--markdown", "/kaggle/working/speech-legibility.md")
# Supplementary only: pin the official Open ASR code, apply its multilingual
# normalizer/scoring to the same cached predictions, and upload nothing.
OPEN_ASR_COMMIT = "4ef4a26b8588ccc140868be36f8c34acc839afe6"
OPEN_ASR = SCRATCH / "open_asr_leaderboard"
if OPEN_ASR.exists():
    shutil.rmtree(OPEN_ASR)
run("git", "clone", "-q", "https://github.com/huggingface/open_asr_leaderboard.git", str(OPEN_ASR))
run("git", "-C", str(OPEN_ASR), "checkout", "-q", OPEN_ASR_COMMIT)
run(sys.executable, "-m", "pip", "install", "-q", "num2words==0.5.14")
OPEN_ASR_JSON = Path("/kaggle/working/open-asr-offline.json")
run(sys.executable, "training/open_asr_offline_score.py", "--predictions", str(WER),
    "--harness", str(OPEN_ASR), "--harness-commit", OPEN_ASR_COMMIT,
    "--json", str(OPEN_ASR_JSON))
'''

SPEECH_PACKAGE = '''\
prev = gate["listeners"]["listen-previous"]
shipped = gate["listeners"]["listen"]
checks = {
    "term_recall_not_below_baseline": shipped["term_recall"] >= prev["term_recall"],
    "croatian_substitution_not_above_baseline": shipped["croatian"] <= prev["croatian"],
}
verdict = "PASS" if all(checks.values()) else "FAIL"
decision = {"schema": 1, "artifact": "shipped lilly-listen-large-v3",
            "fingerprints": PINS, "registered_gates": checks, "verdict": verdict,
            "baseline": prev, "shipped": shipped}
DECISION = Path("/kaggle/working/speech-clean-eval-decision.json")
DECISION.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\\n")
print("registered gate verdict", verdict, checks)
OFF.check_trainproof(TEE)
files = [GATE, WER, DECISION, OPEN_ASR_JSON, Path("/kaggle/working/speech-legibility.md"),
         Path("/kaggle/working/speech-waveform-diagnostics.json"),
         CLONE / "bench/speech/.outputs.json", MANIFEST, SOURCE]
for path in files:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing result {path}")
ZIP = Path("/kaggle/working/lilly-listen-clean-eval.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.name)
if ZIP.stat().st_size < 20_000:
    raise SystemExit(f"result zip too small: {ZIP.stat().st_size}")
OFF.finish("complete", [ZIP.name])
print("wrote", ZIP, ZIP.stat().st_size, "bytes; eval only, no weights or defaults changed")
'''

OCR_PIP = '''\
run(sys.executable, "-m", "pip", "install", "-q", "paddlepaddle-gpu==3.3.1",
    "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/")
run(sys.executable, "-m", "pip", "install", "-q", "paddleocr==3.7.0",
    "easyocr==1.7.2", "opencv-contrib-python==4.10.0.84",
    "opencv-python-headless==4.10.0.84", "pillow==12.3.0")
import cv2, paddle, paddleocr
if not paddle.device.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
    raise SystemExit("paddlepaddle-gpu has no CUDA")
if paddle.__version__ != "3.3.1" or paddleocr.__version__ != "3.7.0" or cv2.__version__ != "4.10.0":
    raise SystemExit(f"OCR runtime drift: paddle={paddle.__version__}, "
                     f"paddleocr={paddleocr.__version__}, cv2={cv2.__version__}")
print("OCR runtime", paddle.__version__, paddleocr.__version__, cv2.__version__)
'''

OCR_DATA = '''\
MANIFEST = CLONE / "training/clean-eval/ocr-commons-40.tsv"
SOURCE = CLONE / "training/clean-eval/ocr-source.json"
source = json.loads(SOURCE.read_text(encoding="utf-8"))
if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != source["manifest_sha256"]:
    raise SystemExit("committed OCR manifest does not match source record")
PHOTOS = SCRATCH / "ocr-commons-originals"
if PHOTOS.exists():
    shutil.rmtree(PHOTOS)
run(sys.executable, "training/fetch_pinned_ocr_photos.py", "--manifest", str(MANIFEST),
    "--out", str(PHOTOS))
if len(list(PHOTOS.iterdir())) != 40:
    raise SystemExit("OCR source fetch is not exactly 40 originals")
# Exact shipped Paddle detector/recogniser bytes, attached as a private Kaggle
# dataset. The notebook does not trust the dataset's own manifest: the six
# expected hashes are fixed in committed code.
OCR_WEIGHT_HASHES = {
    "PP-OCRv6_medium_det/inference.pdiparams": "85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960",
    "PP-OCRv6_medium_det/inference.yml": "7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871",
    "PP-OCRv6_medium_det/inference.json": "0f1a7ec35da36173529c7a60238b7f7919e3831929c3f700ad90ad4896adecd5",
    "PP-OCRv6_medium_rec/inference.pdiparams": "1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319",
    "PP-OCRv6_medium_rec/inference.yml": "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
    "PP-OCRv6_medium_rec/inference.json": "0b2e25e990bd072f1bf77d59d67d508bce6c4bd44af6624e0fb27d6da2cd00e8",
}
weight_manifests = list(Path("/kaggle/input").rglob("weights-sha256.json"))
if len(weight_manifests) != 1:
    raise SystemExit(f"need one attached lilly-ocr-ppocrv6-shipped dataset, found {weight_manifests}")
attached = weight_manifests[0].parent
PADDLE_CACHE = SCRATCH / "paddlex-cache" / "official_models"
for relative, expected in OCR_WEIGHT_HASHES.items():
    source_weight = attached / relative
    if not source_weight.is_file() or hashlib.sha256(source_weight.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"attached shipped OCR weight mismatch: {relative}")
    target_weight = PADDLE_CACHE / relative
    target_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_weight, target_weight)
print("six shipped Paddle files verified and staged", PADDLE_CACHE)
OFF.metric("ocr_photos", 40, stage="data")
OFF.metric("manifest_sha256", source["manifest_sha256"], stage="data")
'''

OCR_SCORE = '''\
os.environ.update({"PADDLE_PDX_MODEL_SOURCE": "huggingface", "LILLY_READER": "paddle",
                   "LILLY_PADDLE_VERSION": "PP-OCRv6", "LILLY_PADDLE_REC_THRESH": "0.9",
                   "PADDLE_PDX_CACHE_HOME": str(PADDLE_CACHE.parent)})
for forbidden in ("LILLY_PADDLE_REC_DIR", "LILLY_PADDLE_CYRILLIC_RESCUE",
                  "LILLY_PADDLE_DET_SIDE_LEN"):
    os.environ.pop(forbidden, None)
from app import ocr
identity = ocr.reader_identity()
expected = "paddle:PP-OCRv6_medium_det+PP-OCRv6_medium_rec:3.7.0:rec>=0.9"
if identity != expected:
    raise SystemExit(f"reader is {identity}, expected {expected}")
print("shipped reader", identity)
CACHE = Path("/kaggle/working/ocr-reader-output.json")
RAW_JSON = Path("/kaggle/working/ocr-overall.json")
run(sys.executable, "training/evaluate_ocr.py", "--truth", "data/ocr/real-photos/truth.json",
    "--photos", str(PHOTOS), "--cache", str(CACHE),
    "--out", "/kaggle/working/ocr-overall.md", "--json", str(RAW_JSON))
raw = json.loads(RAW_JSON.read_text())
if raw["photographs"] != 28 or raw["reader_context"]["treatment"] != "shipped-scan-2mp-cap":
    raise SystemExit(f"OCR did not score the exact shipped path: {raw}")
COHORT_JSON = Path("/kaggle/working/ocr-legibility.json")
run(sys.executable, "training/ocr_legibility_report.py",
    "--truth", "data/ocr/real-photos/truth.json", "--manifest", str(MANIFEST),
    "--cache", str(CACHE), "--json", str(COHORT_JSON),
    "--markdown", "/kaggle/working/ocr-legibility.md")
cohorts = json.loads(COHORT_JSON.read_text())
if cohorts["reader"] != raw["reader"] or cohorts["all_40"]["photographs"] != 40:
    raise SystemExit("OCR cohort report does not cover the exact 40-photo reading cache")
OFF.metric("ocr_reader_fingerprint", raw["reader"], stage="eval")
OFF.metric("ocr_legible_recall", cohorts["legible_overall"]["recall"], stage="eval")
'''

OCR_PACKAGE = '''\
OFF.check_trainproof(TEE)
files = [CACHE, RAW_JSON, COHORT_JSON, Path("/kaggle/working/ocr-overall.md"),
         Path("/kaggle/working/ocr-legibility.md"), MANIFEST, SOURCE]
for path in files:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing result {path}")
ZIP = Path("/kaggle/working/lilly-read-clean-eval.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.name)
if ZIP.stat().st_size < 10_000:
    raise SystemExit(f"result zip too small: {ZIP.stat().st_size}")
OFF.finish("complete", [ZIP.name])
print("wrote", ZIP, ZIP.stat().st_size, "bytes; eval only, no training or default change")
'''


def notebook(markdown: str, job: str, cells: list[str]) -> dict:
    rendered = []
    rendered.append({"cell_type": "markdown", "metadata": {},
                     "source": [line + "\n" for line in markdown.splitlines()]})
    for code in cells:
        code = f'JOB = "{job}"\n' + code if code is SETUP else code
        ast.parse(code)
        rendered.append({"cell_type": "code", "execution_count": None, "metadata": {},
                         "outputs": [], "source": code.splitlines(keepends=True)})
    return {"cells": rendered, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4, "nbformat_minor": 5}


def write(name: str, body: dict) -> None:
    path = TRAINING / name
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def main() -> int:
    listen_md = """# Lilly Listen — clean shipped-model evaluation

Eval only: exact shipped `lilly-listen-large-v3` plus the registered small
baseline, 925 hash-pinned FLEURS Bosnian test recordings, product decode path,
WER by frozen legibility cohort, and the two registered language gates. No
training, install, product-default change, publication, or leaderboard upload.
"""
    read_md = """# Lilly Read — clean shipped-model evaluation

Eval only: shipped PaddleOCR PP-OCRv6 medium at confidence floor 0.9 through
`app.ocr.scan`, on 40 hash-pinned original-resolution Commons photographs.
Cohorts were frozen by human visual inspection before inference. No training,
install, product-default change, or publication.
"""
    write("Lilly_Listen_Clean_Eval_Kaggle.ipynb", notebook(
        listen_md, "listen-clean-eval",
        [SETUP, CLONE, SPEECH_PIP, SPEECH_DATA, SPEECH_MODELS, SPEECH_SCORE, SPEECH_PACKAGE]))
    write("Lilly_Read_Clean_Eval_Kaggle.ipynb", notebook(
        read_md, "read-clean-eval", [SETUP, CLONE, OCR_PIP, OCR_DATA, OCR_SCORE, OCR_PACKAGE]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
