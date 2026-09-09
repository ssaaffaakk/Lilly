#!/usr/bin/env python3
"""Build training/Lilly_Speak_BS_Kaggle.ipynb -- the Bosnian voice, trained and judged on a Kaggle GPU.

The notebook is generated rather than edited by hand so every cell is parsed
before it is committed and the fail-stop rules (docs/kaggle-notebooks.md) are
in one place that a diff can read. What it runs is pre-registered in
training/PREREGISTRATION.md, "v5 -- speak -- a Bosnian voice from FLEURS":

  1. FLEURS bs_ba, all three splits, whole; the shipped listener (fingerprint
     e6bb58483586b06c) from the lilly-listen-large-v3 dataset; the voice the
     app speaks with today (md5-checked); Piper's sr_RS checkpoint (md5-checked).
  2. cluster_speakers.py on the train split, prepare_speak_data.py, then
     piper.train warm-started from sr_RS with espeak-ng's bs, capped in epochs
     and in wall-clock; the last checkpoint is exported.
  3. The served speaker is chosen on VALID sentences; the judgment is on the
     test prefix (first 200 clips): before voice, candidate, human recordings,
     one listener, paired bootstrap over sentences.
  4. Results zip always after the judgment; voice zip only if the bar was cleared.

    python3 training/build_speak_bs_notebook.py
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "Lilly_Speak_BS_Kaggle.ipynb"
STAGING = HERE.parent / "models" / "kaggle-staging" / "lilly-speak-bs"

MARKDOWN_TOP = """# Lilly — a Bosnian voice from FLEURS (`speak-bs`)

**Job:** `speak-bs`. Pre-registered in `training/PREREGISTRATION.md`, "v5 — speak —
a Bosnian voice from FLEURS, written before any run". **Read it before launching.**

**What this run does.** Fine-tunes Piper's `sr_RS-serbski_institut-medium` voice
(the voice the app speaks Bosnian with today, filed under Serbian, trained on
Sorbian recordings) on the FLEURS bs_ba **train** clips, clustered into speakers
on this box. Exports the voice, chooses the served speaker on the **valid**
split, and judges it on the **test** prefix (first 200 clips, 167 sentences)
through the shipped listener against the voice the app speaks with today and
against the human recordings of the same sentences.

**Attach, before Save & Run All** (`scripts/kaggle_train.py speak-bs` does it):
**Add data → Datasets → `lilly-listen-large-v3`** — the shipped listener,
fingerprint `e6bb58483586b06c`, **required**; nothing is heard by any other ear.
Internet **On**: FLEURS, pip, Hugging Face (the sr_RS checkpoint, the before voice).

**Output.** `lilly-speak-bs-results.zip` after the judgment, whichever way it
fell. `lilly-speak-bs.zip` — the voice — **only** if it cleared the bar. ERROR or
CANCEL: nothing here is a result.
"""

CELL_SETUP = '''# 1. Stop here unless the machine is actually set up
import json, os, shutil, subprocess, sys, urllib.error, urllib.request, zipfile
from pathlib import Path

import torch
assert torch.cuda.is_available(), (
    "No GPU. Right panel -> Session options -> Accelerator -> GPU, then Save & Run All again.")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU = torch.cuda.get_device_name(0)
print(torch.cuda.device_count(), "GPU(s) visible, using:", GPU)

def reachable(url):
    try:
        urllib.request.urlopen(url, timeout=20).close()
    except urllib.error.HTTPError:
        pass
    except Exception as exc:
        raise SystemExit(f"Cannot reach {url} ({exc}). Right panel -> Session options -> Internet -> On.")
for host in ("https://github.com", "https://pypi.org", "https://huggingface.co",
             "https://datasets-server.huggingface.co"):
    reachable(host)
print("network ok")

# The child's stdout is NOT the Kaggle log. Tee everything into Output so a run
# whose trainer printed nothing cannot be mistaken for one that trained.
TEE = Path("/kaggle/working/stdout.txt")
TEE.parent.mkdir(parents=True, exist_ok=True)

def run(*cmd, quiet=False, env=None):
    line = "$ " + " ".join(str(c) for c in cmd)
    print(line, flush=True)
    with TEE.open("a", encoding="utf-8") as sink:
        sink.write(line + "\\n")
        child = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 env={**os.environ, **(env or {})})
        for out in child.stdout:
            if not quiet:
                print(out, end="", flush=True)
            sink.write(out)
        code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)
'''

CELL_CLONE = '''# 2. Get the Lilly code -- into scratch, never into Output
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
CLONE = SCRATCH / "Lilly"
subprocess.run(["rm", "-rf", str(CLONE)], check=True)
os.chdir(SCRATCH)
run("git", "clone", "-q", "https://github.com/ssaaffaakk/Lilly.git")
assert (CLONE / "training").is_dir(), "clone produced nothing"
os.chdir(CLONE)
print("working in", os.getcwd())

sys.path.insert(0, str(CLONE))
from training.kaggle_offload import Offload
OFF = Offload("speak-bs", os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "manual"))
OFF.hardware(GPU)
print("offload log started (experiment_log.json, metrics.jsonl):", OFF.body["git"])
'''

CELL_PIP = '''# 3. Install (~3 min): Piper with its training extras, the speaker encoder, the listener.
# Not torch: Kaggle's CUDA build stays, and a pip that replaced it would put the
# run on a card the trainer cannot see. Checked in a fresh interpreter afterwards,
# because the torch already imported in this one would hide the swap.
# onnxscript: torch's ONNX exporter imports it, and piper's train extra does not list it.
NEEDED = ["piper-tts[train]", "onnxscript", "resemblyzer", "setuptools<81", "scikit-learn",
          "faster-whisper", "soundfile", "huggingface_hub", "pyarrow"]
pins = {}
for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
    line = line.split("#")[0].strip()
    if "==" in line:
        pins[line.split("==")[0].strip().lower()] = line
wanted = [pins.get(n.split("[")[0].split("<")[0].lower(), n) for n in NEEDED]
print("installing:", wanted)
TORCH_BEFORE = torch.__version__
run(sys.executable, "-m", "pip", "install", "-q", *wanted)
probe = subprocess.run([sys.executable, "-c",
                        "import torch, lightning, piper, resemblyzer, faster_whisper; "
                        "print(torch.__version__, torch.cuda.is_available(), lightning.__version__)"],
                       text=True, capture_output=True)
print(probe.stdout.strip() or probe.stderr[-1200:])
fields = probe.stdout.split()
if probe.returncode or len(fields) < 3 or fields[0] != TORCH_BEFORE or fields[1] != "True":
    raise SystemExit(f"after pip: {fields[:2]} against torch {TORCH_BEFORE} before -- pip replaced "
                     "torch or an import fails; not training on a card the trainer cannot see")
# resemblyzer drags in the 2015 `typing` backport, which shadows the standard
# library's whenever the working directory is site-packages. Gone, before it can.
run(sys.executable, "-m", "pip", "uninstall", "-y", "-q", "typing")
'''

CELL_ALIGN = '''# 3b. Piper's monotonic alignment is a Cython extension the wheel does not
# carry built: piper1-gpl ships core.pyx in its source tree and a setup.py that
# expects it beside __init__.py, and the trainer imports the built module from a
# nested package. Fetched at the 1.8.0 tag, pinned by sha256, built into scratch,
# copied where the import looks. Without it training stops at the first batch,
# so a failed import stops the run here instead.
import hashlib
import piper
PYX_URL = ("https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/v1.8.0/"
           "src/piper/train/vits/monotonic_align/core.pyx")
PYX_SHA = "8640b303683823a4a1259179547ef476999b1cbb2e46ff656b970763cfbc1157"
MA = Path(piper.__file__).parent / "train" / "vits" / "monotonic_align"
inner = MA / "monotonic_align"
inner.mkdir(exist_ok=True)
(inner / "__init__.py").touch()
pyx = urllib.request.urlopen(PYX_URL, timeout=60).read()
if hashlib.sha256(pyx).hexdigest() != PYX_SHA:
    raise SystemExit("core.pyx from GitHub is not the 1.8.0 file the pre-registration pinned")
(inner / "core.pyx").write_bytes(pyx)
BUILD = SCRATCH / "ma-build"
run(sys.executable, "-c",
    "import sys, numpy; from setuptools import setup; from Cython.Build import cythonize; "
    "setup(name='monotonic_align', script_args=['build_ext', '--build-lib', sys.argv[1], "
    "'--build-temp', sys.argv[2]], ext_modules=cythonize(sys.argv[3], language_level=3), "
    "include_dirs=[numpy.get_include()])",
    str(BUILD / "lib"), str(BUILD / "tmp"), str(inner / "core.pyx"), quiet=True)
built_so = sorted((BUILD / "lib").rglob("core*.so"))
if len(built_so) != 1:
    raise SystemExit(f"expected one built core extension, found {built_so}")
shutil.copy(built_so[0], inner / built_so[0].name)
probe = subprocess.run([sys.executable, "-c",
                        "from piper.train.vits.monotonic_align import maximum_path; print('ok')"],
                       text=True, capture_output=True)
if "ok" not in probe.stdout:
    raise SystemExit("monotonic_align still does not import:\\n" + probe.stderr[-1500:])
print("monotonic_align built:", built_so[0].name)
'''

CELL_SMOKE = '''# 4. Smoke, before anything heavy: the card computes, and espeak-ng's bs voice is
# reachable from Piper's bundled data. (The wheel's data path breaks past ~160
# characters -- seen on the Mac -- so it is checked where the run will read it.)
x = torch.randn(256, 256, device="cuda", requires_grad=True)
(x @ x).sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all(), "CUDA backward failed"
from piper.phonemize_espeak import EspeakPhonemizer
phonemes = EspeakPhonemizer().phonemize("bs", "Sastanak je 2026. godine, dvije žene.")
flat = "".join("".join(s) for s in phonemes)
print("espeak bs:", flat)
if "dvˈije" not in flat or "ʒ" not in flat:
    raise SystemExit(f"espeak-ng bs did not phonemize as expected: {flat!r}")
print("smoke ok")
'''

CELL_DATA = '''# 5. FLEURS bs_ba, all three splits, whole: train to learn from, valid to choose
# the served speaker on, the test prefix to judge on. The test clips never train.
run(sys.executable, "data/scripts/download_speech_data.py")
SPEECH = CLONE / "data" / "speech"
TRAIN, VALID, TEST = SPEECH / "train.tsv", SPEECH / "valid.tsv", SPEECH / "test.tsv"
counts = {p.name: sum(1 for _ in p.open(encoding="utf-8")) for p in (TRAIN, VALID, TEST)}
print("rows:", counts)
if counts["train.tsv"] != 3091 or counts["valid.tsv"] != 400 or counts["test.tsv"] != 925:
    raise SystemExit(f"FLEURS bs_ba came back as {counts}, not 3091 / 400 / 925 -- "
                     "not training on a partial split")
OFF.metric("train_clips", counts["train.tsv"], stage="data")
'''

CELL_LISTENER = '''# 6. The judge: the listener the app ships, from the lilly-listen-large-v3
# dataset, checked by fingerprint before a clip is heard. The voice is judged
# by the ear the product has, and by no other.
import hashlib

def fingerprint(build):
    # identical to scripts/fetch_models.listen_fingerprint and speech_bench.fingerprint
    h = hashlib.md5()
    for name in sorted(p.name for p in build.iterdir()
                       if p.is_file() and p.name != "dataset-metadata.json"):
        h.update(name.encode())
        h.update((build / name).read_bytes())
    return h.hexdigest()[:16]

INPUT = Path("/kaggle/input")
LISTEN = CLONE / "models" / "lilly" / "listen"
if LISTEN.exists():
    shutil.rmtree(LISTEN)
dirs = sorted(p.parent for p in INPUT.rglob("built.json")
              if (p.parent / "model.bin").is_file()) if INPUT.is_dir() else []
larges = [d for d in dirs
          if json.loads((d / "built.json").read_text()).get("base") == "openai/whisper-large-v3"]
if len(larges) != 1:
    raise SystemExit(f"need exactly one whisper-large-v3 listener attached (dataset "
                     f"lilly-listen-large-v3), found {larges}. Relaunch: python3 scripts/kaggle_train.py speak-bs")
shutil.copytree(larges[0], LISTEN, ignore=shutil.ignore_patterns("dataset-metadata.json"))
LISTEN_FP = fingerprint(LISTEN)
print("listener:", LISTEN, LISTEN_FP)
if LISTEN_FP != "e6bb58483586b06c":
    raise SystemExit(f"listener fingerprint {LISTEN_FP} is not the shipped e6bb58483586b06c -- "
                     "the voice would be judged by another ear")
# app.speech puts the listener on CPU int8 -- the product's path. The judgment is
# about which words are heard, not the chip; float16 on the T4 is what makes
# three voices x 167 sentences + 200 clips fit the session, as the instrument did.
GPU_ENV = {"LILLY_SPEECH_DEVICE": "cuda", "LILLY_SPEECH_COMPUTE": "float16",
           "LILLY_IGNORE_GUARD": "1", "PYTHONUNBUFFERED": "1"}
'''

CELL_BEFORE = '''# 7. The voice the app speaks with today -- the "before". Fetched the way every
# install fetches it, and checked to be the bytes the Mac measured 27% with.
run(sys.executable, "scripts/fetch_speak_bs.py")
BEFORE = CLONE / "models" / "lilly" / "speak-bs" / "voice.onnx"
before_md5 = hashlib.md5(BEFORE.read_bytes()).hexdigest()
print("before voice:", BEFORE, before_md5)
if before_md5 != "02c6e27ac7b4dfa84272df89edca9feb":
    raise SystemExit(f"the fetched sr_RS voice is {before_md5}, not the bytes the app ships "
                     "(02c6e27ac7b4dfa84272df89edca9feb)")
'''

CELL_CKPT = '''# 8. The starting point: Piper's sr_RS checkpoint (924 MB), by md5, and a count
# of what the warm start will copy. 804 of 804 tensors match this trainer's
# model for a 2-speaker voice; with more speakers only the speaker table restarts.
from huggingface_hub import hf_hub_download
CKPT = Path(hf_hub_download("rhasspy/piper-checkpoints",
                            "sr/sr_RS/serbski_institut/medium/epoch=1899-step=178600.ckpt",
                            repo_type="dataset", local_dir=str(SCRATCH / "sr-ckpt")))
ckpt_md5 = hashlib.md5(CKPT.read_bytes()).hexdigest()
print("checkpoint:", CKPT, ckpt_md5)
if ckpt_md5 != "3dd3439e5c550d8201a9e7a2b1300a5b":
    raise SystemExit(f"sr_RS checkpoint is {ckpt_md5}, not the one the pre-registration names")
from piper.train.vits.lightning import VitsModel
old = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
new = VitsModel(num_speakers=2, gin_channels=512, sample_rate=22050, num_symbols=256,
                batch_size=8, mos_metric=None).state_dict()
copied = sum(1 for k, v in old.items() if k in new and new[k].shape == v.shape)
print(f"warm start would copy {copied} of {len(old)} tensors into a 2-speaker model")
if copied != len(old) or copied < 800:
    raise SystemExit(f"only {copied} of {len(old)} tensors match -- this trainer does not read that checkpoint")
del old, new
'''

CELL_CLUSTER = '''# 9. Who is speaking: clusters on this box, with the pre-registered constants
# (cosine distance 0.30, at least 20 minutes, at most 8 speakers -- the defaults
# of training/cluster_speakers.py; nothing is passed to override them). The Mac
# saw 8 clusters, 7 of them 54-103 minutes, 14.9% same-sentence collisions; a
# very different picture here is printed, and the script stops on its own rules.
SPK = SCRATCH / "speakers"
run(sys.executable, "training/cluster_speakers.py", "--tsv", str(TRAIN), "--out", str(SPK),
    "--device", "cuda")
speakers = json.loads((SPK / "speakers.json").read_text(encoding="utf-8"))
K = len(speakers["selected"])
print("clusters:", speakers["n_clusters"], "| selected:", speakers["selected"],
      "| minutes:", speakers["selected_minutes"], "| collision:", speakers["collision_rate"])
OFF.metric("speakers_selected", K, stage="data")
OFF.metric("train_minutes", speakers["selected_minutes"], stage="data")
OFF.metric("collision_rate", speakers["collision_rate"], stage="data")
'''

CELL_PREPARE = '''# 10. The training file: wav|speaker|text, the selected speakers only
PIPER = SCRATCH / "piper"
PIPER.mkdir(parents=True, exist_ok=True)
CSV = PIPER / "metadata.csv"
# Clips over 20 s stay out (the Mac counts 72 of 3,057, 2.4%, 27 minutes): VITS
# pays memory for the whole padded batch, and version 1 died at its first
# batches with CUDA out of memory on the T4 with them in.
run(sys.executable, "training/prepare_speak_data.py", "--tsv", str(TRAIN),
    "--speakers", str(SPK / "speakers.json"), "--out", str(CSV), "--max-seconds", "20")
manifest = json.loads(Path(str(CSV) + ".manifest.json").read_text(encoding="utf-8"))
print("training rows:", manifest["rows"], "| by speaker:", manifest["speakers"],
      "| left out over 20 s:", manifest["dropped_long"])
OFF.metric("dropped_long", manifest["dropped_long"], stage="data")
if manifest["rows"] < 500:
    raise SystemExit(f"{manifest['rows']} training rows -- not a voice's worth")
OFF.metric("train_rows", manifest["rows"], stage="data")
'''

CELL_TRAIN = '''# 11. TRAIN -- Piper's own trainer, warm-started from sr_RS, phonemized by
# espeak-ng's bs, at most 60 epochs or 5 h 30 min, whichever comes first. The
# last checkpoint is the candidate: no picking by loss, none by ear
# (PREREGISTRATION.md, "v5 -- speak"). Losses go to metrics.csv and are read
# back: a NaN anywhere, or non-finite weights, and nothing is exported.
# Batch 8 in fp32: batch 16 filled the T4's 14.6 GB at the first batches
# (version 1, 9 Sep) -- the amendment under the pre-registration says so.
# training/train_piper.py is piper.train's own CLI with one checkpoint callback
# (last.ckpt): piper.train's default callbacks watch val_mos, which is never
# logged with the MOS predictor off, and Lightning raised at the first
# validation end (version 2, epoch 5).
EPOCHS, MAX_TIME = "60", "00:05:30:00"
RUN = PIPER / "run"
run(sys.executable, "training/train_piper.py", "fit",
    "--data.csv_path", str(CSV), "--data.cache_dir", str(PIPER / "cache"),
    "--data.config_path", str(PIPER / "config.json"), "--data.voice_name", "bs_BA-fleurs-medium",
    "--data.espeak_voice", "bs", "--data.batch_size", "8", "--data.validation_split", "0.02",
    "--data.num_test_examples", "0", "--data.num_workers", "2",
    "--model.sample_rate", "22050", "--model.num_speakers", str(max(K, 2)),
    "--model.gin_channels", "512", "--model.warmstart_ckpt", str(CKPT), "--model.mos_metric", "none",
    "--trainer.accelerator", "gpu", "--trainer.devices", "1", "--trainer.precision", "32",
    "--trainer.max_epochs", EPOCHS, "--trainer.max_time", MAX_TIME,
    "--trainer.default_root_dir", str(RUN), "--trainer.check_val_every_n_epoch", "5",
    "--trainer.log_every_n_steps", "25", "--trainer.enable_progress_bar", "false",
    "--trainer.logger", "lightning.pytorch.loggers.CSVLogger",
    "--trainer.logger.save_dir", str(RUN), "--trainer.logger.name", "logs",
    env={"PYTHONUNBUFFERED": "1", "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
lasts = sorted(RUN.rglob("last.ckpt"))
if len(lasts) != 1:
    raise SystemExit(f"expected one last.ckpt under {RUN}, found {lasts}")
LAST = lasts[0]
state = torch.load(LAST, map_location="cpu", weights_only=False)
STEPS, EPOCH = int(state["global_step"]), int(state["epoch"])
bad = [k for k, v in state["state_dict"].items()
       if k.startswith("model_g.") and torch.is_tensor(v) and not torch.isfinite(v).all()]
if bad:
    raise SystemExit(f"non-finite weights after training: {bad[:5]} -- training collapsed; not exporting")
del state
metrics_files = sorted(RUN.rglob("metrics.csv"))
if not metrics_files:
    raise SystemExit("no metrics.csv from the trainer -- losses that were never logged cannot be judged")
import csv as _csv
losses = {}
for row in _csv.DictReader(metrics_files[0].open(encoding="utf-8")):
    for k, v in row.items():
        if v not in (None, "") and k.startswith(("loss", "train_")):
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                raise SystemExit(f"{k} is non-finite at step {row.get('step')} -- training collapsed; not exporting")
            losses[k] = f
print(f"trained {STEPS} steps, {EPOCH + 1} epochs; last logged losses: {losses}")
if STEPS < 500:
    raise SystemExit(f"{STEPS} steps is not a fine-tune -- the trainer stopped early; not exporting")
OFF.metric("steps", STEPS, stage="train")
OFF.metric("epochs", EPOCH + 1, stage="train")
for k, v in losses.items():
    OFF.metric(k, v, stage="train")
shutil.copy(metrics_files[0], "/kaggle/working/metrics.csv")
'''

CELL_EXPORT = '''# 12. Export the candidate the way the app loads voices: ONNX beside the config
# Piper wrote. training/export_piper_onnx.py is Piper's export with the
# TorchScript exporter named: the dynamo one refuses VITS's data-dependent
# output length (torch >= 2.9), and piper.train.export_onnx names neither.
CAND = SCRATCH / "cand"
CAND.mkdir(parents=True, exist_ok=True)
run(sys.executable, "training/export_piper_onnx.py", "--checkpoint", str(LAST),
    "--output-file", str(CAND / "voice.onnx"))
shutil.copy(PIPER / "config.json", CAND / "voice.onnx.json")
cfg = json.loads((CAND / "voice.onnx.json").read_text(encoding="utf-8"))
SPEAKER_IDS = cfg["speaker_id_map"]   # cluster name -> index inside the voice
print("speakers in the voice:", SPEAKER_IDS, "| espeak:", cfg["espeak"])
if cfg["espeak"]["voice"] != "bs":
    raise SystemExit("the exported config does not phonemize as bs")
import wave
from piper import PiperVoice, SynthesisConfig
voice = PiperVoice.load(CAND / "voice.onnx", CAND / "voice.onnx.json")
probe_wav = SCRATCH / "probe.wav"
with wave.open(str(probe_wav), "wb") as w:
    voice.synthesize_wav("Dobar dan, kako ste? Sastanak je sutra u devet.", w,
                         syn_config=SynthesisConfig(speaker_id=0))
with wave.open(str(probe_wav)) as w:
    secs = w.getnframes() / w.getframerate()
print(f"probe: {secs:.2f} s of audio")
if secs < 1.0:
    raise SystemExit("the exported voice rendered a two-clause sentence in under a second -- collapsed")
del voice
'''

CELL_SELECT = '''# 13. Which speaker serves: the one the listener hears best on 120 distinct VALID
# sentences -- never the test prefix. Ties go to the lower index.
VALID_JSON = Path("/kaggle/working/speak-bs-valid.json")
voice_args = []
for name, idx in sorted(SPEAKER_IDS.items(), key=lambda kv: kv[1]):
    voice_args += ["--voice", f"spk{idx}={CAND / 'voice.onnx'}:{idx}"]
run(sys.executable, "training/evaluate_speak.py", "--tsv", str(VALID), "--clips", "distinct",
    "--limit", "120", "--listener", str(LISTEN), "--json", str(VALID_JSON), *voice_args, env=GPU_ENV)
valid = json.loads(VALID_JSON.read_text(encoding="utf-8"))
table = sorted((v["wer"], v["speaker"], name) for name, v in valid["voices"].items())
for w_, idx, name in table:
    print(f"  {name}: {w_:.1f}% on {valid['n_sentences']} valid sentences")
CHOSEN = int(table[0][1])
CHOSEN_CLUSTER = [name for name, idx in SPEAKER_IDS.items() if idx == CHOSEN][0]
print("served speaker:", CHOSEN, "(cluster", CHOSEN_CLUSTER + ")")
OFF.metric("chosen_speaker", CHOSEN, stage="select")
OFF.metric("valid_wer_chosen", table[0][0], stage="select")
'''

CELL_JUDGE = '''# 14. THE JUDGMENT -- the test prefix (first 200 clips, 167 sentences): the
# before voice, the candidate's served speaker, the human recordings; one listener.
TEST_JSON = Path("/kaggle/working/speak-bs-test.json")
run(sys.executable, "training/evaluate_speak.py", "--tsv", str(TEST), "--clips", "first200",
    "--voice", f"before={BEFORE}:0", "--voice", f"candidate={CAND / 'voice.onnx'}:{CHOSEN}",
    "--human", "--listener", str(LISTEN), "--json", str(TEST_JSON), env=GPU_ENV)
test = json.loads(TEST_JSON.read_text(encoding="utf-8"))
before, cand, human = test["voices"]["before"], test["voices"]["candidate"], test["human"]
paired = test["paired"]["candidate vs before"]
if test["n_clips"] != 200 or human["n_clips"] != 200 or cand["n"] != test["n_sentences"]:
    raise SystemExit(f"judged {test['n_clips']} clips / {cand['n']} sentences, not the 200-clip prefix")
ships = cand["wer"] < before["wer"] and paired["p"] < 0.05
reached_human = cand["wer"] <= human["wer"]
OFF.check_trainproof(TEE)
for k, v in (("test_wer_before", before["wer"]), ("test_wer_candidate", cand["wer"]),
             ("test_wer_human", human["wer"]), ("paired_delta", paired["delta_points"]),
             ("paired_p", paired["p"]), ("ships", ships), ("reached_human", reached_human)):
    OFF.metric(k, v, stage="judge")
lines = ["# A Bosnian voice from FLEURS -- the judgment", "",
         f"Kaggle, {GPU}. Pre-registered: PREREGISTRATION.md, 'v5 -- speak -- a Bosnian voice from FLEURS'.",
         f"Listener {LISTEN_FP}; test prefix: {test['n_clips']} clips, {test['n_sentences']} sentences.",
         f"Trained {STEPS} steps / {EPOCH + 1} epochs on {manifest['rows']} clips, {K} speakers, "
         f"{speakers['selected_minutes']} minutes; served speaker {CHOSEN} (cluster {CHOSEN_CLUSTER}).", "",
         "| voice | word error | wrong / words |", "|---|---|---|",
         f"| before (Piper sr_RS, as the app ships it) | {before['wer']:.1f}% | {before['edits']} / {before['words']} |",
         f"| candidate (this run) | {cand['wer']:.1f}% | {cand['edits']} / {cand['words']} |",
         f"| human recordings | {human['wer']:.1f}% | {human['edits']} / {human['words']} |", "",
         f"candidate vs before: {paired['delta_points']:+.2f} points, paired bootstrap over sentences p = {paired['p']:.4f}.", "",
         f"**Bar 1 (ships): strictly below the before voice at p < 0.05 -- {'PASS' if ships else 'FAIL'}.**",
         f"**Bar 2 (the owner's ask, reported, not required): at or below the human recordings -- "
         f"{'reached' if reached_human else 'not reached'}.**"]
REPORT = "\\n".join(lines) + "\\n"
Path("/kaggle/working/speak-bs.md").write_text(REPORT, encoding="utf-8")
print(REPORT)
'''

CELL_PACKAGE = '''# 15. Package. The results zip always, now that the judgment exists. The voice
# zip only on PASS: a refused voice leaves no installable file behind.
built = {
    "base": f"rhasspy/piper-checkpoints sr/sr_RS/serbski_institut/medium epoch=1899-step=178600.ckpt (md5 {ckpt_md5})",
    "engine": "piper", "language": "bs (espeak-ng bs)",
    "speaker": CHOSEN, "speaker_cluster": CHOSEN_CLUSTER,
    "voice_md5": hashlib.md5((CAND / "voice.onnx").read_bytes()).hexdigest(),
    "trained_on": {"data": "FLEURS bs_ba train (google/fleurs, CC-BY-4.0)", "clips": manifest["rows"],
                   "speakers": K, "minutes": speakers["selected_minutes"], "steps": STEPS, "epochs": EPOCH + 1},
    "kaggle": {"kernel": "lilly-speak-bs", "gpu": GPU, "git": OFF.body["git"]},
    "measured": {"test_first200": {"before": before["wer"], "candidate": cand["wer"], "human": human["wer"],
                                   "paired_p": paired["p"], "listener": LISTEN_FP},
                 "ships": ships, "reached_human": reached_human},
    "license": "voice: MIT (Piper); training recordings: CC-BY-4.0 (FLEURS)",
}
(CAND / "built.json").write_text(json.dumps(built, indent=2) + "\\n", encoding="utf-8")
RESULTS = Path("/kaggle/working/lilly-speak-bs-results.zip")
with zipfile.ZipFile(RESULTS, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(TEST_JSON, "speak-bs-test.json")
    z.write(VALID_JSON, "speak-bs-valid.json")
    z.write(SPK / "speakers.json", "speakers.json")
    z.write("/kaggle/working/metrics.csv", "metrics.csv")
    z.write(CAND / "voice.onnx.json", "voice.onnx.json")
    z.write(CAND / "built.json", "built.json")
    z.write("/kaggle/working/speak-bs.md", "speak-bs.md")
    z.write(TEE, "stdout.txt")
if RESULTS.stat().st_size < 20_000:
    raise SystemExit(f"results zip is {RESULTS.stat().st_size} bytes -- that is not a result")
artifacts = [RESULTS.name]
if ships:
    VOICE_ZIP = Path("/kaggle/working/lilly-speak-bs.zip")
    with zipfile.ZipFile(VOICE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("voice.onnx", "voice.onnx.json", "built.json"):
            z.write(CAND / name, name)
    artifacts.append(VOICE_ZIP.name)
    print("SHIPS:", VOICE_ZIP, f"{VOICE_ZIP.stat().st_size / 1e6:.0f} MB")
else:
    print("DOES NOT SHIP: no voice zip. The results zip says why.")
OFF.finish("complete", artifacts)
out = list(Path("/kaggle/working").rglob("*"))
print(f"Output holds {len(out)} entries")
assert len(out) < 50, [str(x) for x in out[:50]]
'''

MARKDOWN_END = """**Status ERROR or CANCEL → nothing here is a result.** Read `stdout.txt`, fix the
cause, relaunch. Recovery is not success.

**On COMPLETE:** `scripts/kaggle_train.py speak-bs --fetch`. `lilly-speak-bs-results.zip`
goes to `training/speak-bs/` (the two JSON, `speakers.json`, `metrics.csv`, the
report, `built.json`); write `training/RESULTS-speak-bs.md` and the outcome under
the pre-registration **whichever way it fell**. If `lilly-speak-bs.zip` exists, the
bar was cleared: unzip it over `models/lilly/speak-bs/` and the app speaks with it
(`built.json` carries the speaker); publishing is the owner's act. If it does not
exist, the FLEURS line is a null and the sr_RS voice stays. One run judges this
recipe; a second run needs its own section.
"""


def build():
    cells = []
    for i, (kind, source) in enumerate([
        ("markdown", MARKDOWN_TOP), ("code", CELL_SETUP), ("code", CELL_CLONE),
        ("code", CELL_PIP), ("code", CELL_ALIGN), ("code", CELL_SMOKE), ("code", CELL_DATA),
        ("code", CELL_LISTENER), ("code", CELL_BEFORE), ("code", CELL_CKPT),
        ("code", CELL_CLUSTER), ("code", CELL_PREPARE), ("code", CELL_TRAIN),
        ("code", CELL_EXPORT), ("code", CELL_SELECT), ("code", CELL_JUDGE),
        ("code", CELL_PACKAGE), ("markdown", MARKDOWN_END),
    ]):
        if kind == "code":
            ast.parse(source)
        cells.append({"cell_type": kind, "id": f"cell-{i}", "metadata": {},
                      **({"execution_count": None, "outputs": []} if kind == "code" else {}),
                      "source": source.splitlines(keepends=True)})
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12.0"}},
                "nbformat": 4, "nbformat_minor": 5}
    payload = json.dumps(notebook, indent=1) + "\n"
    STAGING.mkdir(parents=True, exist_ok=True)
    for target in (OUT, STAGING / OUT.name):
        target.write_text(payload, encoding="utf-8")
        written = json.loads(target.read_text(encoding="utf-8"))
        for cell in written["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        print("verified %d cells in %s" % (len(written["cells"]), target))


if __name__ == "__main__":
    build()
