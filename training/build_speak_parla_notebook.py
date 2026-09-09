#!/usr/bin/env python3
"""Build training/Lilly_Speak_Parla_Kaggle.ipynb -- a voice from ParlaSpeech-HR, on a Kaggle GPU.

The second voice line, pre-registered in training/PREREGISTRATION.md as
"v6 -- speak -- a voice from ParlaSpeech-HR". The FLEURS line ("v5") is closed
and its notebook stays as the record; this builder reuses its cells where the
step is the same (setup, clone, pip, the alignment build, the smoke, the
listener, the before voice, the checkpoint) and writes its own where the line
differs:

  * data: the FLEURS test split for the judgment, and the parliament segments
    named by training/speak-parla/selection.json, fetched row group by row
    group (data/scripts/download_parlaspeech_voice.py);
  * training: the same recipe, up to 100 epochs or 9 h -- the FLEURS run was
    still improving when its 5 h 30 min ran out;
  * export: the served voice is the MEAN of the trained speakers' embeddings
    (training/export_piper_onnx.py --add-mean-speaker) -- no member of
    parliament becomes the app's voice;
  * judgment: the before voice, the mean voice, the human recordings; the
    individual speakers are heard for the record and never served.

    python3 training/build_speak_parla_notebook.py
"""
import ast
import json
from pathlib import Path

import build_speak_bs_notebook as fleurs

HERE = Path(__file__).resolve().parent
OUT = HERE / "Lilly_Speak_Parla_Kaggle.ipynb"
STAGING = HERE.parent / "models" / "kaggle-staging" / "lilly-speak-parla"

MARKDOWN_TOP = """# Lilly — a voice from the Croatian parliament (`speak-parla`)

**Job:** `speak-parla`. Pre-registered in `training/PREREGISTRATION.md`, "v6 — speak
— a voice from ParlaSpeech-HR, written before any run". **Read it before launching.**
The FLEURS line (`speak-bs`, "v5") is closed: 53.9% against the before voice's 22.3%.

**What this run does.** Fetches the segments `training/speak-parla/selection.json`
names — five speakers of one gender from ParlaSpeech-HR (CC BY-SA 4.0), up to
three clean hours each, chosen by a rule written before any training — and
fine-tunes Piper's `sr_RS-serbski_institut-medium` on them, phonemized by
espeak-ng's `bs`. Exports the voice with one extra speaker, the **mean** of the
trained speakers' embeddings, which is the only voice this line serves: a
member of parliament's own voice is heard for the record and never shipped.
Judges the mean voice on the FLEURS test prefix (first 200 clips, 167 sentences)
through the shipped listener against the voice the app speaks with today and
against the human recordings.

**Attach, before Save & Run All** (`scripts/kaggle_train.py speak-parla` does it):
**Add data → Datasets → `lilly-listen-large-v3`** — the shipped listener,
fingerprint `e6bb58483586b06c`, **required**. Internet **On**: ParlaSpeech-HR
row groups, FLEURS test, pip, Hugging Face (the sr_RS checkpoint, the before voice).

**Output.** `lilly-speak-parla-results.zip` after the judgment, whichever way it
fell. `lilly-speak-parla.zip` — the voice — **only** if it cleared the bar.
ERROR or CANCEL: nothing here is a result.
"""

CELL_CLONE = fleurs.CELL_CLONE.replace('Offload("speak-bs"', 'Offload("speak-parla"')

CELL_DATA = '''# 5. Two kinds of data, and a hard count on each.
# (a) FLEURS bs_ba test, whole: its first 200 clips judge; nothing trains on it.
run(sys.executable, "data/scripts/download_speech_data.py", "--split", "test")
SPEECH = CLONE / "data" / "speech"
TEST = SPEECH / "test.tsv"
n_test = sum(1 for _ in TEST.open(encoding="utf-8"))
print("test.tsv rows:", n_test)
if n_test != 925:
    raise SystemExit(f"FLEURS bs_ba test came back as {n_test} clips, not 925 -- not judging on a partial split")
# (b) The parliament segments the selection names, and no others: each row
# checked by id, each transcript by the clean rule, each shard by size.
SELECTION = CLONE / "training" / "speak-parla" / "selection.json"
selection = json.loads(SELECTION.read_text(encoding="utf-8"))
print("selection:", len(selection["segments"]), "segments,", selection["minutes_total"], "minutes,",
      len(selection["speakers"]), "speakers:", selection["speakers"])
PARLA = SCRATCH / "parla"
run(sys.executable, "data/scripts/download_parlaspeech_voice.py", "--selection", str(SELECTION),
    "--out", str(PARLA), env={"PYTHONUNBUFFERED": "1"})
CSV = PARLA / "metadata.csv"
manifest = json.loads(Path(str(CSV) + ".manifest.json").read_text(encoding="utf-8"))
print("training rows:", manifest["rows"], "| minutes:", manifest["minutes"], "| by speaker:",
      {k: (v["name"], v["clips"], v["minutes"]) for k, v in manifest["speakers"].items()})
if manifest["rows"] != len(selection["segments"]) or manifest["rows"] < 500:
    raise SystemExit(f"{manifest['rows']} rows against {len(selection['segments'])} selected -- not the pre-registered file")
K = len(manifest["speakers"])
speakers = {"selected_minutes": manifest["minutes"], "selected": list(manifest["speakers"])}
OFF.metric("train_rows", manifest["rows"], stage="data")
OFF.metric("train_minutes", manifest["minutes"], stage="data")
OFF.metric("speakers_selected", K, stage="data")
OFF.metric("transferred_mb", manifest["transferred_mb"], stage="data")
'''

CELL_TRAIN = '''# 11. TRAIN -- Piper's own trainer through training/train_piper.py (one
# checkpoint callback, last.ckpt, the fit's own exit code), warm-started from
# sr_RS, phonemized by espeak-ng's bs, batch 8 in fp32, at most 100 epochs or
# 9 hours, whichever comes first -- the FLEURS run was still improving when its
# 5 h 30 min ran out. The last checkpoint is the candidate: no picking by loss,
# none by ear (PREREGISTRATION.md, "v6 -- speak"). Losses go to metrics.csv and
# are read back: a NaN anywhere, or non-finite weights, and nothing is exported.
EPOCHS, MAX_TIME = "100", "00:09:00:00"
PIPER = SCRATCH / "piper"
PIPER.mkdir(parents=True, exist_ok=True)
RUN = PIPER / "run"
run(sys.executable, "training/train_piper.py", "fit",
    "--data.csv_path", str(CSV), "--data.cache_dir", str(PIPER / "cache"),
    "--data.config_path", str(PIPER / "config.json"), "--data.voice_name", "bs_BA-parla-medium",
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
        if v not in (None, "") and k.startswith(("loss", "train_", "val_")):
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

CELL_EXPORT = '''# 12. Export with the MEAN speaker appended, and name it in the config. The
# mean of the trained speakers' embeddings is the voice this line serves; the
# speakers themselves are heard below for the record and never shipped.
CAND = SCRATCH / "cand"
CAND.mkdir(parents=True, exist_ok=True)
run(sys.executable, "training/export_piper_onnx.py", "--checkpoint", str(LAST),
    "--output-file", str(CAND / "voice.onnx"), "--add-mean-speaker")
cfg = json.loads((PIPER / "config.json").read_text(encoding="utf-8"))
SPEAKER_IDS = dict(cfg["speaker_id_map"])       # cluster name -> index inside the voice
MEAN = max(SPEAKER_IDS.values()) + 1
if MEAN != cfg["num_speakers"]:
    raise SystemExit(f"speaker map {SPEAKER_IDS} does not fill 0..{cfg['num_speakers'] - 1}")
cfg["speaker_id_map"]["mean"] = MEAN
cfg["num_speakers"] = MEAN + 1
if cfg["espeak"]["voice"] != "bs":
    raise SystemExit("the exported config does not phonemize as bs")
(CAND / "voice.onnx.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\\n", encoding="utf-8")
print("speakers in the voice:", SPEAKER_IDS, "| mean at", MEAN, "| espeak:", cfg["espeak"])
import wave
from piper import PiperVoice, SynthesisConfig
voice = PiperVoice.load(CAND / "voice.onnx", CAND / "voice.onnx.json")
if voice.config.num_speakers != MEAN + 1:
    raise SystemExit(f"the voice loads with {voice.config.num_speakers} speakers, not {MEAN + 1}")
probe_wav = SCRATCH / "probe.wav"
with wave.open(str(probe_wav), "wb") as w:
    voice.synthesize_wav("Dobar dan, kako ste? Sastanak je sutra u devet.", w,
                         syn_config=SynthesisConfig(speaker_id=MEAN))
with wave.open(str(probe_wav)) as w:
    secs = w.getnframes() / w.getframerate()
print(f"mean-voice probe: {secs:.2f} s of audio")
if secs < 1.0:
    raise SystemExit("the mean voice rendered a two-clause sentence in under a second -- collapsed")
del voice
'''

CELL_JUDGE = '''# 13. THE JUDGMENT -- the FLEURS test prefix (first 200 clips, 167 sentences):
# the before voice, the mean voice, the human recordings, one listener. The
# individual speakers follow in the same process, for the record only.
TEST_JSON = Path("/kaggle/working/speak-parla-test.json")
record_args = []
for name, idx in sorted(SPEAKER_IDS.items(), key=lambda kv: kv[1]):
    record_args += ["--voice", f"spk{idx}={CAND / 'voice.onnx'}:{idx}"]
run(sys.executable, "training/evaluate_speak.py", "--tsv", str(TEST), "--clips", "first200",
    "--voice", f"before={BEFORE}:0", "--voice", f"candidate={CAND / 'voice.onnx'}:{MEAN}",
    *record_args, "--human", "--listener", str(LISTEN), "--json", str(TEST_JSON), env=GPU_ENV)
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
for name, v in test["voices"].items():
    if name.startswith("spk"):
        OFF.metric(f"test_wer_{name}", v["wer"], stage="record")
lines = ["# A voice from the Croatian parliament -- the judgment", "",
         f"Kaggle, {GPU}. Pre-registered: PREREGISTRATION.md, 'v6 -- speak -- a voice from ParlaSpeech-HR'.",
         f"Listener {LISTEN_FP}; test prefix: {test['n_clips']} clips, {test['n_sentences']} sentences.",
         f"Trained {STEPS} steps / {EPOCH + 1} epochs on {manifest['rows']} clips, {K} speakers, "
         f"{manifest['minutes']} minutes; served voice: the mean of the {K} speakers (index {MEAN}).", "",
         "| voice | word error | wrong / words |", "|---|---|---|",
         f"| before (Piper sr_RS, as the app ships it) | {before['wer']:.1f}% | {before['edits']} / {before['words']} |",
         f"| candidate (the mean voice, this run) | {cand['wer']:.1f}% | {cand['edits']} / {cand['words']} |",
         f"| human recordings | {human['wer']:.1f}% | {human['edits']} / {human['words']} |"]
for name, v in sorted(test["voices"].items(), key=lambda kv: kv[1]["wer"]):
    if name.startswith("spk"):
        who = [n for n, i in SPEAKER_IDS.items() if i == v["speaker"]][0]
        lines.append(f"| for the record: {name} ({manifest['speakers'][who]['name']}) | {v['wer']:.1f}% | {v['edits']} / {v['words']} |")
lines += ["", f"candidate vs before: {paired['delta_points']:+.2f} points, paired bootstrap over sentences p = {paired['p']:.4f}.", "",
          f"**Bar 1 (ships): strictly below the before voice at p < 0.05 -- {'PASS' if ships else 'FAIL'}.**",
          f"**Bar 2 (the owner's ask, reported, not required): at or below the human recordings -- "
          f"{'reached' if reached_human else 'not reached'}.**"]
REPORT = "\\n".join(lines) + "\\n"
Path("/kaggle/working/speak-parla.md").write_text(REPORT, encoding="utf-8")
print(REPORT)
'''

CELL_PACKAGE = '''# 14. Package. The results zip always, now that the judgment exists. The voice
# zip only on PASS: a refused voice leaves no installable file behind.
built = {
    "base": f"rhasspy/piper-checkpoints sr/sr_RS/serbski_institut/medium epoch=1899-step=178600.ckpt (md5 {ckpt_md5})",
    "engine": "piper", "language": "bs (espeak-ng bs)",
    "speaker": MEAN, "speaker_cluster": "mean",
    "voice_md5": hashlib.md5((CAND / "voice.onnx").read_bytes()).hexdigest(),
    "trained_on": {"data": "ParlaSpeech-HR (classla/ParlaSpeech-HR, CC BY-SA 4.0), the segments in "
                           "training/speak-parla/selection.json", "clips": manifest["rows"],
                   "speakers": K, "minutes": manifest["minutes"], "steps": STEPS, "epochs": EPOCH + 1},
    "kaggle": {"kernel": "lilly-speak-parla", "gpu": GPU, "git": OFF.body["git"]},
    "measured": {"test_first200": {"before": before["wer"], "candidate": cand["wer"], "human": human["wer"],
                                   "paired_p": paired["p"], "listener": LISTEN_FP},
                 "ships": ships, "reached_human": reached_human},
    "license": "voice: CC BY-SA 4.0 (derived from ParlaSpeech-HR); served speaker is the mean of the "
               "trained speakers, no member of parliament's own voice",
}
(CAND / "built.json").write_text(json.dumps(built, indent=2) + "\\n", encoding="utf-8")
RESULTS = Path("/kaggle/working/lilly-speak-parla-results.zip")
with zipfile.ZipFile(RESULTS, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(TEST_JSON, "speak-parla-test.json")
    z.write(Path(str(CSV) + ".manifest.json"), "metadata.csv.manifest.json")
    z.write("/kaggle/working/metrics.csv", "metrics.csv")
    z.write(CAND / "voice.onnx.json", "voice.onnx.json")
    z.write(CAND / "built.json", "built.json")
    z.write("/kaggle/working/speak-parla.md", "speak-parla.md")
    z.write(TEE, "stdout.txt")
if RESULTS.stat().st_size < 20_000:
    raise SystemExit(f"results zip is {RESULTS.stat().st_size} bytes -- that is not a result")
artifacts = [RESULTS.name]
if ships:
    VOICE_ZIP = Path("/kaggle/working/lilly-speak-parla.zip")
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

**On COMPLETE:** `scripts/kaggle_train.py speak-parla --fetch`. `lilly-speak-parla-results.zip`
goes to `training/speak-parla/` (the JSON, the manifest, `metrics.csv`, the
report, `built.json`); write `training/RESULTS-speak-parla.md` and the outcome
under the pre-registration **whichever way it fell**. If `lilly-speak-parla.zip`
exists, the bar was cleared: unzip it over `models/lilly/speak-bs/` — `built.json`
names the served speaker, the mean — and the app speaks with it; publishing is
the owner's act, and the voice carries CC BY-SA 4.0. If it does not exist, the
parliament line is a null and the sr_RS voice stays. One run judges this recipe;
a second run needs its own section.
"""


def build():
    cells = []
    for i, (kind, source) in enumerate([
        ("markdown", MARKDOWN_TOP), ("code", fleurs.CELL_SETUP), ("code", CELL_CLONE),
        ("code", fleurs.CELL_PIP), ("code", fleurs.CELL_ALIGN), ("code", fleurs.CELL_SMOKE),
        ("code", CELL_DATA), ("code", fleurs.CELL_LISTENER), ("code", fleurs.CELL_BEFORE),
        ("code", fleurs.CELL_CKPT), ("code", CELL_TRAIN), ("code", CELL_EXPORT),
        ("code", CELL_JUDGE), ("code", CELL_PACKAGE), ("markdown", MARKDOWN_END),
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
