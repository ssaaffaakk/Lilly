#!/usr/bin/env python3
"""Build training/Lilly_Speak_Control_Kaggle.ipynb -- the control the two voice lines never had.

Two lines trained a voice on top of Piper's sr_RS checkpoint and both were
heard near 52% where the checkpoint itself is heard at 22.3%. The recipe has
not been shown to preserve a voice it is handed. This run hands it the sr_RS
voice's OWN 747 training utterances (Sorbian Institute, public) and asks
whether the voice survives a short fine-tune on them, in three arms:

  A  espeak-ng `sr` (the checkpoint's own phonemizer), Piper's default rates
  B  espeak-ng `bs` (what the two lines used), Piper's default rates
  C  espeak-ng `bs`, a gentle fresh optimizer (lr 2e-5 / 1e-5, 2 warm-up epochs)

3,000 steps each, speaker 0 exported, judged beside the before voice and the
human recordings by the same instrument. Pre-registered as "v7 -- speak -- the
control" in training/PREREGISTRATION.md. Ships nothing: a control has no
voice zip, by construction.

    python3 training/build_speak_control_notebook.py
"""
import ast
import json
from pathlib import Path

import build_speak_bs_notebook as fleurs

HERE = Path(__file__).resolve().parent
OUT = HERE / "Lilly_Speak_Control_Kaggle.ipynb"
STAGING = HERE.parent / "models" / "kaggle-staging" / "lilly-speak-control"

MARKDOWN_TOP = """# Lilly — the voice pipeline's control (`speak-control`)

**Job:** `speak-control`. Pre-registered in `training/PREREGISTRATION.md`, "v7 — speak
— the control, written before any run". **Read it before launching.**

**The question.** Two voices fine-tuned from Piper's `sr_RS-serbski_institut-medium`
— on FLEURS (v5) and on the Croatian parliament (v6) — were heard near 52% where
the checkpoint itself is heard at 22.3%. Does the recipe preserve a voice it is
handed? This run fine-tunes the checkpoint on **its own 747 training utterances**
(Sorbian Institute, CC BY-NC-SA 4.0, public; the checkpoint repository carries the
list) for 3,000 steps in three arms — `sr` phonemes, `bs` phonemes, `bs` with a
gentle optimizer — exports speaker 0 of each, and judges them on the FLEURS test
prefix beside the before voice and the human recordings.

**Attach, before Save & Run All** (`scripts/kaggle_train.py speak-control` does it):
**Add data → Datasets → `lilly-listen-large-v3`** (the shipped listener, `e6bb58483586b06c`,
**required**). Internet **On**: GitHub releases (the two FLACs), FLEURS test, pip, Hugging Face.

**Output.** `lilly-speak-control-results.zip` after the judgment, whichever way it
fell. **No voice zip, ever**: nothing trained here is Bosnian and nothing ships.
ERROR or CANCEL: nothing here is a result.
"""

CELL_CLONE = fleurs.CELL_CLONE.replace('Offload("speak-bs"', 'Offload("speak-control"')

CELL_DATA = '''# 5. Three things, each checked. (a) FLEURS bs_ba test, whole: its first 200
# clips judge. (b) The voice's own training list from the checkpoint repository.
# (c) The Sorbian Institute releases -- one FLAC and one YAML per language --
# pinned by sha256 in training/speak-control/sources.json, cut into the 747
# utterances by data/scripts/prepare_sorbian_control.py.
import hashlib
run(sys.executable, "data/scripts/download_speech_data.py", "--split", "test")
TEST = CLONE / "data" / "speech" / "test.tsv"
n_test = sum(1 for _ in TEST.open(encoding="utf-8"))
print("test.tsv rows:", n_test)
if n_test != 925:
    raise SystemExit(f"FLEURS bs_ba test came back as {n_test} clips, not 925 -- not judging on a partial split")
from huggingface_hub import hf_hub_download
LIST = Path(hf_hub_download("rhasspy/piper-checkpoints", "sr/sr_RS/serbski_institut/medium/dataset.jsonl.gz",
                            repo_type="dataset", local_dir=str(SCRATCH / "sr-ckpt")))
sources = json.loads((CLONE / "training" / "speak-control" / "sources.json").read_text(encoding="utf-8"))
SORB = SCRATCH / "sorbian"
SORB.mkdir(parents=True, exist_ok=True)
got = {}
for name, src in sources["files"].items():
    target = SORB / name
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != src["sha256"]:
        run("curl", "-sSL", "-o", str(target), src["url"], quiet=True)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != src["sha256"]:
        raise SystemExit(f"{name}: sha256 {digest}, the pre-registration pinned {src['sha256']} -- not the release the voice was cut from")
    got[name] = target
    print(f"  {name}: {target.stat().st_size / 1e6:.0f} MB, sha256 ok")
DATA = SCRATCH / "sorbian-data"
run(sys.executable, "data/scripts/prepare_sorbian_control.py", "--list", str(LIST),
    "--dsb-flac", str(got["dsb.flac"]), "--dsb-yaml", str(got["dsb.yaml"]),
    "--hsb-flac", str(got["hsb.flac"]), "--hsb-yaml", str(got["hsb.yaml"]), "--out", str(DATA))
CSV = DATA / "metadata.csv"
manifest = json.loads(Path(str(CSV) + ".manifest.json").read_text(encoding="utf-8"))
print("rows:", manifest["rows"], "| minutes:", manifest["minutes"], "| speakers:", manifest["speakers"],
      "| phoneme agreement:", manifest["phoneme_agreement"])
if manifest["rows"] != 747 or manifest["speakers"] != {"dsb": 408, "hsb": 339}:
    raise SystemExit(f"{manifest['rows']} rows / {manifest['speakers']} -- not the voice's own 747")
OFF.metric("train_rows", manifest["rows"], stage="data")
for voice, rec in manifest["phoneme_agreement"].items():
    OFF.metric(f"phoneme_agreement_{voice}", rec["share"], stage="data")
'''

CELL_TRAIN = '''# 11. THREE ARMS, 3,000 steps each, from the same checkpoint on the same 747
# utterances. training/train_piper.py, fp32, 2 speakers so the warm start
# copies the speaker table too. Batch 2, not 8: the voice's own utterances run
# to 56 s and VITS pays for the whole padded batch (batch 16 filled the T4 on
# 20-second clips). Each arm has its own cache: Piper keys the cache on text,
# and sr and bs phonemize the same text differently.
import csv as _csv
PIPER = SCRATCH / "piper"
PIPER.mkdir(parents=True, exist_ok=True)
ARMS = {
    "A": ("sr", []),
    "B": ("bs", []),
    "C": ("bs", ["--model.learning_rate", "2e-5", "--model.learning_rate_d", "1e-5",
                 "--model.warmup_epochs", "2"]),
}
STEPS_EACH, MAX_TIME_EACH = "3000", "00:01:15:00"
trained = {}
for arm, (espeak_voice, extra) in ARMS.items():
    RUN = PIPER / f"run-{arm}"
    run(sys.executable, "training/train_piper.py", "fit",
        "--data.csv_path", str(CSV), "--data.cache_dir", str(PIPER / f"cache-{arm}"),
        "--data.config_path", str(PIPER / f"config-{arm}.json"), "--data.voice_name", f"control-{arm}",
        "--data.espeak_voice", espeak_voice, "--data.batch_size", "2", "--data.validation_split", "0.02",
        "--data.num_test_examples", "0", "--data.num_workers", "2",
        "--model.sample_rate", "22050", "--model.num_speakers", "2",
        "--model.gin_channels", "512", "--model.warmstart_ckpt", str(CKPT), "--model.mos_metric", "none",
        *extra,
        "--trainer.accelerator", "gpu", "--trainer.devices", "1", "--trainer.precision", "32",
        "--trainer.max_steps", STEPS_EACH, "--trainer.max_time", MAX_TIME_EACH,
        "--trainer.default_root_dir", str(RUN), "--trainer.check_val_every_n_epoch", "10",
        "--trainer.log_every_n_steps", "25", "--trainer.enable_progress_bar", "false",
        "--trainer.logger", "lightning.pytorch.loggers.CSVLogger",
        "--trainer.logger.save_dir", str(RUN), "--trainer.logger.name", "logs",
        env={"PYTHONUNBUFFERED": "1", "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    lasts = sorted(RUN.rglob("last.ckpt"))
    if len(lasts) != 1:
        raise SystemExit(f"arm {arm}: expected one last.ckpt under {RUN}, found {lasts}")
    state = torch.load(lasts[0], map_location="cpu", weights_only=False)
    steps = int(state["global_step"])
    bad = [k for k, v in state["state_dict"].items()
           if k.startswith("model_g.") and torch.is_tensor(v) and not torch.isfinite(v).all()]
    del state
    if bad:
        raise SystemExit(f"arm {arm}: non-finite weights: {bad[:5]}")
    if steps < 2500:
        raise SystemExit(f"arm {arm}: {steps} steps -- the wall cut it short of the pre-registered 3,000")
    metrics_files = sorted(RUN.rglob("metrics.csv"))
    if not metrics_files:
        raise SystemExit(f"arm {arm}: no metrics.csv")
    losses = {}
    for row in _csv.DictReader(metrics_files[0].open(encoding="utf-8")):
        for k, v in row.items():
            if v not in (None, "") and k.startswith(("loss", "train_", "val_")):
                f = float(v)
                if f != f or f in (float("inf"), float("-inf")):
                    raise SystemExit(f"arm {arm}: {k} is non-finite at step {row.get('step')}")
                losses[k] = f
    shutil.copy(metrics_files[0], f"/kaggle/working/metrics-{arm}.csv")
    CAND = SCRATCH / "cand" / arm
    CAND.mkdir(parents=True, exist_ok=True)
    run(sys.executable, "training/export_piper_onnx.py", "--checkpoint", str(lasts[0]),
        "--output-file", str(CAND / "voice.onnx"))
    shutil.copy(PIPER / f"config-{arm}.json", CAND / "voice.onnx.json")
    import wave
    from piper import PiperVoice, SynthesisConfig
    voice = PiperVoice.load(CAND / "voice.onnx", CAND / "voice.onnx.json")
    probe_wav = SCRATCH / f"probe-{arm}.wav"
    with wave.open(str(probe_wav), "wb") as w:
        voice.synthesize_wav("Dobar dan, kako ste? Sastanak je sutra u devet.", w,
                             syn_config=SynthesisConfig(speaker_id=0))
    with wave.open(str(probe_wav)) as w:
        secs = w.getnframes() / w.getframerate()
    del voice
    if secs < 1.0:
        raise SystemExit(f"arm {arm}: the probe rendered in under a second -- collapsed")
    trained[arm] = {"steps": steps, "losses": losses, "onnx": CAND / "voice.onnx", "probe_seconds": round(secs, 2)}
    print(f"arm {arm}: {steps} steps, probe {secs:.2f} s, last losses {losses}")
    OFF.metric(f"steps_{arm}", steps, stage="train")
    for k, v in losses.items():
        OFF.metric(f"{k}_{arm}", v, stage="train")
'''

CELL_JUDGE = '''# 12. THE READING -- the FLEURS test prefix, the before voice (speaker 0 of the
# checkpoint as the app fetches it), speaker 0 of each arm, the human recordings.
TEST_JSON = Path("/kaggle/working/speak-control-test.json")
arm_args = []
for arm, rec in trained.items():
    arm_args += ["--voice", f"arm{arm}={rec['onnx']}:0"]
run(sys.executable, "training/evaluate_speak.py", "--tsv", str(TEST), "--clips", "first200",
    "--voice", f"before={BEFORE}:0", *arm_args, "--human", "--listener", str(LISTEN),
    "--json", str(TEST_JSON), env=GPU_ENV)
test = json.loads(TEST_JSON.read_text(encoding="utf-8"))
before, human = test["voices"]["before"], test["human"]
if test["n_clips"] != 200 or human["n_clips"] != 200:
    raise SystemExit(f"judged {test['n_clips']} clips, not the 200-clip prefix")
OFF.check_trainproof(TEE)
readings = {}
for arm in trained:
    v = test["voices"][f"arm{arm}"]
    p = test["paired"][f"arm{arm} vs before"]
    delta, pval = p["delta_points"], p["p"]
    verdict = ("sound" if delta < 5 else "BROKEN" if (delta >= 10 and pval < 0.05) else "inconclusive")
    readings[arm] = {"wer": v["wer"], "delta": delta, "p": pval, "verdict": verdict}
    OFF.metric(f"test_wer_{arm}", v["wer"], stage="judge")
    OFF.metric(f"delta_{arm}", delta, stage="judge")
    OFF.metric(f"verdict_{arm}", verdict, stage="judge")
OFF.metric("test_wer_before", before["wer"], stage="judge")
OFF.metric("test_wer_human", human["wer"], stage="judge")
lines = ["# The voice pipeline's control -- the reading", "",
         f"Kaggle, {GPU}. Pre-registered: PREREGISTRATION.md, 'v7 -- speak -- the control'.",
         f"Listener {LISTEN_FP}; test prefix: {test['n_clips']} clips, {test['n_sentences']} sentences; "
         f"747 utterances of the voice's own data; phoneme agreement {manifest['phoneme_agreement']}.", "",
         "| voice | word error | wrong / words | vs before | p | reading |", "|---|---|---|---|---|---|",
         f"| before (the checkpoint, speaker 0, as fetched) | {before['wer']:.1f}% | {before['edits']} / {before['words']} | | | |"]
for arm, (espeak_voice, extra) in ARMS.items():
    v, r = test["voices"][f"arm{arm}"], readings[arm]
    lines.append(f"| arm {arm} ({espeak_voice}{', gentle' if extra else ''}), {trained[arm]['steps']} steps | "
                 f"{v['wer']:.1f}% | {v['edits']} / {v['words']} | {r['delta']:+.2f} | {r['p']:.4f} | **{r['verdict']}** |")
lines += [f"| human recordings | {human['wer']:.1f}% | {human['edits']} / {human['words']} | | | |", "",
          "Reading rule (pre-registered): sound if the arm is under +5 points from the before voice; "
          "BROKEN if +10 or more at p < 0.05; inconclusive between.", ""]
REPORT = "\\n".join(lines) + "\\n"
Path("/kaggle/working/speak-control.md").write_text(REPORT, encoding="utf-8")
print(REPORT)
'''

CELL_PACKAGE = '''# 13. Package the reading. A control has no voice zip, by construction.
RESULTS = Path("/kaggle/working/lilly-speak-control-results.zip")
with zipfile.ZipFile(RESULTS, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(TEST_JSON, "speak-control-test.json")
    z.write(Path(str(CSV) + ".manifest.json"), "metadata.csv.manifest.json")
    for arm in trained:
        z.write(f"/kaggle/working/metrics-{arm}.csv", f"metrics-{arm}.csv")
        z.write(SCRATCH / "cand" / arm / "voice.onnx.json", f"voice-{arm}.onnx.json")
    z.write("/kaggle/working/speak-control.md", "speak-control.md")
    z.write(TEE, "stdout.txt")
if RESULTS.stat().st_size < 20_000:
    raise SystemExit(f"results zip is {RESULTS.stat().st_size} bytes -- that is not a result")
OFF.finish("complete", [RESULTS.name])
out = list(Path("/kaggle/working").rglob("*"))
print(f"Output holds {len(out)} entries")
assert len(out) < 50, [str(x) for x in out[:50]]
'''

MARKDOWN_END = """**Status ERROR or CANCEL → nothing here is a result.** Read `stdout.txt`, fix the
cause, relaunch. Recovery is not success.

**On COMPLETE:** `scripts/kaggle_train.py speak-control --fetch`;
`lilly-speak-control-results.zip` goes to `training/speak-control/`; write
`training/RESULTS-speak-control.md` and the outcome under the pre-registration
**whichever way it fell**. Nothing is installed from this run. What the reading
means for the next voice line is written in the pre-registration, in advance.
"""


def build():
    cells = []
    for i, (kind, source) in enumerate([
        ("markdown", MARKDOWN_TOP), ("code", fleurs.CELL_SETUP), ("code", CELL_CLONE),
        ("code", fleurs.CELL_PIP), ("code", fleurs.CELL_ALIGN), ("code", fleurs.CELL_SMOKE),
        ("code", CELL_DATA), ("code", fleurs.CELL_LISTENER), ("code", fleurs.CELL_BEFORE),
        ("code", fleurs.CELL_CKPT), ("code", CELL_TRAIN), ("code", CELL_JUDGE),
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
