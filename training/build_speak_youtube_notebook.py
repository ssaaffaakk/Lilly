#!/usr/bin/env python3
"""Build training/Lilly_Speak_YouTube_Kaggle.ipynb -- a voice from Creative-Commons Bosnian YouTube, on a Kaggle GPU.

The third voice line, pre-registered in training/PREREGISTRATION.md as
"v8 -- speak -- a voice from Creative-Commons YouTube". The control ("v7")
showed the recipe carries a voice it is handed on clean single-speaker
recordings; the refused lines were crowd and hall audio. This line brings
what the control asked for from the one place it exists in Bosnian: channels
that publish under CC BY, one person speaking into a microphone for hours.

Reused cells: setup, clone, pip, the alignment build, the smoke, the listener,
the before voice, the checkpoint (build_speak_bs_notebook.py). Its own:

  * data: the FLEURS test split for the judgment; the audio the Mac fetched
    (dataset lilly-youtube-voice-<hash>), every file checked against the
    manifest committed at training/speak-youtube/manifest.json; then
    data/scripts/cut_youtube_voice.py -- the listener hears every video, its
    segments are cut by a rule fixed in advance, and each channel keeps one
    dominant speaker found the way FLEURS's speakers were;
  * training and export as in the parliament line (mean speaker served);
  * judgment as in every line: before, mean, human; individuals for the record.

    python3 training/build_speak_youtube_notebook.py
"""
import ast
import json
from pathlib import Path

import build_speak_bs_notebook as fleurs
import build_speak_parla_notebook as parla

HERE = Path(__file__).resolve().parent
OUT = HERE / "Lilly_Speak_YouTube_Kaggle.ipynb"
STAGING = HERE.parent / "models" / "kaggle-staging" / "lilly-speak-youtube"

MARKDOWN_TOP = """# Lilly — a voice from Creative-Commons Bosnian YouTube (`speak-youtube`)

**Job:** `speak-youtube`. Pre-registered in `training/PREREGISTRATION.md`, "v8 — speak
— a voice from Creative-Commons YouTube, written before any run". **Read it before
launching.** The FLEURS (v5) and parliament (v6) lines are closed; the control (v7)
showed the recipe holds a voice on clean single-speaker recordings.

**What this run does.** Takes the audio the Mac fetched from channels that publish
under the Creative Commons Attribution license (`training/speak-youtube/manifest.json`
is the record: channel, video, title, license, sha256), checks every file, hears every
video with the shipped listener, cuts the segments that pass the pre-registered rule,
keeps one dominant speaker per channel, trains Piper from `sr_RS-serbski_institut-medium`
on them, and serves the **mean** of the speakers' embeddings — no channel's own voice
ships. Judges the mean voice on the FLEURS test prefix beside the voice the app speaks
with today and the human recordings.

**Attach, before Save & Run All** (`scripts/kaggle_train.py speak-youtube` does it):
**Add data → Datasets → `lilly-listen-large-v3`** (the shipped listener, `e6bb58483586b06c`,
**required**) and **`lilly-youtube-voice-…`** (the audio and its manifest, **required**).
Internet **On**: FLEURS test, pip, Hugging Face (the sr_RS checkpoint, the before voice).

**Output.** `lilly-speak-youtube-results.zip` after the judgment, whichever way it fell.
`lilly-speak-youtube.zip` — the voice — **only** if it cleared the bar. ERROR or CANCEL:
nothing here is a result.
"""

CELL_CLONE = fleurs.CELL_CLONE.replace('Offload("speak-bs"', 'Offload("speak-youtube"')

CELL_DATA = '''# 9. The data. (a) FLEURS bs_ba test, whole: its first 200 clips judge; nothing
# trains on it. (b) The Creative-Commons audio the Mac fetched, as a dataset:
# its manifest must be byte-for-byte the one committed in the repository, and
# every file must match the manifest's sha256 (cut_youtube_voice.py checks).
# (c) The listener hears every video and the pre-registered rule cuts it.
import hashlib
run(sys.executable, "data/scripts/download_speech_data.py", "--split", "test")
TEST = CLONE / "data" / "speech" / "test.tsv"
n_test = sum(1 for _ in TEST.open(encoding="utf-8"))
print("test.tsv rows:", n_test)
if n_test != 925:
    raise SystemExit(f"FLEURS bs_ba test came back as {n_test} clips, not 925 -- not judging on a partial split")
COMMITTED = CLONE / "training" / "speak-youtube" / "manifest.json"
hits = sorted(p for p in INPUT.rglob("manifest.json") if "youtube" in str(p).lower()) if INPUT.is_dir() else []
if len(hits) != 1:
    raise SystemExit(f"need exactly one attached lilly-youtube-voice dataset (manifest.json), found {hits}. "
                     "Relaunch: python3 scripts/kaggle_train.py speak-youtube")
MANIFEST = hits[0]
if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != hashlib.sha256(COMMITTED.read_bytes()).hexdigest():
    raise SystemExit("the attached manifest is not the committed training/speak-youtube/manifest.json -- "
                     "the audio would not be the pre-registered selection")
selection = json.loads(MANIFEST.read_text(encoding="utf-8"))
print("manifest:", len(selection), "videos,", round(sum((m["duration_s"] or 0) for m in selection.values()) / 3600, 1), "h,",
      len(set(m["channel"] for m in selection.values())), "channels")
YT = SCRATCH / "yt"
run(sys.executable, "data/scripts/cut_youtube_voice.py", "--manifest", str(MANIFEST), "--audio-dir", str(MANIFEST.parent),
    "--out", str(YT), "--ffmpeg", "ffmpeg", "--device", "cuda", env=GPU_ENV)
CSV = YT / "metadata.csv"
manifest = json.loads(Path(str(CSV) + ".manifest.json").read_text(encoding="utf-8"))
print("training rows:", manifest["rows"], "| minutes:", manifest["minutes"], "| speakers:",
      {k: (v["name"], v["clips"], v["minutes"]) for k, v in manifest["speakers"].items()})
print("channels:", json.dumps(manifest["channels"], ensure_ascii=False, indent=1))
K = len(manifest["speakers"])
if K < 5 or manifest["rows"] < 500:
    raise SystemExit(f"{K} speakers / {manifest['rows']} rows -- under the pre-registered floor")
OFF.metric("train_rows", manifest["rows"], stage="data")
OFF.metric("train_minutes", manifest["minutes"], stage="data")
OFF.metric("speakers_selected", K, stage="data")
OFF.metric("kept_before_speaker_rule", manifest["kept_before_speaker_rule"], stage="data")
shutil.copy(Path(str(CSV) + ".manifest.json"), "/kaggle/working/metadata.csv.manifest.json")
'''

CELL_TRAIN = parla.CELL_TRAIN.replace('"bs_BA-parla-medium"', '"bs_BA-youtube-medium"').replace(
    '(PREREGISTRATION.md, "v6 -- speak")', '(PREREGISTRATION.md, "v8 -- speak")')

CELL_JUDGE = (parla.CELL_JUDGE
              .replace('/kaggle/working/speak-parla-test.json', '/kaggle/working/speak-youtube-test.json')
              .replace("# A voice from the Croatian parliament -- the judgment", "# A voice from Creative-Commons Bosnian YouTube -- the judgment")
              .replace("'v6 -- speak -- a voice from ParlaSpeech-HR'", "'v8 -- speak -- a voice from Creative-Commons YouTube'")
              .replace('/kaggle/working/speak-parla.md', '/kaggle/working/speak-youtube.md'))

CELL_PACKAGE = (parla.CELL_PACKAGE
                .replace('"ParlaSpeech-HR (classla/ParlaSpeech-HR, CC BY-SA 4.0), the segments in "\n                           "training/speak-parla/selection.json"',
                         '"Creative-Commons (CC BY) Bosnian YouTube audio, the videos in training/speak-youtube/manifest.json, "\n                           "transcribed by the shipped listener"')
                .replace('"kernel": "lilly-speak-parla"', '"kernel": "lilly-speak-youtube"')
                .replace('"voice: CC BY-SA 4.0 (derived from ParlaSpeech-HR); served speaker is the mean of the "\n               "trained speakers, no member of parliament\'s own voice"',
                         '"voice: CC BY 4.0 (derived from CC BY YouTube audio; attribution in built.json and the manifest); served speaker "\n               "is the mean of the trained speakers, no channel\'s own voice"')
                .replace('/kaggle/working/lilly-speak-parla-results.zip', '/kaggle/working/lilly-speak-youtube-results.zip')
                .replace('z.write(TEST_JSON, "speak-parla-test.json")', 'z.write(TEST_JSON, "speak-youtube-test.json")\n    z.write(MANIFEST, "manifest.json")')
                .replace('z.write("/kaggle/working/speak-parla.md", "speak-parla.md")', 'z.write("/kaggle/working/speak-youtube.md", "speak-youtube.md")')
                .replace('/kaggle/working/lilly-speak-parla.zip', '/kaggle/working/lilly-speak-youtube.zip'))
assert "parla" not in CELL_PACKAGE.replace("speak-parla", "").replace("parla", "") or True
for needle in ("speak-youtube-results.zip", "lilly-speak-youtube.zip", 'z.write(MANIFEST, "manifest.json")', "v8 -- speak"):
    assert needle in CELL_PACKAGE + CELL_JUDGE + CELL_TRAIN, needle
assert "parla" not in CELL_TRAIN + CELL_JUDGE + CELL_PACKAGE, "a parliament name survived the rename"

MARKDOWN_END = """**Status ERROR or CANCEL → nothing here is a result.** Read `stdout.txt`, fix the
cause, relaunch. Recovery is not success.

**On COMPLETE:** `scripts/kaggle_train.py speak-youtube --fetch`.
`lilly-speak-youtube-results.zip` goes to `training/speak-youtube/`; write
`training/RESULTS-speak-youtube.md` and the outcome under the pre-registration
**whichever way it fell**. If `lilly-speak-youtube.zip` exists, the bar was cleared:
unzip it over `models/lilly/speak-bs/` (`built.json` names the served speaker, the
mean, and the channels the voice was trained on — CC BY asks for that credit to
travel with it); publishing is the owner's act. If it does not exist, the line is a
null and the sr_RS voice stays. One run judges this recipe; a second run needs its
own section.
"""


def build():
    cells = []
    for i, (kind, source) in enumerate([
        ("markdown", MARKDOWN_TOP), ("code", fleurs.CELL_SETUP), ("code", CELL_CLONE),
        ("code", fleurs.CELL_PIP), ("code", fleurs.CELL_ALIGN), ("code", fleurs.CELL_SMOKE),
        ("code", fleurs.CELL_LISTENER), ("code", fleurs.CELL_BEFORE), ("code", fleurs.CELL_CKPT),
        ("code", CELL_DATA), ("code", CELL_TRAIN), ("code", parla.CELL_EXPORT),
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
