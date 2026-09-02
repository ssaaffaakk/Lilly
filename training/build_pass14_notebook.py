#!/usr/bin/env python3
"""Derive the pass-14 OCR notebook from the pass-13 one.

    python3 training/build_pass14_notebook.py

Pass-13 trained in two stages, plates then human. Pass-14 trains once, on the
915 Mapillary shop-sign crops, through `train_ocr.py --train-dir`. Only three
cells change; every gate and every assert that records a past failure is
carried across untouched, because each of them is there for a reason someone
already paid for.

Reads  training/Lilly_OCR_Kaggle.ipynb   (pass-13, left alone)
Writes training/Lilly_OCR_Pass14_Kaggle.ipynb
  and models/kaggle-staging/lilly-ocr-pass14/Lilly_OCR_Pass14_Kaggle.ipynb

Both copies are written here on purpose: `kaggle kernels push` uploads from
the staging directory, and a stale staging copy once made four consecutive
fixes look like they had no effect on Kaggle.
"""
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "training" / "Lilly_OCR_Kaggle.ipynb"
OUT = REPO_ROOT / "training" / "Lilly_OCR_Pass14_Kaggle.ipynb"
STAGING = REPO_ROOT / "models" / "kaggle-staging" / "lilly-ocr-pass14"

TITLE = """\
# Lilly v2 — reader pass-14

Scope: `docs/V2-BOUNDARIES.md`. Latin Bosnian only. No Cyrillic.

Trains once, on the 915 Mapillary shop-sign crops in
`data/ocr/mapillary-train/`, loaded with `train_ocr.py --train-dir` so no
`mly_*` file can reach `data/ocr/train` or `data/ocr/valid`. If one did, the
gate would be scoring EasyOCR against its own output.

The 915 came from 20,240 Mapillary photographs: 13,144 crops, minus Beograd
and Banja Luka for Cyrillic signage, minus dashcam on-screen display, minus
anything under conf 0.85 or three letters. A blind read of 30 of them at that
threshold scored 93% ignoring diacritics, 77% exact.

**The labels keep folded spellings.** `kuca` is labelled `kuca`. The product
bar for this set is meaning through the translator — `kuca` → House is
enough. The crop gate does not share that bar: it requires

    real held-out crops: words strictly up; Bosnian letters NOT down

so a run that trades diacritics for words is refused with *"words rose but
Bosnian letters fell — a trade."* That refusal is a live possibility here and
it is the correct behaviour. **The gate is not to be weakened to get past it.**
An ERROR on a refused gate is the gate working.

Cells 5, 5b still run: they build the held-out valid set the gate measures on
(real crops, and the synthetic floor). The plates land in `data/ocr/train`,
which `--train-dir` then ignores — the 915 are never mixed with plates in one
`train_ocr`.

Do not relaunch pass-8/9/10/11. Pass-10 overfit human-only at 5 epochs; this
is 2 epochs on a smaller set.

915 crops is fewer than the 1,294 human crops that pass-12 and pass-13 both
failed on. Whether it moves the photograph score is not known until this run
returns.
"""

STAGE_MAPILLARY = '''\
# 5e. The 915 Mapillary crops — into their own directory, never into train/.
# data/ocr/mapillary-train/*.png is gitignored, so the clone does not carry
# them; they arrive as the attached dataset lilly-ocr-mapillary-train.
MLY_DIR = Path("data/ocr/mapillary-train")
MLY_DIR.mkdir(parents=True, exist_ok=True)

# Several attached datasets ship a gt.txt. Take the one whose rows are mly_.
mly_gt = None
for root in (input_root, unpack):
    if not root.is_dir():
        continue
    for cand in root.rglob("gt.txt"):
        head = cand.read_text(encoding="utf-8")[:200]
        if head.startswith("mly_"):
            mly_gt = cand
            break
    if mly_gt is not None:
        break
if mly_gt is None:
    raise SystemExit(
        "No mly_ gt.txt under /kaggle/input. Attach "
        "afaksrmeli/lilly-ocr-mapillary-train.")

src_dir = mly_gt.parent
copied_mly = 0
for line in mly_gt.read_text(encoding="utf-8").splitlines():
    parts = line.split("\\t")
    if len(parts) < 2:
        continue
    src = src_dir / parts[0]
    if src.is_file():
        dest = MLY_DIR / parts[0]
        if not dest.exists():
            shutil.copy(src, dest)
        copied_mly += 1
shutil.copy(mly_gt, MLY_DIR / "gt.txt")

mly_rows = [l for l in (MLY_DIR / "gt.txt").read_text(encoding="utf-8").splitlines()
            if l.strip()]
mly_png = len(list(MLY_DIR.glob("mly_*.png")))
print(f"mapillary train dir: {mly_png} PNGs, {len(mly_rows)} rows from {src_dir}")
if mly_png != len(mly_rows):
    raise SystemExit(f"{len(mly_rows)} rows but {mly_png} PNGs — the dataset is partial")
if mly_png < 900:
    raise SystemExit(f"only {mly_png} Mapillary crops, expected 915")

# The whole point of --train-dir. A pseudo-labelled crop in train or valid
# would let the gate score EasyOCR against its own output.
for guard in (Path("data/ocr/train"), Path("data/ocr/valid")):
    leaked = list(guard.glob("mly_*"))
    if leaked:
        raise SystemExit(f"{len(leaked)} mly_ files leaked into {guard}")
print("no mly_ in data/ocr/train or data/ocr/valid")

train_gt = Path("data/ocr/train/gt.txt")
valid_gt = Path("data/ocr/valid/gt.txt")
valid_n = sum(1 for _ in open(valid_gt, encoding="utf-8"))
print(f"held-out valid crops: {valid_n:,}")
if valid_n <= 100:
    raise SystemExit(f"only {valid_n} valid crops — the gate cannot decide")
OFF.metric("mapillary_n", mly_png, stage="split")
OFF.metric("valid_n", valid_n, stage="split")
'''

TRAIN = '''\
# 6. ONE TRAINING, ON THE MAPILLARY CROPS ONLY — pass-14.
# --train-dir replaces the training source outright, so the sign-letter
# plates sitting in data/ocr/train are not read. The plates are still needed:
# they are the synthetic floor the crop gate measures against.
#
# v1 of this pass asked for batch 16 and 2 epochs: 915/16 = 57 steps an epoch,
# 114 in total, and train_ocr refused at MIN_STEPS=200 — "below this a run has
# not trained, it has only run". The fix is a smaller batch, not more epochs.
# batch 8 gives 115 steps an epoch, so 3 epochs clears 200 with room, and each
# crop is still seen three times rather than the five that overfit pass-10.
#
# LR is 3e-6, not the 1e-6 pass-13 used for its human stage. That stage ran
# thousands of steps; 345 steps at 1e-6 would move the weights so little that
# the gate would read "words did not rise" and refuse a run that never really
# happened.
os.environ["LILLY_RUN_ID"] = "heavy-pass14"
os.environ["PYTHONUNBUFFERED"] = "1"
OFF.body["run_id"] = "heavy-pass14"
OFF.flush()
TRAINED = Path("models/lilly/read-trained.pth")

try:
    run("python3", "-u", "training/train_ocr.py",
        "--train-dir", str(MLY_DIR),
        "--epochs", "3", "--batch-size", "8",
        "--lr", "3e-6", "--grad-clip", "1.0", "--warmup-frac", "0.1",
        "--weights", str(INIT), "--keep-trained", str(TRAINED))
except subprocess.CalledProcessError:
    # The crop gate refusing is this cell failing, and that is correct.
    # Folded labels (kuca for kuća) can push Bosnian letters down, which the
    # gate reads as a trade and refuses. Do not weaken the gate to get past
    # it — fix the data or accept the refusal.
    OFF.fail("train_ocr exit 1 — crop gate refused, or collapse")
    raise

assert TRAINED.is_file() and TRAINED.stat().st_size > 100_000, (
    "gate passed but no weights were written")
OFF.check_trainproof()
run("zip", "-j", "/kaggle/working/lilly-read-trained.zip", str(TRAINED))
print("lilly-read-trained.zip saved")
'''


def build():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def find(idx, needle, what):
        src = "".join(cells[idx]["source"])
        if needle not in src:
            raise SystemExit(
                f"cell {idx + 1} is not the {what} cell any more "
                f"(looked for {needle!r}). Pass-13 changed; re-read it."
            )

    # Anchor on content, not position, so a shifted pass-13 fails loudly here
    # rather than silently producing a notebook that trains the wrong thing.
    # The markdown heading still says pass-12 while cell 6 trains pass-13, so
    # anchor on the stable part of the line rather than on a pass number.
    find(0, "Lilly v2 — reader pass", "title")
    find(10, "5e. Split train into two lists", "two-list split")
    find(12, "TWO TRAININGS, NOT MIXED", "training")
    find(13, "The gate that measures what the README claims", "photograph gate")

    cells[0]["source"] = TITLE.splitlines(keepends=True)
    cells[10]["source"] = STAGE_MAPILLARY.splitlines(keepends=True)
    cells[12]["source"] = TRAIN.splitlines(keepends=True)

    # The photograph gate is carried across untouched except for the names of
    # the files it writes. Its thresholds are the shipped reader's numbers and
    # are not this pass's to move.
    gate = "".join(cells[13]["source"])
    gate = gate.replace("RESULTS-ocr-pass13.md", "RESULTS-ocr-pass14.md")
    gate = gate.replace("reader-output-pass13.json", "reader-output-pass14.json")
    cells[13]["source"] = gate.splitlines(keepends=True)

    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))

    payload = json.dumps(nb, indent=1) + "\n"
    STAGING.mkdir(parents=True, exist_ok=True)
    for target in (OUT, STAGING / OUT.name):
        target.write_text(payload, encoding="utf-8")
        written = json.loads(target.read_text(encoding="utf-8"))
        for cell in written["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        print(f"verified {len(written['cells'])} cells in {target}")


if __name__ == "__main__":
    build()
