#!/usr/bin/env python3
"""Generate the Kaggle crop notebook from validated Python sources.

    python3 training/build_crop_notebook.py

Writes training/Lilly_OCR_Crop_Kaggle.ipynb.

Every cell is ast.parse-checked before it is written. Building the notebook
this way instead of hand-editing JSON is deliberate: four consecutive Kaggle
runs died on a newline that was literal inside a string literal because the
.ipynb was edited as JSON. A cell that does not compile never reaches Kaggle.

The notebook does no harvesting. Photos arrive as an attached Kaggle dataset,
so the Mapillary token is never needed on Kaggle at all.
"""
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "training" / "Lilly_OCR_Crop_Kaggle.ipynb"

# `kaggle kernels push` uploads from the staging dir, not from training/.
# Writing both here is deliberate: a stale staging copy is exactly what made
# four fixes in a row look like they had no effect on Kaggle.
STAGING = REPO_ROOT / "models" / "kaggle-staging" / "lilly-ocr-crop"

MARKDOWN = """\
# Lilly — OCR crop pass (GPU)

Crops text regions out of already-harvested Mapillary street photos.

Input: dataset `afaksrmeli/lilly-mapillary-photos` (~20k jpgs, CC BY-SA 4.0).
Output: `crops-mapillary.zip` — PNG crops plus `labels.tsv`.

No Mapillary token is used here. Harvesting happened elsewhere; this pass is
GPU work only.
"""

SETUP = '''\
# 1. Sanity checks and tee
import os, subprocess, sys, shutil, zipfile, time
from pathlib import Path
import torch

assert torch.cuda.is_available(), "No GPU. Enable GPU in Session options."
print(torch.cuda.get_device_name(0))

WORKING = Path("/kaggle/working")
TEE = WORKING / "stdout.txt"


def log(msg):
    """Print to the Kaggle log and to stdout.txt, per the fail-stop rules."""
    print(msg, flush=True)
    with TEE.open("a", encoding="utf-8") as sink:
        print(msg, file=sink)


def run(*cmd, quiet=False):
    """Run a child process, teeing its output to stdout.txt.

    check=True alone is not enough: the Kaggle log never sees the child's fd.
    """
    line = "$ " + " ".join(str(c) for c in cmd)
    print(line, flush=True)
    with TEE.open("a", encoding="utf-8") as sink:
        print(line, file=sink)
        child = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for out in child.stdout:
            if not quiet:
                print(out, end="", flush=True)
            sink.write(out)
            sink.flush()
        code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


log("setup ok")
'''

FIND_PHOTOS = '''\
# 2. Locate the photos, and drop the ones whose signage is Cyrillic
INPUT = Path("/kaggle/input")
assert INPUT.exists(), "No /kaggle/input. Attach the lilly-mapillary-photos dataset."

photos = sorted(INPUT.rglob("mly_*.jpg"))

# Kaggle sometimes leaves an upload as a zip instead of extracting it.
if not photos:
    archives = list(INPUT.rglob("*.zip"))
    log("no loose jpgs; found %d zip(s), extracting" % len(archives))
    unpacked = Path("/kaggle/temp/photos")
    unpacked.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(unpacked)
    photos = sorted(unpacked.rglob("mly_*.jpg"))

log("%d photos in the dataset" % len(photos))

# EasyOCR's latin_g2 recogniser cannot emit Cyrillic at all, so a Cyrillic
# sign does not come back unread — it comes back transcribed into Latin
# lookalikes at ordinary confidence, and that would enter training as ground
# truth. v1 does not read Cyrillic, so these cities are dropped rather than
# mislabelled.
CYRILLIC_CITIES = {"beograd", "novi_sad", "nis", "banjaluka"}

credits = list(INPUT.rglob("CREDITS.tsv"))
if not credits:
    raise SystemExit("No CREDITS.tsv in the dataset — cannot tell which city a photo came from.")

city_of = {}
with credits[0].open(encoding="utf-8") as fh:
    header = fh.readline().rstrip("\\n").split("\\t")
    icity = header.index("city")
    ifile = header.index("file")
    for row in fh:
        parts = row.rstrip("\\n").split("\\t")
        if len(parts) > max(icity, ifile):
            city_of[Path(parts[ifile]).name] = parts[icity]

kept, dropped = [], {}
for photo in photos:
    city = city_of.get(photo.name)
    if city in CYRILLIC_CITIES:
        dropped[city] = dropped.get(city, 0) + 1
        continue
    kept.append(photo)

for city in sorted(dropped):
    log("  dropped %d photos from %s (Cyrillic signage)" % (dropped[city], city))

unknown = sum(1 for p in photos if p.name not in city_of)
if unknown:
    log("  %d photos absent from CREDITS.tsv, kept" % unknown)

photos = kept
log("%d photos to crop" % len(photos))

if len(photos) < 1000:
    raise SystemExit(
        "Only %d photos after filtering. Expected ~13000." % len(photos)
    )
'''

INSTALL = '''\
# 3. Install EasyOCR
run(sys.executable, "-m", "pip", "install", "-q",
    "easyocr", "opencv-python-headless", "pillow")
'''

CROP = '''\
# 4. Crop text regions on the GPU
import easyocr
from PIL import Image

# "bs" and "en" are both in EasyOCR's latin_lang_list, so they share one
# recognition model and may be loaded together.
log("loading EasyOCR on GPU")
reader = easyocr.Reader(["bs", "en"], gpu=True)
log("reader ready")

# Crops are built in scratch, not in /kaggle/working. Tens of thousands of
# loose PNGs in the Output directory is its own failure mode; only the zip
# belongs there.
CROPS_DIR = Path("/kaggle/temp/crops-mapillary")
CROPS_DIR.mkdir(parents=True, exist_ok=True)
LABELS = CROPS_DIR / "labels.tsv"

MIN_CONF = 0.4
MIN_PX = 8

n_crops = 0
n_failed = 0
started = time.time()

with LABELS.open("w", encoding="utf-8") as out:
    out.write("file\\ttext\\tconfidence\\n")

    for i, photo in enumerate(photos):
        try:
            image = Image.open(photo).convert("RGB")
            regions = reader.readtext(str(photo), detail=1)
        except (torch.cuda.OutOfMemoryError, MemoryError):
            # The GPU dying is not a corrupt jpg. Swallowing it here would let
            # the run ship a zip built from however much finished first.
            raise
        except RuntimeError as exc:
            if "CUDA" in str(exc) or "cuDNN" in str(exc):
                raise
            n_failed += 1
            log("  SKIP %s: %s" % (photo.name, exc))
            continue
        except Exception as exc:
            n_failed += 1
            log("  SKIP %s: %s" % (photo.name, exc))
            continue

        for j, (box, text, conf) in enumerate(regions):
            text = text.strip()
            if not text or conf < MIN_CONF:
                continue
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            # Clamp both ends: a slanted box can report corners outside the
            # frame, and PIL pads out-of-bounds crops with black instead of
            # refusing them.
            x0, y0 = max(min(xs), 0), max(min(ys), 0)
            x1 = min(max(xs), image.width)
            y1 = min(max(ys), image.height)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image.crop((x0, y0, x1, y1))
            if crop.width < MIN_PX or crop.height < MIN_PX:
                continue
            name = "%s_%03d.png" % (photo.stem, j)
            crop.save(CROPS_DIR / name)
            out.write("%s\\t%s\\t%.2f\\n" % (name, text, conf))
            n_crops += 1

        if (i + 1) % 500 == 0:
            out.flush()
            rate = (i + 1) / (time.time() - started)
            eta = (len(photos) - i - 1) / rate / 60
            log("  %d/%d photos | %d crops | %.1f img/s | ETA %.0fm"
                % (i + 1, len(photos), n_crops, rate, eta))

log("%d crops from %d photos (%d unreadable)" % (n_crops, len(photos), n_failed))

# A run that could not open one photo in twenty was not measuring the corpus
# it claims to have measured.
if n_failed > 0.05 * len(photos):
    raise SystemExit("%d of %d photos failed to read (>5%%)."
                     % (n_failed, len(photos)))

# A thin harvest is a failure, not a result to zip up.
if n_crops < 5000:
    raise SystemExit("Only %d crops from %d photos. Expected >= 5000."
                     % (n_crops, len(photos)))
'''

PACK = '''\
# 5. Pack the output
# shutil.make_archive takes root_dir explicitly, so the notebook's cwd
# cannot silently point the archiver at the wrong tree.
archive = shutil.make_archive(
    base_name=str(WORKING / "crops-mapillary"),
    format="zip",
    root_dir=str(CROPS_DIR.parent),
    base_dir=CROPS_DIR.name,
)

# Trust the artifact, not the exit code.
size_mb = Path(archive).stat().st_size / 1048576
with zipfile.ZipFile(archive) as zf:
    packed = sum(1 for n in zf.namelist() if n.endswith(".png"))
if packed != n_crops:
    raise SystemExit("Packed %d PNGs but cropped %d." % (packed, n_crops))
log("%s: %.0f MB, %d crops verified inside" % (archive, size_mb, packed))
log("CROPS: %d" % n_crops)
log("PHOTOS: %d" % len(photos))
log("done")
'''

CELLS = [
    ("markdown", "crop-title", MARKDOWN),
    ("code", "cell-setup", SETUP),
    ("code", "cell-photos", FIND_PHOTOS),
    ("code", "cell-install", INSTALL),
    ("code", "cell-crop", CROP),
    ("code", "cell-pack", PACK),
]


def build():
    cells = []
    for kind, cell_id, source in CELLS:
        if kind == "code":
            # The whole point of this generator: a cell that does not
            # compile never reaches Kaggle.
            ast.parse(source)
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "id": cell_id,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            })
        else:
            cells.append({
                "cell_type": "markdown",
                "id": cell_id,
                "metadata": {},
                "source": source.splitlines(keepends=True),
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    payload = json.dumps(notebook, indent=1) + "\n"
    STAGING.mkdir(parents=True, exist_ok=True)
    targets = [OUT, STAGING / OUT.name]
    for target in targets:
        target.write_text(payload, encoding="utf-8")

    # Read back and compile every cell from the files that ship, not from the
    # strings above.
    for target in targets:
        written = json.loads(target.read_text(encoding="utf-8"))
        for cell in written["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        print("verified %d cells in %s" % (len(written["cells"]), target))


if __name__ == "__main__":
    build()
