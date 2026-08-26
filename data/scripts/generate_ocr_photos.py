#!/usr/bin/env python3
"""Make training images that look like a phone photo, not a scan.

data/scripts/generate_ocr_data.py renders flat, sharp text on an almost-white
background. Real photos of Bosnian signs and menus are nothing like that: the
camera is never square to the page, the light is never even, focus is never
perfect, and the phone's JPEG encoder has already thrown away detail before
the file exists. On real photos the shipped reader turns Čaršija into Caršija
and Đačka into Backa — the diacritic strokes are exactly the fine detail that
disappears first. Training on clean synthetic images does not teach the model
to survive that; this script tries to reproduce the failure so training data
can cover it.

Each image goes through the same five insults a photo picks up before it
reaches the model: a camera-angle perspective warp, uneven light and shadow,
focus blur, sensor noise, and a lossy JPEG re-encode — all stacked on a
textured paper/metal background instead of flat colour. Words come from the
existing corpus loader in generate_ocr_data.py, already restricted to the 351
characters the reader has an output class for.

    python3 data/scripts/generate_ocr_photos.py                  # ~4,000 images
    python3 data/scripts/generate_ocr_photos.py --count 10000

Before writing the full set, it calibrates: it renders small probes across a
range of degradation and reads each one back two ways - see
training_pipeline_predictor() and app_ocr_predictor() below for why there
are two and which one actually decides the severity - classifying every
misread. Failing more is not the goal - failing the way real photos fail is:
the diacritic mark gone but the word still recognisable (Čaršija -> Caršija),
not the image destroyed into noise. So it settles on whichever severity
produces the most of that specific pattern while keeping outright-destroyed
readings a minority, not just the first severity that fails often enough -
that bar is trivial to clear by over-degrading. It will not report success
from a batch it never measured.

Writes into the same data/ocr/train and data/ocr/valid folders the trainer
already reads, as new photoNNNNNN.png files appended to gt.txt — the existing
synNNNNNN.png images and their labels are untouched, and (as of
training/prepare_ocr_data.py's build_splits()) a later re-run of
generate_ocr_data.py only clears its own synNNNNNN.png files, so these
photoNNNNNN.png ones survive that too.
"""
import argparse
import io
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import generate_ocr_data as clean_gen  # noqa: E402 - needs sys.path set first

REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = SCRIPT_DIR.parent
OUT_TRAIN = DATA_DIR / "ocr" / "train"
OUT_VALID = DATA_DIR / "ocr" / "valid"
from training.ocr_split import is_valid_text  # noqa: E402 - needs REPO_ROOT on sys.path
BOSNIAN_LETTERS = clean_gen.BOSNIAN_LETTERS
DIACRITIC_SHARE = 0.75          # higher than the clean generator's 0.55 - this
# set exists specifically to stress the letters that go missing on real photos
SEED = 43

# How hard to degrade, tried in order until the shipped reader is actually
# failing on the result THE RIGHT WAY. Each step raises perspective, blur,
# contrast loss, noise and JPEG damage together (see render_photo / degrade /
# warp_perspective).
SEVERITY_STEPS = (0.3, 0.5, 0.7, 1.0, 1.5, 2.2, 3.2)
MIN_SEVERITY = 0.15              # a real floor below the gentlest calibration step,
# so the top-up loop's "back off" can actually reduce severity instead of
# clamping back to where it started (a first version clamped to
# SEVERITY_STEPS[0], which made backing off a no-op whenever the gentlest
# calibrated step was itself already too harsh - exactly what happened the
# first time this ran: severity 1.0 was the softest step tried and it still
# came in at ~38% garbage, so "back off" needs somewhere lower to go)
PROBE_SIZE = 150                 # words per calibration probe - small and fast
DIACRITIC_STRIP = str.maketrans("čćđšžČĆĐŠŽ", "ccdszCCDSZ")

# Failing more is not the goal - failing the SAME WAY real photos fail is. The
# measured real failure is specific: Čaršija -> Caršija, Đačka -> Backa. The
# diacritic mark is gone, the letter that carried it (or the nearest plain
# letter) is still there, and the word stays recognisable to a person. That is
# useful training signal. A word degraded into unrelated noise is not - it
# teaches the model to guess rather than to read through a faint mark, and a
# high enough severity can always reach a low accuracy number that way without
# ever producing anything worth training on. So the bar below is a shape, not
# just a score: enough of the realistic pattern (clean_drop / near_clean_drop /
# substitution - see classify_error), with genuine over-degradation (garbage)
# kept a minority. Measured on a 150-word probe through the actual training
# preprocessing (see training_pipeline_predictor()); the published reader
# through that same preprocessing keeps 68.5% of Bosnian letters on the clean
# synthetic set (data/ocr/valid, checksummed weights, 500 crops) - a useful
# sanity check that these images are at least as hard, though the pattern
# check below is what actually decides the severity.
MIN_REALISTIC_ERRORS = 20        # on a calibration probe
MAX_GARBAGE_SHARE = 0.35         # of all errors on words that carry a diacritic
MIN_FINAL_REALISTIC = 5          # lower bar for the smaller final/top-up checks


def find_coeffs(dst, src):
    """8 coefficients for Image.transform(size, Image.PERSPECTIVE, coeffs).

    For every output pixel, PIL's PERSPECTIVE transform samples the input
    image at (a*x+b*y+c)/(g*x+h*y+1), (d*x+e*y+f)/(g*x+h*y+1). `dst` and `src`
    are four corresponding corners - dst in the output canvas, src in the
    image being warped - and solving that equation at those four points pins
    down all eight coefficients for every other pixel too.
    """
    matrix = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b)


def make_text_layer(text: str, font: ImageFont.FreeTypeFont, rng: random.Random):
    """Ink on a transparent layer, sized tight to the text plus a little pad."""
    box = font.getbbox(text)
    pad = rng.randint(3, 9)
    width = max(box[2] - box[0] + 2 * pad, 1)
    height = max(box[3] - box[1] + 2 * pad, 1)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shade = rng.randint(0, 65)
    tint = rng.choice([(0, 0, 0), (0, 0, 25), (20, 0, 0), (0, 15, 15)])
    ink = tuple(max(0, min(255, shade + t)) for t in tint) + (255,)
    ImageDraw.Draw(layer).text((pad - box[0], pad - box[1]), text, fill=ink, font=font)
    return layer


def warp_perspective(layer: Image.Image, rng: random.Random, severity: float):
    """Tilt the text like a camera that was not square to the page."""
    w, h = layer.size
    margin_frac = min(0.10 + 0.05 * severity, 0.34)
    margin = max(int(max(w, h) * margin_frac), 10)
    canvas_size = (w + 2 * margin, h + 2 * margin)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    base = [(margin, margin), (margin + w, margin),
            (margin + w, margin + h), (margin, margin + h)]
    jitter = margin * min(0.55 + 0.15 * severity, 1.0)
    dst = [(x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter))
           for x, y in base]
    coeffs = find_coeffs(dst, src)
    return layer.transform(canvas_size, Image.PERSPECTIVE, coeffs,
                           resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))


def uneven_light(size, rng: random.Random):
    """A directional gradient (light from one side) plus a soft radial blob
    (a spotlight or a shadow falling across the page) - additive, same shape
    as the image plane."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    angle = rng.uniform(0, 2 * math.pi)
    grad = xx * math.cos(angle) + yy * math.sin(angle)
    spread = max(grad.max() - grad.min(), 1e-6)
    directional = ((grad - grad.min()) / spread - 0.5) * 2 * rng.uniform(-50, 50)

    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    radius = rng.uniform(max(w, h) * 0.5, max(w, h) * 1.4)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radial = np.clip(1 - dist / radius, -1, 1) * rng.uniform(-40, 40)

    return directional + radial


def make_background(size, rng: random.Random, severity: float):
    """Paper or brushed metal: a base tone, fibrous grain, uneven light."""
    w, h = size
    base = rng.randint(140, 245)
    tint = rng.choice([(0, 0, 0), (8, 5, -6), (-6, -3, 8), (5, -3, -6)])
    arr = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = base + tint[c]

    rs = np.random.RandomState(rng.getrandbits(32))
    grain = rs.normal(0, 1, size=(h, w)).astype(np.float32)
    grain_img = Image.fromarray(np.clip(grain * 20 + 128, 0, 255).astype(np.uint8), "L")
    grain_img = grain_img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    grain_amt = (np.asarray(grain_img).astype(np.float32) - 128) * rng.uniform(0.12, 0.35)
    arr += (grain_amt * min(severity, 2.5))[:, :, None]
    arr += (uneven_light(size, rng) * min(0.6 + 0.2 * severity, 1.6))[:, :, None]

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def degrade(image: Image.Image, rng: random.Random, severity: float):
    """Focus blur, low/uneven contrast, sensor noise, then a lossy JPEG
    round-trip - the chain a photo goes through before a model ever sees it."""
    low = max(0.15, 0.75 - 0.12 * severity)
    high = max(low + 0.05, 0.85 - 0.05 * severity)
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(low, high))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.18))

    sigma_max = min(0.5 + 0.9 * severity, 4.5)
    if rng.random() < 0.9:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, sigma_max)))

    amp = int(min(10 + 8 * severity, 45))
    rs = np.random.RandomState(rng.getrandbits(32))
    arr = np.asarray(image).astype(np.int16) + rs.randint(-amp, amp + 1, size=np.asarray(image).shape)
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    if rng.random() < 0.92:
        q_lo = max(5, int(45 - 9 * severity))
        q_hi = max(q_lo + 5, int(65 - 8 * severity))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=rng.randint(q_lo, q_hi))
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
    return image


def render_photo(text: str, font_path: Path, rng: random.Random, severity: float):
    """One word/phrase as a photo-realistic crop, saved out as PNG - the JPEG
    damage above is baked into the pixels, not the file format, so the output
    still matches the .png files the trainer already expects."""
    size = rng.randint(26, 72)
    font = ImageFont.truetype(str(font_path), size)
    layer = make_text_layer(text, font, rng)
    if rng.random() < 0.6:
        layer = layer.rotate(rng.uniform(-7, 7), expand=True, resample=Image.BICUBIC)
    canvas = warp_perspective(layer, rng, severity)
    bg = make_background(canvas.size, rng, severity).convert("RGBA")
    composed = Image.alpha_composite(bg, canvas).convert("RGB")
    return degrade(composed, rng, severity)


def diacritic_counts(truth: str, pred: str) -> tuple:
    """How many Bosnian letters in `truth` are still in `pred` - counted, not
    position-aligned, so one dropped/inserted character elsewhere in the word
    cannot masquerade as a diacritic error. Same approach as the (corrected)
    surviving_diacritics in training/train_ocr.py; reimplemented locally
    rather than imported, since this file is meant to stand on its own."""
    right = total = 0
    for ch in BOSNIAN_LETTERS:
        want = truth.count(ch)
        if want:
            total += want
            right += min(want, pred.count(ch))
    return right, total


def levenshtein(a: str, b: str) -> int:
    """Plain edit distance. These are single words or short phrases, well
    under a hundred characters, so the classic O(len(a)*len(b)) table is fast
    enough without reaching for a library."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def classify_error(truth: str, pred: str) -> str:
    """Which kind of wrong this is, not just that it is wrong.

    "exact" - no error at all.
    "clean_drop" - the diacritic-stripped word exactly, e.g. Caršija for
        Čaršija: the textbook version of the real failure.
    "near_clean_drop" - the mark dropped plus one stray plain-character slip;
        still clearly the same word.
    "substitution" - a letter changed to a different one (Backa for Đačka)
        but the word's length and shape are basically intact - still a
        letter-level failure, not a destroyed image.
    "garbage" - the prediction is not close to either the original or its
        diacritic-stripped form. This is what an over-degraded image looks
        like: the model would be learning to guess, not to read through a
        faint mark, so it does not count toward "the pattern we want."
    """
    if pred == truth:
        return "exact"
    base = truth.translate(DIACRITIC_STRIP)
    if pred == base:
        return "clean_drop"
    len_ratio = len(pred) / max(len(truth), 1)
    if levenshtein(pred, base) <= 1 and 0.7 <= len_ratio <= 1.3:
        return "near_clean_drop"
    if (min(levenshtein(pred, base), levenshtein(pred, truth)) <= max(2, len(truth) // 3)
            and 0.6 <= len_ratio <= 1.4):
        return "substitution"
    return "garbage"


def realistic_count(stats: dict) -> int:
    p = stats["pattern"]
    return p.get("clean_drop", 0) + p.get("near_clean_drop", 0) + p.get("substitution", 0)


def garbage_share(stats: dict) -> float:
    p = stats["pattern"]
    errors = sum(v for k, v in p.items() if k != "exact")
    return p["garbage"] / errors if errors else 0.0


def measure(predict, samples: list) -> dict:
    """Read each (image, truth) back through `predict` and score it, both as
    a raw letter-survival count and as an error-pattern breakdown."""
    exact = 0
    dia_right = dia_total = 0
    pattern = Counter()
    examples = {}
    for image, truth in samples:
        pred = predict(image).strip()
        exact += pred == truth
        right, total = diacritic_counts(truth, pred)
        dia_right += right
        dia_total += total
        if total:  # only words carrying a Bosnian letter go into the pattern breakdown
            kind = classify_error(truth, pred)
            pattern[kind] += 1
            examples.setdefault(kind, (truth, pred))
    return {"n": len(samples), "exact": exact, "dia_right": dia_right,
            "dia_total": dia_total, "pattern": pattern, "examples": examples}


def report(label: str, stats: dict) -> None:
    dropped = stats["dia_total"] - stats["dia_right"]
    rate = 100 * stats["dia_right"] / max(stats["dia_total"], 1)
    p = stats["pattern"]
    print(f"  {label}: {dropped}/{stats['dia_total']} Bosnian letters dropped "
          f"({rate:.1f}% survived), {stats['exact']}/{stats['n']} words exact")
    print(f"    pattern - clean drop: {p.get('clean_drop', 0)}, "
          f"near-clean: {p.get('near_clean_drop', 0)}, "
          f"substitution: {p.get('substitution', 0)}, "
          f"garbage: {p.get('garbage', 0)}, exact: {p.get('exact', 0)}")


def app_ocr_predictor():
    """Wraps app/ocr.py's own reader - what the task named as the reading
    oracle, and still what a photo gets read with if these images were ever
    fed to the app directly without retraining anything.

    IMPORTANT, verified by direct test (a perfectly clean, undistorted image
    reading "čćđšž ČĆĐŠŽ", no degradation at all): calling
    reader.recognize()/readtext() the way app/ocr.py's own scan() does - no
    allowlist - CANNOT produce č, ć, đ, Č, Ć or Đ at all, ever, regardless of
    image quality. easyocr.Reader(["bs","en"], ...) builds a per-language
    character mask (reader.lang_char is missing exactly those six; š and ž
    are NOT masked), and recognize()/readtext() apply it by default, zeroing
    those characters out of the decodable output before greedy decoding ever
    runs. Passing allowlist=reader.character (the full 351-class set) on the
    clean test image fixed it completely. This is an app/ocr.py-level bug
    independent of image content or model quality - reported upstream, not
    fixed here (out of scope: the task says not to touch app/). Left unfixed
    here, it would swamp every measurement below with a constant,
    degradation-independent "failure" on those six letters, so the allowlist
    is passed here purely so this script measures what the degradation
    actually does, not this unrelated masking."""
    from app.ocr import get_reader
    reader = get_reader()

    def predict(image):
        arr = np.array(image.convert("RGB"))
        # recognize(), not readtext(): this is a pre-cropped single word/phrase,
        # so skip the separate text-detector and go straight to the recognizer -
        # a detector box sometimes just doesn't "find" a low-contrast crop and
        # silently reports nothing at all, which would prove nothing here.
        # allowlist=reader.character: see the masking bug in this function's
        # docstring - without it, six of the ten target letters can never
        # come out of this call no matter what the image looks like.
        out = reader.recognize(arr, detail=0, paragraph=False, allowlist=reader.character)
        return out[0] if out else ""
    return predict


def training_pipeline_predictor():
    """The preprocessing training/train_ocr.py's Crops/prepare() actually
    feeds the model - plain grayscale, resize-to-64-tall, [-1,1] normalise,
    one forward pass, greedy decode over the full 351-class output (no
    per-language character masking - see app_ocr_predictor()'s docstring).
    Read-only import from training/ (the task said not to write there;
    reading a pure function is not writing).

    Since the entire point of this dataset is to be trained on, THIS is the
    pipeline that decides whether it will teach the model anything, so it
    drives the severity calibration below. app_ocr_predictor() is reported
    alongside it for the same reason the task named app/ocr.py explicitly,
    not because it is what training will see.
    """
    import torch
    from torchvision import transforms
    import training.train_ocr as trainer
    from easyocr.utils import CTCLabelConverter

    charset = trainer.load_charset()
    converter = CTCLabelConverter(charset, separator_list={}, dict_pathlist={})
    model = trainer.build_model(len(converter.character))
    weights = trainer.READ_DIR / "latin_g2.pth"
    trainer.ensure_pristine(weights)
    trainer.load_weights(model, weights)
    model.eval()

    def predict(image):
        img = image.convert("L")
        ratio = img.width / max(img.height, 1)
        width = min(max(int(trainer.IMG_HEIGHT * ratio), 8), trainer.MAX_WIDTH)
        img = img.resize((width, trainer.IMG_HEIGHT), Image.BICUBIC)
        tensor = transforms.ToTensor()(img).sub_(0.5).div_(0.5).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor, None)
        return trainer.decode(logits, converter)[0]
    return predict


def calibrate(rng: random.Random, fonts: list, with_letters: list):
    """Render probes at every severity, then pick the one that produces the
    most realistic-pattern errors while keeping over-degraded ("garbage")
    errors a minority - not just the first severity that fails hard enough,
    since failing hard enough is trivially reachable by destroying the image.

    Decided against training_pipeline_predictor() (see its docstring for why
    that one, not app_ocr_predictor(), is the one that determines whether
    this dataset actually teaches the model anything)."""
    predict_train = training_pipeline_predictor()
    predict_app = app_ocr_predictor()
    print("calibrating degradation against the actual training pipeline...")
    results = []
    for severity in SEVERITY_STEPS:
        probe_rng = random.Random(rng.getrandbits(32))
        words = [probe_rng.choice(with_letters) for _ in range(PROBE_SIZE)]
        font_choices = [probe_rng.choice(fonts) for _ in range(PROBE_SIZE)]
        probe = [(render_photo(w, f, probe_rng, severity), w)
                 for w, f in zip(words, font_choices)]
        stats = measure(predict_train, probe)
        report(f"severity {severity:.1f}", stats)
        results.append((severity, stats))

    qualifying = [(s, st) for s, st in results
                  if realistic_count(st) >= MIN_REALISTIC_ERRORS
                  and garbage_share(st) <= MAX_GARBAGE_SHARE]
    if qualifying:
        severity, stats = max(qualifying, key=lambda pair: realistic_count(pair[1]))
        print(f"  chosen: severity {severity:.1f} - {realistic_count(stats)} "
              f"realistic diacritic-drop errors, {100 * garbage_share(stats):.0f}% garbage")
        return severity, predict_train, predict_app

    with_errors = [(s, st) for s, st in results if realistic_count(st) > 0]
    severity, stats = min(with_errors or results, key=lambda pair: garbage_share(pair[1]))
    print(f"  no severity cleanly cleared the bar - using {severity:.1f} anyway "
          f"({realistic_count(stats)} realistic errors, "
          f"{100 * garbage_share(stats):.0f}% garbage); proving on the full run below")
    return severity, predict_train, predict_app


def generate_batch(count, severity, rng, fonts, with_letters, without, words_per_image,
                   train_f, valid_f, next_i):
    produced = []
    letters_seen = Counter()
    for i in range(count):
        pool = with_letters if rng.random() < DIACRITIC_SHARE else without
        text = " ".join(rng.choice(pool) for _ in range(words_per_image))
        roll = rng.random()
        if roll < clean_gen.UPPER_SHARE:
            text = text.upper()
        elif roll < clean_gen.UPPER_SHARE + clean_gen.TITLE_SHARE:
            text = text.title()

        image = render_photo(text, rng.choice(fonts), rng, severity)
        # Which side a word belongs on is a property of the word, not of this
        # run's rng. training/prepare_ocr_data.py renders the same strings from
        # the same corpus; a coin flip here put this photographed Čaršija in
        # valid while its synthetic twin went to train, and half the validation
        # set ended up being text the reader had trained on.
        name = f"photo{next_i:06d}.png"
        next_i += 1
        if is_valid_text(text):
            path, sheet = OUT_VALID / name, valid_f
        else:
            path, sheet = OUT_TRAIN / name, train_f
        image.save(path)
        sheet.write(f"{name}\t{text}\n")

        produced.append((path, text))
        letters_seen.update(c for c in text if c in BOSNIAN_LETTERS)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,}/{count:,}")
    return produced, letters_seen, next_i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4000)
    ap.add_argument("--words", type=int, default=1, help="words per image")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--verify-sample", type=int, default=400,
                    help="how many freshly written images to re-read as proof")
    ap.add_argument("--max-topups", type=int, default=2,
                    help="extra higher-severity batches to add if the proof "
                         "sample somehow comes back clean")
    args = ap.parse_args()

    fonts = clean_gen.find_fonts()
    if not fonts:
        print("no fonts with Bosnian letters found on this machine", file=sys.stderr)
        return 1
    with_letters, without = clean_gen.load_words()
    print(f"fonts: {len(fonts)}  words: {len(with_letters):,} with diacritics, "
          f"{len(without):,} without")

    rng = random.Random(args.seed)
    severity, predict_train, predict_app = calibrate(
        random.Random(rng.getrandbits(32)), fonts, with_letters)
    print(f"using severity {severity:.1f} for the full run\n")

    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_VALID.mkdir(parents=True, exist_ok=True)
    # One counter across both folders, not one per split. Numbering per side
    # produced photo000000.png in train and a different photo000000.png in
    # valid — 402 duplicated names — which makes a row impossible to move
    # between splits and silently pairs the wrong picture with a label in
    # anything that joins the two lists.
    # Read the leading number off every existing name, including ones a repair
    # renamed to photoNNNNNNmNNN.png, so the counter cannot restart on top of
    # files that are already there.
    used = [re.match(r"photo(\d+)", f.stem) for folder in (OUT_TRAIN, OUT_VALID)
            for f in folder.glob("photo*.png")]
    next_i = max((int(m.group(1)) for m in used if m), default=-1) + 1

    all_produced = []
    letters_total = Counter()
    with open(OUT_TRAIN / "gt.txt", "a", encoding="utf-8") as train_f, \
         open(OUT_VALID / "gt.txt", "a", encoding="utf-8") as valid_f:
        produced, letters, next_i = generate_batch(
            args.count, severity, rng, fonts, with_letters, without, args.words,
            train_f, valid_f, next_i)
        all_produced += produced
        letters_total += letters

        sample_n = min(args.verify_sample, len(all_produced))
        sample = rng.sample(all_produced, sample_n)
        pairs = [(Image.open(p).convert("RGB"), t) for p, t in sample]
        stats = measure(predict_train, pairs)

        topups = 0
        while topups < args.max_topups and not (
                realistic_count(stats) >= MIN_FINAL_REALISTIC
                and garbage_share(stats) <= MAX_GARBAGE_SHARE):
            topups += 1
            if garbage_share(stats) > MAX_GARBAGE_SHARE and realistic_count(stats) > 0:
                severity = max(severity * 0.7, MIN_SEVERITY)
                why = "too much garbage - backing off"
            else:
                severity = min(severity * 1.4, SEVERITY_STEPS[-1] * 1.6)
                why = "not enough of the realistic pattern - raising severity"
            print(f"\nfinal sample not solid ({why}) -> severity {severity:.1f} "
                  f"({topups}/{args.max_topups})")
            extra, letters, next_i = generate_batch(
                max(200, args.verify_sample), severity, rng, fonts, with_letters,
                without, args.words, train_f, valid_f, next_i)
            all_produced += extra
            letters_total += letters
            sample_n = min(args.verify_sample, len(extra))
            sample = rng.sample(extra, sample_n)
            pairs = [(Image.open(p).convert("RGB"), t) for p, t in sample]
            stats = measure(predict_train, pairs)

    print(f"\n{len(all_produced):,} images -> {OUT_TRAIN} / {OUT_VALID}")
    print("Bosnian letters in the generated labels:")
    for letter in BOSNIAN_LETTERS:
        print(f"  {letter}: {letters_total.get(letter, 0):>6}")

    print(f"\nproof #1 - {stats['n']} freshly written images through the "
          f"ACTUAL TRAINING preprocessing (training/train_ocr.py's "
          f"Crops/prepare/decode - what deciding the severity was based on):")
    report("training pipeline", stats)

    app_stats = measure(predict_app, pairs)
    print(f"\nproof #2 - the same {app_stats['n']} images through app/ocr.py's "
          f"reader (allowlist-corrected - see app_ocr_predictor()):")
    report("app/ocr.py", app_stats)

    if stats["examples"]:
        print("  training-pipeline example per pattern actually produced:")
        for kind, (truth, pred) in stats["examples"].items():
            print(f"    {kind}: {truth!r} -> {pred!r}")

    if realistic_count(stats) == 0:
        print("\nFAILED TO PROVE the real failure pattern (diacritic dropped, "
              "word still recognisable) even after topping up severity - do "
              "not trust this dataset without investigating further.",
              file=sys.stderr)
        return 1
    if garbage_share(stats) > MAX_GARBAGE_SHARE:
        print(f"\nWARNING: {100 * garbage_share(stats):.0f}% of the errors in "
              f"the final sample are over-degraded garbage, not the diacritic-"
              f"drop pattern - some images in this batch are probably too far "
              f"gone to teach the model anything useful.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
