#!/usr/bin/env python3
"""Make training images for Lilly's photo reading.

The reader gets Bosnian letters wrong — Čaršija comes back as Caršija, Đačka as
Backa — because whoever trained it had little Bosnian text. Synthetic data fixes
exactly that, and beats collecting photos for this purpose: you decide how often
each letter appears, and the label is right by construction.

    python3 data/scripts/generate_ocr_data.py                  # 5,000 images
    python3 data/scripts/generate_ocr_data.py --count 20000
    python3 data/scripts/generate_ocr_data.py --words 2        # short phrases

Writes data/ocr/synthetic/*.png and a labels.tsv, then the same splits the
trainer reads. Words come from the cleaned corpus if it is there, otherwise from
a built-in list; either way lines containing č ć đ š ž are over-sampled, since
those are the ones that need the practice.
"""
import argparse
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = DATA_DIR / "ocr" / "synthetic"
CORPUS = DATA_DIR / "clean" / "train.tsv"
BOSNIAN_LETTERS = "čćđšžČĆĐŠŽ"
DIACRITIC_SHARE = 0.55   # aim for this share of samples to contain one
# Signs and menus — the things people actually photograph — are often shouted in
# capitals, but corpus text is almost all lower case. Mix the case in ourselves,
# or the reader never sees an uppercase Č and keeps getting it wrong.
UPPER_SHARE, TITLE_SHARE = 0.25, 0.15
SEED = 41

FALLBACK_WORDS = """
čaršija đačka škola pušenje žene džamija ćevapi burek ulica trg most stari
zabranjeno izlaz ulaz otvoreno zatvoreno radno vrijeme cijena hvala molim
dobrodošli oprez pažnja informacije polazak dolazak peron autobus željeznica
apoteka bolnica policija restoran kafana pekara pijaca banka pošta muzej
džemal ćošak žuti šećer noćenje četvrtak subota nedjelja mjesečno godišnje
""".split()

FONT_DIRS = ("/System/Library/Fonts/Supplemental", "/System/Library/Fonts",
             "/Library/Fonts", "/usr/share/fonts")


def find_fonts(limit: int = 24) -> list:
    """Different fonts is what stops the reader memorising one letter shape."""
    fonts = []
    for folder in FONT_DIRS:
        base = Path(folder)
        if base.is_dir():
            fonts += [p for p in sorted(base.rglob("*"))
                      if p.suffix.lower() in {".ttf", ".otf"}]
    # a font without the Bosnian letters would produce boxes, so check one
    from PIL import ImageFont
    usable = []
    for path in fonts:
        try:
            font = ImageFont.truetype(str(path), 32)
            if font.getbbox("č")[2] > 0 and font.getbbox("đ")[2] > 0:
                usable.append(path)
        except Exception:  # noqa: BLE001 - a font we cannot open is simply skipped
            continue
        if len(usable) >= limit:
            break
    return usable


def load_words() -> tuple:
    """Words from our own corpus if it exists, split into two buckets."""
    words = []
    if CORPUS.exists():
        for line in CORPUS.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                words += [w.strip(".,!?;:()\"'„“") for w in parts[1].split()]
        words = [w for w in words if 2 <= len(w) <= 22]
    if len(words) < 500:
        words = FALLBACK_WORDS * 40
    with_letters = [w for w in words if any(c in BOSNIAN_LETTERS for c in w)]
    without = [w for w in words if not any(c in BOSNIAN_LETTERS for c in w)]
    return (with_letters or FALLBACK_WORDS), (without or FALLBACK_WORDS)


def render(text: str, font_path: Path, rng: random.Random):
    """One line of text as a photo-ish crop: off-white paper, ink, a little grain."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    size = rng.randint(28, 64)
    font = ImageFont.truetype(str(font_path), size)
    pad = rng.randint(6, 18)
    box = font.getbbox(text)
    width, height = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad

    paper = rng.randint(215, 255)
    ink = rng.randint(0, 70)
    image = Image.new("L", (width, height), paper)
    ImageDraw.Draw(image).text((pad - box[0], pad - box[1]), text, fill=ink, font=font)

    if rng.random() < 0.5:
        image = image.rotate(rng.uniform(-3.5, 3.5), expand=True,
                             fillcolor=paper, resample=Image.BILINEAR)
    if rng.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    if rng.random() < 0.5:
        pixels = image.load()
        for _ in range(int(width * height * 0.02)):
            x, y = rng.randrange(image.width), rng.randrange(image.height)
            pixels[x, y] = max(0, min(255, pixels[x, y] + rng.randint(-40, 40)))
    return image.convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--words", type=int, default=1, help="words per image")
    ap.add_argument("--build-splits", action="store_true", default=True,
                    help="also run the split builder when done")
    args = ap.parse_args()

    fonts = find_fonts()
    if not fonts:
        print("no fonts with Bosnian letters found on this machine", file=sys.stderr)
        return 1
    with_letters, without = load_words()
    print(f"fonts: {len(fonts)}  words: {len(with_letters):,} with diacritics, "
          f"{len(without):,} without")

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    letters = Counter()
    with open(OUT_DIR / "labels.tsv", "w", encoding="utf-8") as f:
        for i in range(args.count):
            pool = with_letters if rng.random() < DIACRITIC_SHARE else without
            text = " ".join(rng.choice(pool) for _ in range(args.words))
            roll = rng.random()
            if roll < UPPER_SHARE:
                text = text.upper()
            elif roll < UPPER_SHARE + TITLE_SHARE:
                text = text.title()
            name = f"syn{i:06d}.png"
            render(text, rng.choice(fonts), rng).save(OUT_DIR / name)
            f.write(f"{name}\t{text}\t1.00\n")
            letters.update(c for c in text if c in BOSNIAN_LETTERS)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1:,}/{args.count:,}")

    print(f"\n{args.count:,} images -> {OUT_DIR}")
    print("Bosnian letters in the set:")
    for letter in BOSNIAN_LETTERS:
        print(f"  {letter}: {letters.get(letter, 0):>6}")

    if args.build_splits:
        subprocess.run([sys.executable,
                        str(DATA_DIR.parent / "training" / "prepare_ocr_data.py"),
                        "--labels", str(OUT_DIR / "labels.tsv")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
