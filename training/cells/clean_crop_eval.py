# 6c. Score both readers on the clean human crops — on this GPU, not a laptop.
# labels-human-latin.tsv is the 737 of 1,701 human crops a Latin-only reader
# can fairly be asked to read: no Cyrillic (latin_g2 has none in its character
# set), nothing under 16px tall or 32px wide (EasyOCR resizes everything to
# 64px, and Coca-Cola came back empty from a 22x18 crop), no label under three
# letters.
#
# Diacritics are folded. `kuca` still translates to House, so a missing hat is
# not a wrong word for this product.
import shutil
import unicodedata

import easyocr

CROPS = Path("data/ocr/crops")
CLEAN = CROPS / "labels-human-latin.tsv"
if not CLEAN.is_file():
    raise SystemExit(f"{CLEAN} missing — the clone is stale")

rows = []
for line in CLEAN.read_text(encoding="utf-8").splitlines()[1:]:
    parts = line.split("\t")
    if len(parts) >= 2 and parts[1].strip() and (CROPS / parts[0]).is_file():
        rows.append((parts[0], parts[1].strip()))
print(f"scoring on {len(rows):,} clean human crops")
if len(rows) < 500:
    raise SystemExit(f"only {len(rows)} crops staged — lilly-ocr-crops did not land")


def fold(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


def score(weights):
    shutil.copy(weights, READ / "lilly.pth")
    rd = easyocr.Reader(["bs", "en"], gpu=True, verbose=False,
                        model_storage_directory=str(READ),
                        user_network_directory=str(READ / "user_network"),
                        recog_network="lilly")
    exact = folded = 0
    for name, truth in rows:
        got = " ".join(rd.readtext(str(CROPS / name), detail=0)).strip()
        exact += got == truth
        folded += fold(got) == fold(truth)
    return exact, folded


# Keep the reader the app would load untouched: score a copy, and put it back.
shutil.copy(INIT, READ / "lilly-shipped.pth")
before = score(READ / "lilly-shipped.pth")
after = score(TRAINED)
shutil.copy(READ / "lilly-shipped.pth", READ / "lilly.pth")

n = len(rows)
print("")
print(f"{'':<10}{'exact':>16}{'ignoring diacritics':>26}")
print(f"{'shipped':<10}{before[0]:>7}/{n:<5}{100 * before[0] / n:>6.1f}%"
      f"{before[1]:>12}/{n:<5}{100 * before[1] / n:>6.1f}%")
print(f"{'trained':<10}{after[0]:>7}/{n:<5}{100 * after[0] / n:>6.1f}%"
      f"{after[1]:>12}/{n:<5}{100 * after[1] / n:>6.1f}%")
print("")
print(f"delta: exact {after[0] - before[0]:+d}, folded {after[1] - before[1]:+d}")

OFF.metric("clean_exact_before", before[0], stage="clean-crops")
OFF.metric("clean_exact_after", after[0], stage="clean-crops")
OFF.metric("clean_folded_before", before[1], stage="clean-crops")
OFF.metric("clean_folded_after", after[1], stage="clean-crops")

if after[1] > before[1]:
    print("this reader reads the clean human crops better than the shipped one")
else:
    print("no gain on the clean human crops — the weights are in Output for "
          "inspection, not for installing")
