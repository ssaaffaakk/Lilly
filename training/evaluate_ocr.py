#!/usr/bin/env python3
"""What the reader actually reads off a photograph taken in Bosnia.

Every OCR number in this project so far was measured on text we synthesised
ourselves — rendered onto backgrounds with blur and skew applied. That measures
the reader against our own imitation of a photograph, and if the imitation is too
easy the number is meaningless. This scores it against photographs nobody in this
project produced, transcribed by eye rather than by machine.

    python3 training/evaluate_ocr.py --truth data/ocr/real-photos/truth.json

Three numbers, because one hides the thing that matters:

  word recall           did the reader produce the words that are on the sign
  diacritic word recall the same, over only the words carrying c-caron, c-acute,
                        d-stroke, s-caron or z-caron
  diacritic-blind recall the same words with those letters folded to their plain
                        forms

The gap between the last two is the cost of the diacritics specifically. If
diacritic-blind recall is high and diacritic recall is low, the reader is finding
the words and getting the letters wrong, which is a different repair from not
finding them at all. Reporting a single average would hide exactly that.
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DIACRITICS = "čćđšžČĆĐŠŽ"
FOLD = str.maketrans({
    "č": "c", "ć": "c", "đ": "d", "š": "s", "ž": "z",
    "Č": "c", "Ć": "c", "Đ": "d", "Š": "s", "Ž": "z",
})
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def words(text: str) -> list:
    """Lowercased word tokens, diacritics kept.

    Punctuation is dropped because a reader that returns "ULICA" for a sign
    reading "ULICA:" has not made a mistake anybody cares about. Digits are
    dropped from this particular count because house numbers and years are a
    different skill from reading Bosnian words, and mixing them in would let a
    photograph of a numbered plate flatter the letter accuracy.
    """
    return [w.lower() for w in WORD.findall(unicodedata.normalize("NFC", text))]


def has_diacritic(word: str) -> bool:
    return any(ch in DIACRITICS for ch in word)


def fold(word: str) -> str:
    return word.translate(FOLD)


def recall(truth: list, found: list, key=lambda w: w) -> tuple:
    """How many of the truth words appear among the found words.

    Multiset, not set: a sign reading "ULICA ULICA" needs both. Counter
    intersection does exactly that and is why this is not a set operation.
    """
    want = Counter(key(w) for w in truth)
    got = Counter(key(w) for w in found)
    hit = sum((want & got).values())
    return hit, sum(want.values())


def short(path: Path) -> str:
    """Repository-relative when it can be, absolute when it cannot."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_truth(path: Path) -> dict:
    """The agreed answer key: filename -> list of text lines."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: entry for name, entry in data["photos"].items()}


def read_photo_full(path: Path) -> str:
    """The same read with the shrink removed.

    app.ocr.scan shrinks anything over two megapixels before reading, and the
    people who wrote the answer key enlarged the photographs to read them. If
    the reader is missing words the shrink threw away, that is a product
    decision anyone can change; if it misses them at full resolution too, it is
    the model. The two numbers separate those, and without them the headline
    figure cannot say which repair it is asking for.
    """
    import numpy as np
    from PIL import Image
    from app.ocr import read_regions, MAX_DECLARED_PIXELS

    Image.MAX_IMAGE_PIXELS = MAX_DECLARED_PIXELS
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"))
    return "\n".join(read_regions(array, detail=0, paragraph=True))


def read_photo(path: Path) -> str:
    """What the app itself returns for this photograph.

    `app.ocr.scan` and not `read_regions`, because scan is what the photo
    endpoint serves: it shrinks anything over two megapixels first and groups
    the regions into paragraphs. Calling the region reader directly skips both
    and scores a path no user is on — the same mistake the translation
    evaluation made when it fed whole rows to a model the app feeds sentence by
    sentence, and that one moved the answer by 3.36 chrF2.
    """
    from app.ocr import scan
    return scan(str(path))


def cached_reads(names: list, photos: Path, cache: Path, full=False) -> dict:
    """Read every photograph once, keep the answers.

    Reading is about two minutes per photograph on this machine, so a re-score
    after correcting the answer key would otherwise cost another hour and a
    half. The cache holds the reader's output only; the answer key it is scored
    against lives in a separate file written by people looking at the pictures,
    and the two never meet until here.
    """
    have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [n for n in names if n not in have]
    if todo:
        print(f"reading {len(todo)} photographs ({len(have)} already cached)")
    for i, name in enumerate(todo, 1):
        start = time.time()
        try:
            have[name] = (read_photo_full if full else read_photo)(photos / name)
        except Exception as exc:
            print(f"  {i}/{len(todo)} {name}: FAILED {exc}", file=sys.stderr)
            continue
        cache.write_text(json.dumps(have, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"  {i}/{len(todo)} {name}  {time.time() - start:.0f}s  "
              f"{len(have[name].split())} words", flush=True)
    return have


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/truth.json")
    # The scored photographs used to be read straight out of the harvester's
    # staging area. That area is scratch and the harvester treats it as scratch:
    # a photograph judged `drop` has its rendering unlinked the moment the
    # verdict is written, and the whole directory is emptied at the end of a
    # run. Twenty-five of the forty were already gone when this was noticed, and
    # the run still going would have taken the rest -- leaving the reader's 36%
    # as a number with nothing left to re-measure it against.
    #
    # data/ocr/real-photos/scored/ is not on any harvester's path. What is in it
    # are the same 1280px renderings that were read, restored from the recorded
    # screen_url by data/scripts/restore_scored_photos.py when one goes missing.
    ap.add_argument("--photos", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/scored")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "training/RESULTS-ocr.md")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/reader-output.json")
    ap.add_argument("--full-res", action="store_true",
                    help="read at native resolution instead of the app's "
                         "two-megapixel working size")
    ap.add_argument("--read-only", action="store_true",
                    help="read the sampled photographs into the cache and stop; "
                         "lets the slow half run while the answer key is written")
    ap.add_argument("--sample", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/scored-sample.txt")
    args = ap.parse_args()

    # Reading is the slow half and does not depend on the answer key, so it can
    # run first. Keeping it behind its own flag also keeps the two apart: the
    # people transcribing never see this output, which is the only reason the
    # comparison means anything.
    if args.read_only:
        names = args.sample.read_text(encoding="utf-8").split("\n")
        names = [n for n in names if n.strip()][:args.limit or None]
        cached_reads(names, args.photos, args.cache, args.full_res)
        print(f"\ncached {len(names)} readings in {short(args.cache)}")
        return 0

    if not args.truth.exists():
        raise SystemExit(
            f"no answer key at {args.truth}\n"
            "Transcribe the sampled photographs by eye first — scoring the reader "
            "against another reader's output measures agreement, not accuracy.")

    full = json.loads(args.truth.read_text(encoding="utf-8"))
    agreement = full.get("agreement", {}).get("rate", "?")
    truth = load_truth(args.truth)
    names = sorted(truth)[:args.limit] if args.limit else sorted(truth)
    readings = cached_reads(names, args.photos, args.cache, args.full_res)

    totals = {"plain": [0, 0], "dia": [0, 0], "blind": [0, 0]}
    rows, spurious, empty_truth = [], 0, 0

    for i, name in enumerate(names, 1):
        entry = truth[name]
        want = words(" ".join(entry["lines"]))
        if not want:
            empty_truth += 1
            continue
        if name not in readings:
            print(f"  {i}/{len(names)} {name}: never read", file=sys.stderr)
            continue
        found = words(readings[name])

        hit, need = recall(want, found)
        dia_want = [w for w in want if has_diacritic(w)]
        dhit, dneed = recall(dia_want, found)
        bhit, bneed = recall(dia_want, found, key=fold)

        totals["plain"][0] += hit;  totals["plain"][1] += need
        totals["dia"][0] += dhit;   totals["dia"][1] += dneed
        totals["blind"][0] += bhit; totals["blind"][1] += bneed
        spurious += max(0, len(found) - hit)

        rows.append((name, hit, need, dhit, dneed, bhit, bneed))
        print(f"  {i}/{len(names)} {name}: {hit}/{need} words"
              + (f", {dhit}/{dneed} diacritic" if dneed else ""), flush=True)

    def pct(pair):
        hit, need = pair
        return 100.0 * hit / need if need else float("nan")

    # A rate over pooled words is the rate of whichever photograph happens to
    # carry the most text. One memorial slab in this sample holds 144 of the
    # 373 words in the answer key — 39% of it — and it is in Spanish, so a
    # single picture that is not even the task would be setting the headline.
    # The per-photograph mean weights every photograph the same, which is closer
    # to what a person experiences: they point the camera once and either get
    # the sign or they do not.
    macro = sum(h / n for _n_, h, n, *_r in rows) / len(rows) * 100 if rows else 0
    biggest = max(rows, key=lambda r: r[2]) if rows else None

    lines = [
        "# What the reader reads off a real photograph",
        "",
        f"{len(rows)} photographs from Wikimedia Commons, none of them ours, "
        f"transcribed by eye before the reader was run on them. "
        f"{empty_truth} more carried no legible text and are excluded from the "
        f"rates below — a photograph with nothing to read cannot be read wrongly.",
        "",
        "| | words found | of | rate |",
        "|---|---|---|---|",
        f"| All words | {totals['plain'][0]} | {totals['plain'][1]} | "
        f"**{pct(totals['plain']):.1f}%** |",
        f"| Words with č ć đ š ž | {totals['dia'][0]} | {totals['dia'][1]} | "
        f"**{pct(totals['dia']):.1f}%** |",
        f"| The same words, diacritics folded away | {totals['blind'][0]} | "
        f"{totals['blind'][1]} | **{pct(totals['blind']):.1f}%** |",
        "",
        f"Weighting every photograph equally instead of every word, the reader "
        f"finds **{macro:.1f}%** of the words on a photograph. The two differ "
        f"because the text is not spread evenly: "
        + (f"`{biggest[0]}` alone holds {biggest[2]} of the "
           f"{totals['plain'][1]} words in the answer key. " if biggest else "")
        + "The per-photograph figure is the one that describes pointing a camera "
        "at a sign; the pooled figure describes reading a wall of text.",
        "",
        f"Two independent readers transcribed these photographs without seeing "
        f"each other's work or the machine's, and agreed on {agreement}% of the "
        f"words either of them saw. Only the words both saw are in the answer "
        f"key, so that agreement is also the ceiling on how precise anything "
        f"here can be.",
        "",
        f"The reader also returned {spurious} words that are not on any sign in "
        "these photographs. That is the cost a user pays for text the detector "
        "invented out of brickwork and foliage, and it is not visible in a recall "
        "figure.",
        "",
        "## Per photograph",
        "",
        "| Photograph | words | diacritic words |",
        "|---|---|---|",
    ]
    for name, hit, need, dhit, dneed, _b, _bn in sorted(rows, key=lambda r: r[1] / r[2]):
        lines.append(f"| {name} | {hit}/{need} | " +
                     (f"{dhit}/{dneed} |" if dneed else "— |"))
    lines += ["", "---", "", "Generated by `training/evaluate_ocr.py`."]

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nall words        {pct(totals['plain']):.1f}%  (pooled)")
    print(f"per photograph   {macro:.1f}%")
    print(f"diacritic words  {pct(totals['dia']):.1f}%")
    print(f"  folded         {pct(totals['blind']):.1f}%")
    print(f"\nwrote {short(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
