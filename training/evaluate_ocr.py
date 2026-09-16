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
import hashlib
import json
import os
import platform
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# Refuse to start if the machine has no room. Five of these ran at once on
# 27 August and the kernel panicked: 100% of the compressor limit, fifteen
# swapfiles, watchdog silent for 94 seconds. Each job is reasonable alone and
# none of them knew the others existed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.guard import claim
claim(1.4, "photo scoring")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DIACRITICS = "čćđšžČĆĐŠŽ"
FOLD = str.maketrans({
    "č": "c", "ć": "c", "đ": "d", "š": "s", "ž": "z",
    "Č": "c", "Ć": "c", "Đ": "d", "Š": "s", "Ž": "z",
})
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
CACHE_SCHEMA = 2
HASH_CHUNK_SIZE = 1 << 20


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


def file_sha256(path: Path) -> str:
    """Stable identity for an input or weight file, independent of its mtime."""
    if not path.is_file():
        raise SystemExit(f"cache provenance needs {path}, but it is not a file")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_manifest(choice: str) -> dict:
    """Libraries and execution surface that can change an OCR reading."""
    import cv2
    import easyocr
    import numpy
    import PIL

    runtime = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "cv2": cv2.__version__,
        # Two distributions can expose the same short cv2 version while
        # installing different shared objects. The build text identifies what
        # Python actually imported, not what package metadata claims is there.
        "cv2_build_sha256": hashlib.sha256(
            cv2.getBuildInformation().encode("utf-8")
        ).hexdigest(),
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        # Paddle uses EasyOCR's paragraph grouper after recognition too.
        "easyocr": getattr(easyocr, "__version__", "?"),
    }
    if choice == "paddle":
        import paddle
        import paddleocr
        import paddlex
        runtime.update({
            "paddle": getattr(paddle, "__version__", "?"),
            "paddleocr": getattr(paddleocr, "__version__", "?"),
            "paddlex": getattr(paddlex, "__version__", "?"),
        })
    else:
        import torch
        runtime.update({
            "torch": getattr(torch, "__version__", "?"),
            "cuda_available": bool(torch.cuda.is_available()),
        })
    return runtime


def _hash_weight_set(role: str, directory: Path) -> dict:
    """Hash the three files Paddle inference actually consumes."""
    return {
        f"{role}/{name}": file_sha256(directory / name)
        for name in ("inference.pdiparams", "inference.yml", "inference.json")
    }


def _weight_manifest(choice: str, identity: str) -> dict:
    """Only the selected engine's weights; backups and other engines are noise."""
    from app import ocr

    if choice == "paddle":
        det, rec = ocr.paddle_models()
        cache_home = Path(os.environ.get(
            "PADDLE_PDX_CACHE_HOME", str(Path.home() / ".paddlex")
        ))
        official = cache_home / "official_models"
        weights = _hash_weight_set("detector", official / det)
        custom_rec = ocr.paddle_rec_dir()
        weights.update(_hash_weight_set(
            "recogniser", custom_rec if custom_rec is not None else official / rec
        ))
        if ocr.paddle_cyrillic_rescue():
            weights.update(_hash_weight_set(
                "rescue", official / ocr.CYRILLIC_RESCUE_REC
            ))
        return weights

    read_dir = ocr.READ_DIR
    files = {"detector/craft_mlt_25k.pth": read_dir / "craft_mlt_25k.pth"}
    if identity == "easyocr:stock":
        files["recogniser/latin_g2.pth"] = read_dir / "latin_g2.pth"
    else:
        files.update({
            "recogniser/lilly.pth": read_dir / "lilly.pth",
            "recogniser/lilly.py": read_dir / "user_network" / "lilly.py",
            "recogniser/lilly.yaml": read_dir / "user_network" / "lilly.yaml",
        })
        if choice == "cyrillic":
            files["recogniser/cyrillic_g2.pth"] = read_dir / "cyrillic_g2.pth"
    return {role: file_sha256(path) for role, path in sorted(files.items())}


def reader_cache_context(full: bool = False) -> dict:
    """Everything a cached OCR string depends on, with no paths or mtimes."""
    import inspect
    from app import ocr

    choice = ocr.reader_choice()
    identity = ocr.reader_identity()
    wrapper = read_photo_full if full else read_photo
    return {
        "schema": CACHE_SCHEMA,
        "reader_identity": identity,
        "runtime": _runtime_manifest(choice),
        "weights_sha256": _weight_manifest(choice, identity),
        "implementation_sha256": {
            "app/ocr.py": file_sha256(REPO_ROOT / "app" / "ocr.py"),
            "read_wrapper": hashlib.sha256(
                inspect.getsource(wrapper).encode("utf-8")
            ).hexdigest(),
        },
        "treatment": (
            "native-pixels-read_regions" if full else "shipped-scan-2mp-cap"
        ),
    }


def context_fingerprint(context: dict) -> str:
    """Short display key for a complete, structured cache context."""
    encoded = json.dumps(
        context, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def reader_fingerprint(full: bool = False) -> str:
    """Stable fingerprint of the selected reader, runtime and read treatment."""
    return context_fingerprint(reader_cache_context(full))


def write_reading_cache(cache: Path, context: dict, readings: dict) -> None:
    """Write a complete cache atomically so interruption cannot leave half JSON."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CACHE_SCHEMA,
        "reader": context_fingerprint(context),
        "context": context,
        "readings": readings,
    }
    temporary = cache.with_name(cache.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    temporary.replace(cache)


def load_reading_cache(cache: Path, context: dict) -> dict:
    """Load only v2 cache data produced under this exact measured context."""
    if not cache.exists():
        return {}
    try:
        raw = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{short(cache)} is not a readable OCR cache: {exc}")
    if not isinstance(raw, dict):
        raise SystemExit(f"{short(cache)} is not an OCR cache object")
    have = raw.get("readings", {})
    if have and raw.get("schema") != CACHE_SCHEMA:
        raise SystemExit(
            f"{short(cache)} is a legacy OCR cache (schema "
            f"{raw.get('schema', 1)}). It is evidence from a closed measurement, "
            "not permission to re-read it. Use a new cache path for an "
            "owner-approved run; do not delete or auto-migrate this file."
        )
    if raw.get("schema") not in (None, CACHE_SCHEMA):
        raise SystemExit(
            f"{short(cache)} uses unknown OCR cache schema {raw.get('schema')}"
        )
    if have and raw.get("context") != context:
        old = raw.get("reader", "unknown")
        new = context_fingerprint(context)
        raise SystemExit(
            f"{short(cache)} belongs to OCR context {old}, not {new}. "
            "Refusing to relabel cached text or launch a replacement pass "
            "implicitly; use a new cache path after owner approval."
        )
    if not isinstance(have, dict):
        raise SystemExit(f"{short(cache)} has a non-object readings field")
    return have


def cached_reads(
    names: list,
    photos: Path,
    cache: Path,
    full: bool = False,
    sealed: bool = False,
) -> dict:
    """Read every photograph once, keep the answers.

    Reading is about two minutes per photograph on this machine, so a re-score
    after correcting the answer key would otherwise cost another hour and a
    half. The cache holds the reader's output only; the answer key it is scored
    against lives in a separate file written by people looking at the pictures,
    and the two never meet until here.

    Cache schema 2 binds every string to the selected reader's real weight
    contents, runtime, code path, treatment and source-image contents. A
    mismatch stops instead of silently re-reading a closed experiment. A
    sealed cache is for ephemeral originals: every requested entry must already
    be present and source-hashed, and no missing row may be filled locally.
    """
    context = reader_cache_context(full)
    have = load_reading_cache(cache, context)
    text = {}
    for name in names:
        if name not in have:
            continue
        entry = have[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            raise SystemExit(
                f"{short(cache)} entry {name!r} predates source-bound cache "
                "entries; refusing to reuse it"
            )
        source_hash = entry.get("source_sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise SystemExit(
                f"{short(cache)} entry {name!r} has no source SHA-256"
            )
        if not sealed:
            path = photos / name
            current_hash = file_sha256(path)
            if current_hash != source_hash:
                raise SystemExit(
                    f"{name} changed under the same filename "
                    f"({source_hash[:12]} -> {current_hash[:12]}). Refusing a "
                    "mixed-input score; use a new cache after owner approval."
                )
        text[name] = entry["text"]

    todo = [n for n in names if n not in have]
    if sealed and todo:
        raise SystemExit(
            f"sealed cache {short(cache)} is missing {len(todo)}/{len(names)} "
            "photographs; refusing to fill them from a different photo set: "
            + ", ".join(todo[:5]) + ("..." if len(todo) > 5 else "")
        )
    if todo:
        print(f"reading {len(todo)} photographs ({len(have)} already cached)")
    for i, name in enumerate(todo, 1):
        start = time.time()
        path = photos / name
        source_hash = file_sha256(path)
        try:
            reading = (read_photo_full if full else read_photo)(path)
        except Exception as exc:
            raise SystemExit(
                f"  {i}/{len(todo)} {name}: FAILED {exc}\n"
                f"photo read failed — scoring with holes is not allowed.\n"
                f"Fix the reader or remove '{name}' from the scored set.")
        have[name] = {"source_sha256": source_hash, "text": reading}
        text[name] = reading
        write_reading_cache(cache, context, have)
        print(f"  {i}/{len(todo)} {name}  {time.time() - start:.0f}s  "
              f"{len(reading.split())} words", flush=True)
    return text


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
    # The prose report is for a person. A gate needs the same rates without
    # parsing bold markdown out of a sentence.
    ap.add_argument("--json", type=Path,
                    help="also write the rates where a gate can read them")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/reader-output.json")
    ap.add_argument("--full-res", action="store_true",
                    help="read at native resolution instead of the app's "
                         "two-megapixel working size")
    ap.add_argument("--sealed-cache", action="store_true",
                    help="score a complete schema-2 cache whose source images "
                         "were intentionally removed; any missing row stops "
                         "instead of reading --photos")
    ap.add_argument("--read-only", action="store_true",
                    help="read the sampled photographs into the cache and stop; "
                         "lets the slow half run while the answer key is written")
    ap.add_argument("--sample", type=Path,
                    default=REPO_ROOT / "data/ocr/real-photos/scored-sample.txt")
    args = ap.parse_args()

    if args.sealed_cache and args.read_only:
        raise SystemExit("--sealed-cache cannot create readings; remove --read-only")

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
    readings = cached_reads(
        names, args.photos, args.cache, args.full_res, sealed=args.sealed_cache
    )

    totals = {"plain": [0, 0], "dia": [0, 0], "blind": [0, 0]}
    rows, spurious, empty_truth = [], 0, 0

    for i, name in enumerate(names, 1):
        entry = truth[name]
        want = words(" ".join(entry["lines"]))
        if not want:
            empty_truth += 1
            continue
        if name not in readings:
            print(f"  {i}/{len(names)} {name}: never read — cannot score with holes",
                  file=sys.stderr)
            return 1
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

    # A street sign and a memorial slab are different tasks, and the product
    # is the first (docs/OCR-ROADMAP.md, decision 1). Split by how many words
    # the answer key holds for the photograph; the headline stays the mean over
    # every photograph, so it remains comparable with 36.0% and 54.7%.
    by_class = {}
    for label, lo, hi in (("sign (1-5 words)", 1, 5), ("short board (6-20 words)", 6, 20),
                          ("long board (21+ words)", 21, 10 ** 9)):
        sub = [r for r in rows if lo <= r[2] <= hi]
        by_class[label] = {
            "photographs": len(sub),
            "per_photo": (sum(h / n for _n_, h, n, *_r in sub) / len(sub) * 100) if sub else None,
            "found": sum(r[1] for r in sub), "words": sum(r[2] for r in sub),
            "fully_read": sum(r[1] == r[2] for r in sub)}

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
        "## Signs against boards",
        "",
        "The product reads small signs and street names; long text is reported, not "
        "claimed. Split by how many agreed words the photograph carries.",
        "",
        "| | photographs | words per photograph | words found | read in full |",
        "|---|---|---|---|---|",
    ] + [
        f"| {label} | {c['photographs']} | "
        + (f"**{c['per_photo']:.1f}%**" if c["per_photo"] is not None else "—")
        + f" | {c['found']}/{c['words']} | {c['fully_read']} |"
        for label, c in by_class.items()
    ] + [
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
    if args.json:
        args.json.write_text(json.dumps({
            "pooled": pct(totals["plain"]),
            "per_photo": macro,
            "diacritic": pct(totals["dia"]),
            "folded": pct(totals["blind"]),
            "invented": spurious,
            "photographs": len(rows),
            "reader": reader_fingerprint(full=args.full_res),
            "reader_context": reader_cache_context(full=args.full_res),
            "by_class": by_class,
            # Per photograph, so two readers can be compared pair by pair
            # rather than mean against mean.
            "per_photograph": {name: [hit, need] for name, hit, need, *_r in rows},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {short(args.json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
