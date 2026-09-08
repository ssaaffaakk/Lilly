#!/usr/bin/env python3
"""Put Lilly's models on this machine.

The weights are not in git. They live in the project's model repository on
Hugging Face, and this pulls the four folders the app serves from into
models/lilly/. Then it finishes the install in two ways the download alone
does not:

  * the reader. The app reads photographs with PaddleOCR PP-OCRv6, whose
    weights are not in the bundle: PaddleX fetches them on first use. First
    use is here, through the app's own door, so that once this script has
    finished nothing reaches the network again -- not on the first photograph
    either.
  * the listener. models/lilly/listen is checked against the fingerprint of
    the listener that cleared its pre-registered gate (whisper-small) and says
    which one it was handed. Since the evening of 8 September 2026 the bundle
    carries whisper-large-v3 by the owner's explicit decision: it reads far
    fewer words wrong (14.1% against 39.5% on 925 clips) and it failed the
    gate's Croatian-substitution row (1.1% -> 6.1%), and both facts are on
    the model card. This script says so rather than letting a fresh clone
    find out from the transcripts.
  * the reply direction. translator-en-bs/ (English -> Bosnian) is in the
    bundle since 8 September 2026 and is pulled when present; its absence is
    not a failure, the app answers 503 on /api/reply and says why.

    python3 scripts/fetch_models.py
    python3 scripts/fetch_models.py --skip-reader-warmup   # the Dockerfiles: they warm
                                                           # the reader after copying app/

Point it somewhere else with LILLY_MODELS_REPO, and pass a token in HF_TOKEN if
the repository is private. LILLY_REQUIRE_GATED_LISTEN=1 turns the listener
check from a warning into exit 1.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = os.environ.get("LILLY_MODELS_REPO", "Safak11/lilly")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "models" / "lilly"
# What the app serves from, and what the app needs to run. Not "translate":
# that is the untuned float32 base, which only training reads, fetched by
# scripts/fetch_translate_base.py when training needs it. The bundle happens
# to carry a copy today (475 MB); nothing here downloads it, because the app
# never opens it, and expecting it here once made a fresh clone download the
# whole bundle and then fail with "still missing: translate".
PARTS = ("translator", "listen", "speak", "read")
# Published only once its fine-tune cleared its pre-registered bars and the
# owner put it in the bundle (scripts/publish_to_hf.py --with-reply): pulled
# when the repository has it, never required.
OPTIONAL = ("translator-en-bs",)
# The card and the attribution notice travel with the weights (NOTICE.md is a
# licence condition, not a courtesy). Both are tracked in git as well, so the
# copies here are the published ones and git's are the source.
CARD_FILES = ("README.md", "NOTICE.md")

# The listener that cleared its gate: whisper-small, 34.9% word error on the
# 200-clip prefix, fingerprint printed by training/speech_bench.py
# (training/SPEECHBENCH-gate.txt, "listen-previous: weights a76342f6ab59b382").
# scripts/publish_to_hf.py refuses to upload any other listen/ under its twin,
# LISTEN_FINGERPRINT. Update the two together, in the same commit as the
# results file that says a different listener ships.
GATED_LISTEN_FINGERPRINT = "a76342f6ab59b382"

# What a served translator directory cannot be loaded without. Tested, not
# assumed (8 Sep 2026): drop shared_vocabulary.json and ctranslate2.Translator
# raises "Cannot load the target vocabulary from the model directory"; drop
# tokenizer_config.json and AutoTokenizer falls through to AutoConfig and
# raises "Unrecognized model ... should have a model_type key". Checked here
# because a bundle can be published incomplete -- translator-en-bs/ was, on
# 8 September, and the first anyone knew of it was /api/reply answering 500 on
# a running Space. An install that cannot serve a folder it fetched says so
# here instead.
TRANSLATOR_FILES = ("config.json", "model.bin", "shared_vocabulary.json",
                    "source.spm", "target.spm", "tokenizer_config.json", "vocab.json")


def incomplete_translators() -> dict:
    """{folder: [missing files]} for every translator build that is here but broken."""
    out = {}
    for part in ("translator",) + OPTIONAL:
        build = DEST / part
        if not build.is_dir():
            continue
        missing = [n for n in TRANSLATOR_FILES if not (build / n).is_file()]
        if missing:
            out[part] = missing
    return out


def listen_fingerprint(build: Path) -> str:
    """training/speech_bench.py's fingerprint(), first 16 hex digits: md5 over
    every file in the directory, sorted by name, name then bytes. Kept
    identical so the gate, the publisher and this script name a build the
    same way, and a build whose weights changed cannot match."""
    h = hashlib.md5()
    for name in sorted(p.name for p in build.iterdir() if p.is_file()):
        h.update(name.encode())
        with (build / name).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def listen_base(build: Path) -> str:
    """The base model listen/built.json records, or an empty string."""
    built = build / "built.json"
    if not built.is_file():
        return ""
    try:
        return str(json.loads(built.read_text(encoding="utf-8")).get("base", ""))
    except ValueError:
        return "(built.json unreadable)"


def check_listener(build: Path) -> bool:
    """Say which listener this is. True when it is the one that cleared its gate."""
    if not (build / "model.bin").is_file():
        print(f"listen/: no model.bin at {build}", file=sys.stderr)
        return False
    actual = listen_fingerprint(build)
    base = listen_base(build) or "(no built.json)"
    if actual == GATED_LISTEN_FINGERPRINT:
        print(f"listen/: {base}, fingerprint {actual} -- the listener that cleared its gate")
        return True
    print(f"\nlisten/: {base}, fingerprint {actual} -- NOT the gated listener "
          f"({GATED_LISTEN_FINGERPRINT}).", file=sys.stderr)
    if "large-v3" in base:
        print("""  This is the listener the owner chose to ship on 8 September 2026, knowing its
  pre-registered gate refused it: on all 925 test clips it reads 14.1% of words wrong
  against the gated whisper-small's 39.5%, and it writes the Croatian form of a
  Bosnian-specific word more often (6.1% of decided targets against 1.1%, p = 0.018 --
  the row it failed). Both numbers are on the model card; the decision and its reason
  are in training/PREREGISTRATION.md and training/RESULTS-speech.md.""", file=sys.stderr)
    else:
        print("""  Not the build the published speech numbers were measured on. A candidate that has
  since cleared its pre-registered gate is fine here; one that has not is not the
  product, whatever it scores (training/PREREGISTRATION.md, "Both, not either").""",
              file=sys.stderr)
    print("  Set LILLY_REQUIRE_GATED_LISTEN=1 to make this an error instead of a warning.",
          file=sys.stderr)
    return False


def warm_reader() -> None:
    """Pull the reader's weights into PaddleX's cache, once, here.

    app.ocr.get_paddle_reader() is the only door to the reader, so this uses
    the same model names, the same mirror setting and the same cv2 the app
    will. LILLY_READER=easyocr needs nothing fetched: its weights are in read/.
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from app.ocr import get_paddle_reader, paddle_models, reader_choice
    except ImportError as exc:
        raise SystemExit(
            f"cannot import app.ocr ({exc}). Run this from a checkout that has app/ and "
            f"the requirements installed, or pass --skip-reader-warmup and warm the reader "
            f"yourself once app/ is in place -- the Dockerfiles do the second.")
    if reader_choice() != "paddle":
        print(f"reader: LILLY_READER={reader_choice()}, its weights are in read/; nothing to fetch")
        return
    # The bcebos.com hosts refuse some networks; the Hugging Face mirror is the
    # host the bundle itself came from. A default only -- an explicit choice stands.
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    det, rec = paddle_models()
    print(f"reader: fetching {det} + {rec} through app.ocr "
          f"(PADDLE_PDX_MODEL_SOURCE={os.environ['PADDLE_PDX_MODEL_SOURCE']}) ...", flush=True)
    get_paddle_reader()
    print("reader: ready -- nothing reaches the network at request time now")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-reader-warmup", action="store_true",
                    help="do not fetch the PaddleOCR weights here; the Dockerfiles warm the "
                         "reader themselves after copying app/ into the image")
    args = ap.parse_args()

    have = [p for p in PARTS if (DEST / p).is_dir()]
    if len(have) == len(PARTS):
        print(f"already here: {DEST}")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("pip install huggingface_hub first", file=sys.stderr)
            return 1
        print(f"fetching {REPO} -> {DEST}: {', '.join(PARTS)}, {', '.join(OPTIONAL)} "
              f"if published, and the model card (about 2.3 GB)")
        DEST.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=REPO, repo_type="model", local_dir=str(DEST),
                          allow_patterns=[f"{p}/*" for p in PARTS + OPTIONAL] + list(CARD_FILES),
                          token=os.environ.get("HF_TOKEN") or None)
        missing = [p for p in PARTS if not (DEST / p).is_dir()]
        if missing:
            print(f"still missing: {', '.join(missing)}", file=sys.stderr)
            return 1
        for part in OPTIONAL:
            print(f"{part}/: {'present' if (DEST / part).is_dir() else 'not in this bundle (optional)'}")
        size = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
        print(f"ready: {size / 1073741824:.2f} GB in {DEST}")

    broken = incomplete_translators()
    if broken:
        for part, missing in broken.items():
            print(f"\n{part}/ is in this bundle but cannot be loaded: missing "
                  f"{', '.join(missing)}", file=sys.stderr)
        print("The published bundle is incomplete, not this machine: fetching it again "
              "will fetch the same files. Whoever published it should run "
              "scripts/publish_to_hf.py again (it uploads the whole directory since "
              "8 Sep 2026) and this install should then be re-fetched.", file=sys.stderr)
        return 1

    gated = check_listener(DEST / "listen")
    if not args.skip_reader_warmup:
        warm_reader()
    if not gated and os.environ.get("LILLY_REQUIRE_GATED_LISTEN"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
