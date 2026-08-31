#!/usr/bin/env python3
"""Put a trainable Marian base for one translation direction on this machine.

The float32 bases are not in git and not in the published bundle -- only
training reads them. This pulls one down into models/lilly/ where
training/train_translation.py expects it.

    python3 scripts/fetch_translate_base.py --direction en-bs

bs-en is a single-target model: Bosnian in, English out, nothing to select.
en-bs is not. Its upstream serves five target languages off one decoder and
picks between them from a label on the front of the source sentence, so
`>>bos_Latn<<` is the whole difference between Bosnian and Croatian output. A
label the tokenizer does not recognise is not an error upstream -- it is
silently translated as if it were text, and the model falls back to whatever
its majority target is. That failure produces fluent, plausible, wrong-language
output and no traceback, so it is checked here rather than discovered in a
benchmark six hours later.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "models" / "lilly"

# Upstream for each direction, and the label the source sentence must carry.
# en-bs is opus-mt-tc-base-en-sh: Tatoeba-Challenge, English -> Serbo-Croatian,
# with bos_Latn a declared target (published eng-bos_Latn: chrF2 0.666, BLEU
# 46.3 on tatoeba-test-v2021-08-07). There is no tc-big for this direction --
# Helsinki publishes zls-en and sh-en into English but nothing big back out.
DIRECTIONS = {
    "bs-en": {"repo": "Helsinki-NLP/opus-mt-tc-big-zls-en",
              "dest": MODELS / "translate",
              "tag": None},
    "en-bs": {"repo": "Helsinki-NLP/opus-mt-tc-base-en-sh",
              "dest": MODELS / "translate-en-bs",
              "tag": ">>bos_Latn<<"},
}

# Marian needs both sentencepiece models and the shared vocab; a missing .spm
# does not fail at load, it fails at the first non-ASCII character.
REQUIRED = ("config.json", "source.spm", "target.spm", "vocab.json",
            "tokenizer_config.json")
WEIGHTS = ("pytorch_model.bin", "model.safetensors")


def check_tag(dest: Path, tag: str) -> int:
    """Fail if the target-language label is not a token the model knows."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(dest))
    ids = tok.convert_tokens_to_ids([tag])
    unk = tok.unk_token_id
    if not ids or ids[0] in (None, unk):
        print(f"{tag} is not in the vocabulary of {dest.name} — this base "
              f"cannot be steered to Bosnian, and would answer in whichever "
              f"language its decoder prefers", file=sys.stderr)
        return 1
    # Round-trip a real sentence: the label has to survive tokenisation as one
    # piece and come back out, or it was absorbed into the text.
    encoded = tok.convert_ids_to_tokens(tok.encode(f"{tag} Good day"))
    if tag not in encoded:
        print(f"{tag} did not survive tokenisation of a sentence: {encoded}",
              file=sys.stderr)
        return 1
    print(f"target label {tag} -> id {ids[0]}, survives tokenisation")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="en-bs", choices=sorted(DIRECTIONS))
    args = ap.parse_args()
    spec = DIRECTIONS[args.direction]
    dest, repo, tag = spec["dest"], spec["repo"], spec["tag"]

    if not (dest / "config.json").exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("pip install huggingface_hub first", file=sys.stderr)
            return 1
        print(f"fetching {repo} -> {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, repo_type="model", local_dir=str(dest),
                          allow_patterns=["*.json", "*.spm", "*.bin",
                                          "*.safetensors", "*.txt"],
                          token=os.environ.get("HF_TOKEN") or None)
    else:
        print(f"already here: {dest}")

    absent = [f for f in REQUIRED if not (dest / f).exists()]
    if absent:
        print(f"{dest} is missing {', '.join(absent)}", file=sys.stderr)
        return 1
    if not any((dest / w).exists() for w in WEIGHTS):
        print(f"{dest} has no weights ({' or '.join(WEIGHTS)})", file=sys.stderr)
        return 1

    if tag and check_tag(dest, tag):
        return 1

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"ready: {size / 1048576:.0f} MB in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
