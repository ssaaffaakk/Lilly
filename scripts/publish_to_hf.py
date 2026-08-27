#!/usr/bin/env python3
"""Publish models/lilly/ to a Hugging Face model repository.

The bundle folder holds more than the four models the app serves: backups,
evaluation builds, raw adapters and training leftovers all live beside them.
Uploading the folder wholesale would publish about a gigabyte of scratch work
and, worse, publish weights the model card does not describe. So this script
does not upload a folder — it uploads an explicit list, prints that list with
sizes, and names every single thing it left behind and why.

Nothing is sent anywhere until you ask for it in so many words:

    python3 scripts/publish_to_hf.py <user>/lilly              # dry run, the default
    python3 scripts/publish_to_hf.py <user>/lilly --upload     # actually upload

The token is read from the environment and never written to a file, never
passed on the command line, and never printed:

    export HF_TOKEN=...        # from https://huggingface.co/settings/tokens

A dry run needs no token. Re-running an upload only sends files that changed,
so fixing the model card later is quick.
"""
import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "models" / "lilly"

# The four abilities the app loads, and the files each one cannot start without.
# app/lilly.py:missing() names the same four folders; keep the two in step.
PUBLISH_DIRS = {
    "translator": {
        "what": "Bosnian text -> English text (CTranslate2, int8)",
        "needs": ("built.json", "config.json", "model.bin",
                  "source.spm", "target.spm", "vocab.json"),
    },
    "listen": {
        "what": "spoken Bosnian -> Bosnian text (CTranslate2, int8)",
        "needs": ("config.json", "model.bin", "tokenizer.json"),
    },
    "read": {
        "what": "photo of Bosnian text -> text (EasyOCR)",
        "needs": ("craft_mlt_25k.pth", "latin_g2.pth"),
    },
    "speak": {
        "what": "English text -> spoken English (Kokoro)",
        "needs": ("config.json", "model.pth", "voices/default.pt"),
    },
}

# Documentation that must go up with the weights. NOTICE.md is not optional:
# CC-BY-4.0 on the translation weights and Apache-2.0 on two of the others
# require the attribution to travel with what is redistributed.
PUBLISH_FILES = ("README.md", "NOTICE.md")

# Known scratch, kept on this machine. Matched against top-level names in
# models/lilly/. Anything not matched here and not in the publish list is
# reported as unrecognised rather than quietly swept up either way.
EXCLUDE_RULES = (
    ("keep-*", "dated backup of a working model, kept locally as the rollback"),
    ("*.before-training", "the pre-training weights, kept locally to score against"),
    ("listen-trained", "training-format speech checkpoint; listen/ is the built version"),
    ("translate-merged", "temporary merge directory from scripts/build_translator.py"),
    ("checkpoints", "mid-training checkpoints"),
    ("checkpoints-*", "mid-training checkpoints"),
    ("adapter", "raw LoRA adapter; the merged model is published as translator/"),
    # Not on the brief, but the same argument applies and they are large, so
    # they are named here rather than left to the unrecognised bucket:
    ("translate", "untuned float32 base, read only by training (app/lilly.py); "
                  "upstream at Helsinki-NLP/opus-mt-tc-big-zls-en"),
    ("translator-base", "int8 base built by build_translator.py --no-adapter, "
                        "an evaluation comparison rather than a served model"),
)

# Never interesting, at any depth.
JUNK = ("__pycache__", ".DS_Store", ".ipynb_checkpoints", "*.pyc", ".git")

GLOB_META = set("*?[]")


def human(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / 1024:.1f} KB"


def is_junk(rel: Path) -> bool:
    return any(fnmatch.fnmatch(part, pat) for part in rel.parts for pat in JUNK)


def files_under(directory: Path) -> list:
    """Every non-junk file in a directory, as paths relative to the bundle."""
    return sorted(
        p.relative_to(BUNDLE)
        for p in directory.rglob("*")
        if p.is_file() and not is_junk(p.relative_to(BUNDLE))
    )


def size_of(rels) -> int:
    return sum((BUNDLE / r).stat().st_size for r in rels)


def excuse(name: str):
    """Why this top-level entry stays here, or None if no rule covers it."""
    for pattern, reason in EXCLUDE_RULES:
        if fnmatch.fnmatch(name, pattern):
            return reason
    return None


def preflight() -> tuple:
    """Everything wrong with the bundle, and the files that would go up.

    Returns (problems, publish_map). A non-empty problems list stops the run
    even when --upload was asked for: publishing half a bundle produces a
    repository that loads on nobody's machine.
    """
    problems = []
    publish = {}

    if not BUNDLE.is_dir():
        return [f"no bundle at {BUNDLE}"], {}

    for name in PUBLISH_FILES:
        path = BUNDLE / name
        if not path.is_file():
            problems.append(f"missing {name} — the model card and the attribution "
                            f"notice both ship with the weights")
        else:
            publish[name] = [Path(name)]

    for name, spec in PUBLISH_DIRS.items():
        directory = BUNDLE / name
        if not directory.is_dir():
            problems.append(f"missing {name}/ — {spec['what']}")
            continue
        for needed in spec["needs"]:
            if not (directory / needed).is_file():
                problems.append(f"missing {name}/{needed}")
        found = files_under(directory)
        if not found:
            problems.append(f"{name}/ is empty")
        publish[name] = found

    return problems, publish


def describe_translator() -> str:
    """What translator/built.json says was built, so the card can be checked."""
    built = BUNDLE / "translator" / "built.json"
    if not built.is_file():
        return ""
    try:
        data = json.loads(built.read_text())
    except (ValueError, OSError) as exc:
        return f"  translator/built.json is unreadable: {exc}"
    tuned = data.get("fine_tuned")
    return ("  translator/built.json: "
            f"{'fine-tuned' if tuned else 'UNTUNED BASE'}, "
            f"quantisation {data.get('quantization', 'unrecorded')}")


def report_left_behind() -> list:
    """Print what is not going up, and return the names nothing accounts for."""
    unrecognised = []
    rows = []
    for entry in sorted(BUNDLE.iterdir()):
        name = entry.name
        if name in PUBLISH_DIRS or name in PUBLISH_FILES or is_junk(Path(name)):
            continue
        reason = excuse(name)
        if reason is None:
            reason = "NOT RECOGNISED — nothing in this script accounts for it"
            unrecognised.append(name)
        size = size_of(files_under(entry)) if entry.is_dir() else entry.stat().st_size
        rows.append((name, size, reason))

    if not rows:
        print("\nnothing left behind: the bundle holds only what is published")
        return unrecognised

    print("\nleft behind (not uploaded)")
    for name, size, reason in rows:
        print(f"  {name + '/':24} {human(size):>9}   {reason}")
    return unrecognised


def upload(repo_id: str, publish: dict, public: bool, message: str) -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nHF_TOKEN is not set. Create a write token at "
              "https://huggingface.co/settings/tokens and export it:\n"
              "    export HF_TOKEN=...\n"
              "Do not pass it as an argument and do not put it in a file.",
              file=sys.stderr)
        return 1

    # We upload a list, not a folder, so allow_patterns carries exact relative
    # paths. fnmatch would read a bracket or a star in a filename as a pattern;
    # nothing in the bundle has one, and if that ever changes this stops rather
    # than silently uploading the wrong set.
    paths = [str(p) for group in publish.values() for p in group]
    odd = [p for p in paths if GLOB_META & set(p)]
    if odd:
        print(f"\nfilenames contain glob characters, so the upload filter cannot "
              f"be trusted: {', '.join(odd)}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\nhuggingface_hub is not installed:  "
              "uv pip install -r requirements.txt", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    try:
        who = api.whoami()["name"]
    except Exception as exc:                       # noqa: BLE001 — report, don't raise
        print(f"\nHF_TOKEN was rejected: {exc}", file=sys.stderr)
        return 1
    print(f"\nauthenticated as {who}")

    api.create_repo(repo_id, repo_type="model", private=not public, exist_ok=True)
    api.upload_folder(
        folder_path=str(BUNDLE),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=paths,
        commit_message=message,
    )
    print(f"done: https://huggingface.co/{repo_id}")
    if not public:
        print("the repository is private — make it public from its Settings page "
              "when you are ready")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Publish models/lilly/ to Hugging Face. Lists and stops by "
                    "default; uploading takes --upload.")
    ap.add_argument("repo_id", help="target repository, e.g. yourname/lilly")
    ap.add_argument("--upload", action="store_true",
                    help="actually upload; without it this only lists")
    ap.add_argument("--public", action="store_true",
                    help="create the repository public; the default is private")
    ap.add_argument("--commit-message", default="Lilly model bundle")
    args = ap.parse_args()

    print(f"bundle:  {BUNDLE}")
    print(f"target:  {args.repo_id} ({'public' if args.public else 'private'})")
    print(f"mode:    {'UPLOAD' if args.upload else 'dry run (default) — nothing is sent'}")
    print(f"HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'not set'}"
          f"{'' if args.upload else '  (a dry run does not need it)'}")

    problems, publish = preflight()
    if problems:
        print("\nnot ready to publish:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("\nstopped. Nothing was uploaded.", file=sys.stderr)
        return 1

    print("\nto upload")
    total = 0
    for name in PUBLISH_FILES:
        size = size_of(publish[name])
        total += size
        print(f"  {name:24} {human(size):>9}   1 file")
    for name, spec in PUBLISH_DIRS.items():
        group = publish[name]
        size = size_of(group)
        total += size
        print(f"  {name + '/':24} {human(size):>9}   "
              f"{len(group)} file{'s' if len(group) != 1 else ''}   {spec['what']}")
    print(f"  {'':24} {'':>9}   ")
    print(f"  {'TOTAL':24} {human(total):>9}   "
          f"{sum(len(g) for g in publish.values())} files")
    detail = describe_translator()
    if detail:
        print(detail)

    unrecognised = report_left_behind()
    if unrecognised:
        print(f"\nstopped: {len(unrecognised)} entr"
              f"{'ies are' if len(unrecognised) > 1 else 'y is'} unaccounted for "
              f"({', '.join(unrecognised)}).", file=sys.stderr)
        print("Add it to PUBLISH_DIRS if it belongs in the release, or to "
              "EXCLUDE_RULES with the reason it does not. Nothing was uploaded.",
              file=sys.stderr)
        return 1

    if not args.upload:
        print("\ndry run — nothing was sent. Add --upload to publish.")
        return 0

    return upload(args.repo_id, publish, args.public, args.commit_message)


if __name__ == "__main__":
    raise SystemExit(main())
