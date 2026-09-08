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
import hashlib
import re
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
        # Every file build_translator.py writes. Two of the three this list used
        # to omit are load-critical, tested rather than assumed (8 Sep 2026):
        # without shared_vocabulary.json ctranslate2.Translator raises "Cannot
        # load the target vocabulary from the model directory", and without
        # tokenizer_config.json AutoTokenizer falls through to AutoConfig and
        # raises "Unrecognized model ... should have a model_type key".
        # special_tokens_map.json loads without complaint but is part of the
        # build the digest names, so it belongs here too. The old six-name list
        # called itself "what the app cannot start without" and was not; and
        # translator-en-bs/ was published FROM it -- see REPLY_DIR.
        "needs": ("built.json", "config.json", "model.bin",
                  "shared_vocabulary.json", "source.spm", "special_tokens_map.json",
                  "target.spm", "tokenizer_config.json", "vocab.json"),
    },
    "listen": {
        "what": "spoken Bosnian -> Bosnian text (CTranslate2, int8)",
        "needs": ("config.json", "model.bin", "tokenizer.json"),
    },
    "read": {
        "what": "photo of Bosnian text -> text (EasyOCR)",
        "needs": ("craft_mlt_25k.pth", "latin_g2.pth", "lilly.pth",
                  "user_network/lilly.py", "user_network/lilly.yaml"),
        # Only the files the app loads go up. read/ also holds the reader's
        # backups and evaluation copies -- latin_g2-realcrops.pth,
        # lilly-previous.pth, lilly-before-pass18.pth, cyrillic_g2.pth -- and
        # sweeping the directory would publish several files called
        # something-lilly under a card that describes one reader. The one
        # that scored 54.7% is named by md5 below and checked before upload.
        "only": True,
    },
    "speak": {
        "what": "English text -> spoken English (Kokoro)",
        "needs": ("config.json", "model.pth", "voices/default.pt"),
    },
}

# The reader the model card's numbers were measured on:
# training/RESULTS-ocr-weights.md, 4 Sep 2026 -- 54.7% per photograph / 180
# invented, reproduced exactly from this file and from no other. Update it in
# the same commit as that RESULTS doc when the reader is retrained. Publishing
# any other lilly.pth is what left the 27 Aug reader on Hugging Face for a
# week while every document quoted 54.7%.
READER_MD5 = "2010a2d417e6c253195fa3d95ff11d33"

# The listener the published numbers were measured on. Same shape of failure
# as the reader's, and it happened: the 4-5 September reader publish swept the
# Mac's models/lilly/listen -- by then the whisper-large-v3 candidate, installed
# 1 September, gate never run -- into Safak11/lilly (listen/model.bin
# 1,558,949,857 bytes, LFS sha 8b756776...). The gate ran 7 September and refused
# that build by one word. Nothing here had checked listen/. Now it does.
#
# The value is the fingerprint training/speech_bench.py printed for
# listen-previous, the whisper-small the gate scored at 34.9%:
# training/SPEECHBENCH-gate.txt, "listen-previous: weights a76342f6ab59b382".
# The hash is speech_bench.fingerprint() -- md5 over every file in the
# directory, sorted by name, name then bytes -- so the two agree by
# construction and a build whose weights changed cannot match. Update this in
# the same commit as the results file that says a different listener ships.
LISTEN_FINGERPRINT = "a76342f6ab59b382"

# The reply direction, published only on request (--with-reply). It is bound
# the same way the forward translator is: the build's digest must be the one
# training/RESULTS-en-bs-formrate.md names as the served build, and built.json
# must say fine_tuned with the adapter the numbers were measured on.
REPLY_DIR = {
    "what": "English text -> Bosnian text (CTranslate2, int8), the reply direction",
    # The same nine as translator/, and for the same tested reason. The six-name
    # version of this list was used as the UPLOAD list on 8 September, so the
    # published translator-en-bs/ arrived without shared_vocabulary.json,
    # tokenizer_config.json and special_tokens_map.json and could not be loaded
    # at all: /api/reply answered 500 on every fresh install. check_reply now
    # uploads the directory, like every other published folder.
    "needs": ("built.json", "config.json", "model.bin",
              "shared_vocabulary.json", "source.spm", "special_tokens_map.json",
              "target.spm", "tokenizer_config.json", "vocab.json"),
}
REPLY_RECORD = REPO_ROOT / "training" / "RESULTS-en-bs-formrate.md"

# Documentation that must go up with the weights. NOTICE.md is not optional:
# CC-BY-4.0 on the translation weights and Apache-2.0 on two of the others
# require the attribution to travel with what is redistributed.
PUBLISH_FILES = ("README.md", "NOTICE.md")

# Known scratch, kept on this machine. Matched against top-level names in
# models/lilly/. Anything not matched here and not in the publish list is
# reported as unrecognised rather than quietly swept up either way.
EXCLUDE_RULES = (
    # Candidates and rollbacks from a retraining experiment. The published
    # translator is whichever candidate won; the losers and the build it
    # replaced stay here so a bad result costs nothing to undo, and none of
    # them belongs in a release.
    ("adapter-arm-a", "the losing retraining arm's adapter, kept so its result can be reproduced"),
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
    # The speech retraining kept both sides of its comparison next to the
    # served listener (training/RESULTS-speech.md, training/README.md).
    ("listen-candidate", "the speech retraining candidate, kept to score against listen/"),
    ("listen-previous", "the listener listen/ replaced, kept as the baseline to measure against"),
    # The reply direction. app/lilly.py lists it as OPTIONAL and says why: it
    # is not in the published bundle and is built locally from an upstream
    # base (scripts/fetch_translate_base.py --direction en-bs). Putting it in
    # the release is the owner's call and needs its own model-card entry.
    ("translate-en-bs", "untuned float32 English -> Bosnian base, read only by training "
                        "(training/train_translation.py)"),
    ("translator-en-bs", "English -> Bosnian reply build; published only with --with-reply, "
                         "bound to training/RESULTS-en-bs-formrate.md"),
    ("adapter-en-bs", "raw LoRA adapter for the reply direction; the merged model is published "
                      "as translator-en-bs/ (--with-reply)"),
    # The Bosnian voice. Never published from here: it is Piper's public
    # sr_RS voice, pulled from rhasspy/piper-voices by scripts/fetch_speak_bs.py
    # on every install, and a copy in this bundle would add nothing to it.
    ("speak-bs", "the Bosnian voice (Piper sr_RS-serbski_institut-medium), fetched upstream by "
                 "scripts/fetch_speak_bs.py on every install; not published from here"),
    # The whisper-large-v3 listener refused at its gate (7 Sep) and at the
    # pre-registered last look (8 Sep): closed by rule 3, kept on this machine
    # as the record of what was refused. It is never published; the gated
    # whisper-small is LISTEN_FINGERPRINT above.
    ("listen-large-v3*", "a whisper-large-v3 listener kept aside; whatever sits under listen/ is "
                         "what is published, and only the gated one goes without --allow-listen"),
    # The whisper-small that cleared its gate (LISTEN_FINGERPRINT), kept beside
    # listen/ while the owner ships a different listener by explicit override
    # (8 Sep 2026 evening: large-v3, refused at its gate on Croatian
    # substitution, published with --allow-listen e6bb58483586b06c and the
    # failed row written on the card).
    ("listen-small*", "the gated whisper-small (fingerprint a76342f6ab59b382), kept as the "
                      "baseline every listener is measured against"),
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


def fingerprint_of(rels) -> str:
    """build_fingerprint over an explicit list of bundle-relative paths.

    The point is that the digest is taken of the files that are about to be
    uploaded, not of the directory they happen to sit in. Hashing the directory
    while uploading a subset is how translator-en-bs/ went up incomplete on
    8 September with a digest that matched the record: the guard said "these are
    the scored weights" about files the upload did not carry.
    """
    digest = hashlib.blake2b(digest_size=16)
    for rel in sorted(rels, key=lambda r: Path(r).name):
        name = Path(rel).name
        if name == "built.json":
            continue
        digest.update(name.encode("utf-8"))
        with open(BUNDLE / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def listen_fingerprint(build: Path) -> str:
    """training/speech_bench.py's fingerprint(), first 16 hex digits -- kept
    identical on purpose so the publisher and the gate name a build the same way."""
    h = hashlib.md5()
    for name in sorted(p.name for p in build.iterdir() if p.is_file()):
        h.update(name.encode())
        h.update((build / name).read_bytes())
    return h.hexdigest()[:16]


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excuse(name: str):
    """Why this top-level entry stays here, or None if no rule covers it."""
    for pattern, reason in EXCLUDE_RULES:
        if fnmatch.fnmatch(name, pattern):
            return reason
    return None


def reply_record() -> tuple:
    """(served build digest, adapter md5) the results file names, or ('', '')."""
    if not REPLY_RECORD.exists():
        return "", ""
    text = REPLY_RECORD.read_text(encoding="utf-8")
    build = re.search(r"Served build: `([0-9a-f]{32})`", text)
    adapter = re.search(r"Adapter: md5 `([0-9a-f]{32})`", text)
    return (build.group(1) if build else ""), (adapter.group(1) if adapter else "")


def check_reply(publish: dict, problems: list) -> None:
    """translator-en-bs/ goes up only as the build the results describe."""
    directory = BUNDLE / "translator-en-bs"
    if not directory.is_dir():
        problems.append("--with-reply: no translator-en-bs/ -- build it with "
                        "scripts/build_translator.py --direction en-bs")
        return
    for needed in REPLY_DIR["needs"]:
        if not (directory / needed).is_file():
            problems.append(f"missing translator-en-bs/{needed}")
    try:
        built = json.loads((directory / "built.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problems.append(f"translator-en-bs/built.json unreadable: {exc}")
        return
    want_build, want_adapter = reply_record()
    if not want_build or not want_adapter:
        problems.append(f"{REPLY_RECORD.name} names no served build / adapter for en-bs; "
                        "nothing to bind these weights to")
        return
    if not built.get("fine_tuned") or built.get("direction") != "en-bs":
        problems.append(f"translator-en-bs/built.json says {built} -- not the fine-tuned en-bs build")
    if built.get("adapter_md5") != want_adapter:
        problems.append(f"translator-en-bs was built from adapter {built.get('adapter_md5')}, "
                        f"not {want_adapter}, the one the numbers were measured on")
    # The whole directory, exactly like translator/ and every other published
    # folder that is not marked "only". Never a hand-written subset: a name
    # forgotten here is a file missing from the bundle, and the app finds out
    # at the first request.
    found = files_under(directory)
    publish["translator-en-bs"] = found
    # ... and the digest is taken of that list, so a short upload cannot pass
    # the check that says these are the weights the numbers were measured on.
    actual = fingerprint_of(found)
    if actual != want_build:
        problems.append(f"the translator-en-bs files about to be uploaded are build {actual}, "
                        f"but the results name {want_build}; publishing them would put one "
                        "model's weights behind another's numbers")
    else:
        print(f"translator-en-bs matches the recorded served build: {actual}")


def preflight(allow_listen: str | None = None, with_reply: bool = False) -> tuple:
    """Everything wrong with the bundle, and the files that would go up.

    `allow_listen` is the owner's explicit, named override for listen/: the
    16-hex fingerprint of a listener that has cleared its pre-registered gate
    since LISTEN_FINGERPRINT was written. It is printed, never defaulted.

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
        if spec.get("only"):
            found = [Path(name) / n for n in spec["needs"] if (directory / n).is_file()]
        else:
            found = files_under(directory)
        if not found:
            problems.append(f"{name}/ is empty")
        publish[name] = found
        # The reader is the one file here whose wrong version loads, reads and
        # scores without a single error -- just a different number. Refuse any
        # lilly.pth but the one the published figures were measured on.
        reader = directory / "lilly.pth"
        if name == "read" and reader.is_file():
            actual = md5_of(reader)
            if actual != READER_MD5:
                problems.append(f"read/lilly.pth is md5 {actual}, not {READER_MD5}, the reader that "
                                "scored 54.7% / 180 (training/RESULTS-ocr-weights.md). Install that "
                                "one before publishing; a different reader under the same name is "
                                "exactly what this check exists to stop")
        # The listener is the other file whose wrong version loads and scores
        # without an error. Refuse any listen/ but the gated one, unless the
        # owner names a different fingerprint on the command line -- which is
        # the decision that publishing a listener is, made where it can be seen.
        if name == "listen":
            actual = listen_fingerprint(directory)
            built = directory / "built.json"
            base = ""
            if built.is_file():
                try:
                    base = str(json.loads(built.read_text(encoding="utf-8")).get("base", ""))
                except ValueError:
                    base = "(built.json unreadable)"
            print(f"listen/: fingerprint {actual}  base {base or '(no built.json)'}")
            wanted = allow_listen or LISTEN_FINGERPRINT
            if actual != wanted:
                problems.append(
                    f"listen/ is fingerprint {actual} ({base or 'unknown base'}), not {wanted}"
                    + ("" if allow_listen else
                       " -- the whisper-small the gate scored at 34.9% (training/SPEECHBENCH-gate.txt). "
                       "A listener that has since cleared its pre-registered gate is published with "
                       "--allow-listen <its fingerprint>, named on the command line, never by default. "
                       "This check did not exist on 4 September and an ungated whisper-large-v3 went "
                       "up under listen/; it exists now."))

    # The card and the upload must agree about the reply direction: a card that
    # describes translator-en-bs/ over a bundle without it is a promise the
    # bundle does not keep, and the reverse hides a model behind no numbers.
    card = BUNDLE / "README.md"
    card_says = card.is_file() and "translator-en-bs/" in card.read_text(encoding="utf-8")
    if with_reply:
        check_reply(publish, problems)
        if not card_says:
            problems.append("--with-reply, but the model card never mentions translator-en-bs/; "
                            "the reply direction needs its own card entry before it is published")
    elif card_says:
        problems.append("the model card describes translator-en-bs/ but this publish does not "
                        "include it (no --with-reply); fix one or the other")

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


def report_left_behind(published=()) -> list:
    """Print what is not going up, and return the names nothing accounts for."""
    unrecognised = []
    rows = []
    for entry in sorted(BUNDLE.iterdir()):
        name = entry.name
        if name in PUBLISH_DIRS or name in PUBLISH_FILES or is_junk(Path(name)) or name in published:
            continue
        reason = excuse(name)
        if reason is None:
            reason = "NOT RECOGNISED — nothing in this script accounts for it"
            unrecognised.append(name)
        size = size_of(files_under(entry)) if entry.is_dir() else entry.stat().st_size
        rows.append((name, size, reason))

    # Inside a directory published by list, name what stays, so a backup that
    # was never meant to go up is visibly not going up rather than forgotten.
    for name, spec in PUBLISH_DIRS.items():
        if not spec.get("only") or not (BUNDLE / name).is_dir():
            continue
        wanted = {Path(name) / n for n in spec["needs"]}
        extra = [p for p in files_under(BUNDLE / name) if p not in wanted]
        if extra:
            print(f"\n{name}/: {len(extra)} other file(s) stay here -- backups and evaluation "
                  "copies, not in the release")
            for p in extra:
                print(f"  {str(p):40} {human((BUNDLE / p).stat().st_size):>9}")

    if not rows:
        print("\nnothing left behind: the bundle holds only what is published")
        return unrecognised

    print("\nleft behind (not uploaded)")
    for name, size, reason in rows:
        print(f"  {name + '/':24} {human(size):>9}   {reason}")
    return unrecognised


def stored_token() -> str:
    """The token `hf auth login` saved, or ''.

    HF_TOKEN in the environment still wins. The fallback exists because the
    alternative is typing a write token into a terminal, which on 8 September
    produced two 401s from placeholder text pasted in its place.
    """
    try:
        from huggingface_hub import get_token
    except ImportError:
        return ""
    return get_token() or ""


def stale_in_repo(repo_id: str, publish: dict) -> tuple:
    """(files the repository holds that this release does not, or None if unknown).

    upload_folder adds and replaces; it never removes. So a file that was part
    of an older release stays in the repository for ever, sitting inside a
    folder whose name says it is something else. It has happened twice: a
    float32 translate/ from 22 August lived in the bundle for a fortnight, and
    listen/vocabulary.txt -- the untrained large-v3's token list -- is in the
    published listener now. That one is inert (ctranslate2 reads
    vocabulary.json when both are there, tested 8 Sep: byte-identical
    transcripts), but it changes what the directory hashes to, so every fresh
    install computes a listener fingerprint that no document names.

    Names starting with a dot are the Hub's own (.gitattributes) and are left.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or stored_token() or None)
        if not api.repo_exists(repo_id, repo_type="model"):
            return (), None
        remote = {s.rfilename for s in api.repo_info(repo_id, repo_type="model").siblings}
    except Exception as exc:                       # noqa: BLE001 -- report, don't raise
        return (), f"could not check {repo_id} for files this release does not carry: {exc}"
    sending = {str(rel) for group in publish.values() for rel in group}
    return tuple(sorted(f for f in remote - sending if not f.startswith("."))), None


def upload(repo_id: str, publish: dict, public: bool, message: str,
           prune: tuple = ()) -> int:
    token = os.environ.get("HF_TOKEN") or stored_token()
    if not token:
        print("\nNo Hugging Face token. Either log in once:\n"
              "    .venv/bin/hf auth login\n"
              "or export a write token from https://huggingface.co/settings/tokens:\n"
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
        delete_patterns=list(prune) or None,
        commit_message=message,
    )
    for gone in prune:
        print(f"removed from the repository: {gone}")
    print(f"done: https://huggingface.co/{repo_id}")
    if not public:
        print("the repository is private — make it public from its Settings page "
              "when you are ready")
    return 0



def scored_build() -> str:
    """The build the published scores were measured on, from the report itself."""
    report = REPO_ROOT / "training" / "RESULTS-product.md"
    if not report.exists():
        return ""
    found = re.search(r"Scored build: `([0-9a-f]{32})`",
                      report.read_text(encoding="utf-8"))
    return found.group(1) if found else ""


def build_fingerprint(build: Path) -> str:
    """Identical to training/evaluate_app.py's, so the two can be compared."""
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(f.name for f in build.iterdir() if f.is_file()):
        if name == "built.json":
            continue
        digest.update(name.encode("utf-8"))
        with open(build / name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def refuse_unmeasured_weights(bundle: Path) -> None:
    """Stop if the weights about to be uploaded are not the ones that were scored.

    models/lilly/ holds five directories whose names differ by a word —
    translator, translator-armA, translator-armB, translator-base,
    translator-previous. Publishing the wrong one does not fail: the model loads,
    it translates, and every number on the model card is quietly about different
    weights. There is no way to notice from outside.

    So the report records which build produced its scores and this refuses to
    upload any other. It is the same move as giving readtext one door — the
    mistake stops being expressible rather than being something to be careful
    about.
    """
    expected = scored_build()
    if not expected:
        raise SystemExit(
            "training/RESULTS-product.md names no scored build, so there is "
            "nothing to check these weights against. Run training/evaluate_app.py "
            "against the build you mean to publish first.")
    actual = build_fingerprint(bundle / "translator")
    if actual != expected:
        raise SystemExit(
            f"the translator here is {actual}, but the published scores were "
            f"measured on {expected}. Uploading this would put one model's "
            f"weights behind another model's numbers, and nothing downstream "
            f"could tell. Score this build, or publish the one that was scored.")
    print(f"translator matches the scored build: {expected}")


def check_card_metadata(card: Path) -> None:
    """Fail on a bad model card before sending 900 MB, not after.

    Hugging Face validates the card's YAML header at upload time, and a header
    it rejects aborts the transfer partway. That happened here over one field:
    license_link pointed at NOTICE.md, a relative filename, where an https URI
    is required. A run that dies mid-upload is worse than one that never starts,
    because the repository is left holding some of the files.
    """
    import re
    import urllib.parse

    if not card.exists():
        raise SystemExit(f"no model card at {card}")
    header = re.match(r"^---\n(.*?)\n---\n", card.read_text(encoding="utf-8"), re.S)
    if not header:
        raise SystemExit(f"{card.name} has no YAML header; Hugging Face needs one")
    try:
        import yaml
        meta = yaml.safe_load(header.group(1))
    except Exception as exc:
        raise SystemExit(f"{card.name}'s YAML header does not parse: {exc}")

    link = meta.get("license_link")
    if link and urllib.parse.urlparse(str(link)).scheme != "https":
        raise SystemExit(
            f"license_link is {link!r}. Hugging Face requires an https URI here, "
            f"not a filename — the upload is rejected after it has begun.")
    if not meta.get("license"):
        raise SystemExit("the card names no license; Hugging Face requires one")
    print(f"card metadata: license {meta['license']}, "
          f"{len(meta.get('base_model', []))} base models declared")


def refuse_secrets(bundle: Path) -> None:
    """Refuse to publish anything that looks like a credential.

    Weights are binary and a model card is prose, so neither should contain a
    token — but this repository has handled three of them (Hugging Face,
    Mapillary, Kaggle), and a card is edited by hand. Publishing is the one step
    where a mistake leaves the machine and cannot be recalled: a private repo can
    be made private again, a leaked token cannot be un-copied.

    Only the text files are scanned. The pattern is each provider's own prefix
    followed by enough characters to be a real value, so the placeholders that
    appear in usage instructions do not trip it.
    """
    import re

    patterns = {
        "Hugging Face": re.compile(r"hf_[A-Za-z0-9]{30,}"),
        "Mapillary": re.compile(r"MLY\|[A-Za-z0-9_-]{20,}"),
        "Kaggle": re.compile(r"KGAT_[a-f0-9]{30,}"),
        "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    }
    readable = {".md", ".txt", ".json", ".yaml", ".yml", ".tsv", ".csv", ".py"}
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in readable:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in patterns.items():
            if pattern.search(text):
                raise SystemExit(
                    f"{path.relative_to(bundle)} contains what looks like a "
                    f"{name} credential. Nothing was uploaded. Remove it, and "
                    f"revoke that credential — assume it is already compromised.")
    print("no credentials found in the text files being published")

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
    ap.add_argument("--with-reply", action="store_true",
                    help="also publish translator-en-bs/ (English -> Bosnian), bound to the served "
                         "build training/RESULTS-en-bs-formrate.md names; the card must describe it")
    ap.add_argument("--prune", action="store_true",
                    help="also DELETE files the repository holds that this release does not "
                         "carry -- leftovers from an older publish. Listed first, never "
                         "guessed, and only with --upload.")
    ap.add_argument("--allow-listen", default=None, metavar="FINGERPRINT",
                    help="publish a listen/ other than the gated whisper-small: the 16-hex "
                         "fingerprint speech_bench.py printed for a listener that has cleared "
                         "its pre-registered gate. Named here, never assumed.")
    args = ap.parse_args()

    print(f"bundle:  {BUNDLE}")
    print(f"target:  {args.repo_id} ({'public' if args.public else 'private'})")
    print(f"mode:    {'UPLOAD' if args.upload else 'dry run (default) — nothing is sent'}")
    if os.environ.get("HF_TOKEN"):
        source = "HF_TOKEN"
    elif stored_token():
        source = "the login hf auth login stored"
    else:
        source = "none"
    print(f"token:   {source}"
          f"{'' if args.upload else '  (a dry run does not need one)'}")

    # Before anything else: are these the weights the published scores describe?
    refuse_unmeasured_weights(BUNDLE)
    check_card_metadata(BUNDLE / "README.md")
    refuse_secrets(BUNDLE)

    problems, publish = preflight(args.allow_listen, args.with_reply)
    # refuse_unmeasured_weights above hashed the directory; this hashes the
    # files the upload will actually carry. They agree unless the publish list
    # is short, which is the failure this pair exists to catch.
    if "translator" in publish:
        expected = scored_build()
        sending = fingerprint_of(publish["translator"])
        if expected and sending != expected:
            problems.append(f"the translator files about to be uploaded are build {sending}, "
                            f"but the scored build is {expected}: the upload list is not the "
                            "directory that was scored")
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
    dirs = dict(PUBLISH_DIRS)
    if args.with_reply:
        dirs["translator-en-bs"] = REPLY_DIR
    for name, spec in dirs.items():
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

    stale, why_not = stale_in_repo(args.repo_id, publish)
    if why_not:
        print(f"\n{why_not}")
    elif stale:
        print("\nalready in the repository and NOT part of this release")
        for name in stale:
            print(f"  {name}")
        print("  Left there, upload_folder keeps them for ever: they sit inside a folder "
              "whose name says it is something else, and they change what that folder "
              "hashes to. Add --prune to delete them in this commit.")
        if args.upload and not args.prune:
            print("\nstopped: publish with --prune to remove them, or take them out of the "
                  "repository first. Nothing was uploaded.", file=sys.stderr)
            return 1
    else:
        print("\nthe repository carries nothing this release does not")

    unrecognised = report_left_behind(published=("translator-en-bs",) if args.with_reply else ())
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

    return upload(args.repo_id, publish, args.public, args.commit_message,
                  prune=stale if args.prune else ())


if __name__ == "__main__":
    raise SystemExit(main())
