#!/usr/bin/env python3
"""Put the Bosnian voice on this machine: models/lilly/speak-bs/.

Piper has no Bosnian voice, so this is the one it files under Serbian,
`sr_RS-serbski_institut-medium` from rhasspy/piper-voices. Serbian Latin is
written with the same letters as Bosnian and spoken with the same sounds, and
espeak-ng's `sr` phonemizer -- bundled inside the piper-tts wheel, so nothing
has to be installed beside Python -- reads c, c and d with their diacritics
correctly and spells numbers out, the Serbian way ("dve" where a Bosnian says
"dvije"). Its own MODEL_CARD, kept beside the weights, names the recordings it
was trained on as the Sorbian Institute's Lower Sorbian MaryTTS data, so the
sounds come from Sorbian speakers read through Serbian phonemes: intelligible,
accented, and to be heard before it is relied on. app/tts.py says the same.

Not in the Lilly bundle on purpose. The upstream is public, needs no build and
no gate, so every install pulls the 77 MB from where it lives rather than a
copy of it: scripts/fetch_models.py calls this after the bundle, and it can be
run alone:

    python3 scripts/fetch_speak_bs.py
    python3 scripts/fetch_speak_bs.py --speaker 1     # the voice has two speakers; 0 is the default
    python3 scripts/fetch_speak_bs.py --force         # fetch again over what is here

The files land under fixed names -- voice.onnx, voice.onnx.json -- so app/tts.py
never has to know which voice this is. built.json records it: the upstream
path, the md5 of the weights, the speaker chosen, and the two licences (the
voice is MIT; the recordings it was trained on are CC-BY-NC-SA-4.0, and
MODEL_CARD beside the weights says so in Piper's own words).
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "models" / "lilly" / "speak-bs"

UPSTREAM_REPO = "rhasspy/piper-voices"
UPSTREAM_DIR = "sr/sr_RS/serbski_institut/medium"
UPSTREAM_NAME = "sr_RS-serbski_institut-medium"
# upstream file -> the fixed name the app reads
FILES = {
    f"{UPSTREAM_NAME}.onnx": "voice.onnx",
    f"{UPSTREAM_NAME}.onnx.json": "voice.onnx.json",
    "MODEL_CARD": "MODEL_CARD",
}
NEEDS = ("voice.onnx", "voice.onnx.json")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_complete(dest: Path = DEST) -> bool:
    return all((dest / name).is_file() for name in NEEDS)


def fetch_bosnian_voice(speaker: int = 0, force: bool = False, dest: Path = DEST) -> Path:
    """Fetch the voice into `dest` under fixed names and write built.json. Idempotent."""
    if is_complete(dest) and not force:
        print(f"speak-bs/: already here ({dest})")
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub first")
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest / "_fetch"
    print(f"speak-bs/: fetching {UPSTREAM_REPO}/{UPSTREAM_DIR} -> {dest} (about 77 MB)", flush=True)
    for upstream, fixed in FILES.items():
        got = hf_hub_download(UPSTREAM_REPO, f"{UPSTREAM_DIR}/{upstream}", local_dir=str(staging))
        Path(got).replace(dest / fixed)
    shutil.rmtree(staging, ignore_errors=True)
    if not is_complete(dest):
        raise SystemExit(f"speak-bs/: fetched, but {dest} still lacks one of {NEEDS}")
    (dest / "built.json").write_text(json.dumps({
        "base": f"{UPSTREAM_REPO}/{UPSTREAM_DIR}/{UPSTREAM_NAME}",
        "engine": "piper",
        "language": "sr (Serbian Latin; there is no Bosnian voice in Piper)",
        "speaker": speaker,
        "voice_md5": md5_of(dest / "voice.onnx"),
        "license": "MIT (voice); CC-BY-NC-SA-4.0 (training recordings, see MODEL_CARD)",
    }, indent=2) + "\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in dest.iterdir() if f.is_file())
    print(f"speak-bs/: ready, {size / 1048576:.0f} MB, speaker {speaker}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", type=int, default=0, help="which of the voice's speakers (0 or 1)")
    ap.add_argument("--force", action="store_true", help="fetch again over what is here")
    args = ap.parse_args()
    if args.speaker not in (0, 1):
        print("the voice has two speakers: 0 or 1", file=sys.stderr)
        return 1
    fetch_bosnian_voice(speaker=args.speaker, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
