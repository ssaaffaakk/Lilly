#!/usr/bin/env python3
"""Fetch lilly-speech Kaggle output and install models/lilly/listen/."""
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LISTEN = REPO / "models" / "lilly" / "listen"
PREVIOUS = REPO / "models" / "lilly" / "listen-previous"
OUT = REPO / "models" / "kaggle-output"
KEEP = OUT / "speech-keep"
KAGGLE = REPO / ".venv" / "bin" / "kaggle"
MIN_ZIP = 1_000_000
ZIP_PATTERN = r"lilly-listen.*\.zip"


def kaggle_user() -> str:
    return json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


def find_zip() -> Path | None:
    """Prefer a zip already on disk. Kaggle 404s the large files after cancel."""
    found: list[Path] = []
    for root in (KEEP, OUT):
        if root.is_dir():
            found.extend(p for p in root.rglob("lilly-listen.zip") if p.is_file())
    usable = [p for p in found if p.stat().st_size >= MIN_ZIP]
    if not usable:
        return None
    return max(usable, key=lambda p: p.stat().st_size)


def fetch() -> None:
    existing = find_zip()
    if existing:
        print(f"already have {existing} ({existing.stat().st_size:,} bytes) — skipping Kaggle fetch")
        return
    KEEP.mkdir(parents=True, exist_ok=True)
    # Zip-only. A full `kernels output` walks every audio stub and 429s, then
    # the weights are gone. Status can stay RUNNING after the zip is real.
    subprocess.run(
        [str(KAGGLE), "kernels", "output", f"{kaggle_user()}/lilly-speech",
         "-p", str(KEEP), "--file-pattern", ZIP_PATTERN],
        check=True,
    )


def install() -> Path:
    zpath = find_zip()
    if not zpath:
        raise SystemExit(f"no lilly-listen.zip under {OUT} — speech run may not have finished packaging")
    extract = OUT / "listen-extract"
    if extract.exists():
        shutil.rmtree(extract)
    extract.mkdir(parents=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(extract)
    src = extract / "models" / "lilly" / "listen"
    if not (src / "model.bin").is_file() and not (src / "config.json").exists():
        raise SystemExit(f"zip unpacked but no listener at {src}")
    if LISTEN.exists():
        if PREVIOUS.exists():
            shutil.rmtree(PREVIOUS)
        shutil.copytree(LISTEN, PREVIOUS)
        shutil.rmtree(LISTEN)
    shutil.copytree(src, LISTEN)
    print(f"installed listener -> {LISTEN}")
    if PREVIOUS.exists():
        print(f"previous listener kept at {PREVIOUS}")
    return LISTEN


def main() -> int:
    fetch()
    install()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
