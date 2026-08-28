#!/usr/bin/env python3
"""Fetch lilly-speech Kaggle output and install models/lilly/listen/."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LISTEN = REPO / "models" / "lilly" / "listen"
PREVIOUS = REPO / "models" / "lilly" / "listen-previous"
OUT = REPO / "models" / "kaggle-output"
KAGGLE = REPO / ".venv" / "bin" / "kaggle"


def fetch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(KAGGLE), "kernels", "output", "afaksrmeli/lilly-speech", "-p", str(OUT)],
        check=True,
    )


def find_zip() -> Path | None:
    for path in OUT.rglob("lilly-listen.zip"):
        return path
    return None


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
