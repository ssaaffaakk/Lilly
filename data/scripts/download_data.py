#!/usr/bin/env python3
"""Download the Bosnian-English parallel corpora used for training.

Usage:
    python3 download_data.py            # core corpora (~35 MB, good quality)
    python3 download_data.py --full     # also OpenSubtitles v2024 (large, noisy, conversational)

Files land in data/raw/<corpus>/ as plain text pairs: <name>.bs and <name>.en,
one sentence per line, line N of .bs matches line N of .en.
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "raw"

# name -> (url, approx pairs, why we use it)
CORE = {
    "SETIMES": (
        "https://object.pouta.csc.fi/OPUS-SETIMES/v2/moses/bs-en.txt.zip",
        138_387, "Balkan news, clean formal Bosnian"),
    "WikiMatrix": (
        "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/bs-en.txt.zip",
        210_691, "Wikipedia sentence pairs"),
    "TED2020": (
        "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/bs-en.txt.zip",
        11_638, "TED talk subtitles, natural spoken style"),
    "Tatoeba": (
        "https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/bs-en.txt.zip",
        535, "short everyday sentences, human-checked"),
}

# Verified against the base model's own training manifest: OpenSubtitles-v2018
# contributes 82,298,975 pairs to it. Adding this corpus shows the model
# 18 million sentences it has already been trained on — more data, no new
# information, and a test set that gets harder to keep clean.
FULL = {
    "OpenSubtitles": (
        "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/bs-en.txt.zip",
        18_477_297, "movie/TV subtitles, casual conversational Bosnian (noisy)"),
}


def download(name: str, url: str, pairs: int, note: str) -> None:
    out_dir = RAW_DIR / name
    if any(out_dir.glob("*.bs")):
        print(f"  {name}: already downloaded, skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {name}: ~{pairs:,} pairs — {note}")
    print(f"    downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "lilly-translator"})
    # without a timeout one stalled socket hangs an unattended run for hours
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    print(f"    got {len(blob) / 1048576:.1f} MB, extracting…")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in zf.namelist():
            if member.endswith(".bs") or member.endswith(".en"):
                target = out_dir / (name + Path(member).suffix)
                target.write_bytes(zf.read(member))
                n_lines = sum(1 for _ in open(target, "rb"))
                print(f"    {target.relative_to(RAW_DIR.parent)}: {n_lines:,} lines")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also download OpenSubtitles (large)")
    args = ap.parse_args()

    corpora = dict(CORE)
    if args.full:
        corpora.update(FULL)

    print(f"Downloading {len(corpora)} corpora into {RAW_DIR}")
    for name, (url, pairs, note) in corpora.items():
        try:
            download(name, url, pairs, note)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            print(f"    FAILED: {exc}", file=sys.stderr)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
