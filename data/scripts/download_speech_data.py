#!/usr/bin/env python3
"""Download transcribed Bosnian speech for training Lilly's listening.

There is no huge corpus of transcribed Bosnian, but there is a real one: a
read-speech set of a few thousand clips with human transcripts, split into
train, validation and test. Small by speech standards, big enough to fine-tune
on and — more importantly — big enough to measure with.

    python3 data/scripts/download_speech_data.py                # all three splits
    python3 data/scripts/download_speech_data.py --split test   # just one
    python3 data/scripts/download_speech_data.py --lang hr_hr   # Croatian, as a proxy

Writes data/speech/<split>/*.wav plus data/speech/<split>.tsv, which is exactly
what training/train_speech.py and training/evaluate_speech.py read.

Croatian (hr_hr) and Serbian (sr_rs) work as proxies — the three are close
enough acoustically that training on them helps Bosnian.
"""
import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

DATASET = "google/fleurs"
LISTING = ("https://datasets-server.huggingface.co/parquet"
           f"?dataset={DATASET.replace('/', '%2F')}&config=")
SPEECH_DIR = Path(__file__).resolve().parents[1] / "speech"
SPLITS = ("train", "validation", "test")
# what the training scripts expect the splits to be called
TSV_NAME = {"train": "train", "validation": "valid", "test": "test"}


def list_files(lang: str) -> list:
    with urllib.request.urlopen(LISTING + lang, timeout=60) as resp:
        data = json.load(resp)
    if not data.get("parquet_files"):
        raise SystemExit(f"no files listed for {lang} — check the language code")
    return data["parquet_files"]


def fetch(url: str, note: str) -> bytes:
    print(f"    downloading {note}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "lilly-translator"})
    # without a timeout one stalled socket hangs an unattended run for hours
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    print(f"    got {len(blob) / 1048576:.0f} MB, unpacking…", flush=True)
    return blob


def unpack(blob: bytes, split: str, out_dir: Path) -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(blob))
    names = table.column_names
    audio_col = "audio" if "audio" in names else names[0]
    # raw_transcription keeps capitals and punctuation; transcription is stripped
    # of both. Train on the raw one — the app hands this text to the translator
    # and shows it to the reader, and a listener taught to drop punctuation makes
    # both worse. The scoring script normalises either side anyway.
    text_col = next((c for c in ("raw_transcription", "transcription", "text")
                     if c in names), None)
    if text_col is None:
        raise SystemExit(f"no transcript column in {names}")

    clips_dir = out_dir / split
    clips_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / f"{TSV_NAME[split]}.tsv"
    audio = table.column(audio_col).to_pylist()
    text = table.column(text_col).to_pylist()

    kept = 0
    with open(tsv, "w", encoding="utf-8") as f:
        for i, (clip, said) in enumerate(zip(audio, text)):
            said = (said or "").strip()
            raw = clip.get("bytes") if isinstance(clip, dict) else None
            if not raw or not said:
                continue
            name = f"{i:05d}.wav"
            (clips_dir / name).write_bytes(raw)
            f.write(f"{split}/{name}\t{said}\n")
            kept += 1
    print(f"    {kept:,} clips -> {clips_dir}\n    transcripts -> {tsv}")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="bs_ba",
                    help="bs_ba (Bosnian), hr_hr / sr_rs as proxies")
    ap.add_argument("--split", choices=SPLITS, help="default: all three")
    args = ap.parse_args()

    wanted = (args.split,) if args.split else SPLITS
    files = [f for f in list_files(args.lang) if f["split"] in wanted]
    total_mb = sum(f["size"] for f in files) / 1048576
    print(f"{args.lang}: {len(files)} file(s), {total_mb:.0f} MB total")

    SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    grand = 0
    for entry in files:
        split = entry["split"]
        print(f"  {split}:")
        if (SPEECH_DIR / f"{TSV_NAME[split]}.tsv").exists():
            print("    already here, skipping")
            continue
        try:
            grand += unpack(fetch(entry["url"], f"{entry['size'] / 1048576:.0f} MB"),
                            split, SPEECH_DIR)
        except Exception as exc:  # noqa: BLE001 - report and carry on with the rest
            print(f"    FAILED: {exc}", file=sys.stderr)
    print(f"\n{grand:,} clips ready. Train with:\n"
          f"  python3 training/train_speech.py --data data/speech/train.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
