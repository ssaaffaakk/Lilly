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
import json
import sys
import time
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


def fetch_to_file(url: str, target: Path, note: str) -> Path:
    """Stream a file to disk, picking up where a failed attempt left off.

    The training split is 2.1 GB. Reading that into memory in one call needs
    2 GB of RAM on a machine that has 8, and one stalled socket anywhere in the
    twenty minutes it takes throws the whole thing away. So: write straight to
    disk in chunks, and if the connection drops, ask the server to continue from
    the byte we reached rather than starting over.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")

    for attempt in range(1, 6):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "lilly-translator"}
        if have:
            headers["Range"] = f"bytes={have}-"
            print(f"    resuming {note} at {have / 1048576:.0f} MB", flush=True)
        else:
            print(f"    downloading {note}", flush=True)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                # a server that ignores Range restarts the file, so start over too
                mode = "ab" if resp.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                total = int(resp.headers.get("Content-Length", 0)) + have
                with open(part, mode) as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if have % (100 << 20) < (1 << 20):
                            pct = f" ({100 * have / total:.0f}%)" if total else ""
                            print(f"      {have / 1048576:.0f} MB{pct}", flush=True)
            part.rename(target)
            print(f"    got {target.stat().st_size / 1048576:.0f} MB", flush=True)
            return target
        except Exception as exc:
            print(f"    attempt {attempt} stopped: {type(exc).__name__}: {exc}",
                  flush=True)
            if attempt == 5:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def unpack(path: Path, split: str, out_dir: Path) -> int:
    import pyarrow.parquet as pq

    # Read it a row group at a time. Loading a 2.1 GB parquet whole is the same
    # memory problem the download had, one step later.
    reader = pq.ParquetFile(path)
    names = reader.schema_arrow.names
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

    kept = i = 0
    with open(tsv, "w", encoding="utf-8") as f:
        for batch in reader.iter_batches(batch_size=64,
                                         columns=[audio_col, text_col]):
            audio = batch.column(audio_col).to_pylist()
            text = batch.column(text_col).to_pylist()
            for clip, said in zip(audio, text):
                i += 1
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
        # The parquet lands in scratch space, not under data/speech: keeping a
        # 2 GB copy beside the wav files it was unpacked into is pure waste.
        scratch = SPEECH_DIR / ".download" / f"{split}.parquet"
        try:
            path = fetch_to_file(entry["url"], scratch,
                                 f"{entry['size'] / 1048576:.0f} MB")
            grand += unpack(path, split, SPEECH_DIR)
            path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - report and carry on with the rest
            print(f"    FAILED: {exc}", file=sys.stderr)
    print(f"\n{grand:,} clips ready. Train with:\n"
          f"  python3 training/train_speech.py --data data/speech/train.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
