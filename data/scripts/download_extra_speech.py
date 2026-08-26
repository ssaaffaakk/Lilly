#!/usr/bin/env python3
"""Download extra transcribed speech beyond FLORES bs_ba, as Croatian/Serbian proxies.

There is no Bosnian speech corpus at meaningful scale beyond the 3,091/400/925
clips download_speech_data.py already fetches from google/fleurs bs_ba. The
ParlaSpeech project's own roadmap lists Bosnian only as a v4 "in development"
target (see clarinsi.github.io/parlaspeech, checked 2026-08-26) — not released.
Two real, much larger corpora exist for the two closest languages instead:

  ParlaSpeech-HR 2.0 (Croatian) -- clarin.si/repository/handle/11356/1914
    3,110 hours, 922,679 clips. License: CC BY-SA 4.0.
  ParlaSpeech-RS 1.0 (Serbian)  -- huggingface.co/datasets/classla/ParlaSpeech-RS
    896 hours, 290,778 clips. License: CC BY 4.0.

Both align parliamentary session recordings to the OFFICIAL PARLIAMENTARY
RECORD (ParlaMint) — the "text" column is a human-written transcript, never
machine output. An ASR system was used only to find *where* in the audio each
sentence falls (word-level timestamps ship alongside), not to produce the
words. The honest trade-off: the official record is edited for readability,
so it isn't always a literal transcript of the audio's disfluencies and
repetitions — the ParlaSpeech paper reports ~4-5% word-level mismatch against
the raw recording, similar to a strong ASR system's own error rate. That is
still a real, human-authored transcript, not an ASR guess, and is the reason
this script uses the "text" column rather than a "words"-reconstructed one.

Verified before writing this script (2026-08-26), not taken on faith:
  - Streamed live parquet shards from both datasets and cross-checked clip
    duration (via soundfile, after decoding) against the "audio_length"
    column: exact match on every sample checked.
  - Read the "text" column, not "text_normalised" -- casing and punctuation
    are intact. download_speech_data.py made the same choice for FLORES, and
    for the same reason: this text goes to the translator and the reader, and
    normalised text would teach the listener to drop punctuation.
  - Two lookalikes are not new data: huggingface.co/datasets/
    shunyalabs/bosnian-speech-dataset and .../serbian-speech-dataset have
    exactly 3,091/400/925 and 2,944/290/700 clips -- an exact match to
    google/fleurs' bs_ba and sr_rs splits. They are re-uploads of data
    download_speech_data.py already fetches, not new material.
  - huggingface.co/datasets/Speech-data/Bosnian-Speech-Dataset advertises 169
    hours but ships one sample MP3 file -- a marketing page for a paid
    dataset (speech-data.ai), not a downloadable corpus. Also licensed
    CC BY-NC-ND (no derivatives), which would rule it out on its own.
  - Common Voice's hr/sr configs are gone from the Hugging Face mirror this
    script would otherwise use: as of October 2025 Mozilla moved Common Voice
    behind "Mozilla Data Collective" and stopped serving the free snapshot.
  - MediaSpeech does not cover Bosnian, Croatian or Serbian at all (only
    French, Arabic, Turkish, Spanish).
  - No Bosnian university/OpenSLR/ELRA speech corpus, and no ready dataset of
    Bosnian YouTube subtitles, turned up. YouTube auto-captions are machine
    transcripts anyway -- exactly what this script exists to avoid.
  - VoxPopuli's Croatian config is real and human-transcribed (European
    Parliament official record, CC0) but only 43 transcribed hours -- folded
    in as --lang hr_voxpopuli for completeness, not the main event. Its
    per-clip duration isn't in a metadata column the way ParlaSpeech's is, so
    that path filters by decoded length instead and is less exercised than
    the other two; treat it as a smaller, secondary source.

    python3 data/scripts/download_extra_speech.py                    # HR, 4,000 clips
    python3 data/scripts/download_extra_speech.py --lang sr
    python3 data/scripts/download_extra_speech.py --lang hr --max-clips 8000
    python3 data/scripts/download_extra_speech.py --lang hr_voxpopuli

Writes data/speech/extra_<lang>/<split>/*.wav plus
data/speech/extra_<lang>/<split>.tsv -- the exact "path<TAB>text" format
download_speech_data.py and training/train_speech.py already use. Kept in its
own extra_<lang>/ tree rather than merged into train.tsv/valid.tsv/test.tsv,
so this never overwrites or reshuffles the existing FLORES splits -- combine
them yourself when ready to train on both, e.g.:

    cat data/speech/train.tsv data/speech/extra_hr/train.tsv > combined.tsv
    python3 training/train_speech.py --data combined.tsv \\
        --valid data/speech/valid.tsv   # keep evaluating on real Bosnian

Note on the license: CC BY-SA 4.0 (ParlaSpeech-HR) is share-alike -- anything
you distribute that was built from this data is expected to carry the same
license. CC BY 4.0 (ParlaSpeech-RS) and CC0 (VoxPopuli) do not carry that
requirement. This script prints which license applies to each run; it does
not make the redistribution decision for you, and keeping each --lang in its
own extra_<lang>/ directory (rather than one merged pool) is what lets you
tell later which clips came from which corpus if that ever matters.
"""
import argparse
import io
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

SPEECH_DIR = Path(__file__).resolve().parents[1] / "speech"
LISTING = "https://datasets-server.huggingface.co/parquet?dataset={repo}&config={config}"
SAMPLE_RATE = 16_000
SPLITS = ("train", "valid", "test")
SPLIT_RATIOS = (0.90, 0.05, 0.05)  # matches roughly what FLORES bs_ba gives us

DATASETS = {
    "hr": dict(
        repo="classla/ParlaSpeech-HR", config="default",
        text_cols=("text",), duration_col="audio_length",
        license="CC BY-SA 4.0 -- https://creativecommons.org/licenses/by-sa/4.0/",
    ),
    "sr": dict(
        repo="classla/ParlaSpeech-RS", config="default",
        text_cols=("text",), duration_col="audio_length",
        license="CC BY 4.0 -- https://creativecommons.org/licenses/by/4.0/",
    ),
    "hr_voxpopuli": dict(
        repo="facebook/voxpopuli", config="hr",
        text_cols=("raw_text", "normalized_text", "text"), duration_col=None,
        license="CC0 -- https://creativecommons.org/publicdomain/zero/1.0/",
    ),
}


def list_shards(repo: str, config: str) -> list:
    url = LISTING.format(repo=repo.replace("/", "%2F"), config=config)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    files = [f for f in data.get("parquet_files", []) if f["split"] == "train"]
    if not files:
        raise SystemExit(f"no train shards listed for {repo}/{config} -- "
                          "check the dataset still exists at that path")
    return files


def fetch_to_file(url: str, target: Path, note: str) -> Path:
    """Stream a parquet shard to disk, resuming a dropped connection.

    Same approach as download_speech_data.py's fetch_to_file: these shards run
    300-450 MB each, too big to hold as one in-memory response reliably.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 6):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "lilly-translator"}
        if have:
            headers["Range"] = f"bytes={have}-"
            print(f"      resuming {note} at {have / 1048576:.0f} MB", flush=True)
        else:
            print(f"      downloading {note}", flush=True)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = "ab" if resp.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
            part.rename(target)
            return target
        except Exception as exc:
            print(f"      attempt {attempt} stopped: {type(exc).__name__}: {exc}",
                  flush=True)
            if attempt == 5:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def decode_clip(raw: bytes) -> np.ndarray | None:
    """Decode whatever container the source used, return float32 mono @ 16 kHz."""
    try:
        audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        return None
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, SAMPLE_RATE, rate).astype("float32")
    return audio


def collect(lang: str, max_clips: int, min_sec: float, max_sec: float,
            max_shards: int) -> list:
    """Pull rows from shard after shard until max_clips good ones are found."""
    cfg = DATASETS[lang]
    shards = list_shards(cfg["repo"], cfg["config"])
    print(f"{cfg['repo']} ({cfg['config']}): {len(shards)} shard(s) available, "
          f"license {cfg['license']}")

    import pyarrow.parquet as pq

    scratch = SPEECH_DIR / ".download"
    kept = []
    for shard_i, entry in enumerate(shards):
        if len(kept) >= max_clips or shard_i >= max_shards:
            break
        shard_path = scratch / f"{lang}-{shard_i:04d}.parquet"
        try:
            fetch_to_file(entry["url"], shard_path,
                          f"shard {shard_i + 1}/{len(shards)} "
                          f"({entry['size'] / 1048576:.0f} MB)")
        except Exception as exc:
            print(f"      FAILED, skipping shard: {exc}", file=sys.stderr)
            continue

        reader = pq.ParquetFile(shard_path)
        names = reader.schema_arrow.names
        text_col = next((c for c in cfg["text_cols"] if c in names), None)
        if text_col is None:
            raise SystemExit(f"no transcript column in {names} (looked for "
                              f"{cfg['text_cols']}) -- the dataset schema changed")
        cols = [c for c in ("id", "audio", text_col, cfg["duration_col"]) if c]

        before = len(kept)
        for batch in reader.iter_batches(batch_size=64, columns=cols):
            for row in batch.to_pylist():
                if len(kept) >= max_clips:
                    break
                said = (row.get(text_col) or "").strip()
                if not said:
                    continue
                dur = row.get(cfg["duration_col"]) if cfg["duration_col"] else None
                if dur is not None and not (min_sec <= dur <= max_sec):
                    continue
                raw = (row.get("audio") or {}).get("bytes")
                if not raw:
                    continue
                audio = decode_clip(raw)
                if audio is None:
                    continue
                actual_sec = len(audio) / SAMPLE_RATE
                if not (min_sec <= actual_sec <= max_sec):
                    continue
                kept.append((audio, said))
            if len(kept) >= max_clips:
                break
        print(f"      {len(kept) - before} clip(s) kept from this shard "
              f"({len(kept)}/{max_clips} total)")
        shard_path.unlink(missing_ok=True)

    if scratch.exists() and not any(scratch.iterdir()):
        scratch.rmdir()
    return kept


def write_splits(lang: str, clips: list, seed: int) -> None:
    out_dir = SPEECH_DIR / f"extra_{lang}"
    rng = random.Random(seed)
    order = list(range(len(clips)))
    rng.shuffle(order)  # parliamentary shards are ordered by session/date/speaker

    n = len(order)
    n_train = int(n * SPLIT_RATIOS[0])
    n_valid = int(n * SPLIT_RATIOS[1])
    bounds = {"train": order[:n_train],
              "valid": order[n_train:n_train + n_valid],
              "test": order[n_train + n_valid:]}

    grand = 0
    for split, idxs in bounds.items():
        clips_dir = out_dir / split
        clips_dir.mkdir(parents=True, exist_ok=True)
        tsv = out_dir / f"{split}.tsv"
        with open(tsv, "w", encoding="utf-8") as f:
            for i, idx in enumerate(idxs, start=1):
                audio, text = clips[idx]
                name = f"{i:05d}.wav"
                sf.write(clips_dir / name, audio, SAMPLE_RATE, subtype="FLOAT")
                f.write(f"{split}/{name}\t{text}\n")
        print(f"  {split}: {len(idxs):,} clips -> {clips_dir}\n"
              f"    transcripts -> {tsv}")
        grand += len(idxs)
    print(f"\n{grand:,} clips ready under {out_dir}. Train with, e.g.:\n"
          f"  cat data/speech/train.tsv {out_dir / 'train.tsv'} > /tmp/combined.tsv\n"
          f"  python3 training/train_speech.py --data /tmp/combined.tsv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="hr", choices=sorted(DATASETS),
                    help="hr (ParlaSpeech-HR, default), sr (ParlaSpeech-RS), "
                         "hr_voxpopuli (VoxPopuli Croatian, smaller)")
    ap.add_argument("--max-clips", type=int, default=4000,
                    help="total clips to keep across train/valid/test (default 4000, "
                         "comparable in scale to the FLORES bs_ba set)")
    ap.add_argument("--min-sec", type=float, default=2.0,
                    help="drop clips shorter than this (default 2.0s)")
    ap.add_argument("--max-sec", type=float, default=15.0,
                    help="drop clips longer than this (default 15.0s, Whisper-friendly)")
    ap.add_argument("--max-shards", type=int, default=40,
                    help="give up after this many shards even if --max-clips isn't "
                         "reached (safety net against a very high --max-clips)")
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed before the train/valid/test split")
    args = ap.parse_args()

    if (SPEECH_DIR / f"extra_{args.lang}" / "train.tsv").exists():
        print(f"data/speech/extra_{args.lang}/ already has a train.tsv, skipping. "
              f"Delete it first if you want to re-download.")
        return 0

    clips = collect(args.lang, args.max_clips, args.min_sec, args.max_sec,
                    args.max_shards)
    if not clips:
        print("No usable clips found -- nothing written.", file=sys.stderr)
        return 1
    if len(clips) < args.max_clips:
        print(f"note: only found {len(clips)}/{args.max_clips} clips within "
              f"{args.max_shards} shard(s); raise --max-shards for more.")

    write_splits(args.lang, clips, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
