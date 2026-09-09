#!/usr/bin/env python3
"""Fetch exactly the ParlaSpeech-HR segments a selection names, for the voice.

training/select_parlaspeech_voice.py chooses speakers and segments from the
metadata scan and writes a selection file: the shard URLs and sizes it saw,
and every segment as (shard, row, id, seconds, speaker). This script reads only
the row groups those rows sit in -- 100 rows, about 20 MB each -- through the
same range-reading parquet file download_extra_speech.py uses, checks each
row's id against the selection, decodes the audio to 16 kHz WAV, and writes
the metadata.csv piper.train reads: wav|speaker|text.

Nothing is guessed on the box. A shard whose size differs from the one the
selection recorded, a row whose id is not the one expected, a transcript that
fails the selection's own clean rule, or a segment that never turns up, stops
the run: the training file would not be the one the pre-registration names.

    python3 data/scripts/download_parlaspeech_voice.py \\
        --selection training/speak-parla/selection.json --out /kaggle/temp/parla
"""
import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from download_extra_speech import RemoteParquet, decode_clip  # noqa: E402

DIGIT, BRACKET = re.compile(r"\d"), re.compile(r"[\[\]()]")


def groups_of(reader) -> list:
    """(first_row, last_row_exclusive) per row group, from the footer."""
    out, start = [], 0
    for g in range(reader.metadata.num_row_groups):
        n = reader.metadata.row_group(g).num_rows
        out.append((start, start + n))
        start += n
    return out


def group_index(groups: list, row: int) -> int:
    for g, (a, b) in enumerate(groups):
        if a <= row < b:
            return g
    raise SystemExit(f"row {row} is past the end of the shard ({groups[-1][1]} rows)")


def clean_text(text: str, max_chars: int) -> str:
    text = (text or "").strip().replace("|", " ")
    if not text or len(text) > max_chars or DIGIT.search(text) or BRACKET.search(text):
        raise SystemExit(f"a selected segment fails the clean rule on the box: {text[:80]!r}")
    return text


def write_wav(path: Path, audio, rate: int = 16_000) -> None:
    import numpy as np
    import soundfile as sf
    sf.write(str(path), np.clip(audio, -1.0, 1.0), rate, subtype="PCM_16")


def fetch(selection: dict, out: Path) -> dict:
    import pyarrow.parquet as pq
    clips = out / "train"
    clips.mkdir(parents=True, exist_ok=True)
    by_shard = collections.defaultdict(list)
    for seg in selection["segments"]:
        by_shard[int(seg[0])].append(seg)
    rule = selection["rule"]
    urls, sizes = selection["shard_urls"], selection["shard_sizes"]
    rows_out, per_speaker, secs_speaker = [], collections.Counter(), collections.Counter()
    found = 0
    t0 = time.time()
    transferred = 0
    for k, si in enumerate(sorted(by_shard)):
        handle = RemoteParquet(urls[si])
        if handle.size != sizes[si]:
            raise SystemExit(f"shard {si} is {handle.size} bytes, the selection saw {sizes[si]}: "
                             "the dataset changed under the selection; not fetching")
        reader = pq.ParquetFile(handle)
        groups = groups_of(reader)
        wanted = collections.defaultdict(dict)   # group -> {row: seg}
        for seg in by_shard[si]:
            row = int(seg[1])
            wanted[group_index(groups, row)][row] = seg
        for g in sorted(wanted):
            table = reader.read_row_group(g, columns=["id", "audio", "text_normalised", "speaker_name"])
            start = groups[g][0]
            rows = table.to_pylist()
            for row, seg in sorted(wanted[g].items()):
                r = rows[row - start]
                if r["id"] != seg[2]:
                    raise SystemExit(f"shard {si} row {row}: id {r['id']!r}, the selection named {seg[2]!r}; "
                                     "the dataset changed under the selection; not fetching")
                if r["speaker_name"] != selection["speakers"][str(seg[4])]:
                    raise SystemExit(f"{seg[2]}: spoken by {r['speaker_name']!r}, selected as "
                                     f"{selection['speakers'][str(seg[4])]!r}")
                text = clean_text(r["text_normalised"], rule["max_chars"])
                raw = (r.get("audio") or {}).get("bytes")
                audio = decode_clip(raw) if raw else None
                if audio is None:
                    raise SystemExit(f"{seg[2]}: audio did not decode")
                seconds = len(audio) / 16_000
                if not (rule["min_sec"] <= seconds <= rule["max_sec"] + 0.5):
                    raise SystemExit(f"{seg[2]}: {seconds:.1f} s on the box, outside the rule")
                path = clips / f"{seg[2]}.wav"
                write_wav(path, audio)
                rows_out.append((str(path.resolve()), str(seg[4]), text))
                per_speaker[str(seg[4])] += 1
                secs_speaker[str(seg[4])] += seconds
                found += 1
        transferred += handle.bytes_read
        if (k + 1) % 10 == 0 or k + 1 == len(by_shard):
            print(f"  shard {si} ({k + 1}/{len(by_shard)}): {found} clips, {transferred / 1048576:.0f} MB, "
                  f"{time.time() - t0:.0f} s", flush=True)
    if found != len(selection["segments"]):
        raise SystemExit(f"{found} of {len(selection['segments'])} selected segments fetched; not training on a hole")
    csv = out / "metadata.csv"
    with csv.open("w", encoding="utf-8") as fh:
        for path, speaker, text in rows_out:
            fh.write(f"{path}|{speaker}|{text}\n")
    manifest = {"rows": len(rows_out), "csv": str(csv),
                "speakers": {idx: {"name": selection["speakers"][idx], "clips": per_speaker[idx],
                                   "minutes": round(secs_speaker[idx] / 60, 1)}
                             for idx in sorted(per_speaker, key=int)},
                "minutes": round(sum(secs_speaker.values()) / 60, 1),
                "transferred_mb": round(transferred / 1048576), "seconds": round(time.time() - t0)}
    Path(str(csv) + ".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    print(f"{len(selection['segments'])} segments of {len(selection['speakers'])} speakers, "
          f"{selection['minutes_total']:.0f} minutes, from {len(set(s[0] for s in selection['segments']))} shards")
    manifest = fetch(selection, args.out)
    print(f"{manifest['rows']} rows, {manifest['minutes']:.0f} minutes -> {manifest['csv']} "
          f"({manifest['transferred_mb']} MB in {manifest['seconds']} s)")
    for idx, s in manifest["speakers"].items():
        print(f"  speaker {idx} ({s['name']}): {s['clips']} clips, {s['minutes']:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
