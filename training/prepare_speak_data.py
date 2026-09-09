#!/usr/bin/env python3
"""The Piper training file for the Bosnian voice: wav|speaker|text, one row per clip.

Reads a FLEURS split (data/speech/train.tsv: clip<TAB>raw transcript) and the
speaker clusters training/cluster_speakers.py wrote, and writes the CSV
piper.train reads -- pipe-separated, no header, the cluster id as the speaker
name -- keeping only the clips of the selected speakers. Beside it, a manifest
(<out>.manifest.json) with the counts, so the notebook can assert them and the
results file can quote them.

The text is FLEURS's raw transcription, punctuation and all: espeak reads a
comma as a pause and a full stop as a fall, and a voice trained on text with
neither speaks in one flat line. A '|' inside a sentence would split the row,
so it becomes a space. Nothing else is touched -- the phonemizer is where text
becomes sound, and it runs the same way in training and in the app.

    python3 training/prepare_speak_data.py --tsv data/speech/train.tsv \\
        --speakers data/speech/speakers/speakers.json --out /kaggle/temp/piper/metadata.csv
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.cluster_speakers import read_tsv  # noqa: E402

MIN_ROWS = 500   # fewer clips than this is not a voice's worth of training


def build_rows(rows: list, speakers: dict, audio_root: Path, max_seconds: float = None) -> list:
    """(absolute wav path, speaker name, text) for every clip of a selected speaker.

    max_seconds leaves the longest clips out. VITS pays memory for the whole
    padded batch, so one 35-second clip sets the cost of every clip beside it:
    version 1 on Kaggle died at its first batches with CUDA out of memory. The
    durations are the ones cluster_speakers.py measured, so no audio is read.
    """
    selected = {int(s) for s in speakers["selected"]}
    by_clip = speakers["clips"]
    out = []
    build_rows.dropped_long = 0
    for clip, text in rows:
        rec = by_clip.get(clip)
        if rec is None:
            raise SystemExit(f"{clip} is in the split but not in speakers.json -- "
                             "the clusters were computed on a different file")
        if int(rec["speaker"]) not in selected:
            continue
        if max_seconds is not None and float(rec["seconds"]) > max_seconds:
            build_rows.dropped_long += 1
            continue
        path = (audio_root / clip).resolve()
        if not path.is_file():
            raise SystemExit(f"{path} is not on disk -- not writing a training file with holes")
        out.append((str(path), str(rec["speaker"]), text.replace("|", " ").strip()))
    return out


def write_csv(out: Path, rows: list, dropped_long: int = 0, max_seconds: float = None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for path, speaker, text in rows:
            fh.write(f"{path}|{speaker}|{text}\n")
    per = collections.Counter(speaker for _, speaker, _ in rows)
    manifest = {"rows": len(rows), "speakers": dict(sorted(per.items())),
                "csv": str(out), "max_seconds": max_seconds, "dropped_long": dropped_long}
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                 encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=REPO_ROOT / "data" / "speech" / "train.tsv")
    ap.add_argument("--speakers", type=Path,
                    default=REPO_ROOT / "data" / "speech" / "speakers" / "speakers.json")
    ap.add_argument("--out", type=Path, required=True, help="the metadata.csv piper.train reads")
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="leave out clips longer than this (the pre-registration says 20)")
    args = ap.parse_args()

    rows = read_tsv(args.tsv)
    speakers = json.loads(args.speakers.read_text(encoding="utf-8"))
    if not speakers.get("selected"):
        raise SystemExit(f"{args.speakers} selects no speaker; nothing to train")
    kept = build_rows(rows, speakers, args.tsv.parent, args.max_seconds)
    if len(kept) < args.min_rows:
        raise SystemExit(f"{len(kept)} clips for the selected speakers, under {args.min_rows}; "
                         "not a voice's worth of training")
    manifest = write_csv(args.out, kept, build_rows.dropped_long, args.max_seconds)
    print(f"{manifest['rows']} rows -> {args.out}"
          + (f" ({manifest['dropped_long']} clips over {args.max_seconds:g} s left out)"
             if args.max_seconds is not None else ""))
    for speaker, n in manifest["speakers"].items():
        print(f"  speaker {speaker}: {n} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
