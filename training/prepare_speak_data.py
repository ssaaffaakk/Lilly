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


def build_rows(rows: list, speakers: dict, audio_root: Path) -> list:
    """(absolute wav path, speaker name, text) for every clip of a selected speaker."""
    selected = {int(s) for s in speakers["selected"]}
    by_clip = speakers["clips"]
    out = []
    for clip, text in rows:
        rec = by_clip.get(clip)
        if rec is None:
            raise SystemExit(f"{clip} is in the split but not in speakers.json -- "
                             "the clusters were computed on a different file")
        if int(rec["speaker"]) not in selected:
            continue
        path = (audio_root / clip).resolve()
        if not path.is_file():
            raise SystemExit(f"{path} is not on disk -- not writing a training file with holes")
        out.append((str(path), str(rec["speaker"]), text.replace("|", " ").strip()))
    return out


def write_csv(out: Path, rows: list) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for path, speaker, text in rows:
            fh.write(f"{path}|{speaker}|{text}\n")
    per = collections.Counter(speaker for _, speaker, _ in rows)
    manifest = {"rows": len(rows), "speakers": dict(sorted(per.items())),
                "csv": str(out)}
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
    args = ap.parse_args()

    rows = read_tsv(args.tsv)
    speakers = json.loads(args.speakers.read_text(encoding="utf-8"))
    if not speakers.get("selected"):
        raise SystemExit(f"{args.speakers} selects no speaker; nothing to train")
    kept = build_rows(rows, speakers, args.tsv.parent)
    if len(kept) < args.min_rows:
        raise SystemExit(f"{len(kept)} clips for the selected speakers, under {args.min_rows}; "
                         "not a voice's worth of training")
    manifest = write_csv(args.out, kept)
    print(f"{manifest['rows']} rows -> {args.out}")
    for speaker, n in manifest["speakers"].items():
        print(f"  speaker {speaker}: {n} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
