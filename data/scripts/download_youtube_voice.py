#!/usr/bin/env python3
"""Fetch the Creative-Commons YouTube audio a voice selection names, on the Mac.

training/speak-youtube/selection.json lists channels and video ids chosen by
the rule in training/PREREGISTRATION.md ("v8 -- speak"). For each video this
reads the metadata again and refuses anything whose license is not
"Creative Commons Attribution license (reuse allowed)" -- the license is read
from the video itself on the day it is fetched, not from the search that
found it -- then downloads the best audio stream as YouTube serves it (m4a;
no conversion, no ffmpeg), and writes a manifest beside the files: id, title,
uploader, channel, license, duration, upload date, URL, bytes, sha256. The
manifest is what the attribution CC BY asks for, and what the box checks
before decoding a byte.

The audio stays under data/speech-extra/youtube-voice/ (gitignored) and goes
to Kaggle as a dataset; the manifest is committed.

    python3 data/scripts/download_youtube_voice.py --selection training/speak-youtube/selection.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
YT = REPO_ROOT / ".venv" / "bin" / "yt-dlp"
OUT = REPO_ROOT / "data" / "speech-extra" / "youtube-voice"
CC_BY = "Creative Commons Attribution license (reuse allowed)"


def metadata(video_id: str) -> dict:
    out = subprocess.run([str(YT), "-J", "--no-warnings", "--skip-download",
                          f"https://www.youtube.com/watch?v={video_id}"],
                         text=True, capture_output=True, timeout=180)
    if not out.stdout.strip():
        raise SystemExit(f"{video_id}: no metadata ({out.stderr.strip()[:160]})")
    return json.loads(out.stdout)


def fetch(video_id: str, out_dir: Path) -> Path:
    existing = [f for f in out_dir.glob(f"{video_id}.*") if f.suffix in (".m4a", ".webm", ".opus", ".mp4")]
    if existing:
        return existing[0]
    subprocess.run([str(YT), "-f", "bestaudio[ext=m4a]/bestaudio", "--no-warnings", "-q",
                    "-o", str(out_dir / f"{video_id}.%(ext)s"), f"https://www.youtube.com/watch?v={video_id}"],
                   check=True, timeout=3600)
    got = [f for f in out_dir.glob(f"{video_id}.*") if f.suffix in (".m4a", ".webm", ".opus", ".mp4")]
    if not got:
        raise SystemExit(f"{video_id}: nothing downloaded")
    return got[0]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, default=REPO_ROOT / "training" / "speak-youtube" / "selection.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    t0 = time.time()
    for speaker, rec in selection["speakers"].items():
        for video in rec["videos"]:
            vid, channel = video["id"], video["channel"]
            if vid in manifest and Path(manifest[vid]["file"]).is_file():
                continue
            meta = metadata(vid)
            if meta.get("license") != CC_BY:
                raise SystemExit(f"{vid} ({meta.get('title')!r}): license is {meta.get('license')!r}, not CC BY -- not fetching")
            path = fetch(vid, args.out)
            manifest[vid] = {"speaker": speaker, "channel": channel, "channel_id": meta.get("channel_id"), "uploader": meta.get("uploader"),
                             "title": meta.get("title"), "license": meta.get("license"),
                             "duration_s": meta.get("duration"), "upload_date": meta.get("upload_date"),
                             "url": f"https://www.youtube.com/watch?v={vid}", "file": str(path),
                             "bytes": path.stat().st_size, "sha256": sha256_of(path), "fetched": time.strftime("%Y-%m-%d")}
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"  {speaker} / {channel} {vid}: {meta.get('duration', 0) / 60:.0f} min, {path.stat().st_size / 1e6:.0f} MB "
                  f"({time.time() - t0:.0f} s)", flush=True)
    hours = sum((m["duration_s"] or 0) for m in manifest.values()) / 3600
    print(f"{len(manifest)} videos, {hours:.1f} h, manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
