#!/usr/bin/env python3
"""Screen Creative-Commons YouTube channels for a voice, without ears.

For each candidate video: download the audio stream (CC BY license checked
again from the video's own metadata), cut a 90-second excerpt from the middle,
and ask the shipped listener what it hears -- the language it detects, its
confidence, the words per minute -- and this project's own dialect gate what
share of the words are ijekavian (Bosnian/Croatian) against ekavian (Serbian).
Writes one JSON per run; the speaker check (are the excerpts of a channel one
person?) runs beside it with resemblyzer.

    python3 data/scripts/screen_youtube_voice.py --candidates cand.json --out screen/ --ffmpeg /path/to/ffmpeg
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
from download_extra_data import IJEKAVIAN, EKAVIAN  # noqa: E402

YT = REPO_ROOT / ".venv" / "bin" / "yt-dlp"
EXCERPT_SECONDS = 90


def download_audio(video_id: str, out_dir: Path) -> tuple:
    """(audio file path, metadata dict) -- the stream as served, no conversion."""
    meta = json.loads(subprocess.run([str(YT), "-J", "--no-warnings", "--skip-download",
                                      f"https://www.youtube.com/watch?v={video_id}"],
                                     text=True, capture_output=True, timeout=120).stdout)
    target = out_dir / f"{video_id}.%(ext)s"
    subprocess.run([str(YT), "-f", "bestaudio[ext=m4a]/bestaudio", "--no-warnings", "-q", "-o", str(target),
                    f"https://www.youtube.com/watch?v={video_id}"], check=True, timeout=1800)
    files = sorted(out_dir.glob(f"{video_id}.*"))
    files = [f for f in files if f.suffix != ".json"]
    if not files:
        raise SystemExit(f"{video_id}: nothing downloaded")
    return files[0], meta


def excerpt(ffmpeg: str, src: Path, dst: Path, start: float, seconds: int) -> None:
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds), "-i", str(src),
                    "-ac", "1", "-ar", "16000", str(dst)], check=True)


def dialect(text: str) -> dict:
    words = re.findall(r"\w+", text.lower())
    ije = sum(1 for w in words if IJEKAVIAN.search(w))
    eka = sum(1 for w in words if EKAVIAN.search(w))
    return {"words": len(words), "ijekavian": ije, "ekavian": eka,
            "ijekavian_share": round(ije / (ije + eka), 3) if ije + eka else None}


def listen(path: Path) -> dict:
    from app.speech import get_model
    model = get_model()
    segments, info = model.transcribe(str(path), beam_size=5)
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs)
    logprob = sum(s.avg_logprob for s in segs) / max(len(segs), 1)
    return {"language": info.language, "language_probability": round(info.language_probability, 3),
            "avg_logprob": round(logprob, 3), "no_speech": round(max((s.no_speech_prob for s in segs), default=0.0), 3),
            "words_per_minute": round(60 * len(text.split()) / EXCERPT_SECONDS, 1), "text": text[:400]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True, help='{"channel": ["video_id", ...]}')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ffmpeg", required=True)
    args = ap.parse_args()
    cands = json.loads(args.candidates.read_text(encoding="utf-8"))
    audio_dir, ex_dir = args.out / "audio", args.out / "excerpts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ex_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for channel, ids in cands.items():
        for vid in ids:
            try:
                src, meta = download_audio(vid, audio_dir)
            except Exception as exc:
                report[vid] = {"channel": channel, "error": f"{type(exc).__name__}: {exc}"[:200]}
                print(f"  {channel} {vid}: {report[vid]['error']}", flush=True)
                continue
            duration = float(meta.get("duration") or 0)
            wav = ex_dir / f"{vid}.wav"
            excerpt(args.ffmpeg, src, wav, max(0.0, duration / 2 - EXCERPT_SECONDS / 2), EXCERPT_SECONDS)
            heard = listen(wav)
            rec = {"channel": channel, "title": meta.get("title"), "uploader": meta.get("uploader"),
                   "license": meta.get("license"), "duration_min": round(duration / 60, 1),
                   "upload_date": meta.get("upload_date"), "audio_file": str(src), "excerpt": str(wav),
                   **heard, "dialect": dialect(heard["text"])}
            report[vid] = rec
            print(f"  {channel} {vid}: {rec['license'] and rec['license'][:16]} lang={rec['language']}({rec['language_probability']}) "
                  f"logprob={rec['avg_logprob']} wpm={rec['words_per_minute']} ije={rec['dialect']['ijekavian_share']} | {rec['title'][:50]}", flush=True)
            (args.out / "screen.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'screen.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
