#!/usr/bin/env python3
"""From hours of Creative-Commons YouTube audio to a Piper training file, on the box.

Inputs: the manifest data/scripts/download_youtube_voice.py wrote (one entry
per video: channel, license, sha256, file) beside the audio files it names.
Every file is checked against its sha256 before a byte is decoded.

For each video: decode to 16 kHz mono WAV (ffmpeg), hear it with the shipped
listener through app.speech's own model -- Bosnian, beam 5, Whisper's VAD --
and keep the listener's segments that pass a rule fixed in advance
(training/PREREGISTRATION.md, "v8 -- speak"): 3-20 s long, at most 300
characters, no digit, no bracket, mean log-probability at least LOGPROB, a
no-speech probability under 0.5. Each kept segment is cut to its own WAV with
a little padding; the transcript is the listener's, punctuation and all.

Then, per speaker -- the person the manifest names, the lecturer in the
title, not a cluster -- every kept clip is embedded (resemblyzer), the
speaker's centroid is taken over all of them, and clips under OUTLIER_SIM
cosine to it are dropped: a questioner, a recitation, a second voice. The
screening showed the same lecturer at 0.90-0.97 between recordings and two
lecturers in one hall at 0.62-0.78, so no global cluster threshold would
separate them; a name and a centroid do. Speakers left with under
MIN_MINUTES are left out. At most HOURS_EACH per speaker, in video order.
Fewer than MIN_SPEAKERS speakers, or fewer than 500 clips, stops the run.

The transcripts come from the same listener that will judge the voice. That
is stated in the pre-registration, not hidden here.

    python3 data/scripts/cut_youtube_voice.py --manifest /kaggle/input/.../manifest.json \\
        --audio-dir /kaggle/input/... --out /kaggle/temp/yt --ffmpeg ffmpeg
"""
import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Pre-registered (PREREGISTRATION.md, "v8 -- speak"); change there first.
MIN_SEC, MAX_SEC, MAX_CHARS = 3.0, 20.0, 300
LOGPROB = -0.6
NO_SPEECH = 0.5
PAD = 0.15
OUTLIER_SIM = 0.60
MIN_MINUTES = 60.0
HOURS_EACH = 3.0
MIN_SPEAKERS = 5
DIGIT, BRACKET = re.compile(r"\d"), re.compile(r"[\[\]()]")
CC_BY = "Creative Commons Attribution license (reuse allowed)"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode(ffmpeg: str, src: Path, dst: Path) -> None:
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000",
                    "-sample_fmt", "s16", str(dst)], check=True)


def hear(wav: Path) -> list:
    """The listener's segments: (start, end, text, avg_logprob, no_speech_prob)."""
    from app.speech import get_model
    model = get_model()
    segments, _ = model.transcribe(str(wav), language="bs", beam_size=5, vad_filter=True)
    return [(s.start, s.end, s.text.strip(), s.avg_logprob, s.no_speech_prob) for s in segments]


def passes(start: float, end: float, text: str, logprob: float, no_speech: float) -> str:
    """'' if the segment trains, else the reason it does not."""
    secs = end - start
    if not (MIN_SEC <= secs <= MAX_SEC):
        return "length"
    if not text or len(text) > MAX_CHARS:
        return "chars"
    if DIGIT.search(text) or BRACKET.search(text) or "|" in text:
        return "digits_or_brackets"
    if logprob < LOGPROB:
        return "logprob"
    if no_speech >= NO_SPEECH:
        return "no_speech"
    return ""


def cut(wav: Path, out: Path, start: float, end: float) -> float:
    import soundfile as sf
    with sf.SoundFile(str(wav)) as fh:
        rate = fh.samplerate
        a = max(0, int((start - PAD) * rate))
        b = min(fh.frames, int((end + PAD) * rate))
        fh.seek(a)
        audio = fh.read(b - a, dtype="float32", always_2d=False)
    sf.write(str(out), audio, rate, subtype="PCM_16")
    return len(audio) / rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--device", default="cuda", help="for the speaker embeddings")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips_dir = args.out / "train"
    clips_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.out / "decoded"
    wav_dir.mkdir(parents=True, exist_ok=True)
    kept = []                     # (path, channel, text, seconds, video, order)
    dropped = collections.Counter()
    t0 = time.time()
    videos = sorted(manifest.items(), key=lambda kv: (kv[1]["speaker"], -(kv[1].get("duration_s") or 0), kv[0]))
    for n, (vid, m) in enumerate(videos):
        if m.get("license") != CC_BY:
            raise SystemExit(f"{vid}: manifest license {m.get('license')!r} is not CC BY")
        src = args.audio_dir / Path(m["file"]).name
        if not src.is_file():
            raise SystemExit(f"{vid}: {src} is not attached")
        if sha256_of(src) != m["sha256"]:
            raise SystemExit(f"{vid}: sha256 differs from the manifest -- not the file that was selected")
        wav = wav_dir / f"{vid}.wav"
        decode(args.ffmpeg, src, wav)
        segs = hear(wav)
        got = 0
        for i, (start, end, text, logprob, no_speech) in enumerate(segs):
            why = passes(start, end, text, logprob, no_speech)
            if why:
                dropped[why] += 1
                continue
            path = clips_dir / f"{vid}-{i:05d}.wav"
            secs = cut(wav, path, start, end)
            kept.append((str(path.resolve()), m["speaker"], text.replace("|", " "), secs, vid, len(kept)))
            got += 1
        wav.unlink()
        print(f"  [{n + 1}/{len(videos)}] {m['speaker']} / {m['channel']} {vid}: {len(segs)} segments heard, {got} kept "
              f"({time.time() - t0:.0f} s)", flush=True)
    print(f"kept {len(kept)} clips, {sum(k[3] for k in kept) / 60:.0f} minutes; dropped {dict(dropped)}", flush=True)

    # one voice per named speaker: drop what strays from the speaker's own centroid
    import numpy as np
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder(device=args.device, verbose=False)
    by_speaker = collections.defaultdict(list)
    for k in kept:
        by_speaker[k[1]].append(k)
    speakers, report, rows = {}, {}, []
    for name, clips in by_speaker.items():
        emb = np.zeros((len(clips), 256), dtype=np.float32)
        for i, k in enumerate(clips):
            e = encoder.embed_utterance(preprocess_wav(Path(k[0])))
            emb[i] = e / (np.linalg.norm(e) + 1e-9)
        centroid = emb.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        sims = emb @ centroid
        members = [i for i in range(len(clips)) if sims[i] >= OUTLIER_SIM]
        minutes_all = sum(k[3] for k in clips) / 60
        minutes_kept = sum(clips[i][3] for i in members) / 60
        report[name] = {"clips": len(clips), "minutes": round(minutes_all, 1), "kept_clips": len(members),
                        "kept_minutes": round(minutes_kept, 1), "median_sim": round(float(np.median(sims)), 3),
                        "dropped_as_other_voice": len(clips) - len(members)}
        if minutes_kept < MIN_MINUTES:
            report[name]["verdict"] = f"under {MIN_MINUTES:.0f} min after the centroid filter; left out"
            print(f"  {name}: {report[name]}", flush=True)
            continue
        idx = str(len(speakers))
        speakers[idx] = name
        taken = 0.0
        for i in members:
            k = clips[i]
            if taken + k[3] > HOURS_EACH * 3600:
                break
            rows.append((k[0], idx, k[2]))
            taken += k[3]
        report[name]["verdict"] = f"speaker {idx}, {taken / 60:.0f} min train"
        print(f"  {name}: {report[name]}", flush=True)
    if len(speakers) < MIN_SPEAKERS:
        raise SystemExit(f"{len(speakers)} speakers with a dominant voice and {MIN_MINUTES:.0f}+ minutes; "
                         f"the rule wants {MIN_SPEAKERS}. Channels: {report}")
    if len(rows) < 500:
        raise SystemExit(f"{len(rows)} clips -- not a voice's worth")
    csv = args.out / "metadata.csv"
    with csv.open("w", encoding="utf-8") as fh:
        for path, idx, text in rows:
            fh.write(f"{path}|{idx}|{text}\n")
    per = collections.Counter(idx for _, idx, _ in rows)
    secs = collections.Counter()
    for path, idx, _ in rows:
        secs[idx] += next(k[3] for k in kept if k[0] == path)
    out = {"rows": len(rows), "csv": str(csv), "minutes": round(sum(secs.values()) / 60, 1),
           "speakers": {idx: {"name": speakers[idx], "clips": per[idx], "minutes": round(secs[idx] / 60, 1)} for idx in speakers},
           "kept_before_speaker_rule": len(kept), "dropped": dict(dropped), "channels": report,
           "rule": {"min_sec": MIN_SEC, "max_sec": MAX_SEC, "max_chars": MAX_CHARS, "logprob": LOGPROB,
                    "no_speech": NO_SPEECH, "outlier_sim": OUTLIER_SIM, "min_minutes": MIN_MINUTES,
                    "hours_each": HOURS_EACH, "min_speakers": MIN_SPEAKERS},
           "seconds": round(time.time() - t0)}
    Path(str(csv) + ".manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{out['rows']} rows, {out['minutes']} minutes, {len(speakers)} speakers -> {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
