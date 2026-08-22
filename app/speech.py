#!/usr/bin/env python3
"""Speech input: spoken Bosnian audio -> Bosnian text.

Bosnian ("bs") is supported directly. Bosnian/Croatian/Serbian are
acoustically near-identical, so if "bs" underperforms on real audio we can
fall back to "hr" — kept as an option here.

Usage:
    python3 app/speech.py recording.wav          # any format ffmpeg-free: wav/mp3/m4a/ogg
"""
import sys
from pathlib import Path

MODEL_SIZE = "small"  # ~500 MB, good CPU speed/accuracy balance; "medium" = better, slower

_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str, language: str = "bs") -> str:
    segments, info = get_model().transcribe(audio_path, language=language, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


def main() -> int:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        print("usage: python3 app/speech.py <audio-file>", file=sys.stderr)
        return 1
    print(transcribe(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
