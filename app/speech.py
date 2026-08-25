#!/usr/bin/env python3
"""Speech input: spoken Bosnian audio -> Bosnian text.

Weights are read from models/lilly/listen/. Bosnian ("bs") is supported
directly. Bosnian/Croatian/Serbian are
acoustically near-identical, so if "bs" underperforms on real audio we can
fall back to "hr" — kept as an option here.

Usage:
    python3 app/speech.py recording.wav          # any format ffmpeg-free: wav/mp3/m4a/ogg
"""
import sys
import threading
from pathlib import Path

from app.lilly import LISTEN_DIR, BadInput

class UnreadableAudio(BadInput):
    """Raised when the upload is not audio we can decode."""


_model = None
_model_lock = threading.Lock()
_transcribe_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(str(LISTEN_DIR), device="cpu",
                                      compute_type="int8")
    return _model


def transcribe(audio_path: str, language: str = "bs") -> str:
    with _transcribe_lock:
        try:
            segments, info = get_model().transcribe(audio_path, language=language,
                                                    beam_size=5)
            # segments is a generator: it has to be drained inside the lock
            return " ".join(seg.text.strip() for seg in segments).strip()
        except BadInput:
            raise
        except Exception as exc:
            raise UnreadableAudio("that file is not audio we can read") from exc


def main() -> int:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        print("usage: python3 app/speech.py <audio-file>", file=sys.stderr)
        return 1
    print(transcribe(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
