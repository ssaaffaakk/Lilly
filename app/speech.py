#!/usr/bin/env python3
"""Speech input: spoken Bosnian audio -> Bosnian text.

Weights are read from models/lilly/listen/. Bosnian ("bs") is supported
directly. Bosnian/Croatian/Serbian are
acoustically near-identical, so if "bs" underperforms on real audio we can
fall back to "hr" — kept as an option here.

Usage:
    python3 app/speech.py recording.wav          # any format ffmpeg-free: wav/mp3/m4a/ogg
    python3 app/speech.py recording.wav models/lilly/listen-previous

`transcribe` takes an optional `build`, and that is what the measurement code
uses. The rule on this project is that a score is produced by the path a user's
audio takes, never by a faster-whisper call assembled beside it — both the
translator and the reader have been measured on the wrong path here, and the
translator's moved by 3.36 chrF2 when it was corrected. training/evaluate_speech.py
and training/speech_bench.py used to build their own WhisperModel with their own
decode settings; now they call this function, so "the same code" is a fact about
the code rather than a claim about two copies of it.

One build is held in memory at a time by default. `release()` drops it, which is
what lets a benchmark score two listeners in one process on an 8 GB machine
without holding both.
"""
import sys
import threading
from pathlib import Path

from app.lilly import LISTEN_DIR, BadInput

class UnreadableAudio(BadInput):
    """Raised when the upload is not audio we can decode."""


_models = {}
_model_lock = threading.Lock()
_transcribe_lock = threading.Lock()


def _key(build) -> str:
    """One cache key per build directory, however the caller spelled the path.

    Resolved, because "models/lilly/listen" and the absolute path are the same
    weights and would otherwise load two copies of the same half-gigabyte model
    into an 8 GB machine.
    """
    return str(Path(build).resolve() if build is not None else LISTEN_DIR.resolve())


def get_model(build=None):
    """The listener for `build`, loading it once and keeping it.

    Default is the installed listener, which is what the server serves.
    """
    key = _key(build)
    model = _models.get(key)
    if model is None:
        with _model_lock:
            model = _models.get(key)
            if model is None:
                from faster_whisper import WhisperModel
                model = WhisperModel(key, device="cpu", compute_type="int8")
                _models[key] = model
    return model


def release(build=None) -> None:
    """Drop a loaded listener so its memory goes back.

    Only a benchmark needs this: the server loads one listener and keeps it for
    the life of the process.
    """
    with _model_lock:
        _models.pop(_key(build), None)


def transcribe(audio_path: str, language: str = "bs", build=None) -> str:
    with _transcribe_lock:
        try:
            segments, info = get_model(build).transcribe(audio_path,
                                                         language=language,
                                                         beam_size=5)
            # segments is a generator: it has to be drained inside the lock
            return " ".join(seg.text.strip() for seg in segments).strip()
        except BadInput:
            raise
        except Exception as exc:
            raise UnreadableAudio("that file is not audio we can read") from exc


def main() -> int:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        print("usage: python3 app/speech.py <audio-file> [build]", file=sys.stderr)
        return 1
    build = sys.argv[2] if len(sys.argv) > 2 else None
    print(transcribe(sys.argv[1], build=build))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
