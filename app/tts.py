#!/usr/bin/env python3
"""Voice output: English text -> spoken audio.

Weights and the voice are read from models/lilly/speak/.
Produces a 24 kHz WAV. Runs real-time on CPU.

Usage:
    python3 app/tts.py "Hello, how are you?" out.wav
"""
import sys
import threading

from app.lilly import SPEAK_DIR

VOICE = SPEAK_DIR / "voices" / "default.pt"
# Speech runs at roughly the speed it plays, so the length of the text is the
# length of the CPU burn. Past this the audio is cut rather than refused.
MAX_CHARS = 1200

_pipeline = None
_pipeline_lock = threading.Lock()
_speak_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
      with _pipeline_lock:
        if _pipeline is None:
            from kokoro import KModel, KPipeline
            voice_model = KModel(repo_id="lilly",
                                 config=str(SPEAK_DIR / "config.json"),
                                 model=str(SPEAK_DIR / "model.pth")).eval()
            _pipeline = KPipeline(lang_code="a", repo_id="lilly",  # American English
                                  model=voice_model)
    return _pipeline


def speak_to_file(text: str, out_path: str, voice: str = str(VOICE)) -> str:
    import numpy as np
    import soundfile as sf
    text = text.strip()[:MAX_CHARS]
    if not text:
        raise ValueError("nothing to say")
    with _speak_lock:
        chunks = [audio for _, _, audio in get_pipeline()(text, voice=voice)]
    sf.write(out_path, np.concatenate(chunks), 24000)
    return out_path


def main() -> int:
    if len(sys.argv) < 3:
        print('usage: python3 app/tts.py "text" out.wav', file=sys.stderr)
        return 1
    speak_to_file(sys.argv[1], sys.argv[2])
    print(sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
