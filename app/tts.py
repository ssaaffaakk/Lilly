#!/usr/bin/env python3
"""Voice output: English text -> spoken audio.

Produces a 24 kHz WAV. Runs real-time on CPU.

Usage:
    python3 app/tts.py "Hello, how are you?" out.wav
"""
import sys

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code="a")  # American English
    return _pipeline


def speak_to_file(text: str, out_path: str, voice: str = "af_heart") -> str:
    import numpy as np
    import soundfile as sf
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
