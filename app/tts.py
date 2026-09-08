#!/usr/bin/env python3
"""Voice output: text -> spoken audio, in English or in Bosnian.

English goes through Kokoro, read from models/lilly/speak/: a 24 kHz WAV, about
real time on CPU. Bosnian goes through a Piper voice read from
models/lilly/speak-bs/: a 22.05 kHz WAV, many times faster than real time.
Kokoro has no South Slavic language, so the Bosnian side is a different engine,
not a different voice file for the same one.

The Bosnian voice is Piper's Serbian `sr_RS-serbski_institut-medium`, and that
is its honest name: Piper has no Bosnian voice, and Serbian Latin is written
with the same letters and spoken with the same sounds. Its phonemizer is
espeak-ng's `sr`, which reads c, c and d with their diacritics correctly and
spells numbers out -- in the Serbian, ekavian form ("dve hiljade" where a
Bosnian says "dvije hiljade"). That is the one place the voice is audibly not
Bosnian; a limit to know about, not a defect to hide. scripts/fetch_speak_bs.py
fetches it and says the same.

Usage:
    python3 app/tts.py "Hello, how are you?" out.wav
    python3 app/tts.py --language bs "Dobar dan, kako ste?" out.wav
"""
import json
import sys
import threading
from pathlib import Path

if __name__ == "__main__":
    # Run as a script, sys.path[0] is app/ and `from app.lilly import ...`
    # cannot resolve -- the usage above never worked that way. The repository
    # goes first, as app/lilly.py does.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.lilly import SPEAK_BS_DIR, SPEAK_DIR, BadInput  # noqa: E402

VOICE = SPEAK_DIR / "voices" / "default.pt"
# Fixed names, like voices/default.pt above: the fetcher renames whatever voice
# it pulled to these, so nothing here has to know which Piper voice it is.
# built.json beside them says which one it was and which of its speakers to use.
VOICE_BS = SPEAK_BS_DIR / "voice.onnx"
VOICE_BS_CONFIG = SPEAK_BS_DIR / "voice.onnx.json"
LANGUAGES = ("en", "bs")
# Speech runs at roughly the speed it plays, so the length of the text is the
# length of the CPU burn. Past this the audio is cut rather than refused.
MAX_CHARS = 1200

_pipeline = None
_pipeline_lock = threading.Lock()
_bosnian = None
_bosnian_lock = threading.Lock()
# One voice speaks at a time, whichever language: both are CPU-bound and the
# server hands requests to a wide threadpool.
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


def bosnian_speaker() -> int:
    """Which of the voice's speakers to use: built.json's choice, else the first."""
    built = SPEAK_BS_DIR / "built.json"
    if not built.is_file():
        return 0
    try:
        return int(json.loads(built.read_text(encoding="utf-8")).get("speaker", 0))
    except (ValueError, TypeError):
        return 0


def get_bosnian_voice():
    """The Bosnian voice, loaded once. Absent weights are a missing download,
    reported like the reply direction's: FileNotFoundError, never a 500."""
    global _bosnian
    if _bosnian is None:
        with _bosnian_lock:
            if _bosnian is None:
                if not (VOICE_BS.is_file() and VOICE_BS_CONFIG.is_file()):
                    raise FileNotFoundError(
                        f"no Bosnian voice at {SPEAK_BS_DIR} -- run scripts/fetch_models.py")
                from piper import PiperVoice
                _bosnian = PiperVoice.load(VOICE_BS, VOICE_BS_CONFIG)
    return _bosnian


def speak_to_file(text: str, out_path: str, language: str = "en", voice: str = None) -> str:
    """Write `text` spoken in `language` ("en" or "bs") to `out_path` as WAV."""
    if language not in LANGUAGES:
        raise BadInput(f"Lilly has no {language!r} voice; en or bs")
    text = text.strip()[:MAX_CHARS]
    if not text:
        # Their side of the line: a body of spaces passes the server's
        # min_length, and a plain ValueError here was answered as a 500.
        raise BadInput("nothing to say")
    if language == "bs":
        return _speak_bosnian(text, out_path)
    import numpy as np
    import soundfile as sf
    with _speak_lock:
        chunks = [audio for _, _, audio in get_pipeline()(text, voice=voice or str(VOICE))]
    sf.write(out_path, np.concatenate(chunks), 24000)
    return out_path


def _speak_bosnian(text: str, out_path: str) -> str:
    import wave
    from piper import SynthesisConfig
    engine = get_bosnian_voice()
    config = SynthesisConfig(speaker_id=bosnian_speaker())
    with _speak_lock, wave.open(out_path, "wb") as wav:
        engine.synthesize_wav(text, wav, syn_config=config)
    return out_path


def main() -> int:
    argv = sys.argv[1:]
    language = "en"
    if argv[:1] == ["--language"]:
        language, argv = argv[1], argv[2:]
    if len(argv) < 2:
        print('usage: python3 app/tts.py [--language en|bs] "text" out.wav', file=sys.stderr)
        return 1
    speak_to_file(argv[0], argv[1], language=language)
    print(argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
