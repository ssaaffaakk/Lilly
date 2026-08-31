#!/usr/bin/env python3
"""Lilly — one object, four abilities, one model folder.

    from app.lilly import lilly

    lilly.translate("Dobar dan, kako ste?")   # Bosnian text  -> English text
    lilly.listen("clip.m4a")                  # spoken Bosnian -> Bosnian text
    lilly.speak("Good day", "out.wav")        # English text  -> spoken English
    lilly.read("sign.jpg")                    # photo          -> Bosnian text
    lilly.reply("Good morning")               # English text  -> Bosnian text

Every weight Lilly needs lives under models/lilly/ and is read straight off
this disk — nothing is fetched over the network. Each ability loads the first
time it is used, so starting Lilly costs nothing and unused parts stay unread.

reply() is the one ability that can be absent on a working install. Its weights
are not in the published bundle — they are built locally from an upstream base
(see OPTIONAL below) — so a fresh clone has the other four and not this one.
That is a missing download, not a broken install, and the two are reported
differently: missing() is fatal, missing_optional() is a sentence.
"""
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models" / "lilly"
TRANSLATOR_DIR = MODELS / "translator"  # Bosnian -> English, quantised, what we serve
TRANSLATOR_EN_BS_DIR = MODELS / "translator-en-bs"  # English -> Bosnian, the reply side
TRANSLATE_DIR = MODELS / "translate"   # the trainable copy, only training reads it
LISTEN_DIR = MODELS / "listen"         # speech -> text
SPEAK_DIR = MODELS / "speak"           # text -> speech
READ_DIR = MODELS / "read"             # photo -> text
ADAPTER_DIR = MODELS / "adapter"       # our fine-tuned adapter, once trained


class BadInput(ValueError):
    """What the caller sent cannot be used — their side of the line, not ours.

    Anything raised from this is answered with a plain sentence and a 4xx,
    never a stack trace and never a 500: a wrong answer about whose fault it
    is sends people looking in the wrong place.
    """


# What a working install must have, and what it may not. The reply direction is
# in the second group because it is not in the published bundle: fetch_models.py
# cannot bring it, and it is built here from an upstream base instead. Putting it
# in the first group would make a perfectly good four-ability install exit 1.
REQUIRED = (TRANSLATOR_DIR, LISTEN_DIR, SPEAK_DIR, READ_DIR)
OPTIONAL = {
    TRANSLATOR_EN_BS_DIR: (
        "the reply direction (English -> Bosnian). Build it with:\n"
        "    python3 scripts/fetch_translate_base.py --direction en-bs\n"
        "    python3 scripts/build_translator.py --direction en-bs"),
}


def missing() -> list:
    """Which required parts of the model folder are not on this machine."""
    return [d.name for d in REQUIRED if not d.is_dir()]


def missing_optional() -> dict:
    """Absent parts that cost one ability rather than the install.

    Kept apart from missing() on purpose. Without this, an install lacking the
    reply model reports nothing wrong and then answers /api/reply with a 503,
    which sends the reader looking at the server instead of at the download
    they never made.
    """
    return {d.name: why for d, why in OPTIONAL.items() if not d.is_dir()}


class Lilly:
    """Everything Lilly can do, behind one object."""

    def translate(self, bosnian: str, truncate: bool = False) -> str:
        from app.translate import get_engine
        return get_engine().translate(bosnian, truncate=truncate)

    def reply(self, english: str, truncate: bool = False) -> str:
        """English in, Bosnian out — for answering back, not for reading.

        A separate method rather than a flag on translate(): the two directions
        are different weights with different quality, and the caller should have
        to name which one it wants.
        """
        from app.translate import get_engine
        return get_engine("en-bs").translate(english, truncate=truncate)

    def listen(self, audio_path: str, language: str = "bs") -> str:
        from app.speech import transcribe
        return transcribe(audio_path, language=language)

    def speak(self, english: str, out_path: str) -> str:
        from app.tts import speak_to_file
        return speak_to_file(english, out_path)

    def read(self, image_path: str) -> str:
        from app.ocr import scan
        return scan(image_path)

    # convenience: the two things the app actually does with the other parts
    # truncate=True on both: the reader never typed this text, so a refusal
    # over its length would be baffling. Better a translated beginning.
    def translate_audio(self, audio_path: str) -> tuple:
        bosnian = self.listen(audio_path)
        return bosnian, (self.translate(bosnian, truncate=True) if bosnian else "")

    def translate_photo(self, image_path: str) -> tuple:
        bosnian = self.read(image_path)
        return bosnian, (self.translate(bosnian, truncate=True) if bosnian else "")

    @property
    def status(self) -> str:
        from app.translate import get_engine
        return f"{get_engine().name}, device: {get_engine().device}"


lilly = Lilly()


def main() -> int:
    gaps = missing()
    if gaps:
        print(f"missing from {MODELS}: {', '.join(gaps)}", file=sys.stderr)
        return 1
    print(f"model folder: {MODELS}")
    for part in [d.name for d in REQUIRED] + [d.name for d in OPTIONAL if d.is_dir()]:
        size = sum(f.stat().st_size for f in (MODELS / part).rglob("*") if f.is_file())
        print(f"  {part:<18} {size / 1048576:>6.0f} MB")
    for name, why in missing_optional().items():
        print(f"  {name:<18} not installed — {why}")
    if len(sys.argv) > 1:
        print(lilly.translate(" ".join(sys.argv[1:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
