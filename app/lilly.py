#!/usr/bin/env python3
"""Lilly — one object, five abilities, both directions, one model folder.

    from app.lilly import lilly

    lilly.translate("Dobar dan, kako ste?")   # Bosnian text  -> English text
    lilly.listen("clip.m4a")                  # spoken Bosnian -> Bosnian text
    lilly.speak("Good day", "out.wav")        # English text  -> spoken English
    lilly.read("sign.jpg")                    # photo          -> Bosnian text
    lilly.reply("Good morning")               # English text  -> Bosnian text

    # and the other way round, for answering back
    lilly.listen("clip.m4a", language="en")               # spoken English -> English text
    lilly.speak("Dobar dan", "out.wav", language="bs")    # Bosnian text  -> spoken Bosnian
    lilly.translate_audio("clip.m4a", direction="en-bs")  # spoken English -> (Bosnian, English)
    lilly.translate_photo("sign.jpg", direction="en-bs")  # English photo  -> (Bosnian, English)

Since the evening of 8 September 2026 every ability runs in both directions.
The listener and the reader are the same weights either way -- Whisper is told
which language to hear, and PP-OCRv6 reads Latin script whatever the language
-- so the reverse direction adds exactly one new part: a Bosnian voice, under
speak-bs/, optional in the same way translator-en-bs/ is (below).

Every weight Lilly needs lives under models/lilly/ and is read straight off
this disk — nothing is fetched over the network. Each ability loads the first
time it is used, so starting Lilly costs nothing and unused parts stay unread.

reply() is the one ability that can be absent on a working install. The
published bundle has carried its weights (translator-en-bs/) since 8 September
2026 and fetch_models.py pulls them, but an install fetched before that, or one
built from the upstream base by hand (see OPTIONAL below), may not have them.
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
SPEAK_DIR = MODELS / "speak"           # English text -> speech (Kokoro)
SPEAK_BS_DIR = MODELS / "speak-bs"     # Bosnian text -> speech (Piper), the reply side's voice
READ_DIR = MODELS / "read"             # photo -> text
ADAPTER_DIR = MODELS / "adapter"       # our fine-tuned adapter, once trained


class BadInput(ValueError):
    """What the caller sent cannot be used — their side of the line, not ours.

    Anything raised from this is answered with a plain sentence and a 4xx,
    never a stack trace and never a 500: a wrong answer about whose fault it
    is sends people looking in the wrong place.
    """


# What a working install must have, and what it may not. The reply direction is
# in the second group because the app worked without it for two weeks and older
# fetches do not have it: the bundle carries it since 8 September 2026 and
# fetch_models.py pulls it, but a four-ability install is still a good install
# and must not exit 1.
REQUIRED = (TRANSLATOR_DIR, LISTEN_DIR, SPEAK_DIR, READ_DIR)
OPTIONAL = {
    TRANSLATOR_EN_BS_DIR: (
        "the reply direction (English -> Bosnian). Fetch it with:\n"
        "    python3 scripts/fetch_models.py\n"
        "or build it from the upstream base:\n"
        "    python3 scripts/fetch_translate_base.py --direction en-bs\n"
        "    python3 scripts/build_translator.py --direction en-bs"),
    SPEAK_BS_DIR: (
        "the Bosnian voice (spoken replies). Not in the bundle: it is Piper's "
        "public sr_RS voice, fetched from upstream by\n"
        "    python3 scripts/fetch_models.py\n"
        "or on its own:\n"
        "    python3 scripts/fetch_speak_bs.py"),
}

# The two ways round. "bs-en" reads Bosnian and writes English; "en-bs" is the
# reply: English in, Bosnian out. Whatever the direction, the pair a method
# returns is (bosnian, english) -- which side was typed, said or photographed
# is the difference, and a caller reading `english` should not have to know
# how it was produced. app/server.py and app/feedback.py use the same names.
DIRECTIONS = ("bs-en", "en-bs")


def check_direction(direction: str) -> str:
    if direction not in DIRECTIONS:
        raise BadInput(f"direction must be one of {DIRECTIONS}, not {direction!r}")
    return direction


def required_parts() -> tuple:
    """What this install has to have, given which reader is configured.

    The app reads photographs with PaddleOCR by default, and PaddleX keeps
    those weights in its own cache rather than under models/lilly/, so read/
    -- the EasyOCR way back -- is required only when LILLY_READER names an
    EasyOCR engine. REQUIRED stays the full list: it is what a complete bundle
    holds, and scripts/publish_to_hf.py publishes all four.
    """
    from app.ocr import reader_choice
    return tuple(d for d in REQUIRED if d is not READ_DIR or reader_choice() != "paddle")


def missing() -> list:
    """Which required parts of the model folder are not on this machine."""
    return [d.name for d in required_parts() if not d.is_dir()]


def describe_listener() -> str:
    """Which listener this is, against the one that cleared its gate.

    The bundle has carried an ungated whisper-large-v3 under listen/ since
    the 4-5 September 2026 publish, and nothing a user runs said so. This
    does, in the same words scripts/fetch_models.py uses at download time.
    """
    if not (LISTEN_DIR / "model.bin").is_file():
        return "listen: not installed"
    try:
        from scripts.fetch_models import (GATED_LISTEN_FINGERPRINT, listen_base,
                                          listen_fingerprint)
    except ImportError:
        return "listen: gate status unknown (scripts/fetch_models.py is not beside app/)"
    actual = listen_fingerprint(LISTEN_DIR)
    base = listen_base(LISTEN_DIR) or "unknown base"
    if actual == GATED_LISTEN_FINGERPRINT:
        return f"listen: {base}, fingerprint {actual} -- the listener that cleared its gate"
    return (f"listen: {base}, fingerprint {actual} -- NOT the gated listener "
            f"{GATED_LISTEN_FINGERPRINT}; refused at its gate, training/RESULTS-speech.md")


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
        """Speech in `language` ("bs" or "en") -> text in that language.

        One listener hears both: Whisper is multilingual and is told which
        language to expect, so the reply direction costs no second model.
        """
        from app.speech import transcribe
        return transcribe(audio_path, language=language)

    def speak(self, text: str, out_path: str, language: str = "en") -> str:
        """Text in `language` ("en" or "bs") -> WAV at out_path.

        Two engines behind one door: Kokoro for English, Piper for Bosnian
        (app/tts.py says why). A missing Bosnian voice is a missing download,
        reported the way a missing reply model is.
        """
        from app.tts import speak_to_file
        return speak_to_file(text, out_path, language=language)

    def read(self, image_path: str) -> str:
        from app.ocr import scan
        return scan(image_path)

    # convenience: the two things the app actually does with the other parts.
    # truncate=True on both: the reader never typed this text, so a refusal
    # over its length would be baffling. Better a translated beginning.
    # Both return (bosnian, english) whichever way round they ran -- see
    # DIRECTIONS above.
    def translate_audio(self, audio_path: str, direction: str = "bs-en") -> tuple:
        if check_direction(direction) == "en-bs":
            english = self.listen(audio_path, language="en")
            return (self.reply(english, truncate=True) if english else ""), english
        bosnian = self.listen(audio_path)
        return bosnian, (self.translate(bosnian, truncate=True) if bosnian else "")

    def translate_photo(self, image_path: str, direction: str = "bs-en") -> tuple:
        if check_direction(direction) == "en-bs":
            english = self.read(image_path)
            return (self.reply(english, truncate=True) if english else ""), english
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
    for part in [d.name for d in REQUIRED if d.is_dir()] + [d.name for d in OPTIONAL if d.is_dir()]:
        size = sum(f.stat().st_size for f in (MODELS / part).rglob("*") if f.is_file())
        print(f"  {part:<18} {size / 1048576:>6.0f} MB")
    for name, why in missing_optional().items():
        print(f"  {name:<18} not installed — {why}")
    from app.ocr import reader_identity
    print(f"  reader: {reader_identity()}")
    print(f"  {describe_listener()}")
    if len(sys.argv) > 1:
        print(lilly.translate(" ".join(sys.argv[1:])))
    return 0


if __name__ == "__main__":
    # Run as a script, sys.path[0] is app/ and `from app.ocr import ...` inside
    # the functions above would not resolve. The repository goes first.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
