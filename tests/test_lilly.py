"""Both directions through the one object, with no model on the machine.

The pair every convenience method returns is (bosnian, english) whichever way
it ran; a caller reading `english` never has to know how it was produced. That
is the contract the server and the page are written against, and the one a
swapped tuple would break silently -- the page would show the input as the
answer and read it aloud in the wrong voice.
"""
import pytest

from app.lilly import BadInput, Lilly


class Fake(Lilly):
    """Every part answers with a note of what it was asked, and loads nothing."""

    def __init__(self, hears="", sees=""):
        self.hears, self.sees, self.calls = hears, sees, []

    def listen(self, audio_path, language="bs"):
        self.calls.append(("listen", language))
        return self.hears

    def read(self, image_path):
        self.calls.append(("read",))
        return self.sees

    def translate(self, bosnian, truncate=False):
        self.calls.append(("translate", bosnian, truncate))
        return f"en<{bosnian}>"

    def reply(self, english, truncate=False):
        self.calls.append(("reply", english, truncate))
        return f"bs<{english}>"


def test_forward_audio_hears_bosnian_and_answers_english():
    f = Fake(hears="Dobar dan")
    assert f.translate_audio("clip.wav") == ("Dobar dan", "en<Dobar dan>")
    assert f.calls == [("listen", "bs"), ("translate", "Dobar dan", True)]


def test_reverse_audio_hears_english_and_answers_bosnian_in_the_same_order():
    f = Fake(hears="Good day")
    assert f.translate_audio("clip.wav", direction="en-bs") == ("bs<Good day>", "Good day")
    assert f.calls == [("listen", "en"), ("reply", "Good day", True)]


def test_reverse_photo_reads_then_replies():
    f = Fake(sees="NO ENTRY")
    assert f.translate_photo("sign.jpg", direction="en-bs") == ("bs<NO ENTRY>", "NO ENTRY")
    assert f.calls == [("read",), ("reply", "NO ENTRY", True)]


def test_forward_photo_is_unchanged():
    f = Fake(sees="ZABRANJEN ULAZ")
    assert f.translate_photo("sign.jpg") == ("ZABRANJEN ULAZ", "en<ZABRANJEN ULAZ>")


@pytest.mark.parametrize("direction", ["bs-en", "en-bs"])
def test_nothing_heard_or_seen_translates_nothing(direction):
    f = Fake()
    assert f.translate_audio("clip.wav", direction=direction) == ("", "")
    assert f.translate_photo("sign.jpg", direction=direction) == ("", "")
    assert not [c for c in f.calls if c[0] in ("translate", "reply")]


def test_an_unknown_direction_is_the_callers_fault():
    with pytest.raises(BadInput):
        Fake(hears="x").translate_audio("clip.wav", direction="fr-en")
    with pytest.raises(BadInput):
        Fake(sees="x").translate_photo("sign.jpg", direction="bs-fr")


def test_converse_hears_either_language_and_says_which(monkeypatch):
    """Conversation mode: the listener is not told a language, the detector
    routes the text, and the third value names the side that was heard."""
    from app import detect
    monkeypatch.setattr(detect, "detect_language",
                        lambda text: "en" if text == "Good day" else "bs")

    heard_en = Fake(hears="Good day")
    assert heard_en.converse("clip.wav") == ("bs<Good day>", "Good day", "en")
    assert heard_en.calls == [("listen", None), ("reply", "Good day", True)]

    heard_bs = Fake(hears="Dobar dan")
    assert heard_bs.converse("clip.wav") == ("Dobar dan", "en<Dobar dan>", "bs")
    assert heard_bs.calls == [("listen", None), ("translate", "Dobar dan", True)]


def test_converse_hears_nothing_and_translates_nothing(monkeypatch):
    from app import detect
    monkeypatch.setattr(detect, "detect_language", lambda text: "bs")
    f = Fake(hears="")
    assert f.converse("clip.wav") == ("", "", "")
    assert not [c for c in f.calls if c[0] in ("translate", "reply")]
