"""The language detector and its endpoint, with no model weights needed.

The detector is a committed table of character n-gram counts (built by
`python3 app/detect.py --train` on a machine with data/clean/), so these
tests run against the shipped artifact rather than the training corpora,
which are gitignored and absent on a fresh clone.
"""
import pytest

from app.detect import Detector, detect_language, get_detector

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_the_shipped_detector_loads_and_names_both_languages():
    detector = get_detector()  # raises if app/detect-model.json is missing
    assert detector.predict("Dobar dan, kako ste?") == "bs"
    assert detector.predict("Good morning, how are you?") == "en"


@pytest.mark.parametrize("text,language", [
    ("Dobar dan", "bs"),
    ("Hvala lijepa", "bs"),
    ("Molim vas, možete li ponoviti?", "bs"),
    ("Good day", "en"),
    ("Thank you very much", "en"),
    ("Could you repeat that, please?", "en"),
])
def test_short_real_inputs_are_classified(text, language):
    # The strings a visitor actually types, not corpus sentences: short,
    # sometimes diacritic-free ("Dobar dan" has none), the hard cases.
    assert detect_language(text) == language


def test_diacritics_are_decisive():
    # A diacritic that exists in Bosnian and never in English must tip the
    # scale even on a single short word.
    assert detect_language("čaršija") == "bs"
    assert detect_language("đak") == "bs"
    assert detect_language("station") == "en"


def test_the_detector_is_deterministic():
    text = "Sastanak je sutra u devet u kancelariji."
    assert [detect_language(text) for _ in range(5)] == ["bs"] * 5


def test_an_empty_handmade_table_falls_back_to_bosnian():
    # A text with no grams in the vocabulary scores the languages equally;
    # the tie goes to Bosnian, the language the app faces by default.
    table = {"counts": {"bs": {"a": 1}, "en": {"a": 1}}, "totals": {"bs": 1, "en": 1}}
    assert Detector(table).predict("zz") == "bs"


def test_detect_endpoint_answers_both_languages(client):
    assert client.post("/api/detect", json={"text": "Dobar dan"}).json() == {"language": "bs"}
    assert client.post("/api/detect", json={"text": "Good morning"}).json() == {"language": "en"}


def test_detect_endpoint_is_bounded_like_the_other_text_routes(client):
    # Same constraints as /api/translate: blank is empty, and empty is a 422.
    assert client.post("/api/detect", json={"text": "   "}).status_code == 422
    assert client.post("/api/detect", json={"text": ""}).status_code == 422
    assert client.post("/api/detect", json={"text": "x" * 12_001}).status_code == 422
