"""The photo-boxes and conversation endpoints, with the model work patched out.

The reader and the listener are real weights that may or may not be on the
machine running the tests; the endpoints' own shape — bounds, direction
checks, the response keys the page draws with — needs none of them.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=False)


REGIONS = [{"box": [[10, 10], [200, 10], [200, 40], [10, 40]],
            "source": "ZABRANJEN ULAZ", "translated": "NO ENTRY"}]


def test_photo_boxes_returns_regions_beside_the_pair(client, monkeypatch):
    from app.lilly import lilly
    monkeypatch.setattr(lilly, "translate_photo_regions",
                        lambda path, direction: ("ZABRANJEN ULAZ", "NO ENTRY", REGIONS))
    res = client.post("/api/photo-boxes",
                      files={"file": ("a.jpg", b"patched", "image/jpeg")},
                      data={"direction": "bs-en"})
    assert res.status_code == 200
    body = res.json()
    assert body["bosnian"] == "ZABRANJEN ULAZ"
    assert body["english"] == "NO ENTRY"
    assert body["regions"] == REGIONS


def test_photo_boxes_is_bounded_like_photo(client):
    assert client.post("/api/photo-boxes",
                       files={"file": ("a.jpg", b"", "image/jpeg")}).status_code == 413
    res = client.post("/api/photo-boxes", headers={"content-length": str(13 * 1024 * 1024)},
                      content=b"")
    assert res.status_code == 413
    res = client.post("/api/photo-boxes",
                      files={"file": ("a.jpg", b"x", "image/jpeg")},
                      data={"direction": "fr-en"})
    assert res.status_code == 422


def test_photo_boxes_reports_missing_weights_as_a_503(client, monkeypatch):
    from app.lilly import lilly
    def missing(*a, **k):
        raise FileNotFoundError("no reader on this machine -- run scripts/fetch_models.py")
    monkeypatch.setattr(lilly, "translate_photo_regions", missing)
    res = client.post("/api/photo-boxes",
                      files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert res.status_code == 503
    assert "no reader" in res.json()["error"]


def test_speech_auto_direction_returns_which_side_was_heard(client, monkeypatch):
    from app.lilly import lilly
    monkeypatch.setattr(lilly, "converse",
                        lambda path: ("bs<Good day>", "Good day", "en"))
    res = client.post("/api/speech",
                      files={"file": ("a.webm", b"RIFF....", "audio/webm")},
                      data={"direction": "auto"})
    assert res.status_code == 200
    assert res.json() == {"bosnian": "bs<Good day>", "english": "Good day", "heard": "en"}


def test_speech_auto_still_refuses_a_bad_direction(client):
    res = client.post("/api/speech",
                      files={"file": ("a.webm", b"RIFF....", "audio/webm")},
                      data={"direction": "de-en"})
    assert res.status_code == 422


def test_scan_regions_boxes_are_in_original_pixels(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image

    from app import ocr

    # A 3 MP photo forces the reader's working resize; the box must come back
    # in the original's pixels, not the shrunk copy's.
    big = tmp_path / "big.png"
    Image.new("RGB", (3000, 1000), "white").save(big)
    captured = {}
    def fake_read_regions(image, **kwargs):
        captured["shape"] = np.asarray(image).shape[:2]
        return [[[[10, 10], [50, 10], [50, 30], [10, 30]], "ZABRANJEN ULAZ", 0.95]]
    monkeypatch.setattr(ocr, "read_regions", fake_read_regions)

    text, regions = ocr.scan_regions(str(big))
    assert captured["shape"] != (3000, 1000)          # the reader saw a shrunk copy
    assert text == "ZABRANJEN ULAZ"
    (x0, y0), (x1, _), *_ = regions[0][0]
    assert (x1 - x0) >= (50 - 10)                     # scaled back up, not left shrunk

    # Under the working limit nothing is resized, and the box passes through.
    small = tmp_path / "small.png"
    Image.new("RGB", (400, 300), "white").save(small)
    text, regions = ocr.scan_regions(str(small))
    assert regions[0][0] == [[10, 10], [50, 10], [50, 30], [10, 30]]
