"""The server's error mapping, with no model on the machine.

Every failure below is answered before a model is needed, or by the absence
of one: that is what makes them testable here, and what the page's messages
are written against (app/web/index.html, "what went wrong").
"""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


@pytest.mark.parametrize("path", ["/api/translate", "/api/reply", "/api/speak"])
def test_blank_text_is_a_422_not_a_500(client, path):
    assert client.post(path, json={"text": "   "}).status_code == 422
    assert client.post(path, json={"text": ""}).status_code == 422


def test_too_long_text_is_a_422_the_page_can_name(client):
    res = client.post("/api/translate", json={"text": "x" * 12_001})
    assert res.status_code == 422
    assert any("too_long" in d["type"] for d in res.json()["detail"])


def test_missing_listener_is_a_503_not_a_400(client, tmp_path, monkeypatch):
    from app import speech
    monkeypatch.setattr(speech, "LISTEN_DIR", tmp_path / "no-listener")
    res = client.post("/api/speech", files={"file": ("a.webm", b"RIFF....", "audio/webm")})
    assert res.status_code == 503
    assert "no listener" in res.json()["error"]


def test_empty_upload_is_refused(client):
    res = client.post("/api/speech", files={"file": ("a.webm", b"", "audio/webm")})
    assert res.status_code == 413
    assert "empty" in res.json()["error"]


def test_oversized_declared_upload_is_refused_before_reading(client):
    res = client.post("/api/photo", headers={"content-length": str(13 * 1024 * 1024)},
                      content=b"")
    assert res.status_code == 413


def test_a_file_that_is_not_an_image_is_a_400(client):
    pytest.importorskip("PIL")
    res = client.post("/api/photo", files={"file": ("a.jpg", b"not an image", "image/jpeg")})
    assert res.status_code == 400
    assert "not an image" in res.json()["error"]


def test_feedback_direction_is_validated(client, tmp_path, monkeypatch):
    from app import feedback
    monkeypatch.setattr(feedback, "DB_PATH", tmp_path / "feedback.db")
    assert client.post("/api/feedback", json={"source_text": "x", "direction": "fr-en"}).status_code == 422
    res = client.post("/api/feedback", json={"source_text": "Dobar dan", "model_output": "Good day",
                                             "suggested_translation": "Hello", "direction": "bs-en"})
    assert res.status_code == 200 and res.json()["ok"] is True


def test_speak_language_is_validated(client):
    assert client.post("/api/speak", json={"text": "Dobar dan", "language": "fr"}).status_code == 422


def test_missing_bosnian_voice_is_a_503_not_a_500(client, tmp_path, monkeypatch):
    from app import tts
    monkeypatch.setattr(tts, "VOICE_BS", tmp_path / "voice.onnx")
    monkeypatch.setattr(tts, "VOICE_BS_CONFIG", tmp_path / "voice.onnx.json")
    monkeypatch.setattr(tts, "_bosnian", None)
    res = client.post("/api/speak", json={"text": "Dobar dan", "language": "bs"})
    assert res.status_code == 503
    assert "no Bosnian voice" in res.json()["error"]


@pytest.mark.parametrize("path,name,mime", [("/api/speech", "a.webm", "audio/webm"),
                                            ("/api/photo", "a.jpg", "image/jpeg")])
def test_upload_direction_is_validated(client, path, name, mime):
    res = client.post(path, files={"file": (name, b"RIFF....", mime)}, data={"direction": "fr-en"})
    assert res.status_code == 422


def test_missing_listener_is_a_503_in_the_reply_direction_too(client, tmp_path, monkeypatch):
    from app import speech
    monkeypatch.setattr(speech, "LISTEN_DIR", tmp_path / "no-listener")
    res = client.post("/api/speech", files={"file": ("a.webm", b"RIFF....", "audio/webm")},
                      data={"direction": "en-bs"})
    assert res.status_code == 503
    assert "no listener" in res.json()["error"]


def test_the_page_is_revalidated_not_kept_on_heuristics(client):
    assert client.get("/").headers["cache-control"] == "no-cache"
