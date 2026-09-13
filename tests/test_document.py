"""The document endpoint, without needing translation weights on the machine.

The extraction itself (python-docx / pypdf) is an optional install, so the
tests that need those libraries skip where they are absent; the endpoint's
own behaviour — bounds, suffix check, error mapping — runs anywhere.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_a_file_with_the_wrong_suffix_is_a_400(client):
    # Refused before any reader opens it: the suffix is checked first, so a
    # renamed image is told plainly what Lilly reads, not given a 500.
    res = client.post("/api/document",
                      files={"file": ("notes.txt", b"hello", "text/plain")},
                      data={"direction": "bs-en"})
    assert res.status_code == 400
    assert ".docx" in res.json()["error"]


def test_an_empty_document_upload_is_refused(client):
    res = client.post("/api/document",
                      files={"file": ("a.pdf", b"", "application/pdf")})
    assert res.status_code == 413
    assert "empty" in res.json()["error"]


def test_an_oversized_declared_document_is_refused_before_reading(client):
    res = client.post("/api/document", headers={"content-length": str(26 * 1024 * 1024)},
                      content=b"")
    assert res.status_code == 413


def test_document_direction_is_validated(client):
    res = client.post("/api/document",
                      files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
                      data={"direction": "fr-en"})
    assert res.status_code == 422


def test_a_readable_document_travels_the_translation_path(client, tmp_path, monkeypatch):
    """Extraction is the only part that touches the file; the translation is
    monkeypatched, so this proves the route without loading a model."""
    from app.document import extract_text
    monkeypatch.setattr("app.document.extract_text", lambda path: "Good morning, how are you?")

    from app.lilly import lilly
    monkeypatch.setattr(lilly, "translate", lambda text, truncate=False: f"en<{text}>")

    res = client.post("/api/document",
                      files={"file": ("a.docx", b"whatever", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                      data={"direction": "bs-en"})
    assert res.status_code == 200
    body = res.json()
    assert body["original"] == "Good morning, how are you?"
    assert body["bosnian"] == "Good morning, how are you?"
    assert body["english"] == "en<Good morning, how are you?>"


def test_missing_translation_weights_are_a_503_not_a_500(client, tmp_path, monkeypatch):
    from app.document import extract_text
    monkeypatch.setattr("app.document.extract_text", lambda path: "Good morning")

    from app.lilly import lilly
    def missing(*a, **k):
        raise FileNotFoundError("no bs-en translator at nowhere -- run scripts/fetch_models.py")
    monkeypatch.setattr(lilly, "translate", missing)

    res = client.post("/api/document",
                      files={"file": ("a.docx", b"whatever", "application/pdf")},
                      data={"direction": "bs-en"})
    assert res.status_code == 503
    assert "no bs-en translator" in res.json()["error"]


@pytest.mark.parametrize("suffix", [".docx", ".pdf"])
def test_real_extraction_where_the_libraries_are_installed(tmp_path, suffix):
    """Extraction itself, when the optional reader is on this machine."""
    docx = pytest.importorskip("docx")
    pypdf = pytest.importorskip("pypdf")

    from app.document import extract_text

    path = tmp_path / f"sample{suffix}"
    if suffix == ".docx":
        document = docx.Document()
        document.add_paragraph("Dobar dan")
        document.add_paragraph("Kako ste?")
        document.save(path)
    else:
        # A minimal one-page PDF written by hand is fiddly; instead write a
        # real one through pypdf's writer from a blank page we generate with
        # reportlab only if present — skip cleanly otherwise.
        reportlab = pytest.importorskip("reportlab")
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(path))
        c.drawString(72, 720, "Dobar dan, kako ste?")
        c.save()

    text = extract_text(str(path))
    assert "Dobar dan" in text


def test_extraction_bounds_are_respected(monkeypatch):
    """The cap is applied before the text reaches the translator."""
    from app import document as document_module
    monkeypatch.setattr(document_module, "MAX_DOC_CHARS", 10)
    parts, size, add = document_module._bounded_text()
    add("123456789012345")
    assert document_module._finish(parts, size) == "1234567890"
    parts, size, add = document_module._bounded_text()
    with pytest.raises(document_module.UnreadableDocument):
        document_module._finish(parts, size)  # nothing extracted is refused
