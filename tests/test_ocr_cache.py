"""The OCR cache must bind readings to the thing that actually produced them."""
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("LILLY_IGNORE_GUARD", "1")

from training import evaluate_ocr as evaluated


def _context(runtime="cv2-4.10", treatment="shipped"):
    return {
        "schema": evaluated.CACHE_SCHEMA,
        "reader_identity": "paddle:test:rec>=0.9",
        "runtime": {"cv2": runtime},
        "weights_sha256": {"recogniser/model": "a" * 64},
        "implementation_sha256": {"app/ocr.py": "b" * 64},
        "treatment": treatment,
    }


def _write_v2(path: Path, context: dict, readings: dict):
    path.write_text(json.dumps({
        "schema": evaluated.CACHE_SCHEMA,
        "reader": evaluated.context_fingerprint(context),
        "context": context,
        "readings": readings,
    }), encoding="utf-8")


def test_content_hash_ignores_mtime_but_not_bytes(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"same model")
    before = evaluated.file_sha256(model)

    os.utime(model, (1, 1))
    assert evaluated.file_sha256(model) == before

    model.write_bytes(b"other model")
    assert evaluated.file_sha256(model) != before


def test_context_fingerprint_binds_runtime_and_treatment():
    shipped = evaluated.context_fingerprint(_context())
    assert evaluated.context_fingerprint(_context()) == shipped
    assert evaluated.context_fingerprint(_context(runtime="cv2-5.0")) != shipped
    assert evaluated.context_fingerprint(_context(treatment="native")) != shipped


def test_paddle_manifest_ignores_easyocr_files_and_mtime(tmp_path, monkeypatch):
    cache_home = tmp_path / "paddlex"
    for model in ("det", "rec"):
        directory = cache_home / "official_models" / model
        directory.mkdir(parents=True)
        for filename in ("inference.pdiparams", "inference.yml", "inference.json"):
            (directory / filename).write_text(f"{model}:{filename}", encoding="utf-8")

    irrelevant = tmp_path / "read" / "lilly.pth"
    irrelevant.parent.mkdir()
    irrelevant.write_bytes(b"unused by paddle")

    from app import ocr
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(cache_home))
    monkeypatch.setattr(ocr, "paddle_models", lambda: ("det", "rec"))
    monkeypatch.setattr(ocr, "paddle_rec_dir", lambda: None)
    monkeypatch.setattr(ocr, "paddle_cyrillic_rescue", lambda: False)
    monkeypatch.setattr(ocr, "READ_DIR", irrelevant.parent)

    first = evaluated._weight_manifest("paddle", "paddle:test")
    os.utime(cache_home / "official_models" / "det" / "inference.pdiparams", (2, 2))
    irrelevant.write_bytes(b"changed but still unused")
    assert evaluated._weight_manifest("paddle", "paddle:test") == first

    (cache_home / "official_models" / "det" / "inference.pdiparams").write_bytes(b"changed")
    assert evaluated._weight_manifest("paddle", "paddle:test") != first


def test_legacy_cache_stops_instead_of_relaunching(tmp_path, monkeypatch):
    cache = tmp_path / "legacy.json"
    cache.write_text(json.dumps({
        "reader": "old",
        "readings": {"photo.jpg": "old text"},
    }), encoding="utf-8")
    monkeypatch.setattr(evaluated, "reader_cache_context", lambda full=False: _context())

    with pytest.raises(SystemExit, match="legacy OCR cache"):
        evaluated.cached_reads(["photo.jpg"], tmp_path, cache)


def test_same_filename_with_new_bytes_is_refused(tmp_path, monkeypatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"first image")
    context = _context()
    cache = tmp_path / "cache.json"
    _write_v2(cache, context, {
        photo.name: {
            "source_sha256": evaluated.file_sha256(photo),
            "text": "FIRST",
        }
    })
    photo.write_bytes(b"replacement image")
    monkeypatch.setattr(evaluated, "reader_cache_context", lambda full=False: context)

    with pytest.raises(SystemExit, match="changed under the same filename"):
        evaluated.cached_reads([photo.name], tmp_path, cache)


def test_context_mismatch_stops_instead_of_relaunching(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    _write_v2(cache, _context(runtime="cv2-4.10"), {
        "one.jpg": {"source_sha256": "c" * 64, "text": "ONE"},
    })
    changed = _context(runtime="cv2-5.0")
    monkeypatch.setattr(evaluated, "reader_cache_context", lambda full=False: changed)

    with pytest.raises(SystemExit, match="Refusing to relabel"):
        evaluated.cached_reads(["one.jpg"], tmp_path, cache, sealed=True)


def test_sealed_cache_never_fills_a_missing_row(tmp_path, monkeypatch):
    context = _context()
    cache = tmp_path / "cache.json"
    _write_v2(cache, context, {
        "one.jpg": {"source_sha256": "c" * 64, "text": "ONE"},
    })
    monkeypatch.setattr(evaluated, "reader_cache_context", lambda full=False: context)

    with pytest.raises(SystemExit, match="missing 1/2"):
        evaluated.cached_reads(
            ["one.jpg", "two.jpg"], tmp_path, cache, sealed=True
        )


def test_sealed_complete_cache_does_not_need_source_files(tmp_path, monkeypatch):
    context = _context()
    cache = tmp_path / "cache.json"
    _write_v2(cache, context, {
        "one.jpg": {"source_sha256": "c" * 64, "text": "ONE"},
    })
    monkeypatch.setattr(evaluated, "reader_cache_context", lambda full=False: context)

    assert evaluated.cached_reads(
        ["one.jpg"], tmp_path / "absent", cache, sealed=True
    ) == {"one.jpg": "ONE"}
