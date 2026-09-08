"""app.ocr's environment parsing, which decides what a cache is stamped with."""
import pytest

from app import ocr


def test_default_reader_is_paddle_at_the_shipped_floor(monkeypatch):
    monkeypatch.delenv("LILLY_READER", raising=False)
    monkeypatch.delenv("LILLY_PADDLE_REC_THRESH", raising=False)
    assert ocr.reader_choice() == "paddle"
    assert ocr.paddle_rec_floor() == ocr.DEFAULT_REC_FLOOR == 0.9


def test_lilly_is_an_alias_for_easyocr(monkeypatch):
    monkeypatch.setenv("LILLY_READER", "lilly")
    assert ocr.reader_choice() == "easyocr"


def test_unknown_reader_stops_the_process(monkeypatch):
    monkeypatch.setenv("LILLY_READER", "tesseract")
    with pytest.raises(RuntimeError):
        ocr.reader_choice()


def test_floor_zero_means_no_floor_and_rescue_needs_one(monkeypatch):
    monkeypatch.setenv("LILLY_READER", "paddle")
    monkeypatch.setenv("LILLY_PADDLE_REC_THRESH", "0")
    assert ocr.paddle_rec_floor() is None
    monkeypatch.setenv("LILLY_PADDLE_CYRILLIC_RESCUE", "1")
    with pytest.raises(RuntimeError):
        ocr.paddle_cyrillic_rescue()
    monkeypatch.setenv("LILLY_PADDLE_REC_THRESH", "1.5")
    with pytest.raises(RuntimeError):
        ocr.paddle_rec_floor()
