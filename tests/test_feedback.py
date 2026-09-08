"""The correction store, on a temporary database."""
import pytest

from app import feedback


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "DB_PATH", tmp_path / "feedback.db")
    return feedback.DB_PATH


def test_flatten_keeps_training_rows_on_one_line():
    assert feedback._flatten("a\tb\nc   d") == "a b c d"
    assert feedback._flatten(None) == ""


def test_round_trip_and_export_by_direction(db, tmp_path):
    a = feedback.add_correction("Dobar dan", "Good day", "wrong", "Good afternoon")
    b = feedback.add_correction("Good day", "Dobar\tdan", "", "Dobar dan", direction="en-bs")
    assert [r["id"] for r in feedback.list_corrections()] == [a, b]
    feedback.set_status(a, "approved")
    feedback.set_status(b, "approved")
    out, n = feedback.export_approved(tmp_path / "bs-en.tsv", direction="bs-en")
    assert n == 1 and (tmp_path / "bs-en.tsv").read_text() == "UserCorrections\tDobar dan\tGood afternoon\n"
    out, n = feedback.export_approved(tmp_path / "en-bs.tsv", direction="en-bs")
    assert n == 1 and "Dobar dan" in (tmp_path / "en-bs.tsv").read_text()


def test_bad_direction_and_bad_status_are_refused(db):
    with pytest.raises(ValueError):
        feedback.add_correction("x", "y", direction="fr-en")
    row = feedback.add_correction("x", "y")
    with pytest.raises(ValueError):
        feedback.set_status(row, "maybe")
