"""Attached Commons photos are identified by SHA-1, not by the zip entry name."""
import hashlib
from pathlib import Path

from PIL import Image

from training.fetch_pinned_ocr_photos import fetch_bytes, index_local_source


def _tiny_jpeg(path: Path) -> bytes:
    Image.new("RGB", (8, 8), (20, 40, 60)).save(path, format="JPEG")
    return path.read_bytes()


def test_local_source_finds_photo_when_zip_renames_unicode(tmp_path: Path):
    original = tmp_path / "Međugorje_Banner.jpg"
    data = _tiny_jpeg(original)
    digest = hashlib.sha1(data).hexdigest()
    attached = tmp_path / "kaggle-zip"
    attached.mkdir()
    # What v1 of the Kaggle zip actually stored: ASCII-mangled name, same bytes.
    (attached / "Medugorje_Banner.jpg").write_bytes(data)
    row = {"file": "Međugorje_Banner.jpg", "commons_sha1": digest}
    got = fetch_bytes(row, attached, index_local_source(attached))
    assert got == data


def test_local_source_prefers_sha1_copy(tmp_path: Path):
    data = _tiny_jpeg(tmp_path / "a.jpg")
    digest = hashlib.sha1(data).hexdigest()
    attached = tmp_path / "ds"
    (attached / "by-sha1").mkdir(parents=True)
    (attached / "by-sha1" / digest).write_bytes(data)
    row = {"file": "Trg-žrtava-ŠB03078.JPG", "commons_sha1": digest}
    assert fetch_bytes(row, attached) == data


def test_local_source_still_fail_stops_when_bytes_are_missing(tmp_path: Path):
    attached = tmp_path / "empty"
    attached.mkdir()
    row = {"file": "Međugorje_Banner.jpg", "commons_sha1": "0" * 40}
    try:
        fetch_bytes(row, attached)
    except SystemExit as exc:
        assert "not in attached photo dataset" in str(exc)
    else:
        raise AssertionError("missing photo must fail-stop")
