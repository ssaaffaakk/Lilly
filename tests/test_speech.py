"""The listener's failure modes that need no listener."""
from pathlib import Path

import pytest

from app import speech


def test_missing_build_is_our_fault_not_the_callers(tmp_path):
    with pytest.raises(FileNotFoundError):
        speech.get_model(tmp_path / "no-such-listener")


def test_transcribe_does_not_blame_the_audio_for_a_missing_listener(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF....WAVEfmt ")
    with pytest.raises(FileNotFoundError):
        speech.transcribe(str(clip), build=tmp_path / "no-such-listener")


def test_cache_key_resolves_the_path():
    assert speech._key("models/lilly/listen") == str(Path("models/lilly/listen").resolve())
