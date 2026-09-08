from app.lilly import BadInput
from app import tts

import pytest


def test_blank_text_is_the_callers_problem(tmp_path):
    with pytest.raises(BadInput):
        tts.speak_to_file("   ", str(tmp_path / "out.wav"))


def test_a_language_lilly_cannot_speak_is_refused_before_any_model(tmp_path):
    with pytest.raises(BadInput):
        tts.speak_to_file("Dobar dan", str(tmp_path / "out.wav"), language="fr")


def test_a_missing_bosnian_voice_is_our_fault_not_the_callers(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "VOICE_BS", tmp_path / "no-voice" / "voice.onnx")
    monkeypatch.setattr(tts, "VOICE_BS_CONFIG", tmp_path / "no-voice" / "voice.onnx.json")
    monkeypatch.setattr(tts, "_bosnian", None)
    with pytest.raises(FileNotFoundError, match="no Bosnian voice"):
        tts.speak_to_file("Dobar dan", str(tmp_path / "out.wav"), language="bs")


def test_the_speaker_comes_from_built_json_and_defaults_to_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "SPEAK_BS_DIR", tmp_path)
    assert tts.bosnian_speaker() == 0
    (tmp_path / "built.json").write_text('{"speaker": 1}', encoding="utf-8")
    assert tts.bosnian_speaker() == 1
    (tmp_path / "built.json").write_text('{"speaker": "two"}', encoding="utf-8")
    assert tts.bosnian_speaker() == 0
