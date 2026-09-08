"""The Bosnian voice's training and scoring scripts, with no model on the machine.

The speaker rule, the training file and the instrument's arithmetic are what a
Kaggle run trusts blindly for six hours; each is checked here in seconds.
"""
import json
import wave

import pytest

from training import cluster_speakers, evaluate_speak, prepare_speak_data


def _args(**kw):
    base = dict(threshold=0.3, min_minutes=20.0, max_speakers=8, max_collision=0.5)
    base.update(kw)
    return type("A", (), base)()


def _emb(n, dim=4):
    pytest.importorskip("numpy")
    import numpy as np
    return np.eye(dim)[[i % dim for i in range(n)]].astype("float32")


def test_the_speaker_rule_keeps_minutes_not_clips():
    # speaker 0: 3 long clips (30 min); speaker 1: 40 short clips (10 min); speaker 2: 25 min
    rows = [(f"c{i}.wav", f"sentence {i}") for i in range(68)]
    labels = [0] * 3 + [1] * 40 + [2] * 25
    durations = [600.0] * 3 + [15.0] * 40 + [60.0] * 25
    rec = cluster_speakers.build(rows, labels, durations, _emb(68), _args())
    assert rec["selected"] == [0, 2]
    assert rec["n_clusters"] == 3 and rec["speakers"][0]["id"] == 0
    assert rec["collision_rate"] == 0.0


def test_the_speaker_cap_takes_the_largest():
    rows = [(f"c{i}.wav", f"s{i}") for i in range(30)]
    labels = [i % 3 for i in range(30)]
    durations = [200.0 if lab == 1 else 150.0 for lab in labels]
    rec = cluster_speakers.build(rows, labels, durations, _emb(30), _args(max_speakers=1))
    assert rec["selected"] == [1]


def test_same_sentence_pairs_in_one_cluster_are_counted():
    rows = [("a.wav", "isti tekst"), ("b.wav", "isti tekst"), ("c.wav", "isti tekst"), ("d.wav", "drugi")]
    pairs, same = cluster_speakers.collisions(rows, [0, 0, 1, 0])
    assert (pairs, same) == (3, 1)


def test_prepare_keeps_only_selected_speakers_and_guards_the_pipe(tmp_path):
    audio = tmp_path / "train"
    audio.mkdir()
    for i in range(3):
        (audio / f"{i}.wav").write_bytes(b"RIFF")
    rows = [("train/0.wav", "Dobar | dan"), ("train/1.wav", "Hvala"), ("train/2.wav", "Molim")]
    speakers = {"selected": [5], "clips": {"train/0.wav": {"speaker": 5}, "train/1.wav": {"speaker": 6},
                                           "train/2.wav": {"speaker": 5}}}
    kept = prepare_speak_data.build_rows(rows, speakers, tmp_path)
    assert [k[2] for k in kept] == ["Dobar   dan", "Molim"]
    assert all(k[1] == "5" for k in kept)
    manifest = prepare_speak_data.write_csv(tmp_path / "meta.csv", kept)
    lines = (tmp_path / "meta.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[0].count("|") == 2
    assert manifest["speakers"] == {"5": 2}
    assert json.loads((tmp_path / "meta.csv.manifest.json").read_text())["rows"] == 2


def test_prepare_refuses_a_missing_clip(tmp_path):
    rows = [("train/0.wav", "x")]
    speakers = {"selected": [1], "clips": {"train/0.wav": {"speaker": 1}}}
    with pytest.raises(SystemExit):
        prepare_speak_data.build_rows(rows, speakers, tmp_path)


def test_first200_is_the_prefix_and_distinct_is_one_per_sentence():
    rows = [(f"{i}.wav", f"s{i // 3}") for i in range(600)]
    assert len(evaluate_speak.choose(rows, "first200", 0)) == 200
    assert len(evaluate_speak.choose(rows, "distinct", 0)) == 200
    assert len(evaluate_speak.choose(rows, "distinct", 7)) == 7


def test_voice_spec_parses_with_and_without_speaker():
    assert evaluate_speak.parse_voice("before=/x/voice.onnx:3")[1:] == (evaluate_speak.Path("/x/voice.onnx"), 3)
    assert evaluate_speak.parse_voice("cand=/x/voice.onnx")[2] is None
    with pytest.raises(SystemExit):
        evaluate_speak.parse_voice("no-equals")


def test_word_error_is_counted_per_sentence():
    marks = evaluate_speak.marks_of([("dobar dan svima", "dobar dan"), ("hvala", "hvala")])
    assert [m["edits"] for m in marks] == [1, 0] and [m["words"] for m in marks] == [3, 1]
    assert evaluate_speak.wer(marks) == 25.0
    assert evaluate_speak.summarize(marks)["edits"] == 1


def test_a_silent_rendering_stops_the_run(tmp_path, monkeypatch):
    class Config:
        num_speakers = 1

    class Fake:
        config = Config()

        def synthesize_wav(self, text, wav_file, syn_config=None):
            wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 100)   # 100 frames: silence-length audio

    piper = pytest.importorskip("piper")
    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(lambda a, b: Fake()))
    onnx = tmp_path / "voice.onnx"
    onnx.write_bytes(b"x"); onnx.with_name("voice.onnx.json").write_text("{}")
    with pytest.raises(SystemExit, match="silent"):
        evaluate_speak.synthesize(onnx, None, ["Dobar dan"], tmp_path / "out")
