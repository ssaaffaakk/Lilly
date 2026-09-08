"""build_speech_mix.py: the ratio, the leak filter and the language column."""
import collections
import sys

import pytest

from data.scripts import build_speech_mix as mix


def corpus(tmp_path, sources):
    speech = tmp_path / "speech"
    (speech / "train").mkdir(parents=True)
    for i in range(3):
        (speech / "train" / f"{i}.wav").write_bytes(b"x")
    (speech / "train.tsv").write_text(
        "".join(f"train/{i}.wav\tBosanska rečenica {i}\n" for i in range(3)), encoding="utf-8")
    (speech / "valid.tsv").write_text("valid/0.wav\tHeld out\n", encoding="utf-8")
    (speech / "test.tsv").write_text("test/0.wav\tHeld out too\n", encoding="utf-8")
    extra = tmp_path / "speech-extra"
    for source, n in sources.items():
        (extra / source / "train").mkdir(parents=True)
        for i in range(n + 1):
            (extra / source / "train" / f"{i}.wav").write_bytes(b"x")
        rows = [f"train/{i}.wav\tRečenica {source} {i}\n" for i in range(n)]
        rows.append(f"train/{n}.wav\tHeld out\n")          # leaks a held-out transcript
        (extra / source / "train.tsv").write_text("".join(rows), encoding="utf-8")
    return speech, extra


def run(monkeypatch, speech, extra, out, share="0.5"):
    monkeypatch.setattr(mix, "BOSNIAN", speech / "train.tsv")
    monkeypatch.setattr(mix, "EXTRA_DIR", extra)
    monkeypatch.setattr(mix, "HELD_OUT", (speech / "valid.tsv", speech / "test.tsv"))
    monkeypatch.setattr(sys, "argv", ["build_speech_mix.py", "--share", share, "--out", str(out)])
    return mix.main()


def test_every_row_carries_its_sources_language_token(tmp_path, monkeypatch):
    speech, extra = corpus(tmp_path, {"fleurs_hr": 4})
    out = tmp_path / "train-mix.tsv"
    assert run(monkeypatch, speech, extra, out) == 0
    rows = [line.split("\t") for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(len(r) == 3 for r in rows)
    assert collections.Counter(r[2] for r in rows) == {"bs": 3, "hr": 4}, "the leaked row is gone"
    assert all(r[0].startswith("/") for r in rows), "paths are absolute"


def test_a_source_with_no_registered_token_stops_the_run(tmp_path, monkeypatch):
    speech, extra = corpus(tmp_path, {"youtube_sr": 2})
    with pytest.raises(SystemExit):
        run(monkeypatch, speech, extra, tmp_path / "train-mix.tsv")


def test_bosnian_is_repeated_up_to_the_share(tmp_path, monkeypatch):
    speech, extra = corpus(tmp_path, {"fleurs_hr": 12})
    out = tmp_path / "train-mix.tsv"
    assert run(monkeypatch, speech, extra, out, share="0.5") == 0
    counts = collections.Counter(line.split("\t")[2] for line in out.read_text(encoding="utf-8").splitlines())
    assert counts["hr"] == 12 and counts["bs"] == 12          # 3 clips repeated 4x
