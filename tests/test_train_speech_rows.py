"""train_speech.py's TSV reader and per-row tokenizer choice, with no torch.

The module imports torch, soundfile, scipy and transformers at the top. None
of them is needed to read a TSV or to pick a tokenizer, so where a real one is
missing a bare stub stands in, and the same test runs on a machine that has
them all.
"""
import sys
import types

import numpy as np
import pytest


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


for name in ("torch", "soundfile", "scipy", "transformers"):
    try:
        __import__(name)
    except ImportError:
        if name == "torch":
            torch = _stub("torch")
            torch.utils = _stub("torch.utils")
            torch.utils.data = _stub("torch.utils.data", Dataset=object)
        elif name == "scipy":
            scipy = _stub("scipy")
            scipy.signal = _stub("scipy.signal", resample_poly=None)
        elif name == "transformers":
            _stub("transformers", Seq2SeqTrainer=object, Seq2SeqTrainingArguments=object,
                  TrainerCallback=object, WhisperForConditionalGeneration=object,
                  WhisperProcessor=object)
        else:
            _stub(name)

from training import train_speech  # noqa: E402


def test_read_tsv_two_columns_by_default_three_on_request(tmp_path):
    tsv = tmp_path / "train.tsv"
    tsv.write_text("a.wav\tDobar dan\n"
                   "b.wav\tDobar dan\tBS\n"
                   "/abs/c.wav\tGdje je kolodvor?\thr\n"
                   "one column only\n"
                   "d.wav\t\n", encoding="utf-8")
    plain = train_speech.read_tsv(tsv)
    assert plain == [(tmp_path / "a.wav", "Dobar dan"), (tmp_path / "b.wav", "Dobar dan"),
                     (train_speech.Path("/abs/c.wav"), "Gdje je kolodvor?")]
    with_lang = train_speech.read_tsv(tsv, with_language=True)
    assert [r[2] for r in with_lang] == [None, "bs", "hr"]


class Tok:
    def __init__(self, code):
        self.code, self.prefix_tokens = code, [f"<|{code}|>"]

    def __call__(self, text):
        return types.SimpleNamespace(input_ids=[f"<|{self.code}|>"] + text.split())


def test_each_row_is_tokenised_under_its_own_language(monkeypatch):
    monkeypatch.setattr(train_speech, "load_audio", lambda clip: np.zeros(16_000, dtype="float32"))
    processor = types.SimpleNamespace(
        tokenizer=Tok("bs"),
        feature_extractor=lambda audio, sampling_rate: types.SimpleNamespace(input_features=[audio]))
    rows = [("a.wav", "Dobar dan", None), ("b.wav", "Gdje je kolodvor", "hr"), ("c.wav", "Hvala", "bs")]
    ds = train_speech.ClipDataset(rows, processor, "bs", {"hr": Tok("hr")})
    assert [ds[i]["labels"][0] for i in range(3)] == ["<|bs|>", "<|hr|>", "<|bs|>"]
    assert ds[1]["labels"][1:] == ["Gdje", "je", "kolodvor"]


def test_language_tokenizers_are_built_once_per_code_and_validated(monkeypatch):
    built = []

    class FromPretrained(Tok):
        @classmethod
        def from_pretrained(cls, base, language, task):
            built.append(language)
            if language == "xx":
                raise ValueError("Unsupported language: xx")
            return cls(language)

    processor = types.SimpleNamespace(tokenizer=FromPretrained("bs"))
    rows = [("a", "x", None), ("b", "y", "hr"), ("c", "z", "hr"), ("d", "w", "bs")]
    toks = train_speech.language_tokenizers(processor, "openai/whisper-small", "bs", rows)
    assert list(toks) == ["hr"] and built == ["hr"]
    with pytest.raises(ValueError):
        train_speech.language_tokenizers(processor, "openai/whisper-small", "bs", [("a", "x", "xx")])
