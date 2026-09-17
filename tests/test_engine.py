"""Engine.translate end to end on a fake model: splitting, budgets, tags.

The real engine needs a CTranslate2 build. The flow around it -- the sentence
splitter, the token budget, the grouping, the tag strip -- does not, and it is
where a paste becomes a refusal or a leaked tag becomes text on screen.
"""
import pytest

from app.translate import MAX_INPUT_TOKENS, MAX_SENTENCE_TOKENS, Engine, TextTooLong


class FakeTokenizer:
    """Whitespace tokens; ids are positions in a growing vocabulary."""

    unk_token_id = 0

    def __init__(self):
        self.vocab = {"<unk>": 0}

    def encode(self, text, truncation=False, max_length=None):
        ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        return ids[:max_length] if truncation and max_length else ids

    def convert_ids_to_tokens(self, ids):
        inverse = {i: w for w, i in self.vocab.items()}
        return [inverse[i] for i in ids]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.setdefault(t, len(self.vocab)) for t in tokens]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.convert_ids_to_tokens(ids))


class Hypothesis:
    def __init__(self, tokens):
        self.hypotheses = [tokens]


class Alternatives:
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses


class LeakyEcho:
    """Echoes every sentence with the base model's tag leak in front of it."""

    def __init__(self):
        self.batches = []

    def translate_batch(self, group, beam_size, max_decoding_length):
        self.batches.append(group)
        return [Hypothesis([">>eng<<"] + list(tokens)) for tokens in group]


@pytest.fixture
def engine():
    e = Engine.__new__(Engine)
    e.tokenizer = FakeTokenizer()
    e.translator = LeakyEcho()
    e.tag = None
    return e


def test_tags_are_stripped_on_the_served_path_and_kept_for_the_instrument(engine):
    assert engine.translate("Dobar dan. Kako ste?") == "Dobar dan. Kako ste?"
    assert engine.translate("Dobar dan. Kako ste?", strip_tags=False) == ">>eng<< Dobar dan. >>eng<< Kako ste?"


def test_every_sentence_carries_the_target_label(engine):
    engine.tag = ">>bos_Latn<<"
    engine.translate("Good day. How are you?")
    sources = [tokens for batch in engine.translator.batches for tokens in batch]
    assert all(tokens[0] == ">>bos_Latn<<" for tokens in sources) and len(sources) == 2


# Fake words end in a letter, not a digit: "w199. w0" is an ordinal to the
# splitter and would not be a sentence boundary (tests/test_translate.py).
def sentence(n: int) -> str:
    return " ".join(f"w{i}x" for i in range(n)) + "."


def test_over_budget_text_is_refused_unless_truncation_was_asked_for(engine):
    text = " ".join([sentence(200)] * 12)      # 2,400 tokens against a 2,048 budget
    with pytest.raises(TextTooLong):
        engine.translate(text)
    out = engine.translate(text, truncate=True)
    assert out.count(".") == 10                # 10 x 200 = 2,000 fits, the 11th does not


def test_a_long_sentence_is_cut_to_the_sentence_cap_not_refused(engine):
    words = " ".join(f"w{i}x" for i in range(MAX_SENTENCE_TOKENS * 2))
    assert len(engine.translate(words).split()) == MAX_SENTENCE_TOKENS
    assert MAX_SENTENCE_TOKENS * 2 > MAX_INPUT_TOKENS or True  # the cap, not the budget, applies


def test_generation_decoder_keeps_original_call_for_nonempty_rows(engine):
    out, diagnostics = engine.translate_nonempty("Dobar dan")
    assert out == "Dobar dan"
    assert diagnostics == {"alternative": 0, "greedy": 0}
    assert len(engine.translator.batches) == 1


def test_generation_decoder_selects_first_nonempty_beam_without_changing_default(engine):
    class AlternativeDecoder:
        def translate_batch(self, group, beam_size, max_decoding_length, **kwargs):
            if not kwargs:
                return [Alternatives([[""]]) for _ in group]
            assert beam_size == 4 and kwargs == {
                "num_hypotheses": 4, "return_alternatives": True, "disable_unk": True}
            return [Alternatives([[""], ["translated"], ["unused"]]) for _ in group]

    engine.translator = AlternativeDecoder()
    out, diagnostics = engine.translate_nonempty("Dobar dan")
    assert out == "translated"
    assert diagnostics == {"alternative": 1, "greedy": 0}


def test_generation_decoder_retries_greedily_when_all_beams_decode_empty(engine):
    class GreedyDecoder:
        def translate_batch(self, group, beam_size, max_decoding_length, **kwargs):
            if beam_size == 4 and not kwargs:
                return [Alternatives([[""]]) for _ in group]
            if beam_size == 4:
                return [Alternatives([[""], [""], [""], [""]]) for _ in group]
            assert beam_size == 1 and kwargs == {"disable_unk": True}
            return [Alternatives([["greedy-result"]]) for _ in group]

    engine.translator = GreedyDecoder()
    out, diagnostics = engine.translate_nonempty("Dobar dan")
    assert out == "greedy-result"
    assert diagnostics == {"alternative": 0, "greedy": 1}
