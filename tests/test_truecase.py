"""The truecaser: shouted text in, sentence case out, or nothing at all.

restore_sentence_case is tested without a model, because that is the part that
decides whether a line is rewritten. The wiring tests use the fake engine from
tests/test_engine.py's pattern so they need no CTranslate2 build: they check
that translate() no longer recases at all (the flag moved to the photograph
path), and that the photo restorer recases a shouted source-language line while
leaving a foreign caption alone.
"""
import pytest

from app import truecase
from app.translate import Engine


def restore(text: str) -> str:
    return truecase.restore_sentence_case(text)[0]


# --- the pure function -------------------------------------------------------

def test_all_caps_is_restored_to_sentence_case():
    assert restore("DANGER HIGH VOLTAGE KEEP OUT") == "Danger high voltage keep out"


def test_each_sentence_is_capitalised_after_its_terminator():
    assert restore("NO SMOKING. KEEP OUT! MIND THE GAP?") == \
        "No smoking. Keep out! Mind the gap?"


def test_lines_are_sentence_boundaries_like_the_app_splitter():
    # A sign is three notices on three lines, and each starts a sentence.
    assert restore("ZABRANJEN ULAZ\nRADOVI NA MOSTU\nHVALA NA RAZUMIJEVANJU") == \
        "Zabranjen ulaz\nRadovi na mostu\nHvala na razumijevanju"


@pytest.mark.parametrize("text", [
    "dobar dan, kako ste danas?",
    "Danger High Voltage Keep Out",       # title case, not shouted
    "Dobar dan. Kako ste?",               # ordinary sentences
    "Otvoreno od 8. do 16. sati.",        # mixed case with numbers
])
def test_text_that_is_not_shouted_passes_through_byte_for_byte(text):
    assert restore(text) == text
    assert truecase.restore_sentence_case(text) == (text, False)


@pytest.mark.parametrize("text", [
    "",                                   # empty
    "   \n\t ",                           # no letters at all
    "12345 6789",                         # digits only
    "NASA",                               # too few letters to be a sign
    "STOP",
])
def test_nothing_to_restore_is_left_alone(text):
    assert truecase.restore_sentence_case(text) == (text, False)


def test_diacritics_survive_the_round_trip():
    # The report's caution: capitalising a word that already carries č ć đ š ž
    # must not mangle them. Every one of them appears here, first and elsewhere.
    assert restore("ČAMAC IMA ČETIRI VESLA I ŠEST JEDARA") == \
        "Čamac ima četiri vesla i šest jedara"
    assert restore("ĐAK JE ŠUMA I ĆUPRIJA I ŽITO") == "Đak je šuma i ćuprija i žito"
    # The first letter is the diacritic itself, not a plain Latin one.
    assert restore("ŠUMSKI PUT VODI DO JEZERA") == "Šumski put vodi do jezera"


def test_proper_nouns_are_kept_across_their_cases():
    # Bosnian inflects a name, so the safeguard matches stem plus name ending.
    assert restore("SARAJEVO JE GLAVNI GRAD BOSNE I HERCEGOVINE") == \
        "Sarajevo je glavni grad Bosne i Hercegovine"
    assert restore("U BEOGRADU SE GOVORI SRPSKI") == "U Beogradu se govori srpski"
    assert restore("HRVATSKA I BOSNA") == "Hrvatska i Bosna"


def test_a_derived_adjective_is_not_a_name():
    # The report's caution cuts the other way too: Bosnian writes the adjective
    # lowercase even though the name it comes from is capitalised. Capitalising
    # "srpski" mid-sentence would be English capitalisation, not Bosnian.
    assert restore("GOVORI BOSANSKI I SRPSKI JEZIK") == "Govori bosanski i srpski jezik"


def test_an_ordinal_period_does_not_start_a_sentence():
    # "5. maja 1990. godine" -- capitalising maja would be the English rule, not
    # the Bosnian one. The restorer keeps it lowercase.
    assert restore("ROĐEN JE 5. MAJA 1990. GODINE") == "Rođen je 5. maja 1990. godine"


# --- the flag ----------------------------------------------------------------

def test_flag_is_off_unless_it_is_set(monkeypatch):
    monkeypatch.delenv("LILLY_TRUECASE", raising=False)
    assert truecase.enabled() is False
    for value in ("0", "false", "no", "off"):
        monkeypatch.setenv("LILLY_TRUECASE", value)
        assert truecase.enabled() is False, value
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("LILLY_TRUECASE", value)
        assert truecase.enabled() is True, value


def test_an_unknown_flag_value_stops_the_process(monkeypatch):
    monkeypatch.setenv("LILLY_TRUECASE", "maybe")
    with pytest.raises(RuntimeError):
        truecase.enabled()


# --- the engine no longer recases: the flag moved to the photo path ----------

class FakeTokenizer:
    """Whitespace tokens, as in tests/test_engine.py; ids are vocabulary slots."""

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


class Recorder:
    """Keeps every source token sequence the engine hands the model."""

    def __init__(self):
        self.sources = []

    def translate_batch(self, group, beam_size, max_decoding_length):
        self.sources += [" ".join(tokens) for tokens in group]
        return [Hypothesis(list(tokens)) for tokens in group]


@pytest.fixture
def engine():
    e = Engine.__new__(Engine)
    e.tokenizer = FakeTokenizer()
    e.translator = Recorder()
    e.tag = None
    return e


def test_engine_leaves_shouted_text_untouched_with_the_flag_unset(engine, monkeypatch):
    monkeypatch.delenv("LILLY_TRUECASE", raising=False)
    engine.translate("DANGER HIGH VOLTAGE KEEP OUT")
    assert engine.translator.sources == ["DANGER HIGH VOLTAGE KEEP OUT"], \
        "translate() must not touch the source"


def test_engine_leaves_shouted_text_untouched_even_with_the_flag_set(engine, monkeypatch):
    # Truecasing moved to app.lilly.translate_photo, so translate() never recases
    # whatever the flag says -- typed text and both directions are unaffected.
    monkeypatch.setenv("LILLY_TRUECASE", "1")
    engine.translate("DANGER HIGH VOLTAGE KEEP OUT")
    assert engine.translator.sources == ["DANGER HIGH VOLTAGE KEEP OUT"], \
        "the flag no longer reaches translate(); only the photo path recases"


# --- the photograph path -----------------------------------------------------

def test_photo_text_recases_bosnian_lines_and_keeps_foreign_ones():
    # A Bosnian line and an English caption on one sign: recase the first, leave
    # the second. app.detect reads each line's language from its case-folded grams.
    text = "ŠIROKI BRIJEG OTVORENO SVAKI DAN\nWELCOME TO OUR RESTAURANT PLEASE COME IN"
    out, changed = truecase.restore_photo_text(text, "bs")
    first, second = out.split("\n")
    assert first == "Široki brijeg otvoreno svaki dan"
    assert second == "WELCOME TO OUR RESTAURANT PLEASE COME IN"
    assert changed is True


def test_photo_text_recases_the_source_language_you_ask_for():
    # An en-bs photograph reads an English sign, so there it is the English line
    # to recase; the same restorer, told the source is English.
    text = "WELCOME TO OUR RESTAURANT PLEASE COME IN"
    assert truecase.restore_photo_text(text, "en")[0] == \
        "Welcome to our restaurant please come in"


def test_photo_text_leaves_calm_and_short_lines_alone():
    # Not shouted, too few letters, no letters -- none is touched, changed False.
    text = "Dobar dan\nWC\n1234"
    assert truecase.restore_photo_text(text, "bs") == (text, False)


def test_photo_source_is_recased_only_behind_the_flag(monkeypatch):
    from app.lilly import Lilly
    shout = "ČUVAJ SE OPASAN PAS U DVORIŠTU"
    monkeypatch.delenv("LILLY_TRUECASE", raising=False)
    assert Lilly._recase_photo_source(shout, "bs") == shout
    monkeypatch.setenv("LILLY_TRUECASE", "1")
    assert Lilly._recase_photo_source(shout, "bs") == "Čuvaj se opasan pas u dvorištu"
