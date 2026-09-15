import pytest

from scripts.ekavica_to_ijekavica import EkavicaToIjekavica, EXCLUSIONS


@pytest.fixture(scope="module")
def converter():
    return EkavicaToIjekavica()


@pytest.mark.parametrize(
    "src",
    ["selo", "meso", "more", "srce i duša grade grad", "More je slano."],
)
def test_non_yat_words_unchanged(converter, src):
    assert converter.convert_text(src) == src


@pytest.mark.parametrize(
    "src,expected",
    [
        ("vreme", "vrijeme"),
        ("reka", "rijeka"),
        ("dete", "dijete"),
        ("mesto", "mjesto"),
        ("ovo vreme i ta reka i ovo dete", "ovo vrijeme i ta rijeka i ovo dijete"),
    ],
)
def test_required_yat_positives(converter, src, expected):
    assert converter.convert_text(src) == expected


@pytest.mark.parametrize(
    "src,expected",
    [
        ("vremena", "vremena"),
        ("Devetnaest vijekova", "Devetnaest vijekova"),
        ("deteta", "djeteta"),
        ("reku", "rijeku"),
    ],
)
def test_inflected_forms(converter, src, expected):
    assert converter.convert_text(src) == expected


def test_homograph_exclusions_never_touched(converter):
    assert EXCLUSIONS  # documented collision set
    for word in ["svet", "sveta", "svetu", "sveti", "svete", "svetom", "reci", "zahteva", "rekom", "REKOM"]:
        assert converter.convert_text(word) == word


def test_unambiguous_svjetski_family_rewritten(converter):
    assert converter.convert_text("svetski") == "svjetski"
    assert converter.convert_text("svetsku") == "svjetsku"


def test_case_preserved(converter):
    assert converter.convert_text("VREME") == "VRIJEME"
    assert converter.convert_text("Vreme") == "Vrijeme"
    assert converter.convert_text("Reka") == "Rijeka"
    assert converter.convert_text("gde si reče") == "gdje si reče"


def test_lexicon_keys_lowercase_whole_forms(converter):
    for key, value in converter.lexicon.items():
        assert key.islower()
        assert key.isalpha()
        assert value.islower()
        assert value.isalpha()


def test_punctuation_and_digits_untouched(converter):
    src = "Reč 5, vreme 2! Svet - nije svet."
    out = converter.convert_text(src)
    assert out == "Riječ 5, vrijeme 2! Svet - nije svet."