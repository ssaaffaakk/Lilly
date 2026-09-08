"""The sentence splitter and the batch grouping, which need no model.

The splitter is part of what Lilly does: every served-path score goes through
it. A rule that cuts "5. maja 1990. godine" into three sentences is a product
bug that no corpus metric shows on its own, which is why it has a test.
"""
import pytest

from app.translate import BATCH_TOKEN_BUDGET, SENTENCE_BREAK, Engine


def split(text: str) -> list:
    """Exactly what Engine.translate does with the text before tokenising."""
    return [s for s in SENTENCE_BREAK.split(text.strip()) if s.strip()] or [text]


@pytest.mark.parametrize("text, expected", [
    # ordinals and dates: a period after a digit is not a full stop when a
    # lowercase word follows
    ("Rođen je 5. maja 1990. godine u Sarajevu.",
     ["Rođen je 5. maja 1990. godine u Sarajevu."]),
    ("Otvoreno od 8. do 16. sati.", ["Otvoreno od 8. do 16. sati."]),
    ("U 19. stoljeću grad je rastao.", ["U 19. stoljeću grad je rastao."]),
    ("Rok je 24. jula 1995. godine. Dobro.",
     ["Rok je 24. jula 1995. godine.", "Dobro."]),
    # a real boundary after a number: the next word is capitalised
    ("Na 3. mjestu je Bosna. Slijedi Hrvatska.",
     ["Na 3. mjestu je Bosna.", "Slijedi Hrvatska."]),
    ("Sastanak je u 9.30. Dođite ranije! Ok?",
     ["Sastanak je u 9.30.", "Dođite ranije!", "Ok?"]),
    ("Cijena je 5. Hvala.", ["Cijena je 5.", "Hvala."]),
    # other punctuation after a digit still ends the sentence
    ("Broj 5! dobro", ["Broj 5!", "dobro"]),
    # lines are boundaries whatever the punctuation, for text off a sign
    ("ZABRANJEN ULAZ\nRadovi na mostu\nHvala na razumijevanju",
     ["ZABRANJEN ULAZ", "Radovi na mostu", "Hvala na razumijevanju"]),
    ("Prva.\n\n  Druga.", ["Prva.", "Druga."]),
    # plain sentences
    ("Dobar dan. Kako ste?", ["Dobar dan.", "Kako ste?"]),
    ("Molim vas, možete li ponoviti? Nisam razumio zadaću.",
     ["Molim vas, možete li ponoviti?", "Nisam razumio zadaću."]),
])
def test_splitter(text, expected):
    assert split(text) == expected


def test_whitespace_only_text_is_kept_whole():
    # Engine.translate falls back to the text itself rather than to nothing.
    assert split("   ") == ["   "]


def test_grouped_keeps_every_batch_under_the_budget():
    widths = [10, 200, 50, 60, 5, 256, 1, 1, 1]
    groups = list(Engine._grouped([[0] * w for w in widths]))
    assert [len(t) for g in groups for t in g] == widths, "order and count preserved"
    for group in groups:
        widest = max(len(t) for t in group)
        assert len(group) == 1 or len(group) * widest <= BATCH_TOKEN_BUDGET


def test_grouped_never_drops_a_sentence_wider_than_the_budget():
    groups = list(Engine._grouped([[0] * (BATCH_TOKEN_BUDGET * 2)]))
    assert len(groups) == 1 and len(groups[0]) == 1
