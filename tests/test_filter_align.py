import importlib.util
import sys
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "filter_wikimatrix_align", Path(__file__).parents[1] / "scripts" / "filter_wikimatrix_align.py"
)
align = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = align
SPEC.loader.exec_module(align)


class FakeEncoder:
    def encode(self, texts, batch_size):
        table = {
            "bs-good": [1.0, 0.0], "en-good": [1.0, 0.0],
            "bs-bad": [1.0, 0.0], "en-bad": [0.0, 1.0],
        }
        return [table[text] for text in texts]


def test_scores_crosslingual_pairs_and_filters_tail():
    pairs = [
        align.Pair(1, "WikiMatrix", "bs-good", "en-good"),
        align.Pair(2, "WikiMatrix", "bs-bad", "en-bad"),
    ]
    scored = align.score_pairs(pairs, FakeEncoder(), batch_size=2)
    assert [round(score, 3) for _, score in scored] == [1.0, 0.0]
    assert "drop=1 (50.00%)" in align.summarize(scored, threshold=0.55)


def test_reservoir_sample_is_deterministic_and_counts_population():
    pairs = [align.Pair(i, "WikiMatrix", str(i), str(i)) for i in range(1, 51)]
    first, population = align.reservoir_sample(iter(pairs), 7, seed=11)
    second, _ = align.reservoir_sample(iter(pairs), 7, seed=11)
    assert population == 50
    assert [pair.row for pair in first] == [pair.row for pair in second]


def test_read_pairs_rejects_malformed_rows(tmp_path):
    source = tmp_path / "pairs.tsv"
    source.write_text("WikiMatrix\tb\te\nWikiMatrix\tbroken\n", encoding="utf-8")
    try:
        list(align.read_pairs(source))
    except SystemExit as exc:
        assert "expected source, bs, en" in str(exc)
    else:
        raise AssertionError("malformed row was accepted")
