"""Run B data prep — the guarantee that matters is no leakage.

`scripts/prepare_backtrans_bs.py` must drop every FLORES/bench sentence from the
back-translation sample, by exact match and by normalised near-match, or the
en→bs eval would score its own training data. These tests pin that, plus the
length/dedup/url filters and the empty-holdout refusal.
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_backtrans_bs", REPO / "scripts" / "prepare_backtrans_bs.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_normalize_folds_case_punct_whitespace():
    assert prep.normalize("Dobar   dan!") == prep.normalize("dobar dan")
    assert prep.normalize("Evropa, grad.") == prep.normalize("evropa grad")


def test_exact_holdout_line_is_dropped():
    holdout = {prep.normalize("Ovo je test rečenica za evaluaciju.")}
    kept, rep = prep.prepare(["Ovo je test rečenica za evaluaciju."],
                             holdout, n=0, min_tok=3, max_tok=60, seed=1)
    assert kept == []
    assert rep["dropped"]["holdout"] == 1


def test_near_match_holdout_is_dropped():
    # Same sentence, different case + trailing punctuation + doubled spaces.
    holdout = {prep.normalize("Molim vas ponovite pitanje")}
    kept, rep = prep.prepare(["MOLIM  VAS, ponovite pitanje!!!"],
                             holdout, n=0, min_tok=3, max_tok=60, seed=1)
    assert kept == [], "a case/punctuation variant must not slip past the holdout"
    assert rep["dropped"]["holdout"] == 1


def test_filters_and_dedup():
    holdout = {prep.normalize("held out sentence here")}
    lines = [
        "ok",                                   # too short (2 tokens)
        "ovo je dobra bosanska rečenica danas",  # kept
        "ovo je dobra bosanska rečenica danas",  # duplicate -> dropped
        "poseti http://spam.ba odmah sada",      # url -> dropped
        " ".join(["riječ"] * 70),               # too long -> dropped
    ]
    kept, rep = prep.prepare(lines, holdout, n=0, min_tok=3, max_tok=60, seed=1)
    assert kept == ["ovo je dobra bosanska rečenica danas"]
    assert rep["dropped"]["too short"] == 1
    assert rep["dropped"]["duplicate"] == 1
    assert rep["dropped"]["url/boilerplate"] == 1
    assert rep["dropped"]["too long"] == 1


def test_deterministic_sample():
    holdout = {"x"}
    lines = [f"bosanska rečenica broj {i} ovdje" for i in range(50)]
    a, _ = prep.prepare(list(lines), holdout, n=10, min_tok=3, max_tok=60, seed=7)
    b, _ = prep.prepare(list(lines), holdout, n=10, min_tok=3, max_tok=60, seed=7)
    assert a == b and len(a) == 10


def test_bench_targets_reads_bs_column(tmp_path):
    tsv = tmp_path / "cases.tsv"
    tsv.write_text("case_id\ten\tbs\nc1\tHello\tZdravo svijete danas\n", encoding="utf-8")
    assert list(prep.bench_targets(tsv)) == ["Zdravo svijete danas"]
