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


def test_pathological_long_tokens_are_filtered_before_sampling():
    noisy = "nogomet je " + "1" + "0" * 99 + " puta bolji"
    clean = "nogomet je danas mnogo bolji"
    kept, rep = prep.prepare([noisy, clean], {"x"}, n=1, min_tok=3,
                             max_tok=60, seed=1)
    assert kept == [clean]
    assert rep["dropped"]["pathological source"] == 1
    assert rep["pathological_reasons"] == {"long token/number": 1}


def test_non_latin_and_repeated_letter_noise_are_filtered_before_sampling():
    repeated = (
        "mjauuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu me odbi deckoto ze j**enje "
        "u krevet mjauuuuuuuuuuuuuuuuuuuuuuuuuuu k**va sa p***a pomazite")
    arabic = "مَّثَلُ الْجَنَّةِ الَّتِي وُعِدَ الْمُتَّقُونَ"
    symbols = "*** --- !!!"
    clean = "Čista bosanska rečenica ostaje u uzorku."
    kept, rep = prep.prepare([repeated, arabic, symbols, clean], {"x"}, n=4,
                             min_tok=3, max_tok=60, seed=1)
    assert kept == [clean]
    assert rep["dropped"]["pathological source"] == 3
    assert rep["pathological_reasons"] == {
        "repeated alphanumeric": 1,
        "non-Latin script": 1,
        "no Latin text": 1,
    }


def test_three_repeated_letters_are_filtered_at_measured_decoder_boundary():
    producer0 = "auuu predobar je,stvarno pre pre predobar,neka bude sa srecom"
    producer1 = (
        'skinula prvu, oooo, već vidim, moja duša pjeva, a um zadovoljno '
        'trlja ručice Hvala ti što si me "natjerala" da ju pročitam')
    assert prep.source_rejection_reason(producer0) == "repeated alphanumeric"
    assert prep.source_rejection_reason(producer1) == "repeated alphanumeric"
    assert prep.source_rejection_reason("auu je uzvik sa dva ponovljena slova") is None


def test_forward_input_exposes_joined_sentence_boundaries_only():
    raw = "prva recenica..druga recenica...treca?cetvrta"
    assert prep.forward_input(raw) == (
        "prva recenica. druga recenica. treca? cetvrta")
    ordinary = "Ovo je obična rečenica. Ovo je druga."
    assert prep.forward_input(ordinary) == ordinary


def test_model_vocabulary_gate_replaces_unknown_candidate_before_sample():
    class FakeTokenizer:
        unk_token_id = 99

        def __call__(self, texts, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [[99] if "ᴅᴀɴ" in text else [1, 2]
                                  for text in texts]}

    lines = ["ᴅᴀɴ otvorenih vrata", "prva čista bosanska rečenica",
             "druga čista bosanska rečenica"]
    kept, report = prep.prepare(lines, {"x"}, n=2, min_tok=3, max_tok=60,
                                seed=0, tokenizer=FakeTokenizer())
    assert len(kept) == 2
    assert "ᴅᴀɴ otvorenih vrata" not in kept
    assert report["model_vocabulary_gate"] == {
        "enabled": True, "checked": 3, "rejected_unknown": 1, "accepted": 2}


def test_deterministic_sample():
    holdout = {"x"}
    lines = [f"bosanska rečenica broj {i} ovdje" for i in range(50)]
    a, _ = prep.prepare(list(lines), holdout, n=10, min_tok=3, max_tok=60, seed=7)
    b, _ = prep.prepare(list(lines), holdout, n=10, min_tok=3, max_tok=60, seed=7)
    assert a == b and len(a) == 10


def test_three_shards_are_balanced_gap_free_and_reconstruct_sample():
    holdout = {"x"}
    lines = [f"bosanska rečenica broj {i} ovdje" for i in range(50)]
    whole, _ = prep.prepare(list(lines), holdout, n=10, min_tok=3,
                            max_tok=60, seed=7)
    shards, reports = [], []
    for index in range(3):
        shard, report = prep.prepare(
            list(lines), holdout, n=10, min_tok=3, max_tok=60, seed=7,
            shard_index=index, shard_count=3)
        shards.extend(shard)
        reports.append(report)
    assert shards == whole
    assert [r["kept"] for r in reports] == [4, 3, 3]
    assert all(r["sample_kept"] == 10 for r in reports)


def test_shard_arguments_must_be_paired_and_in_range():
    holdout = {"x"}
    lines = ["bosanska rečenica broj jedan"]
    for kwargs in ({"shard_index": 0}, {"shard_count": 3},
                   {"shard_index": 3, "shard_count": 3}):
        try:
            prep.prepare(lines, holdout, n=0, min_tok=3, max_tok=60, seed=1,
                         **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid shard arguments were accepted: {kwargs}")


def test_bench_targets_reads_bs_column(tmp_path):
    tsv = tmp_path / "cases.tsv"
    tsv.write_text("case_id\ten\tbs\nc1\tHello\tZdravo svijete danas\n", encoding="utf-8")
    assert list(prep.bench_targets(tsv)) == ["Zdravo svijete danas"]
