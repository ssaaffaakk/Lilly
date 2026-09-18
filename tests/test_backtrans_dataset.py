"""Run B producer shards must reconstruct one exact registered sample."""
import gzip, importlib.util, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("backtrans_dataset", REPO / "scripts/backtrans_dataset.py")
dataset = importlib.util.module_from_spec(spec); spec.loader.exec_module(dataset)
import prepare_backtrans_bs as prep  # noqa: E402


def test_kaggle_notebook_package_import_works_from_repo_root():
    probe = subprocess.run(
        [sys.executable, "-c",
         "from scripts.backtrans_dataset import FORMAT_VERSION, pair_hash; "
         "from scripts.prepare_backtrans_bs import forward_input, is_pathological_source, ordered_hash"],
        cwd=REPO, text=True, capture_output=True)
    assert probe.returncode == 0, probe.stderr


def write_three_shards(root: Path):
    class FakeTokenizer:
        unk_token_id = 99

        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [[1, 2] for _ in texts]}

    lines = [f"bosanska rečenica broj {i} ovdje" for i in range(10)]
    for index in range(3):
        source, report = prep.prepare(list(lines), {"x"}, n=10, min_tok=3, max_tok=60,
                                      seed=7, shard_index=index, shard_count=3,
                                      tokenizer=FakeTokenizer())
        rows = [f"backtrans-macocu\t{bs}\tenglish {i}" for i, bs in enumerate(source)]
        data_name = f"backtrans-shard-{index}.tsv.gz"
        with gzip.open(root / data_name, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        report.update({"format_version": dataset.FORMAT_VERSION, "status": "complete",
                       "rows": len(rows), "forward_fingerprint": "forward-test",
                       "forward_normalized": 0,
                       "forward_decode_fallbacks": {"alternative": 0, "greedy": 0},
                       "git": "deadbeef", "data_file": data_name,
                       "source_order_hash": prep.ordered_hash(source),
                       "pair_order_hash": dataset.pair_hash(rows)})
        (root / f"backtrans-shard-{index}.json").write_text(json.dumps(report), encoding="utf-8")


def test_validate_union_proves_exact_gap_free_order(tmp_path):
    write_three_shards(tmp_path)
    got = dataset.validate_union(tmp_path, expected_n=10, shard_count=3,
                                 forward_fingerprint="forward-test", seed=7)
    assert got["rows"] == 10 and got["missing"] == 0 and got["duplicate_sources"] == 0
    assert [s["rows"] for s in got["shards"]] == [4, 3, 3]


def test_validate_union_refuses_a_duplicate_even_when_count_is_full(tmp_path):
    write_three_shards(tmp_path)
    data = tmp_path / "backtrans-shard-2.tsv.gz"
    with gzip.open(data, "rt", encoding="utf-8") as fh: rows = fh.read().splitlines()
    with gzip.open(tmp_path / "backtrans-shard-0.tsv.gz", "rt", encoding="utf-8") as fh:
        rows[0] = fh.readline().rstrip("\n")
    with gzip.open(data, "wt", encoding="utf-8") as fh: fh.write("\n".join(rows) + "\n")
    report_path = tmp_path / "backtrans-shard-2.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_order_hash"] = prep.ordered_hash([r.split("\t")[1] for r in rows])
    report["pair_order_hash"] = dataset.pair_hash(rows)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    try:
        dataset.validate_union(tmp_path, expected_n=10, shard_count=3,
                               forward_fingerprint="forward-test", seed=7)
    except SystemExit as exc:
        assert "duplicate_sources=1" in str(exc)
    else:
        raise AssertionError("duplicate producer source was accepted")


def test_validate_union_refuses_missing_normalization_accounting(tmp_path):
    write_three_shards(tmp_path)
    report_path = tmp_path / "backtrans-shard-1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["forward_normalized"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    try:
        dataset.validate_union(tmp_path, expected_n=10, shard_count=3,
                               forward_fingerprint="forward-test", seed=7)
    except SystemExit as exc:
        assert "invalid forward_normalized=None" in str(exc)
    else:
        raise AssertionError("producer without normalization accounting was accepted")


def test_validate_union_refuses_inconsistent_source_filter_accounting(tmp_path):
    write_three_shards(tmp_path)
    report_path = tmp_path / "backtrans-shard-0.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["pathological_reasons"] = {"non-Latin script": 1}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    try:
        dataset.validate_union(tmp_path, expected_n=10, shard_count=3,
                               forward_fingerprint="forward-test", seed=7)
    except SystemExit as exc:
        assert "inconsistent source-filter accounting" in str(exc)
    else:
        raise AssertionError("inconsistent source-filter accounting was accepted")


def test_validate_union_refuses_disabled_model_vocabulary_gate(tmp_path):
    write_three_shards(tmp_path)
    report_path = tmp_path / "backtrans-shard-0.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["model_vocabulary_gate"]["enabled"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    try:
        dataset.validate_union(tmp_path, expected_n=10, shard_count=3,
                               forward_fingerprint="forward-test", seed=7)
    except SystemExit as exc:
        assert "model-vocabulary gate was not enabled" in str(exc)
    else:
        raise AssertionError("disabled model-vocabulary gate was accepted")


def test_validate_union_refuses_missing_decoder_fallback_accounting(tmp_path):
    write_three_shards(tmp_path)
    report_path = tmp_path / "backtrans-shard-2.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["forward_decode_fallbacks"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    try:
        dataset.validate_union(tmp_path, expected_n=10, shard_count=3,
                               forward_fingerprint="forward-test", seed=7)
    except SystemExit as exc:
        assert "invalid forward_decode_fallbacks" in str(exc)
    else:
        raise AssertionError("missing decoder fallback accounting was accepted")


# --- producer-code equivalence gate: one commit, or a git-verified proof --------
# The shard `git` field is provenance, not integrity: the producer notebook clones
# `main` at run time, so an unrelated commit (an eval job, a doc) can advance it
# between two shards without changing producer output. resolve_producer_git checks
# the real invariant — the bytes of PRODUCER_PATHS — instead of the SHA string.

def test_resolve_producer_git_single_commit_needs_no_git():
    assert dataset.resolve_producer_git(["abc", "abc", "abc"]) == ("abc", None)


def test_resolve_producer_git_refuses_unresolvable_commit():
    # Two well-formed SHAs that no clone can resolve: cannot verify => refuse.
    try:
        dataset.resolve_producer_git(["0" * 40, "1" * 40])
    except SystemExit as exc:
        assert "cannot be checked from git" in str(exc)
    else:
        raise AssertionError("union across unverifiable commits was accepted")


def test_resolve_producer_git_refuses_differing_producer_code(monkeypatch):
    monkeypatch.setattr(dataset, "_producer_paths_tree_hash",
                        lambda sha: {"a": "hash-A", "b": "hash-B"}[sha])
    try:
        dataset.resolve_producer_git(["a", "b"])
    except SystemExit as exc:
        assert "PRODUCER CODE DIFFERS" in str(exc)
    else:
        raise AssertionError("commits with differing producer code were accepted")


def test_resolve_producer_git_accepts_git_verified_equivalent_commits(monkeypatch):
    monkeypatch.setattr(dataset, "_producer_paths_tree_hash", lambda sha: "same-hash")
    git, paths_hash = dataset.resolve_producer_git(["b", "a"])
    assert git == ["a", "b"] and paths_hash == "same-hash"


def test_run_b_union_commits_are_producer_equivalent():
    # The real Run B union spans these two commits (shards 0/1 at ab62ba64,
    # shard 2 at 7b719d06). They must carry byte-identical producer code.
    import pytest
    shards01 = "ab62ba64591d3f389cc17c8d2cbb973741afbb43"
    shard2 = "7b719d06fe7e15811656a685452e3c74d3fd01f9"
    h01 = dataset._producer_paths_tree_hash(shards01)
    h2 = dataset._producer_paths_tree_hash(shard2)
    if h01 is None or h2 is None:
        pytest.skip("Run B commits not resolvable in this checkout (shallow clone)")
    assert h01 == h2, "Run B shard commits no longer carry identical producer code"
    git, paths_hash = dataset.resolve_producer_git([shards01, shard2, shards01])
    assert git == sorted({shards01, shard2}) and paths_hash == h01
