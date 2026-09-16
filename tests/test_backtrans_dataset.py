"""Run B producer shards must reconstruct one exact registered sample."""
import gzip, importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("backtrans_dataset", REPO / "scripts/backtrans_dataset.py")
dataset = importlib.util.module_from_spec(spec); spec.loader.exec_module(dataset)
import prepare_backtrans_bs as prep  # noqa: E402


def write_three_shards(root: Path):
    lines = [f"bosanska rečenica broj {i} ovdje" for i in range(10)]
    for index in range(3):
        source, report = prep.prepare(list(lines), {"x"}, n=10, min_tok=3, max_tok=60,
                                      seed=7, shard_index=index, shard_count=3)
        rows = [f"backtrans-macocu\t{bs}\tenglish {i}" for i, bs in enumerate(source)]
        data_name = f"backtrans-shard-{index}.tsv.gz"
        with gzip.open(root / data_name, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        report.update({"format_version": dataset.FORMAT_VERSION, "status": "complete",
                       "rows": len(rows), "forward_fingerprint": "forward-test",
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
