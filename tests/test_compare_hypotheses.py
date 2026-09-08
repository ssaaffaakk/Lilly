"""compare_hypotheses.py on a fake FLORES: same-build rule and the numbers."""
import json
import random
import sys

import pytest

pytest.importorskip("sacrebleu")
from training import compare_hypotheses, evaluate_app  # noqa: E402


@pytest.fixture
def flores(tmp_path):
    random.seed(1)
    words = "the cat sat on the mat and looked at the dog near the river".split()
    rows = [" ".join(random.choice(words) for _ in range(8)) + "." for _ in range(50)]
    d = tmp_path / "flores"; d.mkdir()
    for name, part in (("devtest", rows[:30]), ("dev", rows[30:])):
        (d / f"{name}.en").write_text("\n".join(part) + "\n")
        (d / f"{name}.bs").write_text("\n".join("bs " + r for r in part) + "\n")
    return d, rows


def saved(path, hyps, build):
    path.write_text(json.dumps({"n": len(hyps), "base": hyps, "base_build": build,
                                "lilly": hyps, "lilly_build": build}))
    return path


def run(argv):
    sys.argv = ["compare_hypotheses.py"] + argv
    return compare_hypotheses.main()


def test_two_recorded_builds_are_refused_unless_asked(tmp_path, flores):
    d, rows = flores
    old = saved(tmp_path / "old.json", rows, "a" * 32)
    new = saved(tmp_path / "new.json", rows, "b" * 32)
    with pytest.raises(SystemExit):
        run([str(old), str(new), "--flores", str(d)])
    assert run([str(old), str(new), "--flores", str(d), "--allow-different-builds"]) == 0


def test_an_unrecorded_fingerprint_is_noted_not_refused(tmp_path, flores, capsys):
    d, rows = flores
    old = saved(tmp_path / "old.json", rows, "")          # the Arm B file's shape
    # Every row leaks a tag, and every fifth row is really different: the
    # tag is stripped before anything is counted, so 10 rows differ, not 50.
    new = saved(tmp_path / "new.json",
                [">>eng<< " + ("changed " + r if i % 5 == 0 else r) for i, r in enumerate(rows)],
                "b" * 32)
    out = tmp_path / "out.json"
    assert run([str(old), str(new), "--flores", str(d), "--json", str(out)]) == 0
    assert "record no build fingerprint" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["splits"]["all"]["changed"] == 10
    assert result["splits"]["all"]["old"]["chrf"] == pytest.approx(100.0)
    assert result["splits"]["all"]["new"]["chrf"] < 100.0
    assert result["splits"]["devtest"]["pairs"] == 30
