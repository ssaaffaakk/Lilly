"""The prompt benchmark's plumbing, with stand-in executors: no model, no network.

The first version of scripts/eval_translation_prompts.py unpacked a tuple from
Engine.translate, which returns a string, and caught the error into the output:
every variant was scored on the text "[Error: too many values to unpack]", and
nothing said so. These tests pin the contract that would have caught it.
"""
import pytest

import scripts.eval_translation_prompts as evaluation
from scripts.prompt_benchmark import (
    GradingRubric, PromptBenchmark, PromptVariant, exact_match_rubric,
    lilly_executor, paired_bootstrap, token_overlap_rubric,
)

CASES = [
    {"input": "Hvala.", "expected_output": "Thank you."},
    {"input": "Dobar dan.", "expected_output": "Good day."},
    {"input": "Ne razumijem.", "expected_output": "I don't understand."},
]
ANSWERS = {case["input"]: case["expected_output"] for case in CASES}


class FakeEngine:
    """What app.translate.Engine is: a string in, a string out."""

    def __init__(self):
        self.seen = []

    def translate(self, text, truncate=False, strip_tags=True):
        self.seen.append(text)
        return ANSWERS.get(text, "")


def test_lilly_executor_returns_the_engine_string_and_sees_only_the_input(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr("app.translate.get_engine", lambda direction="bs-en": engine)
    benchmark = PromptBenchmark(
        name="t", variants={"lilly": PromptVariant("lilly", "{input}", executor=lilly_executor())},
        test_cases=CASES)
    results = benchmark.run()
    assert engine.seen == [case["input"] for case in CASES]
    assert results["lilly"]["mean_accuracy"] == 1.0


def test_an_executor_that_raises_stops_the_run():
    def broken(prompt):
        raise RuntimeError("model missing")

    benchmark = PromptBenchmark(name="t", variants={"v": "{input}"}, test_cases=CASES)
    with pytest.raises(RuntimeError, match="model missing"):
        benchmark.run(executor=broken)


def test_a_template_without_the_placeholder_is_refused():
    with pytest.raises(ValueError, match="placeholder"):
        PromptVariant("v", "Translate this.")


def test_other_braces_in_a_template_are_left_alone():
    variant = PromptVariant("v", 'Answer as {"english": "..."} for: {input}')
    assert variant.prompt("Hvala.") == 'Answer as {"english": "..."} for: Hvala.'


def test_a_variant_uses_its_own_executor_before_the_shared_one():
    benchmark = PromptBenchmark(
        name="t",
        variants={"own": PromptVariant("own", "{input}", executor=lambda p: ANSWERS[p]),
                  "shared": "{input}"},
        test_cases=CASES)
    results = benchmark.run(executor=lambda p: "wrong")
    assert results["own"]["mean_accuracy"] == 1.0
    assert results["shared"]["mean_accuracy"] == 0.0


def test_a_variant_with_no_executor_at_all_is_an_error():
    benchmark = PromptBenchmark(name="t", variants={"v": "{input}"}, test_cases=CASES)
    with pytest.raises(ValueError, match="no executor"):
        benchmark.run()


def test_samples_are_averaged_within_a_case_before_cases_are_averaged():
    calls = {"n": 0}

    def alternating(prompt):
        calls["n"] += 1
        return ANSWERS[prompt] if calls["n"] % 2 else "wrong"

    benchmark = PromptBenchmark(name="t", variants={"v": "{input}"}, test_cases=CASES)
    benchmark.run(executor=alternating, samples=2)
    assert benchmark.case_scores("v") == [0.5, 0.5, 0.5]


def test_cost_comes_from_the_executor_and_stays_unknown_without_one():
    benchmark = PromptBenchmark(
        name="t",
        variants={"priced": PromptVariant("priced", "{input}", executor=lambda p: (ANSWERS[p], 0.01)),
                  "unpriced": PromptVariant("unpriced", "{input}", executor=lambda p: ANSWERS[p])},
        test_cases=CASES)
    results = benchmark.run()
    assert results["priced"]["total_cost_usd"] == pytest.approx(0.03)
    assert results["unpriced"]["total_cost_usd"] is None


def test_paired_bootstrap_calls_identical_scores_a_tie():
    result = paired_bootstrap([0.5, 0.7, 0.9], [0.5, 0.7, 0.9])
    assert result["delta"] == 0.0
    assert result["p"] == 1.0


def test_paired_bootstrap_sees_a_consistent_gain():
    a = [0.2] * 30
    b = [0.8] * 30
    result = paired_bootstrap(a, b)
    assert result["delta"] == pytest.approx(0.6)
    assert result["low"] > 0
    assert result["p"] == 0.0


def test_paired_bootstrap_refuses_unpaired_lists():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_rubrics_stay_in_range():
    assert exact_match_rubric(" Hi ", "Hi") == 1.0
    assert token_overlap_rubric("thank you", "thank you very much") == 1.0
    with pytest.raises(ValueError):
        GradingRubric(lambda e, a: 2.0).grade("x", "y")


def test_chrf_rubric_scores_identical_text_as_one():
    pytest.importorskip("sacrebleu")
    from scripts.prompt_benchmark import chrf_rubric
    assert chrf_rubric("Thank you for your help.", "Thank you for your help.") == pytest.approx(1.0)
    assert chrf_rubric("Thank you for your help.", "Goodbye.") < 0.3


def test_the_example_test_set_is_bosnian_and_complete():
    cases = evaluation.load_test_set()
    assert len(cases) == 10
    inputs = {case["input"] for case in cases}
    assert "Najdite bolnicu" not in inputs           # Slovene, not Bosnian
    assert "Nađite bolnicu" in inputs
    assert dict((c["input"], c["expected_output"]) for c in cases)["Dobar dan"] == "Good day"


def test_the_offline_benchmark_is_lilly_alone_and_every_claude_template_has_its_input():
    benchmark = evaluation.build_benchmark(evaluation.load_test_set(), with_claude=False)
    assert list(benchmark.variants) == ["lilly"]
    for template in evaluation.CLAUDE_VARIANTS.values():
        assert PromptVariant("v", template).prompt("Hvala.").count("Hvala.") == 1
