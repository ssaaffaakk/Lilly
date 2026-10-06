#!/usr/bin/env python3
"""Compare prompt variants -- and fixed systems -- on one frozen test set.

The rules are the repository's own (training/PREREGISTRATION.md): freeze the test
set and the bar before running; score every variant on the same cases; judge a
difference with a paired bootstrap over test cases, never on two means alone; and
report n beside every figure. Ten cases cannot separate two systems, and the
comparison says so rather than ranking them.

    from scripts.prompt_benchmark import (GradingRubric, PromptBenchmark,
                                          PromptVariant, chrf_rubric,
                                          claude_executor, lilly_executor)

    benchmark = PromptBenchmark(
        name="translation_v1",
        variants={
            "baseline": "Translate this Bosnian text to English: {input}",
            "lilly": PromptVariant("lilly", "{input}", executor=lilly_executor()),
        },
        test_cases=[{"input": "Hvala na pomoći.",
                     "expected_output": "Thank you for your help."}],
        rubric=GradingRubric(chrf_rubric),
    )
    benchmark.run(executor=claude_executor(), samples=3)
    benchmark.print_comparison()
    benchmark.compare("lilly", "baseline")

An executor takes the filled prompt and returns the output text, or a
(text, cost_usd) pair when it can price its own call. A variant with an executor
of its own uses it; the others use the one handed to run(). That is how a fixed
system enters the comparison: Lilly's translator is MarianMT, which takes a
sentence and no instructions, so it is not a prompt variant -- it is a baseline
with the template "{input}", scored on exactly the inputs the prompt variants see.

An executor that raises stops the run. A failed call scored as a wrong answer
would make a broken executor look like a bad prompt.
"""

import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

PLACEHOLDER = "{input}"

# Claude Opus 5.5 list prices per million tokens, for the cost column. An
# estimate: a request that falls back to another model is billed at that
# model's rates.
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_USD_PER_MTOK = (4.00, 20.00)  # input, output


@dataclass
class MeasurementPoint:
    """One variant, one test case, one sample."""
    variant_name: str
    test_case_id: str
    sample: int
    input_text: str
    expected_output: str
    actual_output: str
    latency_ms: float
    accuracy: float  # 0.0-1.0
    cost_usd: Optional[float]  # None when the executor cannot price its call
    metadata: Dict[str, Any] = field(default_factory=dict)


def _split_result(result) -> tuple:
    """An executor's return value as (text, cost_usd or None)."""
    if isinstance(result, tuple):
        text, cost = result
        return str(text), (None if cost is None else float(cost))
    if not isinstance(result, str):
        raise TypeError(f"an executor returns str or (str, cost_usd), not {type(result).__name__}")
    return result, None


class PromptVariant:
    """One prompt template, and optionally the executor that runs it."""

    def __init__(self, name: str, template: str, executor: Callable = None):
        # str.format would also read any other braces in the template -- a JSON
        # example, say -- so the placeholder is substituted literally instead,
        # and a template without it is refused here rather than sending every
        # case the same prompt.
        if PLACEHOLDER not in template:
            raise ValueError(f"variant {name!r}: the template has no {PLACEHOLDER} placeholder")
        self.name = name
        self.template = template
        self.executor = executor

    def prompt(self, input_text: str) -> str:
        return self.template.replace(PLACEHOLDER, input_text)

    def run(self, input_text: str, executor: Callable = None) -> tuple:
        """(output, latency_ms, cost_usd or None). Own executor first, else `executor`."""
        actual_executor = self.executor or executor
        if actual_executor is None:
            raise ValueError(f"no executor for variant {self.name!r}")
        prompt = self.prompt(input_text)
        t0 = time.perf_counter()
        result = actual_executor(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000
        output, cost = _split_result(result)
        return output, latency_ms, cost


class GradingRubric:
    """How to grade actual against expected output: a score in [0, 1]."""

    def __init__(self, compare_fn: Callable[[str, str], float]):
        self.compare_fn = compare_fn

    def grade(self, expected: str, actual: str) -> float:
        score = float(self.compare_fn(expected, actual))
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"a rubric returns a score in [0, 1], not {score}")
        return score


def exact_match_rubric(expected: str, actual: str) -> float:
    """1.0 when the stripped strings are identical, else 0.0."""
    return 1.0 if expected.strip() == actual.strip() else 0.0


def token_overlap_rubric(expected: str, actual: str) -> float:
    """Share of the reference's distinct lowercase words found in the output.

    Recall only: an output that repeats the reference and adds a paragraph scores
    1.0. A smoke check, not a translation metric -- use chrf_rubric for that.
    """
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    if not expected_words:
        return 1.0 if not actual_words else 0.0
    return len(expected_words & actual_words) / len(expected_words)


def chrf_rubric(expected: str, actual: str) -> float:
    """Sentence-level chrF2 / 100 -- the metric this project decides translation on.

    Character n-grams give partial credit for a wrong inflection, which word
    overlap does not, and recall weighs twice precision. Sentence-level chrF is
    noisy on one short phrase; the corpus figures in training/ are what decide a
    model, this is for ranking prompts on the same cases.
    """
    try:
        from sacrebleu.metrics import CHRF
    except ImportError as exc:
        raise RuntimeError("chrf_rubric needs sacrebleu: pip install sacrebleu") from exc
    return CHRF(word_order=0).sentence_score(actual.strip(), [expected.strip()]).score / 100.0


def lilly_executor(direction: str = "bs-en") -> Callable[[str], str]:
    """Lilly's served translator as an executor.

    It translates whatever text it is handed, instructions included, so give its
    variant the template "{input}". The engine loads on the first call.
    """
    def run(prompt: str) -> str:
        from app.translate import get_engine
        return get_engine(direction).translate(prompt)
    return run


def claude_executor(model: str = CLAUDE_MODEL, effort: str = "medium",
                    max_tokens: int = 16000) -> Callable[[str], tuple]:
    """Claude through the Anthropic SDK, returning (text, estimated cost in USD).

    Credentials come from the environment (ANTHROPIC_API_KEY, or an
    `ant auth login` profile); nothing is read from the keyboard. Effort is set
    explicitly so every variant runs at the same depth. Server-side fallback is
    on: a request the model declines is re-run by the API on the model Anthropic
    recommends for that refusal category. A refusal that survives the fallback
    comes back as an empty output, scores 0, and is reported on stderr.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("claude_executor needs the Anthropic SDK: pip install anthropic") from exc
    client = anthropic.Anthropic()
    usd_in, usd_out = CLAUDE_USD_PER_MTOK

    def run(prompt: str) -> tuple:
        response = client.beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage
        cost = (usage.input_tokens * usd_in + usage.output_tokens * usd_out) / 1_000_000
        if response.stop_reason == "refusal":
            import sys
            print(f"refusal ({model}): {prompt[:60]!r}", file=sys.stderr)
            return "", cost
        text = "".join(block.text for block in response.content if block.type == "text")
        return text.strip(), cost

    return run


def paired_bootstrap(scores_a: List[float], scores_b: List[float],
                     draws: int = 2000, seed: int = 11) -> Dict[str, float]:
    """b - a over the same test cases, resampling cases.

    Same convention as training/speech_bench.paired_bootstrap: p is the share of
    resamples on the other side of zero from the observed difference, or on it.
    """
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("paired scores need the same, non-empty, list of cases")
    n = len(scores_a)
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    observed = statistics.mean(diffs)
    rng = random.Random(seed)
    resampled, against = [], 0
    for _ in range(draws):
        delta = statistics.mean(diffs[rng.randrange(n)] for _ in range(n))
        resampled.append(delta)
        if (delta <= 0) if observed > 0 else (delta >= 0):
            against += 1
    resampled.sort()
    return {"n": n, "delta": observed,
            "low": resampled[int(0.025 * draws)],
            "high": resampled[min(draws - 1, int(0.975 * draws))],
            "p": against / draws}


class PromptBenchmark:
    """Run every variant on every test case and grade the outputs."""

    def __init__(
        self,
        name: str,
        variants: Dict[str, Union[str, PromptVariant]],
        test_cases: List[Dict[str, str]],
        metrics: List[str] = None,
        rubric: GradingRubric = None,
    ):
        """
        Args:
            name: benchmark name, e.g. "translation_v1"
            variants: {name: template} or {name: PromptVariant}; a PromptVariant
                may carry its own executor
            test_cases: [{"input": ..., "expected_output": ...}, ...], frozen
            metrics: recorded in the export, for the reader
            rubric: how to grade; exact match unless given
        """
        if not variants:
            raise ValueError("no variants to compare")
        if not test_cases:
            raise ValueError("no test cases")
        for i, case in enumerate(test_cases):
            if "input" not in case or "expected_output" not in case:
                raise ValueError(f"test case {i} needs 'input' and 'expected_output'")
        self.name = name
        self.variants = {
            key: value if isinstance(value, PromptVariant) else PromptVariant(key, value)
            for key, value in variants.items()
        }
        self.test_cases = test_cases
        self.metrics = metrics or ["accuracy", "latency_ms", "cost_usd"]
        self.rubric = rubric or GradingRubric(exact_match_rubric)
        self.measurements: List[MeasurementPoint] = []

    def run(
        self,
        executor: Callable = None,
        samples: int = 1,
        cost_per_call_usd: float = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all variants against all test cases, `samples` times each.

        Args:
            executor: used by every variant that has no executor of its own
            samples: calls per variant per case; above 1 for a sampled LLM,
                1 for a deterministic system
            cost_per_call_usd: a flat price for executors that return no cost

        Returns:
            {variant_name: {metric: value, ...}}
        """
        if samples < 1:
            raise ValueError("samples must be at least 1")
        self.measurements = []
        print(f"\n{'=' * 70}\nBENCHMARK: {self.name}\n{'=' * 70}")
        print(f"Variants: {list(self.variants)}")
        print(f"Test cases: {len(self.test_cases)}, samples per case: {samples}\n")

        for variant_name, variant in self.variants.items():
            print(f"Running {variant_name}...")
            for test_idx, test_case in enumerate(self.test_cases):
                input_text = test_case["input"]
                expected = test_case["expected_output"]
                for sample in range(samples):
                    output, latency_ms, cost = variant.run(input_text, executor=executor)
                    if cost is None:
                        cost = cost_per_call_usd
                    self.measurements.append(MeasurementPoint(
                        variant_name=variant_name,
                        test_case_id=f"test_{test_idx}",
                        sample=sample,
                        input_text=input_text,
                        expected_output=expected,
                        actual_output=output,
                        latency_ms=latency_ms,
                        accuracy=self.rubric.grade(expected, output),
                        cost_usd=cost,
                    ))
        return self._aggregate()

    def case_scores(self, variant_name: str) -> List[float]:
        """Mean accuracy per test case, in test-set order, samples averaged."""
        if variant_name not in self.variants:
            raise KeyError(f"no variant {variant_name!r}")
        by_case: Dict[str, List[float]] = {}
        for m in self.measurements:
            if m.variant_name == variant_name:
                by_case.setdefault(m.test_case_id, []).append(m.accuracy)
        if len(by_case) != len(self.test_cases):
            raise RuntimeError(f"{variant_name!r} has not been run on every test case")
        return [statistics.mean(by_case[f"test_{i}"]) for i in range(len(self.test_cases))]

    def _aggregate(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        for variant_name in self.variants:
            points = [m for m in self.measurements if m.variant_name == variant_name]
            if not points:
                continue
            latencies = [m.latency_ms for m in points]
            case_means = self.case_scores(variant_name)
            costs = [m.cost_usd for m in points]
            cost_known = all(c is not None for c in costs)
            results[variant_name] = {
                "cases": len(case_means),
                "samples": len(points),
                "mean_accuracy": statistics.mean(case_means),
                "min_case_accuracy": min(case_means),
                "max_case_accuracy": max(case_means),
                "mean_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "total_cost_usd": sum(costs) if cost_known else None,
                "mean_cost_per_call_usd": statistics.mean(costs) if cost_known else None,
            }
        return results

    def compare(self, variant_a: str, variant_b: str, draws: int = 2000,
                seed: int = 11) -> Dict[str, float]:
        """Paired bootstrap of b against a over the test cases, printed and returned."""
        result = paired_bootstrap(self.case_scores(variant_a), self.case_scores(variant_b),
                                  draws=draws, seed=seed)
        verdict = ("the 95% interval excludes zero" if result["low"] > 0 or result["high"] < 0
                   else "inside the noise: no difference shown")
        print(f"{variant_b} - {variant_a}: {result['delta'] * 100:+.1f} points "
              f"[{result['low'] * 100:+.1f}, {result['high'] * 100:+.1f}], "
              f"p = {result['p']:.3f}, n = {result['n']} cases -- {verdict}")
        return result

    def print_comparison(self):
        """Human-readable table, accuracy as mean over cases."""
        results = self._aggregate()
        print(f"\n{'=' * 70}\nRESULTS: {self.name}  (n = {len(self.test_cases)} cases)\n{'=' * 70}\n")
        print(f"{'Variant':<22} {'Accuracy':<10} {'Latency (ms)':<18} {'Cost/call':<12}")
        print("-" * 70)
        for variant_name, r in sorted(results.items(), key=lambda x: -x[1]["mean_accuracy"]):
            lat = f"{r['mean_latency_ms']:.1f}±{r['stdev_latency_ms']:.1f}"
            cost = ("n/a" if r["mean_cost_per_call_usd"] is None
                    else f"${r['mean_cost_per_call_usd']:.6f}")
            print(f"{variant_name:<22} {r['mean_accuracy']:<10.2%} {lat:<18} {cost:<12}")
        print("\nA ranking is not a result: compare() two variants before choosing one.")

    def export_json(self, filepath: str):
        """All measurements and the aggregates, as JSON."""
        export = {
            "benchmark_name": self.name,
            "test_cases_count": len(self.test_cases),
            "variants": {k: v.template for k, v in self.variants.items()},
            "metrics": self.metrics,
            "aggregated_results": self._aggregate(),
            "raw_measurements": [asdict(m) for m in self.measurements],
        }
        Path(filepath).write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nExported results to {filepath}")

    def pareto_frontier(self) -> Dict[str, Any]:
        """Variants no other variant beats on accuracy, latency and cost at once.

        Cost joins the comparison only when every variant has one. A frontier
        built on means alone is a shortlist, not a decision; compare() decides.
        """
        results = self._aggregate()
        with_cost = all(r["mean_cost_per_call_usd"] is not None for r in results.values())

        def keys(r):
            values = [-r["mean_accuracy"], r["mean_latency_ms"]]
            if with_cost:
                values.append(r["mean_cost_per_call_usd"])
            return values

        frontier = {}
        for name1, r1 in results.items():
            k1 = keys(r1)
            dominated = any(
                all(b <= a for a, b in zip(k1, keys(r2))) and keys(r2) != k1
                for name2, r2 in results.items() if name2 != name1
            )
            if not dominated:
                frontier[name1] = r1

        print(f"\nPareto frontier ({len(frontier)} non-dominated"
              f"{'' if with_cost else ', cost left out: not every variant has one'}):")
        for name, r in frontier.items():
            print(f"  {name}: acc={r['mean_accuracy']:.2%}, lat={r['mean_latency_ms']:.1f}ms")
        return frontier


if __name__ == "__main__":
    # A self-check with a stand-in executor: no model, no network.
    def stand_in(prompt: str) -> str:
        return "Thank you." if "Hvala" in prompt else "Good day."

    demo = PromptBenchmark(
        name="self_check",
        variants={"plain": "{input}", "instructed": "Translate to English: {input}"},
        test_cases=[{"input": "Hvala.", "expected_output": "Thank you."},
                    {"input": "Dobar dan.", "expected_output": "Good day."}],
    )
    demo.run(executor=stand_in)
    demo.print_comparison()
    demo.compare("plain", "instructed")
