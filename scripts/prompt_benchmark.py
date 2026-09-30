#!/usr/bin/env python3
"""Practical prompt evaluation framework: define variants, run them, grade results.

Usage:
    # Define your test set (domain-specific)
    test_cases = [
        {"input": "Dobar dan", "expected_output": "Good morning"},
        {"input": "Kako ste?", "expected_output": "How are you?"},
    ]

    # Define prompt variants (each is a prompt template)
    variants = {
        "baseline": "Translate this Bosnian text to English: {input}",
        "role_context": "You are a professional translator. Translate this Bosnian to English: {input}",
        "reasoning": "Before translating, identify key words. Then translate: {input}",
    }

    # Run benchmark and grade
    benchmark = PromptBenchmark(
        name="translation_v1",
        variants=variants,
        test_cases=test_cases,
        metrics=["latency_ms", "accuracy", "cost_usd"],
    )

    results = benchmark.run(model="claude-opus-5-5", samples=3)
    benchmark.print_comparison()
    benchmark.export_json("results.json")
"""

import json
import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Callable, Any
from pathlib import Path


@dataclass
class MeasurementPoint:
    """Single measurement from one prompt variant on one test case."""
    variant_name: str
    test_case_id: str
    input_text: str
    expected_output: str
    actual_output: str
    latency_ms: float
    accuracy: float  # 0.0-1.0 (0=wrong, 1=perfect)
    cost_usd: float
    metadata: Dict[str, Any] = None


class PromptVariant:
    """One prompt template and its execution context."""

    def __init__(self, name: str, template: str, executor: Callable = None):
        """
        Args:
            name: identifier for this variant ("baseline", "role_context", etc)
            template: prompt template with {input} placeholder
            executor: function that runs the prompt and returns output
                     if None, you must call .run() with executor parameter
        """
        self.name = name
        self.template = template
        self.executor = executor

    def run(self, input_text: str, executor: Callable = None) -> tuple[str, float]:
        """Execute the prompt and return (output, latency_ms).

        Args:
            input_text: the test input to fill into the template
            executor: optional override of self.executor

        Returns:
            (output_text, latency_ms)
        """
        actual_executor = executor or self.executor
        if not actual_executor:
            raise ValueError(f"No executor for variant {self.name}")

        prompt = self.template.format(input=input_text)

        t0 = time.perf_counter()
        output = actual_executor(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        return output, latency_ms


class GradingRubric:
    """How to grade actual vs expected output (domain-specific)."""

    def __init__(self, compare_fn: Callable[[str, str], float]):
        """
        Args:
            compare_fn: function(expected: str, actual: str) -> float (0-1)
                       Returns accuracy score where 1.0 = perfect, 0.0 = wrong
        """
        self.compare_fn = compare_fn

    def grade(self, expected: str, actual: str) -> float:
        """Return accuracy 0.0-1.0."""
        return self.compare_fn(expected, actual)


def exact_match_rubric(expected: str, actual: str) -> float:
    """Simple rubric: 1.0 if exact match, 0.0 otherwise."""
    return 1.0 if expected.strip() == actual.strip() else 0.0


def token_overlap_rubric(expected: str, actual: str) -> float:
    """Rubric: measure token overlap (word matching)."""
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    if not expected_words:
        return 1.0 if not actual_words else 0.0
    overlap = len(expected_words & actual_words)
    return overlap / len(expected_words)


class PromptBenchmark:
    """Run and grade multiple prompt variants against a test set."""

    def __init__(
        self,
        name: str,
        variants: Dict[str, str],  # {name: template}
        test_cases: List[Dict[str, str]],  # [{input, expected_output}, ...]
        metrics: List[str] = None,  # ["latency_ms", "accuracy", "cost_usd"]
        rubric: GradingRubric = None,
    ):
        """
        Args:
            name: benchmark name (e.g., "translation_v1")
            variants: dict of {variant_name: prompt_template}
            test_cases: list of {input, expected_output} dicts
            metrics: which metrics to track
            rubric: GradingRubric for evaluating accuracy
        """
        self.name = name
        self.variants = {name: PromptVariant(name, template)
                        for name, template in variants.items()}
        self.test_cases = test_cases
        self.metrics = metrics or ["latency_ms", "accuracy", "cost_usd"]
        self.rubric = rubric or GradingRubric(exact_match_rubric)
        self.measurements: List[MeasurementPoint] = []

    def run(
        self,
        executor: Callable[[str], str],
        samples: int = 1,
        cost_per_call_usd: float = 0.0001,
    ) -> Dict[str, Dict[str, float]]:
        """Run all variants against all test cases.

        Args:
            executor: function(prompt: str) -> output_text
            samples: run each variant N times per test case (for averaging)
            cost_per_call_usd: cost per API call (for cost metric)

        Returns:
            {variant_name: {metric: value, ...}}
        """
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {self.name}")
        print(f"{'='*70}")
        print(f"Variants: {list(self.variants.keys())}")
        print(f"Test cases: {len(self.test_cases)}")
        print(f"Samples per variant: {samples}")
        print(f"Metrics: {self.metrics}\n")

        for variant_name, variant in self.variants.items():
            print(f"Running {variant_name}...")
            for test_idx, test_case in enumerate(self.test_cases):
                for sample in range(samples):
                    input_text = test_case["input"]
                    expected = test_case.get("expected_output", "")

                    output, latency_ms = variant.run(input_text, executor=executor)
                    accuracy = self.rubric.grade(expected, output)
                    cost_usd = cost_per_call_usd

                    m = MeasurementPoint(
                        variant_name=variant_name,
                        test_case_id=f"test_{test_idx}",
                        input_text=input_text,
                        expected_output=expected,
                        actual_output=output,
                        latency_ms=latency_ms,
                        accuracy=accuracy,
                        cost_usd=cost_usd,
                    )
                    self.measurements.append(m)

        # Aggregate results
        results = self._aggregate()
        return results

    def _aggregate(self) -> Dict[str, Dict[str, float]]:
        """Aggregate measurements into summary stats per variant."""
        by_variant = {}

        for m in self.measurements:
            if m.variant_name not in by_variant:
                by_variant[m.variant_name] = []
            by_variant[m.variant_name].append(m)

        results = {}
        for variant_name, measurements in by_variant.items():
            latencies = [m.latency_ms for m in measurements]
            accuracies = [m.accuracy for m in measurements]
            costs = [m.cost_usd for m in measurements]

            results[variant_name] = {
                "mean_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,

                "mean_accuracy": statistics.mean(accuracies),
                "median_accuracy": statistics.median(accuracies),
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),

                "total_cost_usd": sum(costs),
                "mean_cost_per_call_usd": statistics.mean(costs),

                "samples": len(measurements),
            }

        return results

    def print_comparison(self):
        """Print human-readable comparison table."""
        results = self._aggregate()

        print(f"\n{'='*70}")
        print(f"RESULTS: {self.name}")
        print(f"{'='*70}\n")

        # Table headers
        print(f"{'Variant':<20} {'Accuracy':<12} {'Latency (ms)':<20} {'Cost':<12}")
        print("-" * 70)

        for variant_name in sorted(results.keys()):
            r = results[variant_name]
            acc = f"{r['mean_accuracy']:.2%}"
            lat = f"{r['mean_latency_ms']:.1f}±{r['stdev_latency_ms']:.1f}"
            cost = f"${r['mean_cost_per_call_usd']:.6f}"
            print(f"{variant_name:<20} {acc:<12} {lat:<20} {cost:<12}")

        # Ranking
        print(f"\n{'='*70}")
        print("RANKINGS (by metric)")
        print(f"{'='*70}\n")

        sorted_by_acc = sorted(results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
        print("Accuracy (higher is better):")
        for i, (name, r) in enumerate(sorted_by_acc, 1):
            print(f"  {i}. {name:<20} {r['mean_accuracy']:.2%}")

        sorted_by_lat = sorted(results.items(), key=lambda x: x[1]['mean_latency_ms'])
        print("\nLatency (lower is better):")
        for i, (name, r) in enumerate(sorted_by_lat, 1):
            print(f"  {i}. {name:<20} {r['mean_latency_ms']:.1f}ms")

        sorted_by_cost = sorted(results.items(), key=lambda x: x[1]['mean_cost_per_call_usd'])
        print("\nCost per call (lower is better):")
        for i, (name, r) in enumerate(sorted_by_cost, 1):
            print(f"  {i}. {name:<20} ${r['mean_cost_per_call_usd']:.6f}")

    def export_json(self, filepath: str):
        """Export all measurements and aggregates to JSON."""
        results = self._aggregate()

        export = {
            "benchmark_name": self.name,
            "test_cases_count": len(self.test_cases),
            "variants": list(self.variants.keys()),
            "metrics": self.metrics,
            "aggregated_results": results,
            "raw_measurements": [asdict(m) for m in self.measurements],
        }

        Path(filepath).write_text(json.dumps(export, indent=2))
        print(f"\nExported results to {filepath}")

    def pareto_frontier(self) -> Dict[str, Any]:
        """Find non-dominated solutions on accuracy vs latency vs cost tradeoff.

        A solution is on the Pareto frontier if no other solution is better
        in ALL metrics simultaneously.
        """
        results = self._aggregate()
        variants_list = list(results.items())

        frontier = {}
        for name1, r1 in variants_list:
            dominated = False
            for name2, r2 in variants_list:
                if name1 == name2:
                    continue
                # r2 dominates r1 if:
                # - r2 accuracy >= r1 accuracy AND
                # - r2 latency <= r1 latency AND
                # - r2 cost <= r1 cost AND
                # - at least one is strictly better
                if (r2['mean_accuracy'] >= r1['mean_accuracy'] and
                    r2['mean_latency_ms'] <= r1['mean_latency_ms'] and
                    r2['mean_cost_per_call_usd'] <= r1['mean_cost_per_call_usd'] and
                    (r2['mean_accuracy'] > r1['mean_accuracy'] or
                     r2['mean_latency_ms'] < r1['mean_latency_ms'] or
                     r2['mean_cost_per_call_usd'] < r1['mean_cost_per_call_usd'])):
                    dominated = True
                    break

            if not dominated:
                frontier[name1] = r1

        print(f"\nPareto frontier ({len(frontier)} non-dominated solutions):")
        for name in frontier:
            r = frontier[name]
            print(f"  {name}: acc={r['mean_accuracy']:.2%}, "
                  f"lat={r['mean_latency_ms']:.1f}ms, "
                  f"cost=${r['mean_cost_per_call_usd']:.6f}")

        return frontier


if __name__ == "__main__":
    # Example: compare translation prompts
    # (In practice, you'd use a real LLM executor)

    def mock_executor(prompt: str) -> str:
        """Mock executor for testing. In real use, call Claude API."""
        # Simulates different outputs based on prompt content
        if "identify key words" in prompt:
            return "Hello"  # Slightly more careful
        return "Hi"  # Basic response

    variants = {
        "baseline": "Translate: {input}",
        "role": "You are a translator. Translate: {input}",
        "reasoning": "Identify key words first. Then translate: {input}",
    }

    test_cases = [
        {"input": "Dobar dan", "expected_output": "Good morning"},
        {"input": "Kako ste?", "expected_output": "How are you?"},
    ]

    benchmark = PromptBenchmark(
        name="translation_test",
        variants=variants,
        test_cases=test_cases,
    )

    # Run with mock executor
    results = benchmark.run(executor=mock_executor, samples=2)
    benchmark.print_comparison()
    benchmark.pareto_frontier()
    benchmark.export_json("/tmp/benchmark_results.json")
