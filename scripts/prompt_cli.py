#!/usr/bin/env python3
"""Interactive CLI for prompt evaluation. Test variants without writing code.

Usage:
    python3 scripts/prompt_cli.py

Then follow the prompts to:
    1. Load test cases from CSV/JSON
    2. Enter prompt variants
    3. Run benchmark
    4. View results
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Callable

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prompt_benchmark import (
    PromptBenchmark, GradingRubric, exact_match_rubric, token_overlap_rubric
)


class InteractivePromptEvaluator:
    """Interactive CLI for prompt benchmarking."""

    def __init__(self):
        self.test_cases: List[Dict] = []
        self.variants: Dict[str, str] = {}
        self.rubric: GradingRubric = None
        self.executor: Callable = None

    def load_test_set(self) -> bool:
        """Load test cases from file or manual entry."""
        print("\n" + "="*70)
        print("STEP 1: Load Test Cases")
        print("="*70)
        print("\nOptions:")
        print("  1. Load from CSV file")
        print("  2. Load from JSON file")
        print("  3. Enter manually")

        choice = input("\nChoose (1-3): ").strip()

        if choice == "1":
            return self._load_csv()
        elif choice == "2":
            return self._load_json()
        elif choice == "3":
            return self._load_manual()
        else:
            print("Invalid choice")
            return False

    def _load_csv(self) -> bool:
        """Load test cases from CSV (columns: input, expected_output)."""
        filepath = input("Path to CSV file: ").strip()
        if not Path(filepath).exists():
            print(f"File not found: {filepath}")
            return False

        try:
            with open(filepath) as f:
                reader = csv.DictReader(f)
                self.test_cases = list(reader)
            print(f"Loaded {len(self.test_cases)} test cases from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    def _load_json(self) -> bool:
        """Load test cases from JSON."""
        filepath = input("Path to JSON file: ").strip()
        if not Path(filepath).exists():
            print(f"File not found: {filepath}")
            return False

        try:
            with open(filepath) as f:
                self.test_cases = json.load(f)
            print(f"Loaded {len(self.test_cases)} test cases from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return False

    def _load_manual(self) -> bool:
        """Enter test cases manually."""
        count = int(input("How many test cases? "))

        for i in range(count):
            print(f"\nTest case {i+1}:")
            input_text = input("  Input: ").strip()
            expected = input("  Expected output: ").strip()

            self.test_cases.append({
                "input": input_text,
                "expected_output": expected,
            })

        print(f"\nLoaded {len(self.test_cases)} test cases")
        return True

    def define_variants(self) -> bool:
        """Define prompt variants interactively."""
        print("\n" + "="*70)
        print("STEP 2: Define Prompt Variants")
        print("="*70)
        print("\nEnter prompt variants. Use {input} as placeholder for test input.")
        print("Example: 'Translate to English: {input}'")
        print("Leave name blank when done.\n")

        while True:
            name = input("Variant name (or press Enter to continue): ").strip()
            if not name:
                break

            template = input(f"  Prompt template for '{name}': ").strip()
            self.variants[name] = template

        if not self.variants:
            print("No variants defined")
            return False

        print(f"\nDefined {len(self.variants)} variants:")
        for name in self.variants:
            print(f"  - {name}")

        return True

    def choose_rubric(self) -> bool:
        """Choose how to grade accuracy."""
        print("\n" + "="*70)
        print("STEP 3: Choose Grading Rubric (Accuracy Metric)")
        print("="*70)
        print("\nOptions:")
        print("  1. Exact match (strictest)")
        print("  2. Token/word overlap (more lenient)")
        print("  3. Custom function")

        choice = input("\nChoose (1-3): ").strip()

        if choice == "1":
            self.rubric = GradingRubric(exact_match_rubric)
            print("Using exact match rubric")
            return True
        elif choice == "2":
            self.rubric = GradingRubric(token_overlap_rubric)
            print("Using token overlap rubric")
            return True
        elif choice == "3":
            return self._create_custom_rubric()
        else:
            print("Invalid choice")
            return False

    def _create_custom_rubric(self) -> bool:
        """Create a custom grading function."""
        print("\nEnter Python function to grade outputs.")
        print("Template:")
        print("  def grade(expected: str, actual: str) -> float:")
        print("      # Return 0.0-1.0 where 1.0 = perfect")
        print("      return 1.0 if expected == actual else 0.0")
        print("\nEnter your function (end with blank line):")

        lines = []
        while True:
            line = input()
            if not line and lines:
                break
            lines.append(line)

        code = "\n".join(lines)

        try:
            # Execute the function definition
            namespace = {}
            exec(code, namespace)
            grade_fn = namespace.get("grade")
            if not grade_fn:
                print("Function 'grade' not found")
                return False

            self.rubric = GradingRubric(grade_fn)
            print("Custom rubric loaded")
            return True
        except Exception as e:
            print(f"Error in custom rubric: {e}")
            return False

    def choose_executor(self) -> bool:
        """Choose how to run prompts."""
        print("\n" + "="*70)
        print("STEP 4: Choose Executor (How to Run Prompts)")
        print("="*70)
        print("\nOptions:")
        print("  1. Mock/test (returns dummy outputs)")
        print("  2. Lilly translator (bs-en)")
        print("  3. Claude API (claude-opus-5-5)")
        print("  4. Custom function")

        choice = input("\nChoose (1-4): ").strip()

        if choice == "1":
            self.executor = self._mock_executor
            print("Using mock executor")
            return True
        elif choice == "2":
            return self._setup_lilly_executor()
        elif choice == "3":
            return self._setup_claude_executor()
        elif choice == "4":
            return self._setup_custom_executor()
        else:
            print("Invalid choice")
            return False

    def _mock_executor(self, prompt: str) -> str:
        """Mock executor for testing."""
        return f"[Mock response to: {prompt[:50]}...]"

    def _setup_lilly_executor(self) -> bool:
        """Setup Lilly translator."""
        try:
            from app.translate import get_engine
            engine = get_engine("bs-en")

            def lilly_executor(prompt: str) -> str:
                # Extract the Bosnian text from prompt
                lines = prompt.split('\n')
                bosnian_text = lines[-1].strip()

                # Try to translate
                output, diag = engine.translate(bosnian_text)
                return output

            self.executor = lilly_executor
            print("Lilly translator loaded")
            return True
        except Exception as e:
            print(f"Error loading Lilly: {e}")
            return False

    def _setup_claude_executor(self) -> bool:
        """Setup Claude API."""
        api_key = input("Enter your ANTHROPIC_API_KEY (or leave blank to use env): ").strip()

        try:
            from anthropic import Anthropic

            if api_key:
                client = Anthropic(api_key=api_key)
            else:
                client = Anthropic()  # Uses ANTHROPIC_API_KEY env var

            def claude_executor(prompt: str) -> str:
                response = client.messages.create(
                    model="claude-opus-5-5",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            self.executor = claude_executor
            print("Claude API configured")
            return True
        except Exception as e:
            print(f"Error setting up Claude: {e}")
            return False

    def _setup_custom_executor(self) -> bool:
        """Setup custom executor function."""
        print("\nEnter Python function to execute prompts.")
        print("Template:")
        print("  def executor(prompt: str) -> str:")
        print("      # Call your LLM, model, API, etc.")
        print("      return output")
        print("\nEnter your function (end with blank line):")

        lines = []
        while True:
            line = input()
            if not line and lines:
                break
            lines.append(line)

        code = "\n".join(lines)

        try:
            namespace = {}
            exec(code, namespace)
            executor_fn = namespace.get("executor")
            if not executor_fn:
                print("Function 'executor' not found")
                return False

            self.executor = executor_fn
            print("Custom executor loaded")
            return True
        except Exception as e:
            print(f"Error in custom executor: {e}")
            return False

    def run_benchmark(self) -> bool:
        """Run the benchmark."""
        print("\n" + "="*70)
        print("STEP 5: Run Benchmark")
        print("="*70)

        samples = int(input("\nHow many samples per variant? (1-5, default 2): ") or "2")
        samples = max(1, min(5, samples))

        benchmark = PromptBenchmark(
            name="interactive_benchmark",
            variants=self.variants,
            test_cases=self.test_cases,
            rubric=self.rubric,
        )

        print(f"\nRunning {len(self.variants)} variants × {len(self.test_cases)} test cases × {samples} samples...")
        print("This may take a while...\n")

        try:
            results = benchmark.run(
                executor=self.executor,
                samples=samples,
                cost_per_call_usd=0.0001,
            )

            benchmark.print_comparison()
            benchmark.pareto_frontier()

            # Export results
            export_path = "prompt_benchmark_results.json"
            benchmark.export_json(export_path)

            print(f"\n✓ Results exported to {export_path}")
            return True

        except Exception as e:
            print(f"✗ Error during benchmark: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run(self) -> int:
        """Run the interactive evaluator."""
        print("\n" + "="*70)
        print("LILLY PROMPT EVALUATION CLI")
        print("="*70)

        if not self.load_test_set():
            return 1

        if not self.define_variants():
            return 1

        if not self.choose_rubric():
            return 1

        if not self.choose_executor():
            return 1

        if not self.run_benchmark():
            return 1

        print("\n" + "="*70)
        print("✓ BENCHMARK COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("1. Review results in 'prompt_benchmark_results.json'")
        print("2. Choose best variant based on your priorities")
        print("3. Deploy to production")
        print("4. Monitor performance")

        return 0


def main():
    evaluator = InteractivePromptEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    sys.exit(main())
