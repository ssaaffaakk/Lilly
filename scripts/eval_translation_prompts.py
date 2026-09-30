#!/usr/bin/env python3
"""Example: Evaluate different translation prompt variants on Lilly.

This script defines multiple prompt approaches, runs them through your test set,
and grades them systematically. Use this as a template for your own benchmarks.

Usage:
    python3 scripts/eval_translation_prompts.py
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prompt_benchmark import (
    PromptBenchmark, GradingRubric, token_overlap_rubric
)
from app.translate import get_engine


# ============================================================================
# DEFINE YOUR TEST SET (domain-specific examples)
# ============================================================================

TRANSLATION_TEST_SET = [
    {
        "input": "Dobar dan, kako ste?",
        "expected_output": "Good morning, how are you?",
    },
    {
        "input": "Gdje je autobuska stanica?",
        "expected_output": "Where is the bus station?",
    },
    {
        "input": "Hvala na pomoći.",
        "expected_output": "Thank you for your help.",
    },
    {
        "input": "Koja je temperatura vani?",
        "expected_output": "What is the temperature outside?",
    },
    {
        "input": "Ne razumijem.",
        "expected_output": "I don't understand.",
    },
]


# ============================================================================
# DEFINE PROMPT VARIANTS (different approaches to the same task)
# ============================================================================

PROMPT_VARIANTS = {
    # Variant 1: Minimal baseline
    "baseline": "Translate to English: {input}",

    # Variant 2: Role context - tell the model what role to take
    "role_context": (
        "You are a professional translator specializing in South Slavic languages. "
        "Translate the following Bosnian to English, preserving meaning and nuance:\n\n{input}"
    ),

    # Variant 3: Task clarity - be very explicit about what you want
    "explicit_task": (
        "Task: Translate the following Bosnian sentence to natural English.\n"
        "Input: {input}\n"
        "Output (translation only, no explanation):"
    ),

    # Variant 4: Constraints - tell the model what NOT to do
    "with_constraints": (
        "Translate this Bosnian text to English. Do not add explanations, "
        "do not change the meaning, do not expand or contract the text:\n\n{input}"
    ),

    # Variant 5: Examples - show examples of good translations
    "with_examples": (
        "Examples:\n"
        "- Bosnian: Dobar dan\n  English: Good morning\n"
        "- Bosnian: Hvala\n  English: Thank you\n\n"
        "Now translate this:\n{input}"
    ),

    # Variant 6: Chain-of-thought - ask model to reason first
    "chain_of_thought": (
        "Translate this Bosnian to English. First, identify key words and grammar. "
        "Then produce the translation.\n\nInput: {input}\n\nTranslation:"
    ),
}


# ============================================================================
# DEFINE GRADING RUBRIC (how to score accuracy)
# ============================================================================

def translation_rubric(expected: str, actual: str) -> float:
    """Grade translation accuracy based on token/word overlap.

    This is a simple heuristic. In real use, you might:
    - Use a semantic similarity model (sentence-transformers)
    - Call an LLM to grade ("is this a valid translation?")
    - Use BLEU or chrF2 scores
    - Manual human review (gold standard)
    """
    # Normalize: lowercase, strip whitespace
    exp = expected.lower().strip()
    act = actual.lower().strip()

    # Exact match is best
    if exp == act:
        return 1.0

    # Otherwise use word overlap
    return token_overlap_rubric(exp, act)


# ============================================================================
# CREATE EXECUTOR (how to run each prompt)
# ============================================================================

def executor_with_lilly(prompt: str) -> str:
    """Execute prompt using Lilly's translation engine.

    In a real scenario, you'd:
    - Call Claude API: anthropic.Anthropic().messages.create(...)
    - Call a local LLM: ollama, llama.cpp, etc.
    - Call any other API

    For this example, we're using Lilly's existing engine to see how
    different prompt phrasings affect its behavior (if at all).
    """
    # Extract just the input from the prompt (assume it's at the end after "Input:")
    lines = prompt.split('\n')
    input_text = lines[-1].strip()

    if input_text.startswith('{input}'):
        # Some prompts might not have been formatted
        # Fall back to guessing the Bosnian text
        for line in reversed(lines):
            if any(c in line for c in 'čćđšž'):  # Bosnian special chars
                input_text = line.strip()
                break

    try:
        engine = get_engine("bs-en")
        output, diag = engine.translate(input_text)
        return output
    except Exception as e:
        return f"[Error: {e}]"


# ============================================================================
# RUN THE BENCHMARK
# ============================================================================

def main():
    print("\n" + "="*70)
    print("LILLY PROMPT EVALUATION")
    print("="*70)
    print(f"\nEvaluating {len(PROMPT_VARIANTS)} prompt variants")
    print(f"Against {len(TRANSLATION_TEST_SET)} test cases")
    print(f"Grading by: token overlap rubric")

    # Create benchmark
    benchmark = PromptBenchmark(
        name="lilly_translation_prompts_v1",
        variants=PROMPT_VARIANTS,
        test_cases=TRANSLATION_TEST_SET,
        metrics=["latency_ms", "accuracy", "cost_usd"],
        rubric=GradingRubric(translation_rubric),
    )

    # Run evaluation
    # Note: Lilly is deterministic, so samples=1 is fine
    # For LLM evaluation, use samples=3+ to capture variance
    results = benchmark.run(
        executor=executor_with_lilly,
        samples=1,
        cost_per_call_usd=0.0001,  # Adjust based on your model pricing
    )

    # Print results
    benchmark.print_comparison()

    # Find non-dominated solutions
    frontier = benchmark.pareto_frontier()

    # Export for further analysis
    export_path = "scripts/prompt_eval_results.json"
    benchmark.export_json(export_path)

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"1. Review detailed results in: {export_path}")
    print(f"2. Select best variant based on your priorities:")
    print(f"   - Highest accuracy: use 'accuracy' ranking")
    print(f"   - Fastest response: use 'latency' ranking")
    print(f"   - Best tradeoff: check Pareto frontier")
    print(f"3. Deploy winning variant to production")
    print(f"4. Monitor performance in real usage")
    print(f"5. Re-run benchmark periodically to catch regressions")

    return 0


if __name__ == "__main__":
    sys.exit(main())
