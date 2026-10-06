#!/usr/bin/env python3
"""Lilly's translator against Claude prompt variants, on one frozen test set.

Lilly's Bosnian -> English translator is MarianMT. It takes a sentence and no
instructions, so it is not a prompt variant: it enters as a fixed baseline whose
template is "{input}", and it sees exactly the Bosnian text each prompt wraps.
The prompt experiment is the Claude variants. Every output is graded with
sentence-level chrF2 against the same reference, and each Claude variant is
compared with Lilly by a paired bootstrap over the test sentences.

    python3 scripts/eval_translation_prompts.py              # Lilly only: offline, free
    python3 scripts/eval_translation_prompts.py --claude     # + six Claude variants: API calls, paid
    python3 scripts/eval_translation_prompts.py --claude --samples 3 --effort low --out results.json

The test set is examples/test_set_translation.csv: ten everyday phrases, a smoke
test of the harness and not a measurement. Ten sentences cannot separate two
systems, and the comparison prints its interval so nobody reads it as one. A
decision needs FLORES-200 (training/evaluate_app.py scores Lilly there) or a
held-out set of real sentences, frozen before the run, with the bar written down
first (training/PREREGISTRATION.md).
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prompt_benchmark import (  # noqa: E402
    CLAUDE_MODEL, GradingRubric, PromptBenchmark, PromptVariant, chrf_rubric,
    claude_executor, lilly_executor,
)

TEST_SET = ROOT / "examples" / "test_set_translation.csv"
BASELINE = "lilly"

# One change per variant against the first, so a difference can be attributed.
CLAUDE_VARIANTS = {
    "baseline": "Translate this Bosnian text to English:\n\n{input}",
    "role_context": (
        "You are a professional translator from Bosnian to English. "
        "Translate the following text, preserving meaning and register:\n\n{input}"),
    "output_only": (
        "Translate this Bosnian text to English. "
        "Reply with the translation only, no notes or quotation marks:\n\n{input}"),
    "constraints": (
        "Translate this Bosnian text to English. Keep names and numbers unchanged, "
        "do not add or drop content, and reply with the translation only:\n\n{input}"),
    "with_examples": (
        "Translate Bosnian to English. Reply with the translation only.\n\n"
        "Bosnian: Dobro jutro.\nEnglish: Good morning.\n\n"
        "Bosnian: Gdje je željeznička stanica?\nEnglish: Where is the railway station?\n\n"
        "Bosnian: {input}\nEnglish:"),
    "variety_note": (
        "The following text is Bosnian (ijekavian, Latin script), not Croatian or "
        "Serbian. Translate it to English and reply with the translation only:\n\n{input}"),
}


def load_test_set(path: Path = TEST_SET) -> list:
    """The frozen cases: rows of input, expected_output."""
    with path.open(encoding="utf-8", newline="") as fh:
        cases = [{"input": row["input"].strip(), "expected_output": row["expected_output"].strip()}
                 for row in csv.DictReader(fh)]
    if not cases:
        raise SystemExit(f"no test cases in {path}")
    return cases


def build_benchmark(cases: list, with_claude: bool, claude=None) -> PromptBenchmark:
    """Lilly as the baseline, plus the Claude variants when asked for."""
    variants = {BASELINE: PromptVariant(BASELINE, "{input}", executor=lilly_executor("bs-en"))}
    if with_claude:
        for name, template in CLAUDE_VARIANTS.items():
            variants[f"claude_{name}"] = PromptVariant(f"claude_{name}", template, executor=claude)
    return PromptBenchmark(
        name="lilly_vs_claude_prompts",
        variants=variants,
        test_cases=cases,
        metrics=["chrF2 (sentence)", "latency_ms", "cost_usd"],
        rubric=GradingRubric(chrf_rubric),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--claude", action="store_true",
                    help="add the Claude prompt variants (needs Anthropic credentials; costs money)")
    ap.add_argument("--model", default=CLAUDE_MODEL)
    ap.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--samples", type=int, default=1,
                    help="calls per variant per case; Lilly is deterministic, Claude is not")
    ap.add_argument("--test-set", type=Path, default=TEST_SET)
    ap.add_argument("--out", type=Path, default=None, help="write every measurement as JSON")
    args = ap.parse_args()

    cases = load_test_set(args.test_set)
    claude = None
    if args.claude:
        claude = claude_executor(model=args.model, effort=args.effort)
        print(f"Claude: {args.model}, effort {args.effort}, server-side fallback on")
    benchmark = build_benchmark(cases, args.claude, claude)
    benchmark.run(samples=args.samples)
    benchmark.print_comparison()
    if args.claude:
        print()
        for name in CLAUDE_VARIANTS:
            benchmark.compare(BASELINE, f"claude_{name}")
    if args.out:
        benchmark.export_json(str(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
