# Prompt evaluation

`scripts/prompt_benchmark.py` runs several prompt variants, and fixed systems
beside them, on one frozen test set and compares them the way the rest of this
repository compares models: same cases for every variant, a score per case, and
a paired bootstrap before any difference is called real.

It replaces a first version (commit `4ebfde6`) whose example scored nothing: its
Lilly executor unpacked a tuple from `Engine.translate`, which returns a string,
and turned the error into the output text, so every variant was graded on
`[Error: too many values to unpack]`. It also cut the "input" out of the last
line of each prompt, which for three of the six templates was an instruction,
not Bosnian. `tests/test_prompt_benchmark.py` pins the contract that would have
caught both.

## The one thing to know about Lilly here

Lilly's translator is MarianMT, not an LLM. It translates whatever text it is
given, instructions included. So it is not a prompt variant: it enters as a
**fixed baseline** with the template `{input}`, scored on exactly the Bosnian
text the prompt variants wrap. The prompt experiment is the LLM variants.

## Run it

```bash
python3 scripts/eval_translation_prompts.py              # Lilly only: offline, free
python3 scripts/eval_translation_prompts.py --claude     # + six Claude variants: API calls, paid
python3 scripts/eval_translation_prompts.py --claude --samples 3 --effort low --out results.json
python3 scripts/prompt_cli.py                            # the same, interactively
python3 scripts/prompt_benchmark.py                      # self-check, no model
```

`--claude` needs the Anthropic SDK (`pip install anthropic`) and credentials in
the environment (`ANTHROPIC_API_KEY`, or an `ant auth login` profile); nothing
asks for a key on the keyboard. The default model is `claude-opus-5-5` at an
explicit effort (`medium` unless `--effort` says otherwise), with server-side
fallback on: a request the model declines is re-run on the model Anthropic
recommends for that refusal category, and a refusal that survives comes back
empty, scores 0 and is printed to stderr. Cost is estimated from the token usage
at Claude Opus 5.5 list prices.

## The test set is a smoke test

`examples/test_set_translation.csv` holds ten everyday phrases. Ten cases cannot
separate two systems, and `compare()` prints its interval so nobody reads it as
if it could. A decision needs FLORES-200 (`training/evaluate_app.py` scores Lilly
there) or a held-out set of real sentences, frozen before the run, with the bar
written in `training/PREREGISTRATION.md` first.

## In code

```python
from scripts.prompt_benchmark import (GradingRubric, PromptBenchmark, PromptVariant,
                                      chrf_rubric, claude_executor, lilly_executor)

benchmark = PromptBenchmark(
    name="translation_v1",
    variants={
        "lilly": PromptVariant("lilly", "{input}", executor=lilly_executor()),
        "plain": "Translate this Bosnian text to English:\n\n{input}",
        "output_only": "Translate this Bosnian text to English. "
                       "Reply with the translation only:\n\n{input}",
    },
    test_cases=[{"input": "Hvala na pomoći.", "expected_output": "Thank you for your help."}],
    rubric=GradingRubric(chrf_rubric),
)
benchmark.run(executor=claude_executor(effort="medium"), samples=3)
benchmark.print_comparison()
benchmark.compare("lilly", "output_only")   # paired bootstrap over the test cases
benchmark.export_json("results.json")
```

| Piece | Contract |
| --- | --- |
| `PromptVariant(name, template, executor=None)` | The template must contain `{input}`; it is substituted literally, so other braces (a JSON example) are safe. A variant's own executor is used before the shared one. |
| executor | `executor(prompt) -> str`, or `-> (str, cost_usd)` when it can price its call. One that raises stops the run: a failed call is never scored as a wrong answer. |
| `GradingRubric(fn)` | `fn(expected, actual) -> float` in [0, 1]; anything else is an error. Built in: `exact_match_rubric`, `token_overlap_rubric` (recall only, a smoke check), `chrf_rubric` (sentence-level chrF2, the metric this project decides translation on). |
| `run(executor, samples)` | Every variant on every case, `samples` calls each. Samples are averaged within a case before cases are averaged, so a sampled LLM is not counted as more evidence than it is. |
| `compare(a, b)` | Mean difference b − a over cases, 95% bootstrap interval and p, 2,000 draws, seed 11 — the convention of `training/speech_bench.paired_bootstrap`. Inside the interval is "no difference shown". |
| `pareto_frontier()` | A shortlist on mean accuracy, latency and (when every variant has one) cost. Not a decision; `compare()` is. |

What the self-check prints, for the shape of the output:

```
Variant                Accuracy   Latency (ms)       Cost/call
----------------------------------------------------------------------
plain                  100.00%    0.0±0.0            n/a
instructed             100.00%    0.0±0.0            n/a

A ranking is not a result: compare() two variants before choosing one.
instructed - plain: +0.0 points [+0.0, +0.0], p = 1.000, n = 2 cases -- inside the noise: no difference shown
```

## Rules that carry over from the rest of the project

- Freeze the test set and write the bar before the first run; never tune a
  variant on the cases that then judge it.
- One change per variant against the first, so a difference can be attributed.
- Report n beside every figure, and the interval beside every difference.
- An LLM grader is a model under test too: check it against human judgments on
  a sample before its scores decide anything.
