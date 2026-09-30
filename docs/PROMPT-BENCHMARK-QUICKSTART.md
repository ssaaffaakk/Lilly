# Prompt Evaluation Quickstart

Three ways to benchmark your prompts. Pick one based on your comfort level.

## 1. Interactive CLI (No Coding Required) ⭐ START HERE

```bash
python3 scripts/prompt_cli.py
```

Walks you through:
1. Load test cases (CSV, JSON, or manual entry)
2. Define prompt variants
3. Choose grading metric
4. Choose executor (mock, Claude, Lilly, custom)
5. Run and view results

**Takes 10 minutes. No Python knowledge required.**

### Example: Test 3 translation prompts

```
STEP 1: Load test cases
  → Choose "3. Enter manually"
  → Enter "Dobar dan" / "Good morning"
  → Enter "Kako ste?" / "How are you?"
  → (2 test cases)

STEP 2: Define variants
  → Name: baseline
    Template: Translate to English: {input}
  → Name: with_role
    Template: You are a professional translator. Translate: {input}
  → Name: reasoning
    Template: First list key words, then translate: {input}

STEP 3: Choose rubric
  → 1 (exact match)

STEP 4: Choose executor
  → 2 (Lilly translator)

STEP 5: Run
  → Samples: 1
  → Benchmark runs...
  → Results printed
  → JSON exported to prompt_benchmark_results.json
```

## 2. Python Script (Copy-Paste Template)

Create `my_benchmark.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.prompt_benchmark import (
    PromptBenchmark, GradingRubric, token_overlap_rubric
)
from anthropic import Anthropic

# Test set
test_cases = [
    {"input": "Dobar dan", "expected_output": "Good morning"},
    {"input": "Kako ste?", "expected_output": "How are you?"},
]

# Prompt variants
variants = {
    "baseline": "Translate to English: {input}",
    "with_role": "You are a translator. Translate: {input}",
    "reasoning": "First identify key words, then translate: {input}",
}

# Grading function
def grade_fn(expected: str, actual: str) -> float:
    return 1.0 if expected.lower() == actual.lower() else 0.8 * (
        len(set(expected.lower().split()) & set(actual.lower().split())) /
        max(len(set(expected.lower().split())), 1)
    )

# Executor (Claude API)
client = Anthropic()

def executor(prompt: str) -> str:
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Run benchmark
benchmark = PromptBenchmark(
    name="translation_v1",
    variants=variants,
    test_cases=test_cases,
    rubric=GradingRubric(grade_fn),
)

results = benchmark.run(executor=executor, samples=2)

# View results
benchmark.print_comparison()
benchmark.pareto_frontier()
benchmark.export_json("results.json")
```

Run it:

```bash
python3 my_benchmark.py
```

## 3. Full Custom Framework (Complete Control)

See `docs/PROMPT-EVALUATION.md` for advanced topics:
- Multi-objective optimization
- Statistical significance testing
- Production monitoring
- LLM-based grading

## Common Recipes

### Compare Claude vs Lilly vs Another API

```python
def claude_executor(prompt: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def lilly_executor(prompt: str) -> str:
    from app.translate import get_engine
    engine = get_engine("bs-en")
    output, _ = engine.translate(prompt)
    return output

# Run benchmark with both
benchmark.run(executor=claude_executor, samples=2)
# Then run again with lilly_executor and compare JSON results
```

### Grade Using Another LLM

```python
def llm_rubric(expected: str, actual: str) -> float:
    """Have Claude judge if translation is good."""
    from anthropic import Anthropic
    client = Anthropic()
    
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": f"""
            Reference translation: {expected}
            Model's translation: {actual}
            
            Is the model's translation correct? (0-10)
            """
        }]
    )
    score = int(response.content[0].text.strip())
    return score / 10.0

rubric = GradingRubric(llm_rubric)
```

### Load Test Set from Files

```python
import json

# From JSON
test_cases = json.loads(Path("test_set.json").read_text())

# From CSV
import csv
with open("test_set.csv") as f:
    test_cases = list(csv.DictReader(f))
```

Sample files are in `examples/`:
- `examples/test_set_translation.csv`
- `examples/test_set_translation.json`

## Running the Translation Example

The repo includes a complete working example:

```bash
# Run the translation prompt benchmark
python3 scripts/eval_translation_prompts.py

# View results
cat scripts/prompt_eval_results.json | python3 -m json.tool
```

This evaluates 6 different translation prompts against 5 test cases.

## What You Get

After running a benchmark, you'll have:

1. **Console output** showing rankings
   ```
   Variant              Accuracy     Latency (ms)
   reasoning            85.60%       52.3±4.1
   with_role            82.30%       47.8±1.9
   ```

2. **Pareto frontier** showing best tradeoffs
   ```
   Pareto frontier (2 non-dominated solutions):
     reasoning: acc=85.60%, lat=52.3ms
     constrained: acc=79.10%, lat=43.1ms
   ```

3. **JSON export** with all raw data
   ```json
   {
     "benchmark_name": "translation_v1",
     "aggregated_results": {
       "reasoning": {
         "mean_accuracy": 0.856,
         "mean_latency_ms": 52.3,
         ...
       }
     },
     "raw_measurements": [...]
   }
   ```

## Next Steps

1. **Try the CLI** (fastest way to learn)
   ```bash
   python3 scripts/prompt_cli.py
   ```

2. **Copy a Python template** and customize for your domain

3. **Read** `docs/PROMPT-EVALUATION.md` for advanced techniques

4. **Deploy** the winning prompt to production

5. **Monitor** real-world performance and re-benchmark periodically

## Architecture

```
PromptVariant
  └─ One prompt template + executor
  
GradingRubric
  └─ Function to score accuracy
  
PromptBenchmark
  ├─ Runs all variants on all test cases
  ├─ Collects measurements (latency, accuracy, cost)
  ├─ Aggregates to statistics (mean, stdev, p50, p95, p99)
  ├─ Ranks variants by each metric
  ├─ Computes Pareto frontier (non-dominated solutions)
  └─ Exports results to JSON
```

## API Reference

### PromptBenchmark

```python
benchmark = PromptBenchmark(
    name="my_benchmark",
    variants={"v1": "template 1", "v2": "template 2"},
    test_cases=[{"input": "...", "expected_output": "..."}, ...],
    rubric=GradingRubric(my_grade_fn),
)

# Run benchmark
results = benchmark.run(
    executor=my_executor_fn,  # (prompt: str) -> str
    samples=3,                 # samples per test case
    cost_per_call_usd=0.0001,  # for cost metric
)

# View results
benchmark.print_comparison()        # Print table
benchmark.pareto_frontier()         # Find non-dominated solutions
benchmark.export_json("out.json")   # Export for analysis
```

### GradingRubric

```python
# Built-in rubrics
from scripts.prompt_benchmark import exact_match_rubric, token_overlap_rubric

# Custom rubric
def my_rubric(expected: str, actual: str) -> float:
    # Return 0.0-1.0 where 1.0 = perfect
    ...

rubric = GradingRubric(my_rubric)
```

## Common Questions

**Q: How many test cases do I need?**
A: At least 10-20. More is better (50-100 ideal). But 5 is fine to start.

**Q: How many samples should I use?**
A: 1-2 for deterministic systems (like Lilly), 3-5 for LLMs (for variance).

**Q: Can I compare prompts from different models?**
A: Yes. Create separate benchmarks or use different executors.

**Q: How do I know if a result is statistically significant?**
A: Check the stdev. Low stdev = reliable. High stdev = run more samples.

**Q: Can I use this with my own model?**
A: Yes. Just write an executor function that calls your model.

## See Also

- `docs/PROMPT-EVALUATION.md` - Advanced guide
- `scripts/prompt_benchmark.py` - Source code
- `scripts/eval_translation_prompts.py` - Complete example
- `scripts/prompt_cli.py` - Interactive CLI
