# Prompt Evaluation Framework

A practical system for measuring, comparing, and grading multiple prompt variants. This is what Symphony.is does at scale—now you can do it locally on your domain.

## Quick Start (5 minutes)

```python
from scripts.prompt_benchmark import PromptBenchmark, GradingRubric

# 1. Define your test set
test_cases = [
    {"input": "...", "expected_output": "..."},
    {"input": "...", "expected_output": "..."},
]

# 2. Define prompt variants
variants = {
    "baseline": "Do X with {input}",
    "role": "You are Y. Do X with {input}",
    "constraints": "Do X with {input}. Constraint: Z",
}

# 3. Define grading rubric (accuracy metric)
def grade_fn(expected: str, actual: str) -> float:
    return 1.0 if expected == actual else 0.0

# 4. Run benchmark
benchmark = PromptBenchmark(
    name="my_experiment",
    variants=variants,
    test_cases=test_cases,
    rubric=GradingRubric(grade_fn),
)

results = benchmark.run(executor=my_llm_call, samples=3)
benchmark.print_comparison()
benchmark.pareto_frontier()
benchmark.export_json("results.json")
```

## The Framework

### Components

**PromptVariant**: One prompt template
- Takes a template with `{input}` placeholder
- Runs it through an executor (LLM API, local model, etc.)
- Returns output + latency

**GradingRubric**: How to score accuracy
- Custom function: `(expected: str, actual: str) -> float` (0.0 to 1.0)
- Built-in rubrics: exact_match, token_overlap
- You define what "correct" means for your domain

**PromptBenchmark**: Orchestrates the evaluation
- Runs all variants against all test cases
- Collects measurements: latency, accuracy, cost
- Aggregates results with stats (mean, median, stdev)
- Ranks variants by each metric
- Computes Pareto frontier (non-dominated solutions)

### Workflow

```
1. Define test set (inputs + expected outputs)
   ↓
2. Write prompt variants (different templates)
   ↓
3. Create grading rubric (what is "correct"?)
   ↓
4. Build executor (how to run prompts)
   ↓
5. Run benchmark (execute all variants on all tests)
   ↓
6. Analyze results (which variant is best?)
   ↓
7. Deploy winner (use in production)
   ↓
8. Monitor & re-benchmark periodically
```

## How to Use It

### Step 1: Create Your Test Set

The test set defines the problem you're solving. Quality matters—a bad test set gives meaningless results.

```python
# Translation example
test_cases = [
    {"input": "Dobar dan", "expected_output": "Good morning"},
    {"input": "Gdje je stanica?", "expected_output": "Where is the station?"},
    # ... more examples covering edge cases
]

# Summarization example
test_cases = [
    {
        "input": "Long article text...",
        "expected_output": "Summary: ..."
    },
]

# Code generation example
test_cases = [
    {
        "input": "Write a function that checks if a string is a palindrome",
        "expected_output": "def is_palindrome(s): return s == s[::-1]"
    },
]
```

**Guidelines:**
- Include typical cases (happy path)
- Include edge cases (empty input, special characters, very long text)
- Include tricky cases (ambiguous, context-dependent, multiple valid answers)
- Aim for 10-50 test cases (more data = more confidence)
- Use real examples from your domain

### Step 2: Define Prompt Variants

Write multiple approaches to the same task. Vary one thing at a time to isolate what works.

```python
# Good: Varies one thing (role context)
variants = {
    "baseline": "Translate: {input}",
    "with_role": "You are a translator. Translate: {input}",
}

# Good: Varies constraints
variants = {
    "open_ended": "Translate: {input}",
    "constrained": "Translate, keeping names and numbers unchanged: {input}",
}

# Good: Varies reasoning
variants = {
    "direct": "Translate: {input}",
    "reasoning": "First list key words, then translate: {input}",
}

# Bad: Changes too many things
variants = {
    "v1": "Translate: {input}",
    "v2": "You are a translator who is an expert in historical texts. Translate to modern English, preserving nuance: {input}",
    # ^ This changes role, expertise, instructions, and style all at once
}
```

**Prompt engineering techniques to try:**

1. **ROLE** - Tell the model what to be
   ```
   "You are a {role}. {task}: {input}"
   ```

2. **CONTEXT** - Provide background
   ```
   "Context: {background}. {task}: {input}"
   ```

3. **CONSTRAINTS** - What NOT to do
   ```
   "{task}: {input}. Do NOT {constraint}. Do NOT {constraint}."
   ```

4. **TASK** - Be explicit about the job
   ```
   "Task: {task}\nInput: {input}\nOutput:"
   ```

5. **SUCCESS CRITERIA** - Define what success looks like
   ```
   "{task}: {input}\nYour translation should be {criterion1} and {criterion2}."
   ```

6. **OUTPUT FORMAT** - Specify format
   ```
   "{task}: {input}\nRespond with only the output, nothing else."
   ```

7. **EXAMPLES** - Show what good looks like
   ```
   "Examples:\n- Input: X → Output: Y\n- Input: A → Output: B\n\n{task}: {input}"
   ```

8. **REASONING** - Ask model to think before responding
   ```
   "Reason through this step by step, then provide your answer: {input}"
   ```

### Step 3: Create a Grading Rubric

Define how to score accuracy. This is critical—garbage grading = garbage results.

**Option A: Exact Match** (strictest)
```python
from scripts.prompt_benchmark import exact_match_rubric

rubric = GradingRubric(exact_match_rubric)
# Returns 1.0 only if outputs match exactly
```

**Option B: Token/Word Overlap** (loose)
```python
from scripts.prompt_benchmark import token_overlap_rubric

rubric = GradingRubric(token_overlap_rubric)
# Returns ratio of words that match
```

**Option C: Custom Function** (recommended)
```python
def my_rubric(expected: str, actual: str) -> float:
    """
    Grade a translation.
    
    Args:
        expected: reference translation
        actual: model's translation
    
    Returns:
        float 0.0-1.0 where 1.0 = perfect
    """
    # Exact match is perfect
    if expected.lower().strip() == actual.lower().strip():
        return 1.0
    
    # Minor variations are OK
    exp_words = set(expected.lower().split())
    act_words = set(actual.lower().split())
    if len(exp_words) == 0:
        return 1.0 if len(act_words) == 0 else 0.0
    
    overlap = len(exp_words & act_words)
    coverage = overlap / len(exp_words)
    
    # Partial credit
    return min(1.0, coverage + 0.1)

rubric = GradingRubric(my_rubric)
```

**Advanced: Use an LLM to Grade**
```python
def llm_rubric(expected: str, actual: str) -> float:
    """Have Claude grade the output."""
    grade_prompt = f"""
    Reference: {expected}
    Output: {actual}
    
    Is the output correct? Reply with just a number 0-10.
    """
    # Call Claude API
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=10,
        messages=[{"role": "user", "content": grade_prompt}]
    )
    score = int(response.content[0].text.strip())
    return score / 10.0

rubric = GradingRubric(llm_rubric)
```

### Step 4: Build an Executor

Tell the benchmark how to run each prompt.

**Option A: Call Claude API**
```python
from anthropic import Anthropic

client = Anthropic()

def executor(prompt: str) -> str:
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

results = benchmark.run(executor=executor, samples=3)
```

**Option B: Call Lilly's Translation Engine**
```python
from app.translate import get_engine

def executor(prompt: str) -> str:
    engine = get_engine("bs-en")
    output, diag = engine.translate(prompt)
    return output

results = benchmark.run(executor=executor, samples=1)
```

**Option C: Mock/Test**
```python
def mock_executor(prompt: str) -> str:
    # For testing framework without calling LLM
    if "role" in prompt:
        return "response_variant_1"
    return "response_variant_0"

results = benchmark.run(executor=mock_executor, samples=1)
```

### Step 5: Run the Benchmark

```python
benchmark = PromptBenchmark(
    name="translation_v1",
    variants=variants,
    test_cases=test_cases,
    rubric=GradingRubric(my_rubric),
)

# Run all variants on all tests
# samples=3 means run each variant 3 times per test (to measure variance)
results = benchmark.run(
    executor=executor,
    samples=3,
    cost_per_call_usd=0.0001  # Adjust based on your model
)

# Display results
benchmark.print_comparison()

# Find non-dominated solutions
frontier = benchmark.pareto_frontier()

# Export for further analysis
benchmark.export_json("results.json")
```

### Step 6: Analyze Results

The output tells you which variant is best for which metric.

**Example output:**
```
Variant              Accuracy     Latency (ms)     Cost
baseline             78.50%       45.2±2.1         $0.000100
with_role            82.30%       47.8±1.9         $0.000100
constrained          79.10%       43.1±3.2         $0.000100
reasoning            85.60%       52.3±4.1         $0.000100

Rankings (by metric)

Accuracy (higher is better):
  1. reasoning           85.60%
  2. with_role           82.30%
  3. constrained         79.10%
  4. baseline            78.50%

Latency (lower is better):
  1. constrained         43.1ms
  2. baseline            45.2ms
  3. with_role           47.8ms
  4. reasoning           52.3ms

Pareto frontier (2 non-dominated solutions):
  reasoning: acc=85.60%, lat=52.3ms, cost=$0.000100
  constrained: acc=79.10%, lat=43.1ms, cost=$0.000100
```

**How to choose:**

1. **If accuracy matters most**: Use the highest-accuracy variant
2. **If latency matters most**: Use the fastest variant
3. **If you want tradeoff**: Pick from Pareto frontier based on your priorities
4. **If all matter equally**: Weighted score = `0.3*accuracy - 0.3*latency_rank - 0.3*cost_rank`

## Real-World Example

See `scripts/eval_translation_prompts.py` for a complete example:

```bash
# Run it
python3 scripts/eval_translation_prompts.py

# View results
cat scripts/prompt_eval_results.json
```

## Advanced Topics

### Measuring Statistical Significance

When using `samples=3+`, you get standard deviation. Larger stdev = less reliable.

```python
results = benchmark.run(executor=executor, samples=5)
# Results include stdev for each metric
# High stdev = consider increasing samples or investigating variance source
```

### Weighted Multi-Objective Optimization

If you care about accuracy AND latency AND cost simultaneously:

```python
def weighted_score(results: Dict) -> Dict:
    """Combine metrics into one score."""
    scores = {}
    for variant, r in results.items():
        # Normalize each metric to 0-1 scale
        acc = r['mean_accuracy']
        lat_norm = 1.0 / (1.0 + r['mean_latency_ms'] / 100.0)  # Lower is better
        cost_norm = 1.0 / (1.0 + r['mean_cost_per_call_usd'] * 10000)
        
        # Weighted combination (adjust weights to your priorities)
        score = (
            0.5 * acc +         # Accuracy is most important
            0.3 * lat_norm +    # Latency matters
            0.2 * cost_norm     # Cost matters least
        )
        scores[variant] = score
    
    return scores
```

### Monitoring in Production

After deploying a prompt variant, track real-world performance:

```python
# Log every inference
def production_executor(prompt: str) -> str:
    t0 = time.time()
    result = call_llm(prompt)
    latency_ms = (time.time() - t0) * 1000
    
    # Log for analysis
    log_event({
        "prompt_variant": "reasoning",
        "latency_ms": latency_ms,
        "timestamp": time.time(),
    })
    
    return result

# Weekly: re-run benchmark on production queries
# If performance drifts, go back to evaluation mode
```

## Common Pitfalls

1. **Bad test set**: Too small, not representative, or biased toward one variant
   - **Fix**: Collect at least 20-30 diverse examples

2. **Over-optimizing for test set**: Variants that score high on your tests but fail in production
   - **Fix**: Hold out test set, don't tune variants against it

3. **Single run variance**: Running once and trusting the result
   - **Fix**: Use `samples=3+` to measure variance

4. **Wrong grading rubric**: Metric doesn't match what users care about
   - **Fix**: Validate rubric against human judgment first

5. **Changing test set mid-evaluation**: Makes comparisons invalid
   - **Fix**: Freeze test set before running variants

6. **Not tracking cost**: Only measuring accuracy/latency
   - **Fix**: Include cost in your metrics from the start

## See Also

- `docs/PROFILING.md` - Measuring Lilly's performance
- `scripts/prompt_benchmark.py` - Framework source code
- `scripts/eval_translation_prompts.py` - Complete working example
