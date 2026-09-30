# Prompt Evaluation Framework

**The practical system for measuring, testing, and grading multiple prompts. What Symphony.is does—now local to your codebase.**

## What This Is

A complete framework for:
- ✅ Defining multiple prompt variants
- ✅ Running them against test cases
- ✅ Measuring accuracy, latency, cost
- ✅ Comparing results systematically
- ✅ Finding best-performing variants
- ✅ Exporting results for production deployment

Think of it as **A/B testing for prompts**.

## Files

| File | Purpose |
|------|---------|
| `scripts/prompt_benchmark.py` | Core framework (PromptBenchmark, PromptVariant, GradingRubric) |
| `scripts/eval_translation_prompts.py` | Complete working example for translation |
| `scripts/prompt_cli.py` | Interactive CLI (no coding required) |
| `docs/PROMPT-EVALUATION.md` | Full documentation + advanced topics |
| `docs/PROMPT-BENCHMARK-QUICKSTART.md` | Quick recipes and templates |
| `examples/test_set_*.csv/.json` | Sample test sets you can use |

## Quick Start (Choose One)

### Option 1: Interactive CLI (Recommended for First-Time Users)

```bash
python3 scripts/prompt_cli.py
```

Walk through loading test cases, defining variants, choosing grading metrics, and running benchmarks. **10 minutes, no coding.**

### Option 2: Run Example

```bash
python3 scripts/eval_translation_prompts.py
```

This evaluates 6 different translation prompts against 5 test cases. See results immediately.

### Option 3: Copy Python Template

```python
from scripts.prompt_benchmark import PromptBenchmark, GradingRubric
from anthropic import Anthropic

# Your test cases
test_cases = [
    {"input": "...", "expected_output": "..."},
]

# Your prompt variants
variants = {
    "v1": "Do X with {input}",
    "v2": "You are Y. Do X with {input}",
}

# Your grading function
def grade_fn(expected, actual):
    return 1.0 if expected == actual else 0.0

# Your executor
client = Anthropic()
def executor(prompt):
    return client.messages.create(
        model="claude-opus-5-5",
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text

# Run it
benchmark = PromptBenchmark(
    name="my_test",
    variants=variants,
    test_cases=test_cases,
    rubric=GradingRubric(grade_fn),
)

results = benchmark.run(executor=executor, samples=2)
benchmark.print_comparison()
benchmark.export_json("results.json")
```

## How It Works

### 1. Define Test Set
Collection of (input, expected_output) pairs that represent your problem.

```python
test_cases = [
    {"input": "Dobar dan", "expected_output": "Good morning"},
    {"input": "Kako ste?", "expected_output": "How are you?"},
]
```

### 2. Write Prompt Variants
Different approaches to the same task. Vary one thing at a time.

```python
variants = {
    "baseline": "Translate: {input}",
    "role": "You are a translator. Translate: {input}",
    "reasoning": "Identify key words, then translate: {input}",
}
```

### 3. Create Grading Rubric
Function that scores how good the output is (0.0-1.0).

```python
def grade_fn(expected, actual):
    # Your domain-specific grading logic
    return 1.0 if expected == actual else 0.0

rubric = GradingRubric(grade_fn)
```

### 4. Build Executor
Function that runs a prompt and returns output.

```python
def executor(prompt):
    # Call Claude, Lilly, local model, etc.
    response = client.messages.create(...)
    return response.content[0].text
```

### 5. Run Benchmark
All variants × all test cases × multiple samples.

```python
benchmark = PromptBenchmark(
    name="translation_v1",
    variants=variants,
    test_cases=test_cases,
    rubric=rubric,
)

results = benchmark.run(executor=executor, samples=3)
```

### 6. Analyze Results

```
Variant              Accuracy     Latency (ms)
reasoning            85.60%       52.3±4.1
role                 82.30%       47.8±1.9
baseline             78.50%       45.2±2.1

Pareto frontier (non-dominated solutions):
  reasoning: acc=85.60%, lat=52.3ms
  baseline: acc=78.50%, lat=45.2ms
```

### 7. Export & Deploy

```python
benchmark.export_json("results.json")
# Use best variant in production
```

## Concepts

### Prompt Variants
Different templates for the same task. The framework runs them all and compares.

```python
"baseline": "Translate: {input}"
"role": "You are a translator. Translate: {input}"
"constraints": "Translate: {input}. Do NOT expand."
"examples": "Examples: [examples]. Now: {input}"
"cot": "Reason step-by-step, then translate: {input}"
```

### Grading Rubric
Defines what "correct" means for your domain. Critical for meaningful results.

```python
# Exact match (strictest)
def exact_match(expected, actual):
    return 1.0 if expected == actual else 0.0

# Word overlap (more lenient)
def overlap(expected, actual):
    exp_words = set(expected.split())
    act_words = set(actual.split())
    return len(exp_words & act_words) / len(exp_words)

# LLM-based grading (most sophisticated)
def llm_grading(expected, actual):
    # Have Claude judge if it's correct
    return float(claude_evaluate(expected, actual)) / 10.0
```

### Pareto Frontier
Set of non-dominated solutions. If variant A has better accuracy AND latency than variant B, then B is dominated. Frontier shows the true tradeoffs.

```
Pareto frontier (2 solutions):
  fast: acc=78%, lat=40ms     (best latency)
  accurate: acc=86%, lat=55ms (best accuracy)
  
middle: acc=82%, lat=48ms is DOMINATED (worse in both)
```

### Metrics

| Metric | Meaning |
|--------|---------|
| **Accuracy** | How often the output is correct (0-1) |
| **Latency** | Time to get response (ms) |
| **Cost** | Cost per API call (USD) |
| **Stdev** | Variance across samples (higher = less reliable) |

## Real-World Example: Translation Prompts

The repo includes a complete example that tests 6 translation approaches:

```bash
python3 scripts/eval_translation_prompts.py
```

Tests these variants:
1. **baseline** - Minimal: "Translate: {input}"
2. **role_context** - Role: "You are a professional translator. Translate: {input}"
3. **explicit_task** - Clear task: "Task: Translate Bosnian to English.\nInput: {input}\nOutput:"
4. **with_constraints** - Constraints: "Translate. Do NOT expand or change meaning: {input}"
5. **with_examples** - Examples: "Examples: [examples]. Now: {input}"
6. **chain_of_thought** - Reasoning: "Identify key words. Then translate: {input}"

Output shows which variant:
- Has highest accuracy
- Is fastest
- Offers best latency/accuracy tradeoff

## Workflow

```
Start
  ↓
1. Collect test cases (10-50 examples)
  ↓
2. Write 3-5 prompt variants
  ↓
3. Define grading rubric (what is correct?)
  ↓
4. Choose executor (Claude, Lilly, local model, etc.)
  ↓
5. Run benchmark (python script or CLI)
  ↓
6. Get results:
   - Ranking by accuracy
   - Ranking by latency
   - Ranking by cost
   - Pareto frontier
   - JSON export
  ↓
7. Pick winner based on your priorities
  ↓
8. Deploy to production
  ↓
9. Monitor real-world performance
  ↓
10. Re-benchmark periodically to catch regressions
```

## Customization

### Custom Test Domain
Replace test cases with YOUR domain:

```python
# Classification
test_cases = [
    {"input": "This product is great!", "expected_output": "positive"},
    {"input": "Terrible experience", "expected_output": "negative"},
]

# Code generation
test_cases = [
    {"input": "fibonacci function", "expected_output": "def fib(n): ..."},
]

# Summarization
test_cases = [
    {"input": "long article...", "expected_output": "summary..."},
]
```

### Custom Metrics
Add your own scoring logic:

```python
def semantic_similarity_grade(expected, actual):
    """Use embeddings to measure semantic distance."""
    from sentence_transformers import util, SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    emb_exp = model.encode(expected)
    emb_act = model.encode(actual)
    similarity = util.pytorch_cos_sim(emb_exp, emb_act)[0][0]
    return float(similarity)

rubric = GradingRubric(semantic_similarity_grade)
```

### Custom Executor
Use any LLM/model:

```python
# Ollama local model
def ollama_executor(prompt):
    import requests
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={'model': 'llama2', 'prompt': prompt}
    )
    return response.json()['response']

# OpenAI
def openai_executor(prompt):
    import openai
    return openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content
```

## Prompt Engineering Techniques

The framework lets you test these approaches:

1. **Role** - "You are X"
2. **Context** - Provide background information
3. **Constraints** - "Do NOT X", "Do Y only"
4. **Task Clarity** - Be explicit about what you want
5. **Success Criteria** - Define what good looks like
6. **Output Format** - Specify exact format expected
7. **Examples** - Show 2-3 examples of correct behavior
8. **Reasoning** - "Think step-by-step first"
9. **Few-Shot** - Provide examples inline

## Advanced Topics

See `docs/PROMPT-EVALUATION.md` for:
- Multi-objective optimization (accuracy vs latency vs cost)
- Statistical significance testing
- Production deployment strategies
- Monitoring and re-benchmarking
- Custom grading functions
- Handling multiple valid answers

## FAQ

**Q: How many test cases?**
A: Start with 10. Ideal is 50-100. At least 5 to get started.

**Q: How many samples?**
A: 1 for deterministic (Lilly), 3-5 for LLMs (measure variance).

**Q: What if results are tied?**
A: Use secondary metrics. If accuracy is same, pick fastest. If latency is same, pick cheapest.

**Q: Can I test multiple executors?**
A: Yes. Run benchmark once per executor, then compare JSON results.

**Q: How do I know if variant A is better than B?**
A: Look at Pareto frontier. If A dominates B (better in all metrics), A is better. Otherwise it's a tradeoff—depends on your priorities.

**Q: Can I add new variants mid-benchmark?**
A: Yes, just re-run with updated variants dict. Old measurements are overwritten.

## Files Organization

```
scripts/
  ├─ prompt_benchmark.py       ← Core framework
  ├─ eval_translation_prompts.py ← Translation example
  ├─ prompt_cli.py              ← Interactive CLI
  └─ prompt_eval_results.json    ← Results (auto-generated)

docs/
  ├─ PROMPT-EVALUATION.md        ← Full reference
  ├─ PROMPT-BENCHMARK-QUICKSTART.md ← Quick recipes
  └─ README-PROMPT-FRAMEWORK.md  ← This file

examples/
  ├─ test_set_translation.csv
  └─ test_set_translation.json
```

## Getting Started

1. **Learn the concepts** (5 min)
   - Read this file

2. **Try the CLI** (10 min)
   ```bash
   python3 scripts/prompt_cli.py
   ```

3. **Run the example** (5 min)
   ```bash
   python3 scripts/eval_translation_prompts.py
   ```

4. **Write your own benchmark** (30 min)
   - Copy Python template from quickstart
   - Replace test cases with your domain
   - Run and analyze results

5. **Deploy winner to production**
   - Use best-performing variant
   - Monitor in real-world usage
   - Re-benchmark periodically

## Next Steps

- **Quickstart**: `docs/PROMPT-BENCHMARK-QUICKSTART.md`
- **Full Docs**: `docs/PROMPT-EVALUATION.md`
- **Example**: `python3 scripts/eval_translation_prompts.py`
- **CLI**: `python3 scripts/prompt_cli.py`
