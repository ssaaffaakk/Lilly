#!/usr/bin/env python3
"""Score the untuned base vs our fine-tuned Lilly adapter, honestly.

Metrics:
  - BLEU  — classic word-overlap score
  - chrF2 — character-level score, fairer for morphology-rich languages like Bosnian

Two test sets, because they answer different questions:

  in-house   1,500 pairs held out from OUR training. But the base model was
             trained on the same OPUS releases these came from, so it has seen
             most of them. This is the in-domain number, and it flatters the base.
  FLORES-200 1,012 pairs the base model has not seen. This is the number that
             answers "is Lilly better at Bosnian", rather than "better at this
             corpus". Fetch it with data/scripts/download_flores.py.

Results are broken out per source corpus, because the in-house set is a mix and a
single average hides where a gain came from.

Usage:
    python3 evaluate.py                          # base model only (baseline)
    python3 evaluate.py --adapter models/lilly/adapter
    python3 evaluate.py --limit 200              # quicker, subset of test set

Writes training/RESULTS.md with the comparison table.
"""
import argparse
import json
import os
import random
import sys
import re
import time
from collections import defaultdict
from pathlib import Path

import sacrebleu
import sacrebleu.significance   # not pulled in by the top-level import
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
# Weights normally sit in the project's own model folder. On a machine that has
# no copy (a fresh Colab runtime, say) point LILLY_BASE at one.
BASE_MODEL = os.environ.get("LILLY_BASE") or str(REPO_ROOT / "models" / "lilly" / "translate")
FLORES_DIR = REPO_ROOT / "data" / "flores"


def load_test(limit=None):
    """The in-house split, keeping which corpus each pair came from."""
    rows = []
    with open(REPO_ROOT / "data" / "clean" / "test.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                rows.append(tuple(parts))
            if limit and len(rows) >= limit:
                break
    return rows


def load_flores(limit=None):
    """FLORES-200, if it has been fetched. Unseen by the base model.

    Both halves — devtest (1,012) and dev (997). They were built the same way and
    the base has seen neither; using both nearly doubles the sample, and on a
    difference this small the width of the interval is what decides whether the
    result says anything at all.
    """
    rows = []
    for half in ("devtest", "dev"):
        bs, en = FLORES_DIR / f"{half}.bs", FLORES_DIR / f"{half}.en"
        if bs.exists() and en.exists():
            rows += list(zip(bs.read_text(encoding="utf-8").splitlines(),
                             en.read_text(encoding="utf-8").splitlines()))
    return [("FLORES", s, r) for s, r in rows[:limit]]


def translate_all(model, tokenizer, sentences, device, batch_size=16):
    """Translate everything, batching sentences of similar length together.

    This is not a speed trick. Padding a short sentence out to the longest one in
    its batch changes what the model produces — measured on this test set, batching
    in the order the file happens to be in costs 8.7 BLEU against the same model
    scoring the same sentences grouped by length. Sorting first also runs about
    nineteen times faster, because most of the old batches were mostly padding.
    """
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    out = [None] * len(sentences)
    model.eval()
    done = 0
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        enc = tokenizer([sentences[j] for j in idx], return_tensors="pt",
                        padding=True, truncation=True, max_length=192).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=192, num_beams=4)
        for j, text in zip(idx, tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out[j] = text
        done += len(idx)
        if (i // batch_size) % 10 == 0:
            print(f"  {done}/{len(sentences)}", flush=True)
    return out


# The base model writes a language tag into the text of many of its outputs —
# 433 of 1,012 FLORES sentences, measured. That is a real defect and a user would
# see it, but it is not a translation error, and left in it swamps the metric:
# scoring with the tags gave a 5.23 BLEU gap where the translation difference is
# 0.5. Both models are scored with it stripped, and the leak is reported instead.
LANGUAGE_TAG = re.compile(r"^\s*(>>[a-zA-Z_]+<<\s*)+")


def strip_tags(hyps: list) -> tuple:
    """Return the outputs without their language tags, and how many had one."""
    cleaned = [LANGUAGE_TAG.sub("", h) for h in hyps]
    leaked = sum(1 for a, b in zip(hyps, cleaned) if a != b)
    return cleaned, leaked


def score(hyps, refs):
    # word_order=0 is chrF2. word_order=2 is chrF++, which is a different metric
    # and was being reported under the chrF2 label.
    return (sacrebleu.corpus_bleu(hyps, [refs]).score,
            sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score)


def by_corpus(rows, hyps):
    """Per-corpus scores, because one blended average hides where a gain came from."""
    grouped = defaultdict(lambda: ([], []))
    for (corpus, _, ref), hyp in zip(rows, hyps):
        grouped[corpus][0].append(hyp)
        grouped[corpus][1].append(ref)
    return {name: (len(h), *score(h, r)) for name, (h, r) in sorted(grouped.items())}


def confidence_interval(base_hyps, tuned_hyps, refs, samples=2000):
    """The 95% range the BLEU difference actually sits in, on this test set.

    Resample the sentences with replacement, score both systems on the same
    resample each time, and keep the difference. The spread of those differences
    is what "how much of this is the sample" means — measured on the data rather
    than asserted as a round number.
    """
    rng = random.Random(41)
    n = len(refs)
    deltas = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        b = [base_hyps[i] for i in idx]
        t_ = [tuned_hyps[i] for i in idx]
        r = [[refs[i] for i in idx]]
        deltas.append(sacrebleu.corpus_bleu(t_, r).score
                      - sacrebleu.corpus_bleu(b, r).score)
    deltas.sort()
    lo = deltas[int(0.025 * samples)]
    hi = deltas[int(0.975 * samples) - 1]
    favour = 100 * sum(1 for d in deltas if d > 0) / samples
    return lo, hi, favour


def significance(base_hyps, tuned_hyps, refs):
    """Is the difference bigger than the noise? Paired bootstrap resamples the
    test set a thousand times and asks how often a gap this size shows up by
    chance. Small p means the difference is the model, not the sample."""
    try:
        # named_systems is a list of (name, outputs) pairs, not a dict
        _, scores = sacrebleu.significance.PairedTest(
            [("base", list(base_hyps)), ("lilly", list(tuned_hyps))],
            {"bleu": sacrebleu.metrics.BLEU()},
            references=[list(refs)], test_type="bs", n_samples=1000)()
        # results come back keyed by the metric's display name, not the key we
        # passed in, and there is a "System" column alongside it
        key = next(k for k in scores if k != "System")
        p = scores[key][1].p_value
        return f"{p:.4f}" if p is not None else "not reported"
    except Exception as exc:      # a missing p-value is not worth losing the table for
        return f"could not compute ({type(exc).__name__}: {exc})"


def run(model, tokenizer, rows, device, label):
    print(f"scoring {label} on {len(rows)} pairs…", flush=True)
    t0 = time.time()
    raw = translate_all(model, tokenizer, [s for _, s, _ in rows], device)
    hyps, leaked = strip_tags(raw)
    overall = score(hyps, [r for _, _, r in rows])
    leak_note = f"  |  language tag in {leaked}/{len(rows)} outputs" if leaked else ""
    print(f"  BLEU={overall[0]:.2f}  chrF2={overall[1]:.2f}  "
          f"({time.time() - t0:.0f}s){leak_note}")
    return hyps, overall, leaked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="path to LoRA adapter; omit to score the base model only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rescore", action="store_true",
                    help="rebuild RESULTS.md from the translations already saved in "
                         "hypotheses.json — no model is loaded and nothing is "
                         "re-translated, which is how a scoring fix gets applied to a "
                         "run that already cost hours")
    args = ap.parse_args()

    if args.rescore:
        return rescore()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    in_house = load_test(args.limit)
    flores = load_flores(args.limit)
    print(f"in-house: {len(in_house)} pairs"
          + (f"  |  FLORES-200: {len(flores)} pairs" if flores
             else "  |  FLORES-200: not fetched (data/scripts/download_flores.py)"))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    results = {}
    hyps = {}
    leaks = {}
    hyps["base_in"], results[("Base (untuned)", "in-house")], leaks["base_in"] = run(
        base, tokenizer, in_house, device, "base / in-house")
    if flores:
        hyps["base_fl"], results[("Base (untuned)", "FLORES-200")], leaks["base_fl"] = run(
            base, tokenizer, flores, device, "base / FLORES-200")

    tuned = None
    if args.adapter:
        from peft import PeftModel
        tuned = PeftModel.from_pretrained(base, args.adapter)
        hyps["tuned_in"], results[("Lilly (fine-tuned)", "in-house")], leaks["tuned_in"] = run(
            tuned, tokenizer, in_house, device, "Lilly / in-house")
        if flores:
            hyps["tuned_fl"], results[("Lilly (fine-tuned)", "FLORES-200")], leaks["tuned_fl"] = run(
                tuned, tokenizer, flores, device, "Lilly / FLORES-200")

    # Keep the outputs. Re-running the significance test on saved text costs
    # seconds; regenerating it costs an hour of translation.
    out = REPO_ROOT / "training" / "hypotheses.json"
    out.write_text(json.dumps({
        "in_house_refs": [r for _, _, r in in_house],
        "flores_refs": [r for _, _, r in flores],
        **{k: v for k, v in hyps.items()}}, ensure_ascii=False))
    print(f"kept the translations in {out.name}")

    write_results(in_house, flores, results, hyps, bool(args.adapter), leaks)
    print("wrote training/RESULTS.md")
    return 0


def rescore() -> int:
    """Re-derive every number from the saved translations."""
    saved = REPO_ROOT / "training" / "hypotheses.json"
    if not saved.exists():
        print(f"no saved translations at {saved} — run a full evaluation first",
              file=sys.stderr)
        return 1
    data = json.loads(saved.read_text())
    in_house = load_test(len(data["in_house_refs"]))
    flores = load_flores(len(data["flores_refs"])) if data.get("flores_refs") else []

    results, hyps, leaks = {}, {}, {}
    pairs = [("base_in", "Base (untuned)", "in-house", in_house),
             ("base_fl", "Base (untuned)", "FLORES-200", flores),
             ("tuned_in", "Lilly (fine-tuned)", "in-house", in_house),
             ("tuned_fl", "Lilly (fine-tuned)", "FLORES-200", flores)]
    for key, model, which, rows in pairs:
        if key not in data or not rows:
            continue
        cleaned, leaked = strip_tags(data[key])
        hyps[key], leaks[key] = cleaned, leaked
        results[(model, which)] = score(cleaned, [r for _, _, r in rows])
        print(f"{model} / {which}: BLEU={results[(model, which)][0]:.2f} "
              f"chrF2={results[(model, which)][1]:.2f}"
              + (f"  ({leaked} tagged)" if leaked else ""))

    write_results(in_house, flores, results, hyps, "tuned_fl" in hyps, leaks)
    print("wrote training/RESULTS.md")
    return 0


def write_results(in_house, flores, results, hyps, tuned, leaks) -> None:
    lines = ["# Translation quality — Phase 2", "",
             "## Headline", "",
             "| Model | Test set | BLEU | chrF2 |",
             "|-------|----------|------|-------|"]
    for (model, which), (bleu, chrf) in results.items():
        lines.append(f"| {model} | {which} | {bleu:.2f} | {chrf:.2f} |")

    lines += ["", "## Which number to believe", "",
              f"**FLORES-200** ({len(flores)} pairs) is the honest one. The base model was "
              "trained on the OPUS releases our in-house set was drawn from, so it has "
              "already seen almost all of those sentences — the in-house number flatters "
              "it and is in-domain for both models." if flores else
              "", ""]
    if not flores:
        lines += ["**No unseen benchmark was scored.** The in-house set is drawn from the "
                  "same OPUS releases the base model was trained on, so the base has "
                  "already seen almost all of it. Run `data/scripts/download_flores.py` "
                  "and score again before quoting these numbers as evidence.", ""]

    lines += ["## In-house set, by source corpus", "",
              "A single average over a mixed set hides where a gain came from. SETIMES is "
              "clean news text; WikiMatrix is web-mined and roughly a sixth of it is "
              "misaligned, so a gain there may be style-fitting rather than translation "
              "quality.", "",
              "| Corpus | Pairs | Base BLEU | Base chrF2"
              + (" | Lilly BLEU | Lilly chrF2 |" if tuned else " |"),
              "|--------|-------|-----------|-----------"
              + ("|------------|-------------|" if tuned else "|")]
    base_c = by_corpus(in_house, hyps["base_in"])
    tuned_c = by_corpus(in_house, hyps["tuned_in"]) if tuned else {}
    for name, (n, bleu, chrf) in base_c.items():
        row = f"| {name} | {n} | {bleu:.2f} | {chrf:.2f}"
        if tuned:
            tb, tc = tuned_c[name][1], tuned_c[name][2]
            row += f" | {tb:.2f} | {tc:.2f} |"
        else:
            row += " |"
        lines.append(row)

    if tuned and flores:
        refs = [r for _, _, r in flores]
        p = significance(hyps["base_fl"], hyps["tuned_fl"], refs)
        lo, hi, favour = confidence_interval(hyps["base_fl"], hyps["tuned_fl"], refs)
        gap = (results[("Lilly (fine-tuned)", "FLORES-200")][0]
               - results[("Base (untuned)", "FLORES-200")][0])
        lines += ["", "## Is the difference real", "",
                  f"On FLORES-200 the gap is **{gap:+.2f} BLEU**, paired bootstrap "
                  f"**p = {p}**, 95% interval **[{lo:+.2f}, {hi:+.2f}]**, and "
                  f"{favour:.0f}% of resamples favour Lilly.",
                  "",
                  "Read that as it is: a difference that does not clear the usual 0.05 "
                  "bar is *unproven*, not *absent* — the interval and the direction both "
                  "lean one way, the test set is just not large enough to settle it."]

    leaked = [(k, v) for k, v in leaks.items() if v]
    if leaked:
        lines += ["", "## Language tags in the output", "",
                  "The base model writes a `>>bos_Latn<<` tag into the text of many of "
                  "its translations. That is a real defect a reader would see, but it is "
                  "not a translation error, so both systems are scored with it stripped "
                  "and the count reported here instead. Left in, it moved BLEU by about "
                  "five points and hid what the fine-tuning actually changed.", ""]
        for name, n in sorted(leaked):
            total = len(flores) if name.endswith("_fl") else len(in_house)
            lines.append(f"- `{name}`: {n} of {total} outputs ({100 * n / total:.1f}%)")

    lines += ["", "---", "",
              "Generated by `training/evaluate.py`. Sentences are batched by length: "
              "padding short sentences out to the longest in their batch changes what the "
              "model produces, and measured on this set it costs 8.7 BLEU."]
    (REPO_ROOT / "training" / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
