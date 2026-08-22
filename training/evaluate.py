#!/usr/bin/env python3
"""Score the untuned base vs our fine-tuned Lilly adapter on the held-out test set.

Metrics:
  - BLEU  — classic word-overlap score
  - chrF2 — character-level score, fairer for morphology-rich languages like Bosnian

Usage:
    python3 evaluate.py                          # base model only (baseline)
    python3 evaluate.py --adapter models/lilly/adapter
    python3 evaluate.py --limit 200              # quicker, subset of test set

Writes training/RESULTS.md with the comparison table.
"""
import argparse
import os
import time
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
# Weights normally sit in the project's own model folder. On a machine that has
# no copy (a fresh Colab runtime, say) point LILLY_BASE at one.
BASE_MODEL = os.environ.get("LILLY_BASE") or str(REPO_ROOT / "models" / "lilly" / "translate")


def load_test(limit=None):
    src, ref = [], []
    with open(REPO_ROOT / "data" / "clean" / "test.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                src.append(parts[1])
                ref.append(parts[2])
            if limit and len(src) >= limit:
                break
    return src, ref


def translate_all(model, tokenizer, sentences, device, batch_size=16):
    out = []
    model.eval()
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=192).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=192, num_beams=4)
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch)}/{len(sentences)}")
    return out


def score(name, hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    print(f"{name}: BLEU={bleu:.2f}  chrF2={chrf:.2f}")
    return bleu, chrf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="path to LoRA adapter; omit to score the base model only")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    src, ref = load_test(args.limit)
    print(f"test sentences: {len(src)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    results = {}

    print("scoring base model…")
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
    t0 = time.time()
    results["Base (untuned)"] = score("base", translate_all(base, tokenizer, src, device), ref)
    print(f"  took {time.time() - t0:.0f}s")

    if args.adapter:
        from peft import PeftModel
        print("scoring Lilly (fine-tuned)…")
        tuned = PeftModel.from_pretrained(base, args.adapter)
        results["Lilly (fine-tuned)"] = score(
            "lilly", translate_all(tuned, tokenizer, src, device), ref)

    lines = ["# Translation quality — Phase 2", "",
             f"Test set: {len(src)} held-out sentence pairs (never seen in training).", "",
             "| Model | BLEU | chrF2 |", "|-------|------|-------|"]
    for name, (bleu, chrf) in results.items():
        lines.append(f"| {name} | {bleu:.2f} | {chrf:.2f} |")
    (REPO_ROOT / "training" / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote training/RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
