#!/usr/bin/env python3
"""Fine-tune our Bosnian -> English translation model with LoRA.

Designed for a free Colab T4 GPU (16 GB). LoRA trains a small "adapter"
(~10-20 MB) on top of the frozen base — that adapter IS our model.

Usage:
    python3 train_translation.py                 # full training run
    python3 train_translation.py --quick-test    # 200 pairs, 1 tiny epoch, just to verify the pipeline

Expects data/clean/{train,valid}.tsv from data/scripts/clean_data.py.
Writes the LoRA adapter to models/lilly/adapter/, where the app picks it up.
"""
import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MAX_LEN = 128
REPO_ROOT = Path(__file__).resolve().parents[1]
# Weights normally sit in the project's own model folder. On a machine that has
# no copy (a fresh Colab runtime, say) point LILLY_BASE at one.
BASE_MODEL = os.environ.get("LILLY_BASE") or str(REPO_ROOT / "models" / "lilly" / "translate")


def read_tsv(path: Path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                rows.append((parts[1], parts[2]))  # (bosnian, english)
            if limit and len(rows) >= limit:
                break
    return rows


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tok = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        bs, en = self.pairs[idx]
        enc = self.tok(bs, text_target=en, max_length=MAX_LEN, truncation=True)
        return {k: enc[k] for k in ("input_ids", "attention_mask", "labels")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--output", default=str(REPO_ROOT / "models" / "lilly" / "adapter"))
    ap.add_argument("--quick-test", action="store_true")
    args = ap.parse_args()

    if args.quick_test:
        # small enough to fit an 8 GB laptop — the full run needs a 16 GB GPU
        args.batch_size, args.grad_accum = 2, 1
        # never into the real adapter: a 200-pair toy sitting at that path would
        # be picked up by the app, and evaluated and shipped as the finished model
        args.output = str(REPO_ROOT / "models" / "quicktest-adapter")

    train_pairs = read_tsv(REPO_ROOT / "data" / "clean" / "train.tsv",
                           limit=200 if args.quick_test else None)
    valid_pairs = read_tsv(REPO_ROOT / "data" / "clean" / "valid.tsv",
                           limit=20 if args.quick_test else 500)
    random.Random(41).shuffle(train_pairs)
    print(f"train={len(train_pairs):,}  valid={len(valid_pairs):,}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        # attention + feed-forward of both encoder and decoder
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    use_cuda = torch.cuda.is_available()
    train_args = Seq2SeqTrainingArguments(
        output_dir=str(REPO_ROOT / "models" / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        fp16=use_cuda,
        # Roughly half the compute was going into padding: batches were filled in
        # file order, so one long sentence padded fifteen short ones out to match.
        # Grouping by length before batching cuts the run without touching quality.
        group_by_length=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=41,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=PairDataset(train_pairs, tokenizer),
        eval_dataset=PairDataset(valid_pairs, tokenizer),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Saved LoRA adapter to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
