#!/usr/bin/env python3
"""Fine-tune our Bosnian -> English translation model.

Two ways to train the same base, chosen by --full-finetune:

  LoRA (default)   trains a ~4.3 M-parameter adapter on top of frozen weights.
  full fine-tune   trains all 237.7 M parameters.

LoRA's advantage is small data against a large model. This project is the other
way round -- 361,621 pairs against 237.7 M parameters -- and the measured
capacity arithmetic says r=16 is at the edge of what it can hold:

    LoRA r=16 on this model  =  270,336 x r  =  4,325,376 parameters
    at ~2 bits/parameter     =  8.65 M bits of capacity
    corpus target side       =  9.87 M tokens, ~1 bit/token upper bound

So the two numbers are the same size, which is the regime where the literature
says LoRA starts losing to full fine-tuning. --full-finetune exists to measure
that here rather than argue about it.

Usage:
    python3 train_translation.py                        # LoRA, the shipped recipe
    python3 train_translation.py --full-finetune        # all 237.7 M parameters
    python3 train_translation.py --quick-test           # 200 pairs, proves the pipeline
    python3 train_translation.py --preflight            # one real step, prints peak VRAM

Expects data/clean/{train,valid}.tsv from data/scripts/clean_data.py.
LoRA writes an adapter to models/lilly/adapter/; a full fine-tune writes whole
weights to models/lilly/translate-fullft/. scripts/build_translator.py turns
either into the int8 build the app serves.
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
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

# The two directions do not share a base model, and the corpus is the same file
# read the other way round. bs-en is single-target: Bosnian in, English out.
# en-bs is one decoder serving five South Slavic languages, chosen by a label on
# the front of the source sentence -- so `>>bos_Latn<<` is the only thing
# separating Bosnian output from Croatian, and it has to be on every row of
# training data as well as every sentence at inference. Train without it and the
# adapter learns to translate the label-free distribution, which is the wrong
# distribution to serve.
DIRECTIONS = {
    "bs-en": {"base": REPO_ROOT / "models" / "lilly" / "translate",
              "tag": None, "adapter": "adapter", "fullft": "translate-fullft",
              "probe": "riječ"},
    "en-bs": {"base": REPO_ROOT / "models" / "lilly" / "translate-en-bs",
              "tag": ">>bos_Latn<<", "adapter": "adapter-en-bs",
              "fullft": "translate-en-bs-fullft", "probe": "word"},
}


def base_model(direction: str) -> str:
    """Where the trainable base for this direction is.

    Weights normally sit in the project's own model folder. On a machine that
    has no copy (a fresh Kaggle runtime, say) point LILLY_BASE at one.
    """
    return os.environ.get("LILLY_BASE") or str(DIRECTIONS[direction]["base"])

# LoRA's own learning rate is not full fine-tuning's. Schulman et al. measure the
# optimum for LoRA at about ten times the optimum for full fine-tuning, over 14
# Llama and Qwen models -- https://thinkingmachines.ai/blog/lora/ . Ten times
# 2e-4 the wrong way round would walk the base weights off a cliff in the first
# hundred steps, so the default is resolved from the mode and printed, never
# inherited silently. (The shipped adapter's one epoch came from exactly that
# kind of unexamined default; see training/PREREGISTRATION.md.)
LORA_LR = 2e-4
FULL_LR = 2e-5


def read_tsv(path: Path, limit=None, direction="bs-en"):
    """Rows as (source, target) for this direction.

    The corpus file is `corpus \t bosnian \t english` either way; which column
    is the source is the direction, not a different dataset.
    """
    rows = []
    reverse = direction == "en-bs"
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                bosnian, english = parts[1], parts[2]
                rows.append((english, bosnian) if reverse else (bosnian, english))
            if limit and len(rows) >= limit:
                break
    return rows


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, tag=None):
        self.pairs = pairs
        self.tok = tokenizer
        self.tag = tag

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        if self.tag:
            source = f"{self.tag} {source}"
        enc = self.tok(source, text_target=target, max_length=MAX_LEN,
                       truncation=True)
        return {k: enc[k] for k in ("input_ids", "attention_mask", "labels")}


def chrf_metric(tokenizer):
    """Score the validation set the way the product is scored, not by loss.

    Cross-entropy and translation quality come apart: the loss keeps falling
    while the model learns to be confident about the same words, and this
    project has already paid for the difference once -- the shipped fine-tune
    gained BLEU and lost 0.22 chrF2, and the cause was output getting shorter.
    Loss cannot see length at all, because it never generates. chrF2 can: at
    beta=2 it weights recall twice as heavily as precision, so an output that
    drops words is penalised twice over.

    So the checkpoint that gets kept is chosen by generating, with the same beam
    width the app decodes with.
    """
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=0)          # chrF2, matching training/evaluate_app.py

    def compute(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        # The collator writes -100 into label padding so the loss ignores it.
        # Handed to the tokenizer that is not a token id, it is an IndexError --
        # and a metric that raises inside evaluate() takes the whole run down
        # after however many hours it had already spent.
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        hyp = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref = tokenizer.batch_decode(labels, skip_special_tokens=True)
        hyp = [h.strip() for h in hyp]
        ref = [r.strip() for r in ref]
        return {
            "chrf": chrf.corpus_score(hyp, [ref]).score,
            # Terseness is this project's known failure mode, so it is measured
            # every evaluation instead of being discovered at the end.
            "len_ratio": (sum(len(h) for h in hyp) / max(1, sum(len(r) for r in ref))),
        }

    return compute


def build_model(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model(args.direction))
    if args.full_finetune:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"full fine-tune: {total:,} trainable parameters "
              f"({100.0:.2f}% of the model)")
        return model

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        # Attention and feed-forward, in both stacks. The name suffixes match the
        # decoder's cross-attention (encoder_attn.q_proj and friends) as well as
        # self-attention, so every linear layer that carries weight is adapted --
        # which is the first of the two conditions under which LoRA is measured
        # to track full fine-tuning. The second is having enough rank for the
        # data, and that is what --lora-r is for.
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def preflight(args, model, tokenizer, pairs) -> None:
    """Run one real optimizer step and report what it cost.

    A full fine-tune of this model needs roughly 4.2 GB before a single
    activation exists -- fp32 weights, fp32 gradients and two fp32 Adam moments
    -- and the rest depends on batch size and sequence length in ways that are
    easy to get wrong by a factor of two. Guessing costs a T4 session; measuring
    costs twenty seconds.

    Deliberately a real step of the real model at the real batch size, not a toy:
    a toy that fits proves nothing about the job that follows.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # The widest batch is what sets the peak, and group_by_length guarantees one
    # batch of nothing but full-length sequences. So the probe is that batch, not
    # an average one: MAX_LEN tokens on both sides, every row.
    spec = DIRECTIONS[args.direction]
    long = f"{spec['probe']} " * (MAX_LEN * 2)
    batch = collator([{**PairDataset([(long, long)], tokenizer,
                                     tag=spec["tag"])[0]}
                      for _ in range(args.batch_size)])
    batch = {k: v.to(device) for k, v in batch.items()}
    print(f"probe batch: input_ids {tuple(batch['input_ids'].shape)}, "
          f"labels {tuple(batch['labels'].shape)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    scaler = torch.amp.GradScaler(device) if device == "cuda" else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(2):                      # step 2 is the one with Adam state
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
            loss = model(**batch).loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        print(f"  step {step + 1}: loss {loss.item():.4f}")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\npeak allocated {peak:.2f} GB of {total:.2f} GB on "
              f"{torch.cuda.get_device_name(0)}")
        # A step that fits with nothing to spare fits until the first batch that
        # is one token wider. The margin is the point of the check.
        assert peak < 0.80 * total, (
            f"one training step peaked at {peak:.2f} GB of {total:.2f} GB. That "
            f"leaves no room for evaluation or fragmentation, and the run would "
            f"die hours in. Lower --batch-size, or pass "
            f"--gradient-checkpointing.")
        print(f"headroom {total - peak:.2f} GB — the long run has room")
    else:
        print("\nno GPU here, so no VRAM figure. What this proved is that the "
              "graph builds and a gradient reaches the weights.")
    print(f"trainable tensors that took a gradient: "
          f"{sum(1 for p in model.parameters() if p.grad is not None):,}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="bs-en", choices=sorted(DIRECTIONS),
                    help="bs-en is what Lilly shipped; en-bs is the reply side")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None,
                    help=f"default {LORA_LR:g} for LoRA, {FULL_LR:g} for --full-finetune")
    ap.add_argument("--lora-r", type=int, default=16)
    # Marian bases are trained with --label-smoothing 0.1. Fine-tuning at 0 --
    # the HuggingFace default, which the shipped recipe inherited without
    # choosing it — changes the objective the base was built under and sharpens
    # the output distribution, and a sharper distribution reaches the end-of-
    # sentence token sooner. This project's chrF2 shortfall is entirely
    # terseness, so this is the first knob to try after the LoRA question.
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--full-finetune", action="store_true",
                    help="train all 237.7 M parameters instead of a LoRA adapter")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="trade ~30%% speed for activation memory")
    ap.add_argument("--metric", default="chrf", choices=("chrf", "loss"),
                    help="what picks the checkpoint that gets kept")
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--output", default=None)
    ap.add_argument("--quick-test", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="one real step at --batch-size, print peak VRAM, stop")
    # Named rather than hardcoded so an experiment can change the corpus without
    # editing the file that trains on it — and so the run's own command line
    # records which corpus produced which adapter.
    ap.add_argument("--data", type=Path,
                    default=REPO_ROOT / "data" / "clean" / "train.tsv")
    ap.add_argument("--valid", type=Path,
                    default=REPO_ROOT / "data" / "clean" / "valid.tsv")
    ap.add_argument("--valid-limit", type=int, default=500,
                    help="0 for all of it")
    args = ap.parse_args()

    if args.lr is None:
        args.lr = FULL_LR if args.full_finetune else LORA_LR
        print(f"learning rate {args.lr:g} — the default for "
              f"{'a full fine-tune' if args.full_finetune else 'LoRA'}, not the "
              f"other one")
    elif args.full_finetune and args.lr > 1e-4:
        # 2e-4 is right for an adapter and catastrophic for the base weights.
        # The two modes now share a flag, so the mistake is one word away.
        raise SystemExit(
            f"--lr {args.lr:g} with --full-finetune. LoRA's rate is about ten "
            f"times a full fine-tune's; applied to all 237.7 M weights it "
            f"destroys the pre-training in the first few hundred steps. Use "
            f"{FULL_LR:g}, or say so explicitly by staying under 1e-4.")

    spec = DIRECTIONS[args.direction]
    if args.output is None:
        args.output = str(REPO_ROOT / "models" / "lilly" /
                          (spec["fullft"] if args.full_finetune else spec["adapter"]))

    if args.quick_test:
        # small enough to fit an 8 GB laptop — the full run needs a 16 GB GPU
        args.batch_size, args.grad_accum = 2, 1
        args.eval_steps = 10
        # never into the real output: a 200-pair toy sitting at that path would
        # be picked up by the app, and evaluated and shipped as the finished model
        kind = "fullft" if args.full_finetune else "adapter"
        args.output = str(REPO_ROOT / "models" /
                          f"quicktest-{args.direction}-{kind}")

    base = base_model(args.direction)
    print(f"direction {args.direction} | base {base}"
          + (f" | target label {spec['tag']}" if spec["tag"] else ""))
    tokenizer = AutoTokenizer.from_pretrained(base)
    if spec["tag"] and tokenizer.convert_tokens_to_ids([spec["tag"]])[0] in (
            None, tokenizer.unk_token_id):
        # Not a warning. An unknown label is translated as text, the decoder
        # falls back to its majority target, and every number this run produces
        # would describe a Croatian model wearing a Bosnian name.
        raise SystemExit(
            f"{spec['tag']} is not in the vocabulary of {base}. This base "
            f"cannot be steered to Bosnian — run "
            f"scripts/fetch_translate_base.py --direction {args.direction}")
    model = build_model(args)

    if args.preflight:
        preflight(args, model, tokenizer,
                  read_tsv(args.data, limit=args.batch_size * 4,
                           direction=args.direction))
        return 0

    train_pairs = read_tsv(args.data, limit=200 if args.quick_test else None,
                           direction=args.direction)
    valid_pairs = read_tsv(args.valid,
                           limit=20 if args.quick_test
                           else (args.valid_limit or None),
                           direction=args.direction)
    print(f"training on {args.data.name}: {len(train_pairs):,} pairs | "
          f"validating on {args.valid.name}: {len(valid_pairs):,}")
    random.Random(41).shuffle(train_pairs)
    print(f"train={len(train_pairs):,}  valid={len(valid_pairs):,}")

    if args.gradient_checkpointing:
        # Recomputing activations and caching keys for generation are the same
        # tensors read two different ways; leaving the cache on makes the
        # checkpointing silently do nothing and prints a warning nobody reads.
        model.config.use_cache = False

    # Checkpoints go to scratch, not into the repo. A full fine-tune's checkpoint
    # is ~950 MB of weights, and on Kaggle everything under /kaggle/working is
    # copied into the version Output — two of those is 1.9 GB of upload for files
    # nobody downloads. save_only_model drops the optimizer state on top of that
    # (another ~1.9 GB each): it costs the ability to resume, which no run here
    # has ever used, and load_best_model_at_end still works.
    scratch = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else None
    checkpoints = (scratch / "lilly-checkpoints") if scratch else (REPO_ROOT / "models" / "checkpoints")

    use_cuda = torch.cuda.is_available()
    by_chrf = args.metric == "chrf"
    train_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        label_smoothing_factor=args.label_smoothing,
        fp16=use_cuda,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Roughly half the compute was going into padding: batches were filled in
        # file order, so one long sentence padded fifteen short ones out to match.
        # Grouping by length before batching cuts the run without touching quality.
        group_by_length=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="chrf" if by_chrf else "eval_loss",
        greater_is_better=by_chrf,
        # Generating to score costs about a minute per evaluation on the 462-pair
        # holdout, and buys the only validation number that can see length.
        predict_with_generate=by_chrf,
        generation_max_length=MAX_LEN,
        generation_num_beams=4,          # what app/translate.py decodes with
        report_to="none",
        seed=41,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=PairDataset(train_pairs, tokenizer, tag=spec["tag"]),
        eval_dataset=PairDataset(valid_pairs, tokenizer, tag=spec["tag"]),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=chrf_metric(tokenizer) if by_chrf else None,
    )
    trainer.train()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    kind = "weights" if args.full_finetune else "LoRA adapter"
    print(f"Saved {kind} to {out}")

    # The number the checkpoint was chosen by, printed where the log records it.
    best = trainer.state.best_metric
    if best is not None:
        print(f"best {'valid chrF2' if by_chrf else 'valid loss'}: {best:.4f} "
              f"({trainer.state.best_model_checkpoint})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
