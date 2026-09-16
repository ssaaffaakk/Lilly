#!/usr/bin/env python3
"""Score Bosnian--English WikiMatrix pairs with LaBSE cosine similarity.

This is deliberately a *filtering instrument*, not a training-data rewrite.
It reads source/bs/en TSV rows, samples only the WikiMatrix portion by default,
and prints distribution and tail examples.  ``--write-scores`` and
``--write-kept`` are explicit so an exploratory sample cannot silently alter
the training corpus.  The full-corpus pass is owner-gated.

LaBSE is multilingual: both sides are embedded in one vector space and cosine
similarity is therefore an alignment signal, unlike monolingual word overlap.

Examples:
  python3 scripts/filter_wikimatrix_align.py --sample-size 3000 --seed 20260915
  python3 scripts/filter_wikimatrix_align.py --input path/to/wiki.tsv --all \\
      --write-scores scores.tsv --write-kept kept.tsv
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "data" / "clean" / "train.tsv"
DEFAULT_MODEL = "sentence-transformers/LaBSE"
DEFAULT_THRESHOLD = 0.55
TARGET_CORPUS = "WikiMatrix"


@dataclass(frozen=True)
class Pair:
    row: int
    source: str
    bs: str
    en: str


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int) -> Sequence[Sequence[float]]: ...


class LaBSEEncoder:
    """Minimal mean-pooling wrapper; avoids a separate sentence-transformers dep."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment error
            raise SystemExit("LaBSE needs the repository's torch and transformers dependencies") from exc
        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def encode(self, texts: Sequence[str], batch_size: int) -> Sequence[Sequence[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(list(texts[start:start + batch_size]), padding=True,
                                    truncation=True, max_length=256, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine with a fail-closed zero-vector result."""
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def read_pairs(path: Path, corpus: str = TARGET_CORPUS) -> Iterable[Pair]:
    """Yield only valid rows from one named corpus; malformed input is fatal."""
    with path.open(encoding="utf-8") as fh:
        for row, line in enumerate(fh, start=1):
            # This corpus is plain TSV, not CSV.  Quotes are ordinary sentence
            # characters (and are sometimes unmatched), so csv.reader would
            # incorrectly consume the next physical row.
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise SystemExit(f"{path}:{row}: expected source, bs, en (got {len(fields)} fields)")
            source, bs, en = fields
            if source == corpus:
                if not bs.strip() or not en.strip():
                    raise SystemExit(f"{path}:{row}: empty side in {corpus} pair")
                yield Pair(row, source, bs, en)


def reservoir_sample(items: Iterable[Pair], size: int, seed: int) -> tuple[list[Pair], int]:
    """Uniform sample without loading the 145k-row source into memory."""
    rng = random.Random(seed)
    sample: list[Pair] = []
    count = 0
    for count, item in enumerate(items, start=1):
        if len(sample) < size:
            sample.append(item)
        else:
            replacement = rng.randrange(count)
            if replacement < size:
                sample[replacement] = item
    return sample, count


def score_pairs(pairs: Sequence[Pair], encoder: Encoder, batch_size: int) -> list[tuple[Pair, float]]:
    bs_vectors = encoder.encode([pair.bs for pair in pairs], batch_size)
    en_vectors = encoder.encode([pair.en for pair in pairs], batch_size)
    if len(bs_vectors) != len(pairs) or len(en_vectors) != len(pairs):
        raise ValueError("encoder returned a vector count different from its input")
    return [(pair, cosine(bs, en)) for pair, bs, en in zip(pairs, bs_vectors, en_vectors)]


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize no scores")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(scored: Sequence[tuple[Pair, float]], threshold: float) -> str:
    values = [score for _, score in scored]
    dropped = sum(score < threshold for score in values)
    return (f"scored={len(values)} threshold={threshold:.3f} drop={dropped} ({dropped / len(values):.2%})\n"
            f"min={min(values):.3f} p01={percentile(values, .01):.3f} "
            f"p05={percentile(values, .05):.3f} p10={percentile(values, .10):.3f} "
            f"median={percentile(values, .50):.3f} p90={percentile(values, .90):.3f} "
            f"p95={percentile(values, .95):.3f} max={max(values):.3f}")


def write_scores(path: Path, scored: Sequence[tuple[Pair, float]], threshold: float, kept_only: bool) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("source\tbs\ten\tlabse_cosine\tkeep\n")
        for pair, score in scored:
            if not kept_only or score >= threshold:
                if "\t" in pair.bs or "\t" in pair.en or "\n" in pair.bs or "\n" in pair.en:
                    raise ValueError(f"row {pair.row} contains a TSV control character")
                fh.write(f"{pair.source}\t{pair.bs}\t{pair.en}\t{score:.6f}\t{str(score >= threshold).lower()}\n")


def rewrite_train_tsv(input_path: Path, output_path: Path, scored: Sequence[tuple[Pair, float]], threshold: float) -> int:
    scores_by_pair = {(pair.bs, pair.en): score for pair, score in scored}
    kept_count = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                continue
            source, bs, en = fields
            if source == TARGET_CORPUS:
                score = scores_by_pair.get((bs, en))
                if score is not None and score < threshold:
                    continue
            fout.write(f"{source}\t{bs}\t{en}\n")
            kept_count += 1
    return kept_count


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--sample-size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--all", action="store_true", help="score all WikiMatrix rows (owner-gated)")
    ap.add_argument("--write-scores", type=Path)
    ap.add_argument("--write-kept", type=Path)
    ap.add_argument("--rewrite-train-tsv", type=Path, help="rewrite train.tsv keeping only WikiMatrix pairs with score >= threshold")
    ap.add_argument("--examples", type=int, default=20)
    args = ap.parse_args(argv)
    if not 0 < args.threshold < 1:
        ap.error("--threshold must be between 0 and 1")
    if args.sample_size < 1 or args.batch_size < 1 or args.examples < 0:
        ap.error("sample size, batch size, and examples must be positive (examples may be zero)")
    if args.all:
        print("FULL-CORPUS MODE: owner-gated; do not run without explicit owner approval.", file=sys.stderr)
        pairs = list(read_pairs(args.input))
        population = len(pairs)
    else:
        pairs, population = reservoir_sample(read_pairs(args.input), args.sample_size, args.seed)
    if not pairs:
        raise SystemExit(f"no {TARGET_CORPUS} rows found in {args.input}")
    print(f"model={args.model} input={args.input} population={population} sample={len(pairs)} seed={args.seed}")
    scored = score_pairs(pairs, LaBSEEncoder(args.model), args.batch_size)
    print(summarize(scored, args.threshold))
    for pair, score in sorted(scored, key=lambda item: item[1])[:args.examples]:
        print(f"\n[row {pair.row} cosine={score:.3f}]\nBS: {pair.bs}\nEN: {pair.en}")
    if args.write_scores:
        write_scores(args.write_scores, scored, args.threshold, kept_only=False)
    if args.write_kept:
        write_scores(args.write_kept, scored, args.threshold, kept_only=True)
    if args.rewrite_train_tsv:
        total_kept = rewrite_train_tsv(args.input, args.rewrite_train_tsv, scored, args.threshold)
        print(f"rewrote {args.rewrite_train_tsv}: {total_kept:,} total rows kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
