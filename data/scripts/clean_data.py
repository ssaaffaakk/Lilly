#!/usr/bin/env python3
"""Clean the raw Bosnian-English pairs and build train/valid/test splits.

Why each filter exists:
  - empty / whitespace-only lines ......... broken alignments
  - length ratio > 2.5x ................... sentence pairs that don't actually match
  - > 250 words ........................... not sentences, breaks training batches
  - Cyrillic on the Bosnian side .......... almost always Serbian, we want exact Bosnian
  - < 30% letters ......................... tables, timestamps, markup junk
  - identical bs == en .................... untranslated lines
  - exact duplicates ...................... corpus overlap would skew training

Output (data/clean/):  train.tsv / valid.tsv / test.tsv
  tab-separated: corpus \t bosnian \t english   (corpus tag kept for domain weighting)
Also writes data/STATS.md — the report we commit to the repo.

Usage:  python3 clean_data.py [--max-per-corpus N]
"""
import argparse
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

SEED = 41
VALID_SIZE = 1500
TEST_SIZE = 1500

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keep(bs: str, en: str, drops: Counter) -> bool:
    if not bs or not en:
        drops["empty"] += 1
        return False
    bs_words, en_words = len(bs.split()), len(en.split())
    if bs_words > 250 or en_words > 250:
        drops["too_long"] += 1
        return False
    if bs_words >= 4 and en_words >= 4:
        ratio = max(bs_words, en_words) / min(bs_words, en_words)
        if ratio > 2.5:
            drops["length_ratio"] += 1
            return False
    if CYRILLIC.search(bs):
        drops["cyrillic"] += 1
        return False
    for text in (bs, en):
        letters = len(LETTERS.findall(text))
        if letters < 0.3 * len(text):
            drops["mostly_symbols"] += 1
            return False
    if bs.lower() == en.lower():
        drops["untranslated"] += 1
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-corpus", type=int, default=None,
                    help="cap pairs taken from any single corpus (for huge noisy ones)")
    args = ap.parse_args()

    corpora = sorted(d for d in RAW_DIR.iterdir() if (d / f"{d.name}.bs").exists())
    if not corpora:
        raise SystemExit(f"No raw corpora in {RAW_DIR} — run download_data.py first")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    pairs = []
    seen = set()
    per_corpus = {}
    for corpus_dir in corpora:
        name = corpus_dir.name
        drops = Counter()
        kept = 0
        raw_total = 0
        bs_lines = open(corpus_dir / f"{name}.bs", encoding="utf-8", errors="replace")
        en_lines = open(corpus_dir / f"{name}.en", encoding="utf-8", errors="replace")
        for bs, en in zip(bs_lines, en_lines):
            raw_total += 1
            bs, en = normalize(bs), normalize(en)
            if not keep(bs, en, drops):
                continue
            key = (bs.lower(), en.lower())
            if key in seen:
                drops["duplicate"] += 1
                continue
            seen.add(key)
            pairs.append((name, bs, en))
            kept += 1
            if args.max_per_corpus and kept >= args.max_per_corpus:
                break
        per_corpus[name] = (raw_total, kept, drops)
        print(f"{name}: {raw_total:,} raw -> {kept:,} kept "
              f"({', '.join(f'{k}={v:,}' for k, v in drops.most_common())})")

    rng.shuffle(pairs)
    test, valid, train = (pairs[:TEST_SIZE],
                          pairs[TEST_SIZE:TEST_SIZE + VALID_SIZE],
                          pairs[TEST_SIZE + VALID_SIZE:])

    for split_name, rows in (("train", train), ("valid", valid), ("test", test)):
        path = CLEAN_DIR / f"{split_name}.tsv"
        with open(path, "w", encoding="utf-8") as f:
            for corpus, bs, en in rows:
                f.write(f"{corpus}\t{bs}\t{en}\n")
        print(f"wrote {path}: {len(rows):,} pairs")

    write_stats(per_corpus, len(train), len(valid), len(test))
    return 0


def write_stats(per_corpus, n_train, n_valid, n_test) -> None:
    lines = ["# Data report — Phase 1", "",
             "| Corpus | Raw pairs | Kept | Top reasons dropped |",
             "|--------|-----------|------|---------------------|"]
    for name, (raw, kept, drops) in sorted(per_corpus.items()):
        top = ", ".join(f"{k} {v:,}" for k, v in drops.most_common(3)) or "—"
        lines.append(f"| {name} | {raw:,} | {kept:,} | {top} |")
    total_kept = n_train + n_valid + n_test
    lines += ["",
              f"**Final dataset:** {total_kept:,} clean pairs "
              f"— train {n_train:,} / valid {n_valid:,} / test {n_test:,}",
              "",
              "Filters: empty lines, >2.5x length-ratio mismatches, >250-word lines, "
              "Cyrillic on the Bosnian side (almost always Serbian), mostly-symbol lines, "
              "untranslated lines, exact duplicates.",
              "",
              f"Split seed {SEED}, so the split is reproducible."]
    (DATA_DIR / "STATS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {DATA_DIR / 'STATS.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
