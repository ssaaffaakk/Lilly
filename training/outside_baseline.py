#!/usr/bin/env python3
"""Score an outside translation system on the same FLORES pairs Lilly is scored on.

Pre-registered in `training/PREREGISTRATION.md`, "v3 — an outside comparison,
written before any outside number exists". Every number this project has
published compares Lilly to itself; this is the first that compares it to a
system built by someone else, so that "42.14 BLEU" has a scale attached.

Everything that touches a number is imported from `training/evaluate.py` — the
same FLORES loader, the same length-sorted batching (batching in file order
costs 8.7 BLEU, measured), the same sacrebleu call. The outside system gets
Lilly's decoding settings, not settings chosen for it, in either direction.

    .venv/bin/python3 training/outside_baseline.py --direction bs-en
    .venv/bin/python3 training/outside_baseline.py --direction en-bs

NLLB selects its target language with a forced first token, not with a tag on
the source, so that is the one thing that differs from Lilly's path and it is
the system's own documented interface rather than a choice made here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# `training` goes in FRONT of the repo root, and both go in front of
# site-packages, ON PURPOSE. `evaluate` is also a PyPI package -- HuggingFace's
# -- and Kaggle ships it. Reorder these two lines and `from evaluate import
# score` silently binds to that package instead of the repo's evaluate.py, so
# the run would report numbers from a different scorer with no error anywhere.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

MODEL = REPO / "models" / "external" / "nllb-600M"
OUT = REPO / "training" / "outside"
# NLLB's own language codes. bos_Latn is Bosnian, not "Serbo-Croatian": the
# model was trained with the three varieties as separate targets.
LANG = {"bs-en": ("bos_Latn", "eng_Latn"), "en-bs": ("eng_Latn", "bos_Latn")}
# Lilly's published FLORES numbers, quoted from the results files so the
# comparison is against what the project actually claims in public.
LILLY = {
    "bs-en": {"bleu": 42.14, "chrf2": 66.79, "params": "~230M base + LoRA",
              "source": "training/RESULTS.md, Lilly (fine-tuned), FLORES-200"},
    "en-bs": {"bleu": 29.57, "chrf2": 58.96, "params": "~77M base, NOT fine-tuned",
              "source": "training/RESULTS-en-bs.md, base (untuned), FLORES-200"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="bs-en", choices=("bs-en", "en-bs"))
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test only; a limited run is not the comparison")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if not (MODEL / "config.json").is_file():
        print(f"no model at {MODEL} — snapshot_download facebook/"
              f"nllb-200-distilled-600M into it first", file=sys.stderr)
        return 1

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from evaluate import load_flores, score, translate_all

    rows = load_flores(args.limit, args.direction)
    if not rows:
        print("no FLORES on disk — data/scripts/download_flores.py", file=sys.stderr)
        return 1
    src_lang, tgt_lang = LANG[args.direction]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"NLLB-200-distilled-600M | {src_lang} -> {tgt_lang} | "
          f"{len(rows)} FLORES pairs | {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL)).to(device)

    # NLLB picks its target with a forced first generated token. This is the
    # system's own interface; everything else below is Lilly's path unchanged.
    forced = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced is None or forced == tokenizer.unk_token_id:
        print(f"tokenizer does not know {tgt_lang}", file=sys.stderr)
        return 1
    model.generation_config.forced_bos_token_id = forced
    print(f"  forced_bos_token_id = {forced} ({tgt_lang})", flush=True)

    t0 = time.time()
    hyps = translate_all(model, tokenizer, [s for _, s, _ in rows], device,
                         batch_size=args.batch_size)
    bleu, chrf2 = score(hyps, [r for _, _, r in rows])
    took = time.time() - t0

    lilly = LILLY[args.direction]
    print(f"\n=== FLORES-200, {args.direction}, {len(rows)} pairs ===")
    print(f"  NLLB-600M (600M params)           BLEU {bleu:6.2f}   chrF2 {chrf2:6.2f}")
    print(f"  Lilly ({lilly['params']})   "
          f"BLEU {lilly['bleu']:6.2f}   chrF2 {lilly['chrf2']:6.2f}")
    print(f"  difference (Lilly - NLLB)         BLEU {lilly['bleu'] - bleu:+6.2f}   "
          f"chrF2 {lilly['chrf2'] - chrf2:+6.2f}")
    print(f"  ({took:.0f}s)")
    if args.limit:
        print("  --limit was set: this is a smoke test, NOT the pre-registered "
              "comparison, which is all 2,009 pairs")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"nllb-600M-{args.direction}.json"
    path.write_text(json.dumps(
        {"system": "facebook/nllb-200-distilled-600M", "params": "600M",
         "direction": args.direction, "src_lang": src_lang, "tgt_lang": tgt_lang,
         "pairs": len(rows), "bleu": bleu, "chrf2": chrf2, "seconds": took,
         "limited": bool(args.limit), "lilly": lilly,
         "hyps": hyps if not args.limit else hyps[:20]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
