#!/usr/bin/env python3
"""Score what Lilly actually serves, not the file the training produced.

training/evaluate.py scores the adapter as float32 PyTorch and feeds each test
row to the model whole. Neither is what a user meets. The app serves a quantised
CTranslate2 build and splits input on sentence boundaries first, because the
model drops a clause when handed several at once — and the difference between
those two paths is not a rounding error:

    raw adapter, whole rows      BLEU +0.54   chrF2 -0.79
    through the app's own path   BLEU +1.23   chrF2 -0.22

Everything here goes through app.translate.Engine, so the splitting, the token
budgets and the quantisation are the product's own rather than a second copy
that can drift from it. Both models are built the same way — see
scripts/build_translator.py --no-adapter — because comparing an int8 build
against a float32 one measures the quantisation as much as the training.

    python3 training/evaluate_app.py
    python3 training/evaluate_app.py --limit 200      # a quick look

Roughly forty minutes on a laptop for the full 2,009 pairs, most of it the base
model: it writes longer output and so decodes for longer.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FLORES = REPO_ROOT / "data" / "flores"
MODELS = REPO_ROOT / "models" / "lilly"
BASE_BUILD = MODELS / "translator-base"
TUNED_BUILD = MODELS / "translator"
SAVED = REPO_ROOT / "training" / "app-hypotheses.json"
REPORT = REPO_ROOT / "training" / "RESULTS-product.md"

# The base model writes its language tag into the text of many translations.
# That is a real defect a user sees, so the score is reported both ways: with the
# tags, which is what arrives on screen, and without, which is the translation
# quality underneath. Reporting only the first credits the fine-tuning with
# fixing a bug; only the second hides that it did.
LANGUAGE_TAG = re.compile(r"^\s*(>>[a-zA-Z_]+<<\s*)+")


def pairs(limit=None):
    """FLORES-200 devtest and dev — 2,009 pairs the base model has not seen."""
    src, ref = [], []
    for split in ("devtest", "dev"):
        bs = (FLORES / f"{split}.bs").read_text(encoding="utf-8").splitlines()
        en = (FLORES / f"{split}.en").read_text(encoding="utf-8").splitlines()
        if len(bs) != len(en):
            raise SystemExit(f"{split}: {len(bs)} Bosnian against {len(en)} English")
        src += [s.strip() for s in bs]
        ref += [s.strip() for s in en]
    keep = [i for i, s in enumerate(src) if s and ref[i]]
    if limit:
        keep = keep[:limit]
    return [src[i] for i in keep], [ref[i] for i in keep]


def translate_all(build: Path, src: list, label: str) -> list:
    from app.translate import Engine

    if not (build / "model.bin").exists():
        raise SystemExit(
            f"no build at {build}. For the base: python3 scripts/build_translator.py "
            f"--no-adapter --dest {build}")
    engine = Engine(directory=build)
    out, started = [], time.time()
    for i, text in enumerate(src):
        out.append(engine.translate(text, truncate=True))
        if (i + 1) % 200 == 0:
            print(f"  {label} {i + 1}/{len(src)}  ({time.time() - started:.0f}s)",
                  flush=True)
    print(f"  {label} done in {time.time() - started:.0f}s", flush=True)
    return out


def score(hyps, refs):
    from sacrebleu.metrics import BLEU, CHRF
    return (BLEU().corpus_score(hyps, [refs]).score,
            CHRF(word_order=0).corpus_score(hyps, [refs]).score)


def significance(base, tuned, refs):
    """How often a gap this size comes out of resampling the test set alone."""
    from sacrebleu.metrics import BLEU, CHRF
    # Imported by name: sacrebleu does not expose .significance as an attribute of
    # the package, so `import sacrebleu` then `sacrebleu.significance` raises — and
    # the except below would have swallowed it into a report with no p-values and
    # no complaint, which is the shape of missing evidence that reads as evidence.
    from sacrebleu.significance import PairedTest

    out = {}
    try:
        _, scores = PairedTest(
            [("base", list(base)), ("lilly", list(tuned))],
            {"BLEU": BLEU(), "chrF2": CHRF(word_order=0)},
            references=[list(refs)], test_type="bs", n_samples=1000)()
        for metric, values in scores.items():
            if metric == "System":
                continue
            for value in values:
                p = getattr(value, "p_value", None)
                if p is not None:
                    out[metric] = p
    except Exception as exc:                      # a missing p is not a wrong p
        print(f"  could not test significance: {exc}", file=sys.stderr)
    return out


def table(refs, base, tuned, strip: bool):
    clean = (lambda x: LANGUAGE_TAG.sub("", x).strip()) if strip else (lambda x: x)
    b, t = [clean(x) for x in base], [clean(x) for x in tuned]
    bb, bc = score(b, refs)
    tb, tc = score(t, refs)
    p = significance(b, t, refs)
    return {"base_bleu": bb, "base_chrf": bc, "lilly_bleu": tb, "lilly_chrf": tc,
            "p": p,
            "base_len": sum(len(x) for x in b) / sum(len(x) for x in refs),
            "lilly_len": sum(len(x) for x in t) / sum(len(x) for x in refs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="score only the first N pairs")
    ap.add_argument("--fresh", action="store_true", help="re-translate, ignore saved")
    # A candidate build can be scored where it stands, without being installed
    # first. Deciding whether to replace the served model should not require
    # having already replaced it.
    ap.add_argument("--tuned", type=Path, default=TUNED_BUILD,
                    help="the build to score against the base")
    ap.add_argument("--saved", type=Path, default=SAVED,
                    help="where to cache the translations")
    args = ap.parse_args()

    src, refs = pairs(args.limit)
    print(f"{len(src):,} pairs, both models through app.translate.Engine")

    # The two sides are cached separately because only one of them changes. The
    # base build is the same in every comparison, and re-translating it costs
    # half an hour per candidate for an answer already on disk. Any cache with
    # the right number of rows can supply it, so a new candidate borrows the
    # base from whatever earlier run produced it.
    def cached(path):
        if args.fresh or not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data.get("n") == len(src) else {}

    saved = cached(args.saved)
    base = saved.get("base") or []
    if len(base) != len(src):
        for other in sorted(REPO_ROOT.glob("training/app-hypotheses*.json")):
            spare = cached(other).get("base") or []
            if len(spare) == len(src):
                print(f"  base translations reused from {other.name}")
                base = spare
                break
    if len(base) != len(src):
        base = translate_all(BASE_BUILD, src, "base")

    tuned = saved.get("lilly") or []
    if len(tuned) != len(src):
        tuned = translate_all(args.tuned, src, args.tuned.name)
    else:
        print("  candidate translations reused — --fresh to redo them")

    args.saved.write_text(json.dumps({"n": len(src), "base": base, "lilly": tuned},
                                     ensure_ascii=False), encoding="utf-8")

    leaked = sum(1 for x in base if LANGUAGE_TAG.match(x))
    leaked_t = sum(1 for x in tuned if LANGUAGE_TAG.match(x))
    print(f"\nlanguage tag printed into the translation: base {leaked}/{len(base)} "
          f"({100 * leaked / len(base):.1f}%), Lilly {leaked_t}/{len(tuned)} "
          f"({100 * leaked_t / len(tuned):.1f}%)")

    tables = {}
    for title, strip in (("as the user sees it", False),
                         ("with the language tag stripped", True)):
        r = table(refs, base, tuned, strip)
        print(f"\n{title}")
        print(f"  {'':<6} {'BLEU':>7} {'chrF2':>7} {'length':>8}")
        print(f"  {'base':<6} {r['base_bleu']:>7.2f} {r['base_chrf']:>7.2f} "
              f"{r['base_len']:>8.3f}")
        print(f"  {'Lilly':<6} {r['lilly_bleu']:>7.2f} {r['lilly_chrf']:>7.2f} "
              f"{r['lilly_len']:>8.3f}")
        print(f"  {'gap':<6} {r['lilly_bleu'] - r['base_bleu']:>+7.2f} "
              f"{r['lilly_chrf'] - r['base_chrf']:>+7.2f}")
        for metric, p in r["p"].items():
            print(f"    {metric} p = {p:.4f}"
                  + ("" if p < 0.05 else "   (does not clear 0.05)"))
        tables[title] = r

    write_report(len(src), leaked, leaked_t, tables)
    print(f"\nwritten to {REPORT.relative_to(REPO_ROOT)}")
    return 0


def reading(gap: float, p: float, metric: str) -> str:
    """The sentence has to come out of the number, not sit next to it."""
    if p is None:
        return f"{metric} moves {gap:+.2f}, untested."
    if p >= 0.05:
        return (f"{metric} moves {gap:+.2f} at p = {p:.3f} — that does not clear the "
                f"usual 0.05 bar, so it is unproven rather than absent.")
    return (f"{metric} moves {gap:+.2f} at p = {p:.3f}, which clears the usual 0.05 "
            f"bar: measured, not leaned towards.")


def write_report(n, leaked_base, leaked_tuned, tables) -> None:
    lines = [
        "# Translation quality — what Lilly actually serves", "",
        f"{n:,} FLORES-200 pairs the base model was not trained on. Both models are "
        "int8 CTranslate2 builds and both go through `app.translate.Engine`, so the "
        "sentence splitting and the quantisation are the product's own. The only "
        "difference between the two columns is the fine-tuning.", "",
        "This is the number to quote. `training/RESULTS.md` scores the raw adapter "
        "on whole rows, which is a useful diagnostic and not what anyone runs: on "
        "the same pairs that path reads +0.54 BLEU and −0.79 chrF2, because feeding "
        "several sentences at once makes the model drop a clause and the app never "
        "does that.", "",
        f"The base model prints its language tag into the translation itself — "
        f"`>>eng<<` and friends — in {leaked_base:,} of {n:,} outputs "
        f"({100 * leaked_base / n:.1f}%). Lilly does it in {leaked_tuned:,} "
        f"({100 * leaked_tuned / n:.1f}%). That is a defect a reader sees, so the "
        f"scores are given both with it and without: with, because it is what "
        f"arrives on screen; without, because otherwise the fine-tuning gets credit "
        f"for translation quality it did not gain.", "",
    ]
    for title, r in tables.items():
        lines += [f"## {title.capitalize()}", "",
                  "| Model | BLEU | chrF2 | Length vs reference |",
                  "|---|---|---|---|",
                  f"| Base (untuned) | {r['base_bleu']:.2f} | {r['base_chrf']:.2f} | "
                  f"{r['base_len']:.3f} |",
                  f"| Lilly (fine-tuned) | {r['lilly_bleu']:.2f} | "
                  f"{r['lilly_chrf']:.2f} | {r['lilly_len']:.3f} |",
                  f"| **Gap** | **{r['lilly_bleu'] - r['base_bleu']:+.2f}** | "
                  f"**{r['lilly_chrf'] - r['base_chrf']:+.2f}** | |", ""]
        for metric, key in (("BLEU", "bleu"), ("chrF2", "chrf")):
            gap = r[f"lilly_{key}"] - r[f"base_{key}"]
            lines.append(reading(gap, r["p"].get(metric), metric))
        lines.append("")
    lines += ["---", "",
              "Generated by `training/evaluate_app.py`. Build the base to compare "
              "against with `python3 scripts/build_translator.py --no-adapter "
              "--dest models/lilly/translator-base`."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
