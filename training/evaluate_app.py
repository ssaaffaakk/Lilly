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
import hashlib
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
    """FLORES-200 devtest and dev — 2,009 pairs the base model has not seen.

    devtest first, always, because `split_range` slices this list rather than
    re-reading the files: the two halves are translated once together and then
    scored separately, which is what makes a devtest-only score free instead of
    another hour of decoding.
    """
    src, ref, bounds = [], [], {}
    for split in ("devtest", "dev"):
        start = len(src)
        bs = (FLORES / f"{split}.bs").read_text(encoding="utf-8").splitlines()
        en = (FLORES / f"{split}.en").read_text(encoding="utf-8").splitlines()
        if len(bs) != len(en):
            raise SystemExit(f"{split}: {len(bs)} Bosnian against {len(en)} English")
        src += [s.strip() for s in bs]
        ref += [s.strip() for s in en]
        bounds[split] = (start, len(src))
    keep = [i for i, s in enumerate(src) if s and ref[i]]
    if limit:
        keep = keep[:limit]
    pairs.bounds = {name: (sum(1 for i in keep if i < lo),
                           sum(1 for i in keep if i < hi))
                    for name, (lo, hi) in bounds.items()}
    return [src[i] for i in keep], [ref[i] for i in keep]


def split_range(name: str, total: int) -> slice:
    """Which rows of the scored set belong to a FLORES split.

    Published scores for this language pair are on devtest alone — 1,012
    segments — so a number measured over devtest and dev together cannot be put
    beside them, however carefully it was produced. Scoring both halves at once
    gives a tighter interval for our own base-against-Lilly comparison and is
    the right default for that; it is the wrong number to quote against anybody
    else's system. Both are reported, and each says which it is.
    """
    if name == "all":
        return slice(0, total)
    bounds = getattr(pairs, "bounds", None)
    if not bounds or name not in bounds:
        raise SystemExit(f"unknown split {name!r} — call pairs() first")
    lo, hi = bounds[name]
    return slice(lo, hi)


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


def build_fingerprint(build: Path) -> str:
    """What is in this build, as one number.

    Five directories in models/lilly/ have names within a word of each other —
    translator, translator-armA, translator-armB, translator-base,
    translator-previous — and publishing the wrong one does not fail. The model
    loads, it translates, and every figure on the model card is quietly about a
    different set of weights. Nothing downstream could notice.

    So the score and the weights are tied together by this: the report records
    which build produced it, and the publisher refuses to upload weights whose
    fingerprint is not the one in the report.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(f.name for f in build.iterdir() if f.is_file()):
        if name == "built.json":          # records the build, is not the build
            continue
        digest.update(name.encode("utf-8"))
        with open(build / name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


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
    ap.add_argument("--out", type=Path, help="where to write the report")
    ap.add_argument("--split", choices=("all", "devtest", "dev"), default="all",
                    help="which FLORES half to score; devtest is the one "
                         "published leaderboards use")
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

    # The two sides are cached separately because only one of them changes, and
    # re-translating the base costs half an hour per candidate for an answer
    # already on disk. THE CACHE IS KEYED ON THE BUILD, NOT ONLY ON THE ROW
    # COUNT, and that is the whole point of the next thirty lines.
    #
    # It used to be keyed on `n == len(src)` alone. Every FLORES run has 2,009
    # rows, so every saved file matched every build, and a candidate silently
    # inherited whatever translations the last run happened to leave behind.
    # Found live: training/app-hypotheses.json held the PREVIOUS fine-tune's
    # output (42.04 / 67.11) while models/lilly/translator is Arm B
    # (42.18 / 67.47). Running this file with no flags would have reported the
    # old model's score, and `build_fingerprint` below would have stamped the
    # INSTALLED build's fingerprint onto it — binding one model's number to
    # another model's weights, which is the exact failure the fingerprint was
    # added to prevent, defeated by the cache sitting above it.
    #
    # This is the same bug as the reader's photograph cache keyed on the file
    # name alone, which HANDOFF.md already lists as a trap this project fell
    # into. Same shape, different file: a cache whose key does not mention the
    # thing that changed.
    #
    # So a side is reused only when the saved fingerprint equals the fingerprint
    # of the build being asked for. A file written before this change carries no
    # fingerprint, and is not reused — for either side. That costs one base
    # re-translation, once. Reusing text whose provenance cannot be checked is
    # how the wrong number gets published, and half an hour is cheaper than that.
    def fingerprint_of(build: Path) -> str:
        try:
            return build_fingerprint(build)
        except (FileNotFoundError, NotADirectoryError):
            return ""

    base_fp, tuned_fp = fingerprint_of(BASE_BUILD), fingerprint_of(args.tuned)

    def cached_side(path, side, want_fp):
        """Saved translations for `side`, but only if they came from `want_fp`."""
        if args.fresh or not path.exists() or not want_fp:
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("n") != len(src):
            return []
        have = data.get(f"{side}_build")
        rows = data.get(side) or []
        if len(rows) != len(src):
            return []
        if have != want_fp:
            print(f"  {path.name}: {side} translations are from build "
                  f"{have or 'unrecorded'}, not {want_fp[:16]} — re-translating")
            return []
        return rows

    base = cached_side(args.saved, "base", base_fp)
    if not base:
        # The base build really is the same in every comparison, so a cache
        # written by an earlier candidate can still supply it — but only one
        # that says so by fingerprint.
        for other in sorted(REPO_ROOT.glob("training/app-hypotheses*.json")):
            spare = cached_side(other, "base", base_fp)
            if spare:
                print(f"  base translations reused from {other.name}")
                base = spare
                break
    if not base:
        base = translate_all(BASE_BUILD, src, "base")

    tuned = cached_side(args.saved, "lilly", tuned_fp)
    if tuned:
        print(f"  candidate translations reused (build {tuned_fp[:16]})"
              f" — --fresh to redo them")
    else:
        tuned = translate_all(args.tuned, src, args.tuned.name)

    args.saved.write_text(json.dumps({"n": len(src),
                                      "base": base, "base_build": base_fp,
                                      "lilly": tuned, "lilly_build": tuned_fp},
                                     ensure_ascii=False), encoding="utf-8")

    # Everything above translated the whole set, because the cache is keyed on
    # its length and half a set would throw the other half away. Scoring is
    # where the split applies.
    if args.split != "all":
        cut = split_range(args.split, len(src))
        src, refs = src[cut], refs[cut]
        base, tuned = base[cut], tuned[cut]
        print(f"  scoring the {args.split} half alone: {len(src):,} pairs")

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

    fingerprint = tuned_fp or build_fingerprint(args.tuned)
    print(f"\nscored build: {fingerprint}")
    write_report(len(src), leaked, leaked_t, tables, fingerprint,
                 split=args.split, out=args.out)
    written = (args.out or REPORT).resolve()
    try:
        written = written.relative_to(REPO_ROOT)
    except ValueError:
        pass
    print(f"\nwritten to {written}")
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


def write_report(n, leaked_base, leaked_tuned, tables, fingerprint,
                 split="all", out=None) -> None:
    lines = [
        "# Translation quality — what Lilly actually serves", "",
        f"Scored build: `{fingerprint}`", "",
        (f"{n:,} FLORES-200 devtest pairs — the set published leaderboards for "
         f"this language pair use, so these numbers can be put beside theirs. "
         if split == "devtest" else
         f"{n:,} FLORES-200 pairs the base model was not trained on. ")
        + "Both models are "
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
    (out or REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
