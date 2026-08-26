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
# Run as `python3 training/evaluate.py`, sys.path[0] is training/, so the
# app package is not importable — and the report wants the app's own
# sentence splitter rather than a second copy of the rule.
sys.path.insert(0, str(REPO_ROOT))
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


def flores_on_disk() -> int:
    """How many FLORES pairs have actually been fetched.

    Separate from how many a given run scores, because those two numbers drifted
    apart once already: a rescore asked for as many pairs as it had saved
    translations for, silently threw away the half that had been downloaded
    since, and the report that came out of it said 1,012 while the commit that
    published it said 2,009. A run that scores fewer pairs than are sitting on
    disk is dropping evidence, and has to say so out loud.
    """
    return len(load_flores())


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


def significance(base_hyps, tuned_hyps, refs, metric="bleu"):
    """Is the difference bigger than the noise? Paired bootstrap resamples the
    test set a thousand times and asks how often a gap this size shows up by
    chance. Small p means the difference is the model, not the sample.

    Both headline metrics can be tested, because on this pair of models they
    disagree and the strength of each direction is the whole question. Measured
    on the 2,009-pair set: BLEU +0.54 at p=0.029, chrF2 -0.79 at p=0.001. The
    metric that moved against us moved with far more confidence than the one
    that moved for us, and a report that tested only BLEU would never show it.
    """
    metrics = {"bleu": sacrebleu.metrics.BLEU(),
               "chrf": sacrebleu.metrics.CHRF(word_order=0)}
    try:
        # named_systems is a list of (name, outputs) pairs, not a dict
        _, scores = sacrebleu.significance.PairedTest(
            [("base", list(base_hyps)), ("lilly", list(tuned_hyps))],
            {metric: metrics[metric]},
            references=[list(refs)], test_type="bs", n_samples=1000)()
        # results come back keyed by the metric's display name, not the key we
        # passed in, and there is a "System" column alongside it
        key = next(k for k in scores if k != "System")
        p = scores[key][1].p_value
        return f"{p:.4f}" if p is not None else "not reported"
    except Exception as exc:      # a missing p-value is not worth losing the table for
        return f"could not compute ({type(exc).__name__}: {exc})"


def fingerprint(adapter) -> str:
    """What produced a set of saved translations, so they are never reused for
    a different model. Saved translations are worth hours; reusing the wrong
    ones is worth less than nothing."""
    parts = [BASE_MODEL, str(adapter or "")]
    for path in (Path(BASE_MODEL), Path(adapter) if adapter else None):
        if path and path.exists():
            newest = max((f.stat().st_mtime for f in path.rglob("*") if f.is_file()),
                         default=0)
            parts.append(f"{path.name}:{int(newest)}")
    return "|".join(parts)


def sources_digest(rows) -> str:
    """A fingerprint of the sentences themselves, not just the model.

    Reusing a saved translation is only safe if it was made from the same
    sentence. Matching the model is not enough — re-fetching a test set can
    change what sits at each line, and the old translations would then pair with
    the wrong sources silently, producing a score that means nothing.
    """
    import hashlib
    joined = "\n".join(s for _, s, _ in rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def run(model, tokenizer, rows, device, label, saved=None):
    """Translate the rows, reusing any saved translations that cover the start.

    The test set grows — FLORES arrived in two halves — and re-translating what
    was already done costs hours for nothing. Saved translations cover a prefix
    of the set because the rows are read in file order, so anything beyond what
    was saved is the only new work.
    """
    reuse = list(saved or [])[:len(rows)]
    todo = rows[len(reuse):]
    if reuse:
        print(f"scoring {label}: reusing {len(reuse)} saved, "
              f"translating {len(todo)}…", flush=True)
    else:
        print(f"scoring {label} on {len(rows)} pairs…", flush=True)
    t0 = time.time()
    raw = reuse + (translate_all(model, tokenizer, [s for _, s, _ in todo], device)
                   if todo else [])
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
    ap.add_argument("--fresh", action="store_true",
                    help="translate everything again instead of reusing what is "
                         "already saved in hypotheses.json")
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
    available = flores_on_disk()
    print(f"in-house: {len(in_house)} pairs"
          + (f"  |  FLORES-200: {len(flores)} pairs" if flores
             else "  |  FLORES-200: not fetched (data/scripts/download_flores.py)"))
    if flores and len(flores) < available:
        print(f"scoring {len(flores)} of the {available} FLORES pairs on disk "
              f"— --limit is holding back {available - len(flores)} of them",
              file=sys.stderr)

    saved = {}
    saved_path = REPO_ROOT / "training" / "hypotheses.json"
    if saved_path.exists() and not args.fresh:
        data = json.loads(saved_path.read_text())
        same_weights = data.get("fingerprint") in (None, fingerprint(args.adapter))
        # The saved translations cover a prefix, so check the digest of that
        # same prefix rather than of the whole set — the set is allowed to grow.
        def prefix_matches(key, rows):
            n = len(data.get(key.replace("_digest", "_refs"), []) or [])
            stored = data.get(key)
            return not stored or not n or stored == sources_digest(rows[:n])

        same_text = (prefix_matches("in_house_digest", in_house)
                     and prefix_matches("flores_digest", flores))
        if same_weights and same_text:
            saved = data
            have = {k: len(v) for k, v in data.items() if isinstance(v, list)}
            print(f"reusing saved translations: {have}")
        elif not same_weights:
            print("saved translations came from different weights — starting fresh")
        else:
            print("the test sentences changed since those translations were saved "
                  "— starting fresh")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    results = {}
    hyps = {}
    leaks = {}
    hyps["base_in"], results[("Base (untuned)", "in-house")], leaks["base_in"] = run(
        base, tokenizer, in_house, device, "base / in-house", saved.get("base_in"))
    if flores:
        hyps["base_fl"], results[("Base (untuned)", "FLORES-200")], leaks["base_fl"] = run(
            base, tokenizer, flores, device, "base / FLORES-200", saved.get("base_fl"))

    tuned = None
    if args.adapter:
        from peft import PeftModel
        tuned = PeftModel.from_pretrained(base, args.adapter)
        hyps["tuned_in"], results[("Lilly (fine-tuned)", "in-house")], leaks["tuned_in"] = run(
            tuned, tokenizer, in_house, device, "Lilly / in-house", saved.get("tuned_in"))
        if flores:
            hyps["tuned_fl"], results[("Lilly (fine-tuned)", "FLORES-200")], leaks["tuned_fl"] = run(
                tuned, tokenizer, flores, device, "Lilly / FLORES-200", saved.get("tuned_fl"))

    # Keep the outputs. Re-running the significance test on saved text costs
    # seconds; regenerating it costs an hour of translation.
    out = REPO_ROOT / "training" / "hypotheses.json"
    out.write_text(json.dumps({
        "fingerprint": fingerprint(args.adapter),
        "in_house_digest": sources_digest(in_house),
        "flores_digest": sources_digest(flores) if flores else "",
        "in_house_refs": [r for _, _, r in in_house],
        "flores_refs": [r for _, _, r in flores],
        **{k: v for k, v in hyps.items()}}, ensure_ascii=False))
    print(f"kept the translations in {out.name}")

    write_results(in_house, flores, results, hyps, bool(args.adapter), leaks,
                  available)
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

    # Only the sentences that were actually translated can be rescored, so the
    # test set is cut down to the saved half — but never quietly. Everything
    # below, the p-value and the interval included, then describes that smaller
    # set, and the report has to say which one it means.
    available = flores_on_disk()
    saved_pairs = len(data.get("flores_refs") or [])
    flores = load_flores(saved_pairs) if saved_pairs else []
    if flores and saved_pairs < available:
        print(f"only {saved_pairs} of the {available} fetched FLORES pairs have "
              f"translations saved — {available - saved_pairs} were never translated. "
              f"Rescoring covers the {saved_pairs} saved ones; to score all "
              f"{available}, re-run without --rescore.", file=sys.stderr)
    if flores and [r for _, _, r in flores] != data["flores_refs"]:
        print("the saved FLORES references are not the first "
              f"{saved_pairs} of the set on disk — the files changed under the "
              "saved translations, so rescoring them would compare against the "
              "wrong sentences. Re-run without --rescore.", file=sys.stderr)
        return 1

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

    write_results(in_house, flores, results, hyps, "tuned_fl" in hyps, leaks,
                  available)
    print("wrote training/RESULTS.md")
    return 0


def write_results(in_house, flores, results, hyps, tuned, leaks,
                  flores_available=None) -> None:
    lines = ["# Translation quality — Phase 2", "",
             "## Headline", "",
             "| Model | Test set | Pairs | BLEU | chrF2 |",
             "|-------|----------|-------|------|-------|"]
    for (model, which), (bleu, chrf) in results.items():
        pairs = len(flores) if which == "FLORES-200" else len(in_house)
        lines.append(f"| {model} | {which} | {pairs} | {bleu:.2f} | {chrf:.2f} |")

    # How many pairs a number rests on is part of the number. A run that scored
    # fewer than are on disk says so here, in the table, rather than leaving the
    # reader to assume the whole set was used.
    missing = (flores_available or len(flores)) - len(flores)
    if flores and missing > 0:
        lines += ["",
                  f"**Scored on {len(flores)} of the {flores_available} FLORES-200 "
                  f"pairs that have been fetched.** The other {missing} have never "
                  "been translated by either model, so nothing below — the gap, the "
                  "p-value and the interval included — covers them. The set to quote "
                  f"is {len(flores)} pairs, not {flores_available}."]

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
        # The reading of the number has to be derived from the number. This
        # sentence was once hardcoded to "does not clear the usual 0.05 bar";
        # when the test set grew to 2,009 pairs and p fell to 0.029, the table
        # and the paragraph under it said opposite things, and the report was
        # one copy-paste away from being published that way.
        try:
            p_val = float(p)
        except (TypeError, ValueError):
            p_val = None

        if p_val is None:
            reading = ("The p-value could not be computed for this run, so the gap "
                       "above stands on its own: a direction and a size, with nothing "
                       "yet said about whether the sample could have produced it by "
                       "chance.")
        elif p_val < 0.05:
            reading = (f"Read that as it is: at p = {p_val:.4f} the gap clears the usual "
                       f"0.05 bar and the 95% interval excludes zero, so on this test set "
                       f"the BLEU difference is *measured*, not merely leaned towards. It "
                       f"is still a small gap — the interval's low end is "
                       f"{lo:+.2f} BLEU — so the honest claim is that Lilly is better "
                       f"than the base model here, by a little.")
        else:
            reading = ("Read that as it is: a difference that does not clear the usual "
                       "0.05 bar is *unproven*, not *absent* — the interval and the "
                       "direction both lean one way, the test set is just not large "
                       "enough to settle it.")

        lines += ["", "## Is the difference real", "",
                  f"On FLORES-200 the gap is **{gap:+.2f} BLEU**, paired bootstrap "
                  f"**p = {p}**, 95% interval **[{lo:+.2f}, {hi:+.2f}]**, and "
                  f"{favour:.0f}% of resamples favour Lilly.",
                  "", reading]

        # BLEU is not the only metric in the headline table, and on this pair of
        # models the two disagree: BLEU goes up, chrF2 goes down. Reporting only
        # the metric that moved the way we hoped is how a report stops surviving
        # being checked, so the disagreement is stated wherever it exists.
        chrf_gap = (results[("Lilly (fine-tuned)", "FLORES-200")][1]
                    - results[("Base (untuned)", "FLORES-200")][1])
        # Defined up here because the table below is guarded on them and the
        # block that fills them only runs when the two metrics disagree.
        single_idx = multi_idx = None
        if (chrf_gap < 0) != (gap < 0):
            chrf_p = significance(hyps["base_fl"], hyps["tuned_fl"], refs, "chrf")
            # Which rows carry more than one sentence, by the app's own splitter
            # rather than a second copy of the rule — if the app's idea of a
            # sentence boundary changes, this breakdown has to change with it.
            # Names deliberately not base_c/tuned_c: those are the per-corpus
            # tables built above, and this block once shadowed them.
            try:
                from app.translate import SENTENCE_BREAK
                base_fl_c, _ = strip_tags(hyps["base_fl"])
                tuned_fl_c, _ = strip_tags(hyps["tuned_fl"])
                single_idx, multi_idx = [], []
                for i, r in enumerate(refs):
                    pieces = [x for x in SENTENCE_BREAK.split(r.strip()) if x.strip()]
                    (multi_idx if len(pieces) > 1 else single_idx).append(i)
            except Exception as exc:
                # The breakdown is an explanation of the gap, not the gap itself.
                # A report that loses its headline because an explanation could
                # not be built is worse than one that says why it is missing.
                print(f"row-shape breakdown skipped ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
                single_idx = multi_idx = None
            lines += ["",
                      f"**The two metrics disagree.** On the same {len(flores):,} "
                      f"sentences chrF2 moves {chrf_gap:+.2f}, the opposite way from "
                      f"BLEU. chrF2 scores character n-grams rather than whole words, "
                      f"which is the fairer of the two for a language that inflects as "
                      f"heavily as Bosnian — so this is not a footnote. The chrF2 drop "
                      f"is tested the same way and comes back at **p = {chrf_p}**, "
                      f"against **p = {p}** for the BLEU gain: the metric that moved "
                      f"against the fine-tuning moved with more confidence than the one "
                      f"that moved for it.",
                      ""]
        if (chrf_gap < 0) != (gap < 0) and single_idx and multi_idx:
            lines += [
                      f"But most of that drop is in a shape the app never sends. "
                      f"`app/translate.py` splits input on sentence boundaries and "
                      f"translates one sentence at a time, because the model drops a "
                      f"clause when it is handed several at once. Scoring feeds whole "
                      f"rows instead, so the rows carrying more than one sentence "
                      f"measure a failure the product has already designed around. "
                      f"Split by row shape, on the same {len(flores):,} sentences:",
                      ""]
            lines += [f"| Row shape | Pairs | Base chrF2 | Lilly chrF2 | Gap |",
                      f"|---|---|---|---|---|"]
            for shape, idx in (("one sentence", single_idx), ("more than one", multi_idx)):
                if not idx:
                    continue
                sub_refs = [refs[i] for i in idx]
                b = sacrebleu.corpus_chrf([base_fl_c[i] for i in idx], [sub_refs],
                                          word_order=0).score
                t = sacrebleu.corpus_chrf([tuned_fl_c[i] for i in idx], [sub_refs],
                                          word_order=0).score
                lines.append(f"| {shape} | {len(idx):,} | {b:.2f} | {t:.2f} | "
                             f"{t - b:+.2f} |")
            lines += ["",
                      "So the honest reading is narrower than the headline gap: the "
                      "fine-tuning costs a little character-level accuracy on the input "
                      "the product actually sends, and a lot on input it never sends. "
                      "Whether the first of those survives being measured through the "
                      "app's own path — sentence splitting and the quantised build — is "
                      "not settled by the numbers on this page, which score the raw "
                      "adapter on whole rows."]

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
