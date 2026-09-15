#!/usr/bin/env python3
"""§2 — decode-time Croatian suppression, measured on the SERVED en->bs build.

A candidate lever from docs (the "what would raise the numbers" report and the
14 Sep strategy note): pass the token sequences of the Croatian forms into
CTranslate2's `suppress_sequences` at decode time, so the decoder is forced to
the Bosnian alternative without any training. Zero GPU, done in the served path.

This script MEASURES it rather than assuming it. It reuses the committed,
pre-registered form-rate machinery (`training/bosnian_form_rate.py`: the same
bench cases, audit filter, and Bosnian/variant/silent matcher) but drives it
through `app.translate.Engine` (int8 CTranslate2, the path a user meets), which
is where `suppress_sequences` applies. Two builds, two conditions each:

  served (LoRA)   models/lilly/translator-en-bs        the shipped reply model
  base (untuned)  models/lilly/translator-en-bs-base   its int8 comparison

The suppression list is the ORACLE UPPER BOUND: every Croatian counterpart the
surviving targets weigh against, tokenised to the model's own tokens (nominative
plus case folds). A deployable fixed list cannot do better than knowing every
counterpart in advance, so if this barely moves, §2 has no headroom.

Reported per build, OFF vs ON:
  form rate     Bosnian form / decided, on the surviving bench targets
  chrF2         whole output vs each case's professional `bs` reference (a
                harm check: does forcing a token away cost fluency?)
  flips         variant->bosnian (the win §2 promises) and bosnian->variant (harm)

Result, 15 Sep 2026 (`training/RESULTS-suppress-en-bs.md`): on the served build
0 of 308 outputs change and form rate holds at 99.6% — the LoRA already writes
the Bosnian forms, so there is nothing to suppress. §2 is subsumed by the
fine-tune and is not worth shipping.

    .venv/bin/python3 training/measure_suppress_en_bs.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

import sacrebleu
from app.translate import Engine
from bosnian_form_rate import load_targets, audit, mark, fold, _is_croatian

BUILDS = {
    "served (LoRA)": None,                                     # DIRECTIONS default dir
    "base (untuned)": REPO / "models" / "lilly" / "translator-en-bs-base",
}


class SuppressWrapper:
    """Wraps a CT2 Translator so every translate_batch injects suppress_sequences."""
    def __init__(self, translator, suppress):
        self._t = translator
        self._suppress = suppress or None

    def translate_batch(self, batch, **kw):
        if self._suppress:
            kw["suppress_sequences"] = self._suppress
        return self._t.translate_batch(batch, **kw)


def tokens_of(engine, word):
    ids = engine.tokenizer.encode(word, add_special_tokens=False)
    return engine.tokenizer.convert_ids_to_tokens(ids)


def form_rate(marks):
    b = marks.count("bosnian"); v = marks.count("variant"); s = marks.count("silent")
    decided = b + v
    return (b / decided if decided else float("nan")), b, v, s


def one_build(label, directory, kept, suppress_words):
    eng = Engine(direction="en-bs", directory=directory)
    real = eng.translator
    suppress = []
    for w in suppress_words:
        for form in (w, w.capitalize(), w.lower()):
            toks = tokens_of(eng, form)
            if toks and toks not in suppress:
                suppress.append(toks)

    sources = sorted({t["en"] for t in kept})
    eng.translator = real
    out_off = {en: eng.translate(en) for en in sources}
    eng.translator = SuppressWrapper(real, suppress)
    out_on = {en: eng.translate(en) for en in sources}
    eng.translator = real

    fr_off, b0, v0, s0 = form_rate([mark(out_off[t["en"]], t) for t in kept])
    fr_on, b1, v1, s1 = form_rate([mark(out_on[t["en"]], t) for t in kept])
    flips_good = sum(1 for t in kept
                     if mark(out_off[t["en"]], t) == "variant" and mark(out_on[t["en"]], t) == "bosnian")
    flips_bad = sum(1 for t in kept
                    if mark(out_off[t["en"]], t) == "bosnian" and mark(out_on[t["en"]], t) == "variant")
    changed = sum(1 for en in sources if out_off[en] != out_on[en])
    refs = {t["en"]: t["bs"] for t in kept}
    keys = sorted(refs)
    chrf_off = sacrebleu.corpus_chrf([out_off[k] for k in keys], [[refs[k] for k in keys]], word_order=0).score
    chrf_on = sacrebleu.corpus_chrf([out_on[k] for k in keys], [[refs[k] for k in keys]], word_order=0).score

    print(f"\n===== {label} — suppress A/B (oracle upper bound, {len(suppress)} seqs) =====")
    print(f"{'':16}{'OFF':>12}{'ON':>12}")
    print(f"{'form rate':16}{fr_off*100:>11.1f}%{fr_on*100:>11.1f}%")
    print(f"{'  bosnian':16}{b0:>12}{b1:>12}")
    print(f"{'  variant':16}{v0:>12}{v1:>12}")
    print(f"{'  silent':16}{s0:>12}{s1:>12}")
    print(f"{'chrF2 (bench)':16}{chrf_off:>12.2f}{chrf_on:>12.2f}")
    print(f"outputs changed: {changed}/{len(sources)}  |  variant->bosnian: {flips_good}  |  bosnian->variant: {flips_bad}")


def main():
    _, targets = load_targets()
    kept, dropped = audit(targets)
    print(f"surviving targets: {len(kept)} (dropped {sum(dropped.values())}: {dropped})")
    cro = sorted({t["variant"] for t in kept if t["variant"] and _is_croatian(fold(t["variant"]))})
    print(f"Croatian counterparts in oracle list: {len(cro)} words")
    print(f"unique en sources: {len({t['en'] for t in kept})} (x2 conditions x{len(BUILDS)} builds)")
    for label, directory in BUILDS.items():
        one_build(label, directory, kept, cro)


if __name__ == "__main__":
    main()
