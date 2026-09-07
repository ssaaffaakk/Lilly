#!/usr/bin/env python3
"""Bosnian form rate — English to Bosnian, scored on the output side.

`training/bosnian_bench.py` says plainly what it cannot do: "The direction is
bs->en, so nothing Bosnian survives into the output -- you cannot look at English
and ask whether it is ijekavica." That limitation is why the forward direction's
Bosnian claim came back +0.5 points at p = 0.360 and stayed unproven.

It does not apply in this direction. Here Bosnian **is** the output, so the
question is directly checkable: given an English sentence whose professional
Bosnian translation used a Bosnian-only term, does the model write that term or
its Croatian or Serbian counterpart? One target, one bit, nothing averaged away.

`training/PREREGISTRATION.md`, "v3 -- reply -- English to Bosnian", fixes three
things about this measure, and this file is where they are implemented:

  1. The baseline is measured on the base BEFORE the fine-tune is launched.
  2. The cases come from bench/cases.tsv, reused, not rebuilt. `en` is the
     source to feed, `terms` is the Bosnian-only target, `variant_swaps` names
     the counterpart it is weighed against.
  3. A case whose Bosnian and Croatian forms score the same is dropped BEFORE
     the run. `--audit` runs that pass on its own, without a model, so the
     surviving set is fixed and committed before any output exists.

## The metric, defined before the number

Every surviving target is read twice against the same output, with the same
matcher: does the Bosnian surface form appear (word boundary, case folded), and
does its counterpart appear?

    decided        the output contains exactly one of the two
    form rate  =   decided-for-Bosnian / decided
    silent     =   attempted - decided, reported beside it and never averaged in

A target where neither form appears is the instrument being unable to say, not a
miss, and it is reported as its own column -- the same honesty bosnian_bench.py
applies to its 27-of-85. A target where BOTH appear is also undecided: the
output hedged, and counting it either way would be a choice made after the fact.

The matcher is deliberately literal: the exact surface form, word boundary, case
folded, both sides the same way. The pairs in bench/ are inflection matched
(`pobjedu>pobedu`, `dvije>dve`), so a model that writes a different inflection is
lost from BOTH columns equally, and the loss lands in `silent` where it can be
seen. Loosening the matcher to stems is a degree of freedom, and one chosen with
a number in hand is one chosen to be passed.

Usage — the baseline, before any fine-tune exists:

    .venv/bin/python3 training/bosnian_form_rate.py --audit          # no model
    .venv/bin/python3 training/bosnian_form_rate.py --label base

and afterwards, on the candidate:

    .venv/bin/python3 training/bosnian_form_rate.py --adapter models/lilly/adapter-en-bs --label tuned
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

CASES = REPO / "bench" / "cases.tsv"
OUT = REPO / "training" / "form-rate"
LANGUAGE_TAG = re.compile(r"^\s*(>>[a-zA-Z_]+<<\s*)+")

# Counterparts that are Croatian rather than Serbian, by stem. Written here so
# the split can be counted; nothing in the metric depends on it. The list is the
# distinct Croatian side of the pairs bench/ already contains -- month names,
# the Croatian lexicon -- and it is a label on the report, not a filter.
CROATIAN_FORMS = {
    f for stem in (
        "obitelj", "europ", "zrakoplov", "tisuć", "sudjelov", "vlak", "tjedan",
        "tjedn", "siječ", "veljač", "ožuj", "travnj", "svibnj", "lipnj", "srpnj",
        "kolovoz", "rujn", "listopad", "studen", "prosinc", "tvrtk", "glazb",
        "kazališ", "nogomet", "znanstven", "gospodarstv", "tisuc", "točn",
    ) for f in (stem,)
}


def _is_croatian(form: str) -> bool:
    return any(form.startswith(stem) for stem in CROATIAN_FORMS)


def fold(s: str) -> str:
    return s.strip().lower()


def present(word: str, text: str) -> bool:
    """Exact surface form, word boundary, case folded. Both sides, same way."""
    return re.search(rf"(?<!\w){re.escape(fold(word))}(?!\w)", text) is not None


def load_targets():
    """Every (case, bosnian form, counterpart) the bench can offer, pre-audit."""
    with open(CASES, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    targets = []
    for r in rows:
        swaps = {}
        for pair in r["variant_swaps"].split("|"):
            if ">" in pair:
                bos, var = pair.split(">", 1)
                swaps[fold(bos)] = var.strip()
        # `categories` is pipe-aligned with `terms`, one label per target: `yat`
        # marks an ijekavica/ekavica pair whose alternative is Serbian, `lex` a
        # genuine lexical choice. Which of the two a target is decides what a
        # hit can be claimed to show, so it is carried per target, not per case.
        cats = r["categories"].split("|")
        for i, term in enumerate(r["terms"].split("|")):
            term = term.strip()
            if not term:
                continue
            targets.append({
                "case_id": r["case_id"], "source": r["source"],
                "en": r["en"], "bs": r["bs"],
                "term": term, "variant": swaps.get(fold(term), ""),
                "category": cats[i] if i < len(cats) else "",
            })
    return rows, targets


def audit(targets):
    """Point 3 of the pre-registration, run before any model output exists.

    A target survives only if all four hold, and each drop is counted so the
    surviving set can be read as a filter and not as a selection:

      has a counterpart   `variant_swaps` names the Croatian/Serbian form. With
                          no counterpart there is nothing to weigh against.
      forms differ        the two surface forms are not the same string folded.
                          Identical forms score identically by construction.
      not nested          neither form contains the other as a substring under
                          the matcher, so a hit cannot be claimed by both.
      the professional wrote it
                          the Bosnian form is in this case's own `bs` reference.
                          If the translator did not use it, the model writing
                          something else is not being compared to anything.
    """
    kept, dropped = [], {"no counterpart": 0, "same form": 0,
                         "nested forms": 0, "not in reference": 0}
    for t in targets:
        if not t["variant"]:
            dropped["no counterpart"] += 1
            continue
        b, v = fold(t["term"]), fold(t["variant"])
        if b == v:
            dropped["same form"] += 1
            continue
        if present(b, v) or present(v, b):
            dropped["nested forms"] += 1
            continue
        if not present(b, fold(t["bs"])):
            dropped["not in reference"] += 1
            continue
        kept.append(t)
    return kept, dropped


def mark(output: str, target: dict) -> str:
    """bosnian | variant | silent — one bit per target, or the instrument's silence."""
    text = fold(LANGUAGE_TAG.sub("", output))
    b = present(target["term"], text)
    v = present(target["variant"], text)
    if b and not v:
        return "bosnian"
    if v and not b:
        return "variant"
    return "silent"          # neither form, or both — undecided either way


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def translate(sources, adapter=None, tag=">>bos_Latn<<", batch_size=16):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from train_translation import base_model

    base = base_model("en-bs")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"base {base} | adapter {adapter or 'none'} | tag {tag} | {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).to(device)
    model.eval()

    texts = [f"{tag} {s}" if tag else s for s in sources]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    t0 = time.time()
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        enc = tokenizer([texts[j] for j in idx], return_tensors="pt",
                        padding=True, truncation=True, max_length=192).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=192, num_beams=4)
        for j, text in zip(idx, tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out[j] = text
        if (i // batch_size) % 5 == 0:
            print(f"  {min(i + batch_size, len(order))}/{len(order)}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true",
                    help="run the pre-registered case audit only; loads no model")
    ap.add_argument("--adapter", default=None, help="LoRA adapter to put on the base")
    ap.add_argument("--label", default="base", help="names the output files")
    ap.add_argument("--tag", default=">>bos_Latn<<")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows, targets = load_targets()
    kept, dropped = audit(targets)
    cases = len({t["case_id"] for t in kept})
    print(f"bench cases {len(rows)} | targets {len(targets)} | "
          f"discriminating {len(kept)} across {cases} cases")
    for reason, n in dropped.items():
        print(f"  dropped, {reason}: {n}")

    # What a hit can be claimed to show depends on which alternative the target
    # is weighed against, so both splits are reported with the count and not
    # recovered afterwards. bosnian_bench.py's own honest asterisk is that 73 of
    # its 85 targets are yat pairs whose alternative is Serbian, leaving drift
    # toward Croatian almost nothing to land on. The same split is printed here.
    cats = Counter(t["category"] for t in kept)
    croat = sum(1 for t in kept if _is_croatian(fold(t["variant"])))
    print(f"  by category: {dict(cats)}")
    print(f"  counterpart is a listed Croatian form: {croat}/{len(kept)}")

    audit_path = OUT / "audit.json"
    audit_path.write_text(json.dumps(
        {"cases": len(rows), "targets": len(targets), "kept": len(kept),
         "kept_cases": cases, "dropped": dropped, "by_category": dict(cats),
         "croatian_counterparts": croat,
         "distinct_pairs": len({(fold(t["term"]), fold(t["variant"])) for t in kept}),
         "target_ids": [f"{t['case_id']}:{t['term']}" for t in kept]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {audit_path.relative_to(REPO)}")
    if args.audit:
        return 0
    if len(kept) < 30:
        print("fewer than 30 discriminating targets — the instrument cannot speak; "
              "report that, do not report a rate")
        return 1

    use = kept[:args.limit] if args.limit else kept
    outputs = translate([t["en"] for t in use], adapter=args.adapter, tag=args.tag)
    marks = [mark(o, t) for o, t in zip(outputs, use)]
    bos = marks.count("bosnian")
    var = marks.count("variant")
    silent = marks.count("silent")
    decided = bos + var
    rate = bos / decided if decided else 0.0
    lo, hi = wilson(bos, decided)

    print(f"\n=== Bosnian form rate, {args.label} ===")
    print(f"  attempted : {len(use)}")
    print(f"  decided   : {decided}   (silent {silent} — neither form, or both)")
    print(f"  bosnian   : {bos}")
    print(f"  variant   : {var}")
    print(f"  form rate : {rate * 100:.1f}%   95% {lo * 100:.1f}–{hi * 100:.1f}%")

    res = OUT / f"{args.label}.json"
    res.write_text(json.dumps(
        {"label": args.label, "adapter": str(args.adapter) if args.adapter else None,
         "tag": args.tag, "attempted": len(use), "decided": decided,
         "bosnian": bos, "variant": var, "silent": silent,
         "form_rate": rate, "ci95": [lo, hi],
         "marks": [{"case_id": t["case_id"], "term": t["term"],
                    "variant": t["variant"], "mark": m, "output": o}
                   for t, m, o in zip(use, marks, outputs)]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {res.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
