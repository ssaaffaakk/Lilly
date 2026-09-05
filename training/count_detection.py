#!/usr/bin/env python3
"""The count behind detection recall R_d — two blind counters, agreed words only.

`training/measure_detection.py` draws every box a detector produced onto the
photograph and computes nothing, because deciding whether a box covers a
particular word of the answer key means reading the box, and the only reader
available is the thing under test. This script hands that decision to two
counters who read with their eyes (a person, or a vision agent that never sees
the reader's text), and combines their answers the way the answer key itself
was built: a word is covered when both counters say so.

    python3 training/count_detection.py sheet --arm <overlay dir name> --counter a [--size 14]
        writes detection-count/<arm>/counter-a-NN.todo.json: one entry per
        photograph with agreed words — the photograph, its overlay, the key's
        words, and three empty lists to sort them into

    (each counter fills covered / missed / not_located for every word and
     saves counter-a-NN.json beside it; counter b is a different person or
     agent, same procedure, no peeking — training/transcribe/COUNT-BRIEF.md)

    python3 training/count_detection.py check --arm <arm> --counter a
        refuses holes, invented words, and a photograph whose three lists do
        not add up to the key's multiset

    python3 training/count_detection.py validity --arm <arm> --cache data/ocr/real-photos/reader-output-<x>.json
        the pre-registered check: the text in the arm's boxes file (what the
        detector-plus-recogniser returned when the overlays were drawn) must
        carry, photograph by photograph, the same words as the scorer's cache
        for that arm; otherwise the detector drawn is not the one scored

    python3 training/count_detection.py merge --arm <arm> [--pooled label=value ...]
        prints, per counter and agreed, the covered count out of 373 (R_d),
        the disagreements, and recognition-given-detection = pooled / R_d for
        every --pooled figure named; appends a section to --out

The three buckets, per word of the key (a multiset: "the" twice is two
judgements):
  covered      a red box overlaps the glyphs of that word, even partially
  missed       the word is visible and no box touches it
  not_located  the counter could not find the word on the photograph at all
For R_d only `covered` counts; `not_located` is reported so it cannot hide
inside `missed`.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL = REPO_ROOT / "data" / "ocr" / "real-photos"
BUCKETS = ("covered", "missed", "not_located")


def key_words(truth: Path) -> dict:
    data = json.loads(truth.read_text(encoding="utf-8"))
    return {name: [w for line in e.get("lines", []) for w in line.split()]
            for name, e in data["photos"].items()}


def arm_dir(arm: str) -> Path:
    d = REAL / "detection-count" / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_sheet(a) -> int:
    words = key_words(a.truth)
    names = [n for n in a.sample.read_text(encoding="utf-8").split() if words.get(n)]
    overlays = REAL / "detection-overlays" / a.arm
    missing = [n for n in names if not (overlays / f"{Path(n).stem}.jpg").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} overlays missing under {overlays}: {missing[:3]}")
    if a.counter == "b":
        names = names[a.offset:] + names[:a.offset]
    out = arm_dir(a.arm)
    for i in range(0, len(names), a.size):
        batch = [{"filename": n, "photo": str(a.photos / n),
                  "overlay": str(overlays / f"{Path(n).stem}.jpg"),
                  "words": words[n], "opened": False,
                  "covered": [], "missed": [], "not_located": []}
                 for n in names[i:i + a.size]]
        p = out / f"counter-{a.counter}-{i // a.size:02d}.todo.json"
        p.write_text(json.dumps(batch, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(p, len(batch), "photographs,", sum(len(e["words"]) for e in batch), "words")
    print(f"{len(names)} photographs with agreed words, {sum(len(words[n]) for n in names)} words in the key")
    return 0


def load_counter(arm: str, counter: str) -> dict:
    entries = {}
    for f in sorted(arm_dir(arm).glob(f"counter-{counter}-*.json")):
        if not re.fullmatch(rf"counter-{counter}-\d\d\.json", f.name):
            continue
        for e in json.loads(f.read_text(encoding="utf-8")):
            entries[e.get("filename")] = e
    return entries


def problems(entries: dict, words: dict, names: list, label: str) -> list:
    out = []
    for n in names:
        e = entries.get(n)
        if e is None:
            out.append(f"{label}: {n} has no entry")
            continue
        if not e.get("opened"):
            out.append(f"{label}: {n} not marked opened")
        sorted_ = Counter(w for b in BUCKETS for w in e.get(b, []))
        if sorted_ != Counter(words[n]):
            extra = list((sorted_ - Counter(words[n])).elements())[:3]
            lost = list((Counter(words[n]) - sorted_).elements())[:3]
            out.append(f"{label}: {n} lists do not add up to the key (extra {extra}, missing {lost})")
    for n in entries:
        if n not in names:
            out.append(f"{label}: {n} is not a photograph with agreed words")
    return out


def cmd_check(a) -> int:
    words = key_words(a.truth)
    names = [n for n in a.sample.read_text(encoding="utf-8").split() if words.get(n)]
    bad = problems(load_counter(a.arm, a.counter), words, names, f"counter {a.counter}")
    for line in bad:
        print("  " + line)
    print(f"counter {a.counter}: {len(bad)} problems")
    return 1 if bad else 0


def cmd_validity(a) -> int:
    boxes = json.loads((REAL / f"detection-boxes-{a.arm}.json").read_text(encoding="utf-8"))
    cache = json.loads(a.cache.read_text(encoding="utf-8"))
    readings = cache.get("readings", cache)
    names = [n for n in a.sample.read_text(encoding="utf-8").split()]
    print(f"boxes file: {boxes.get('_reader')}; cache stamp: {cache.get('reader')}")
    same, differ, absent = 0, [], []
    for n in names:
        if n not in boxes:
            absent.append(n)
            continue
        drawn = Counter(w for b in boxes[n]["boxes"] for w in str(b["text"]).split())
        scored = Counter(str(readings.get(n, "")).split())
        if drawn == scored:
            same += 1
        else:
            differ.append((n, list((drawn - scored).elements())[:4], list((scored - drawn).elements())[:4]))
    for n, extra, lost in differ:
        print(f"  DIFFERS {n}: drawn-only {extra}  scored-only {lost}")
    print(f"{same} photographs identical, {len(differ)} differ, {len(absent)} not drawn yet")
    return 1 if differ or absent else 0


def cmd_merge(a) -> int:
    words = key_words(a.truth)
    names = [n for n in a.sample.read_text(encoding="utf-8").split() if words.get(n)]
    ca, cb = load_counter(a.arm, "a"), load_counter(a.arm, "b")
    bad = problems(ca, words, names, "counter a") + problems(cb, words, names, "counter b")
    if bad:
        for line in bad:
            print("  " + line)
        return 1
    total = sum(len(words[n]) for n in names)
    rows, tot = [], Counter()
    for n in names:
        k = Counter(words[n])
        ac, bc = Counter(ca[n]["covered"]), Counter(cb[n]["covered"])
        agreed = sum((ac & bc).values())
        agreed_not = sum(min(k[w] - ac[w], k[w] - bc[w]) for w in k)
        r = {"words": len(words[n]), "a": sum(ac.values()), "b": sum(bc.values()), "agreed": agreed,
             "disagree": len(words[n]) - agreed - agreed_not,
             "not_located": len(ca[n]["not_located"]) + len(cb[n]["not_located"])}
        rows.append((n, r)); tot.update(r)
    rd = tot["agreed"] / total
    lines = [f"## Detection recall — {a.arm}", "",
             f"Key: {total} agreed words on {len(names)} photographs (`{a.truth.relative_to(REPO_ROOT)}`). "
             "Two blind counters; a word is covered when both say a box overlaps it.", "",
             "| | covered | of | R_d |", "|---|---|---|---|",
             f"| counter a | {tot['a']} | {total} | {100 * tot['a'] / total:.1f}% |",
             f"| counter b | {tot['b']} | {total} | {100 * tot['b'] / total:.1f}% |",
             f"| **agreed (R_d)** | **{tot['agreed']}** | {total} | **{100 * rd:.1f}%** |", "",
             f"Disagreements: {tot['disagree']} of {total} words ({100 * tot['disagree'] / total:.1f}%); "
             f"not located by a counter: {tot['not_located']} judgements.", ""]
    if a.pooled:
        lines += ["| pooled recall | R_d | recognition given detection |", "|---|---|---|"]
        for spec in a.pooled:
            label, value = spec.split("=", 1)
            v = float(value)
            lines.append(f"| {label}: {v:.1f}% | {100 * rd:.1f}% | {100 * (v / 100) / rd:.1f}% |")
        lines.append("")
    lines += ["| photograph | words | a | b | agreed | disagree |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {r['words']} | {r['a']} | {r['b']} | {r['agreed']} | {r['disagree']} |" for n, r in rows]
    lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with a.out.open("a", encoding="utf-8") as fh:
            fh.write("\n" + text)
        print(f"appended to {a.out.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, default=REAL / "truth.json")
    ap.add_argument("--sample", type=Path, default=REAL / "scored-sample.txt")
    ap.add_argument("--photos", type=Path, default=REAL / "scored")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet"); s.add_argument("--arm", required=True); s.add_argument("--counter", choices=["a", "b"], required=True)
    s.add_argument("--size", type=int, default=14); s.add_argument("--offset", type=int, default=7)
    c = sub.add_parser("check"); c.add_argument("--arm", required=True); c.add_argument("--counter", choices=["a", "b"], required=True)
    v = sub.add_parser("validity"); v.add_argument("--arm", required=True); v.add_argument("--cache", type=Path, required=True)
    m = sub.add_parser("merge"); m.add_argument("--arm", required=True)
    m.add_argument("--pooled", action="append", metavar="LABEL=PERCENT", help="pooled recall figures to divide by R_d")
    m.add_argument("--out", type=Path, help="markdown file to append the section to")
    a = ap.parse_args()
    return {"sheet": cmd_sheet, "check": cmd_check, "validity": cmd_validity, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
