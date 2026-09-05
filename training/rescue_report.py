#!/usr/bin/env python3
"""Report a Cyrillic-rescue run against the shipped configuration, both bars.

    python3 training/rescue_report.py --set the40 \\
        --shipped training/paddle-floor/the40-floor0.9.json --rescue training/paddle-rescue/the40.json \\
        --shipped-cache data/ocr/real-photos/reader-output-paddle-v6-floor0.9.json \\
        --rescue-cache data/ocr/real-photos/reader-output-paddle-v6-rescue.json \\
        --log training/paddle-rescue/the40-rescued.jsonl [--out training/RESULTS-ocr-cyrillic-rescue.md]

Pre-registered in training/PREREGISTRATION.md, "Cyrillic rescue under
PP-OCRv6". The rule can only add words, so the report is the difference:
- the mechanical check that rescue-off equals shipped: every word the shipped
  cache holds for a photograph is in the rescue cache's reading of it, and no
  photograph's hit count fell;
- words found per photograph, pooled, invented, diacritic, folded, both arms;
- the paired per-photograph delta with a 95% percentile bootstrap interval
  (bar 1: the interval excludes zero) and invented against the shipped count
  (bar 2: no increase);
- from the rescue log, how many dropped boxes the Cyrillic recogniser was
  asked about and how many it kept.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.bakeoff_ocr import paired  # noqa: E402


def words_of(text: str) -> Counter:
    return Counter(w.lower() for w in text.replace("\n", " ").split() if w.strip())


def containment(shipped_cache: Path, rescue_cache: Path) -> tuple:
    """Photographs where the shipped reading is not a sub-multiset of the rescue reading."""
    s = json.loads(shipped_cache.read_text(encoding="utf-8"))["readings"]
    r = json.loads(rescue_cache.read_text(encoding="utf-8"))["readings"]
    broken = []
    for name, text in s.items():
        if name not in r:
            broken.append(f"{name}: not in rescue cache")
            continue
        missing = words_of(text) - words_of(r[name])
        if missing:
            broken.append(f"{name}: shipped words missing from rescue reading: {list(missing.elements())[:5]}")
    return len(s), broken


def interval(arm: dict, shipped: dict, resamples: int = 10000, seed: int = 0) -> tuple:
    names = sorted(set(arm["per_photograph"]) & set(shipped["per_photograph"]))
    deltas = [100 * (arm["per_photograph"][n][0] / arm["per_photograph"][n][1]
                     - shipped["per_photograph"][n][0] / shipped["per_photograph"][n][1])
              for n in names if arm["per_photograph"][n][1] and shipped["per_photograph"][n][1]]
    rng = random.Random(seed)
    k = len(deltas)
    means = sorted(sum(rng.choice(deltas) for _ in range(k)) / k for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="label: the40 or test-v2")
    ap.add_argument("--shipped", type=Path, required=True, help="evaluate_ocr --json of the shipped arm")
    ap.add_argument("--rescue", type=Path, required=True, help="evaluate_ocr --json of the rescue arm")
    ap.add_argument("--shipped-cache", type=Path, required=True)
    ap.add_argument("--rescue-cache", type=Path, required=True)
    ap.add_argument("--log", type=Path, help="LILLY_PADDLE_RESCUE_LOG of the rescue run")
    ap.add_argument("--invented-bar", type=int, help="invented words must not exceed this (default: the shipped count)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    s = json.loads(a.shipped.read_text(encoding="utf-8"))
    r = json.loads(a.rescue.read_text(encoding="utf-8"))
    n_cached, broken = containment(a.shipped_cache, a.rescue_cache)
    fell = [n for n in s["per_photograph"] if n in r["per_photograph"]
            and r["per_photograph"][n][0] < s["per_photograph"][n][0]]
    changed = [(n, s["per_photograph"][n][0], r["per_photograph"][n][0], r["per_photograph"][n][1])
               for n in s["per_photograph"] if n in r["per_photograph"]
               and r["per_photograph"][n][0] != s["per_photograph"][n][0]]
    p = paired(r, s)
    lo, hi = interval(r, s)
    bar = a.invented_bar if a.invented_bar is not None else s["invented"]
    asked = kept = 0
    if a.log and a.log.exists():
        for line in a.log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                asked += 1
                kept += bool(json.loads(line).get("kept"))
    rise_ok = lo > 0
    invented_ok = r["invented"] <= bar
    lines = [f"## {a.set} — Cyrillic rescue against the shipped configuration", "",
             f"Shipped arm `{s['reader']}`; rescue arm `{r['reader']}`.", "",
             "**Rescue-off equals shipped, mechanically:** "
             + (f"every shipped word is in the rescue reading on all {n_cached} photographs, and no photograph's "
                f"hit count fell." if not broken and not fell else
                f"**FAILED** — {len(broken)} photographs with shipped words missing, {len(fell)} with fewer hits: "
                f"{broken[:3]} {fell[:3]}"), "",
             "| arm | words per photograph | pooled | invented | diacritic | folded |", "|---|---|---|---|---|---|",
             f"| shipped | **{s['per_photo']:.1f}%** | {s['pooled']:.1f}% | {s['invented']} | {s['diacritic']:.1f}% | {s['folded']:.1f}% |",
             f"| rescue | **{r['per_photo']:.1f}%** | {r['pooled']:.1f}% | {r['invented']} | {r['diacritic']:.1f}% | {r['folded']:.1f}% |", "",
             f"Paired per photograph (n={p['n']}): mean Δ **{p['mean_delta']:+.1f}** points, 95% {lo:+.1f} to {hi:+.1f}, "
             f"{p['up']} up, {p['down']} down, bootstrap p {p['p']:.3f}.", "",
             f"Dropped boxes the Cyrillic recogniser was asked about: {asked}; kept at ≥ 0.9: {kept}."
             if a.log else "No rescue log given.", "",
             f"Photographs whose hit count changed ({len(changed)}):", ""]
    lines += [f"- {n}: {before} → {after} of {need}" for n, before, after, need in changed] or ["- none"]
    lines += ["", "**Bars** (pre-registered): words per photograph must rise with the 95% interval excluding zero — "
              + ("**holds**" if rise_ok else "**does not hold**")
              + f"; invented ≤ {bar} — " + ("**holds**" if invented_ok else f"**does not hold** ({r['invented']})") + ".", ""]
    text = "\n".join(lines)
    print(text)
    if a.out:
        with a.out.open("a", encoding="utf-8") as fh:
            fh.write("\n" + text)
        print(f"appended to {a.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
