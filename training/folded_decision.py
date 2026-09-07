#!/usr/bin/env python3
"""Decide the step-7 fine-tune on the product's meaning metric (folded).

Pre-registered in training/PREREGISTRATION.md, "the fine-tune on the meaning
(folded) metric". Re-scores the already-read caches — no photograph is read
again — for the fine-tune at the step-7a floor 0.94 against the shipped reader
at 0.9, on test-v2, and reports the paired per-photograph interval on three
metrics:

  exact        all truth words, diacritics kept   (the step-7a decider; sanity)
  folded-all   all truth words, diacritics folded  (the meaning metric — DECIDES)
  folded-dia   diacritic truth words only, folded  (evaluate_ocr's "blind" row)

The bar (pre-registered): meaning found per photograph must rise, the paired
95% bootstrap interval of the folded-all per-photograph difference excluding
zero; strict exact invented ≤ 450 (unchanged, computed here for completeness).

    python3 training/folded_decision.py
"""
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "data" / "ocr" / "real-photos" / "test-v2"
SHIP_CACHE = V2 / "reader-output-paddle-v6-floor0.9.json"
CAND_CACHE = REPO / "training" / "paddle-finetune-floor" / "cache-test-v2-floor0.94.json"
TRUTH = V2 / "truth-v2.json"
OUT = REPO / "training" / "paddle-finetune-floor" / "folded-decision.json"

DIACRITICS = "čćđšžČĆĐŠŽ"
FOLD = str.maketrans({"č": "c", "ć": "c", "đ": "d", "š": "s", "ž": "z",
                      "Č": "c", "Ć": "c", "Đ": "d", "Š": "s", "Ž": "z"})
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def words(text: str) -> list:
    return [w.lower() for w in WORD.findall(unicodedata.normalize("NFC", text))]


def has_diacritic(w: str) -> bool:
    return any(c in DIACRITICS for c in w)


def fold(w: str) -> str:
    return w.translate(FOLD)


def recall(want: list, found: list, key=lambda w: w) -> tuple:
    a = Counter(key(w) for w in want)
    b = Counter(key(w) for w in found)
    return sum((a & b).values()), sum(a.values())


def readings(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return (raw.get("readings", raw), raw.get("reader"))


def interval(per_ship: dict, per_cand: dict, resamples=10000, seed=0) -> dict:
    names = sorted(set(per_ship) & set(per_cand))
    deltas = [100 * (per_cand[n][0] / per_cand[n][1] - per_ship[n][0] / per_ship[n][1])
              for n in names if per_ship[n][1] and per_cand[n][1]]
    k = len(deltas)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(deltas) for _ in range(k)) / k for _ in range(resamples))
    lo, hi = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    frac_le0 = sum(m <= 0 for m in means) / resamples
    p = 2 * min(frac_le0, 1 - frac_le0)
    return {"n": k, "mean_delta": sum(deltas) / k, "lo": lo, "hi": hi,
            "up": sum(d > 0 for d in deltas), "down": sum(d < 0 for d in deltas), "p": p}


def pooled(per: dict) -> float:
    h = sum(v[0] for v in per.values()); n = sum(v[1] for v in per.values())
    return 100 * h / n if n else float("nan")


def per_photo(per: dict) -> float:
    vals = [v[0] / v[1] for v in per.values() if v[1]]
    return 100 * sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    ship, ship_id = readings(SHIP_CACHE)
    cand, cand_id = readings(CAND_CACHE)
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))["photos"]

    metrics = ("exact", "folded_all", "folded_dia")
    ship_per = {m: {} for m in metrics}
    cand_per = {m: {} for m in metrics}
    ship_inv = cand_inv = 0
    scored = 0
    for name, entry in truth.items():
        want = words(" ".join(entry["lines"]))
        if not want or name not in ship or name not in cand:
            continue
        scored += 1
        fs, fc = words(ship[name]), words(cand[name])
        dia = [w for w in want if has_diacritic(w)]
        for per, found in ((ship_per, fs), (cand_per, fc)):
            per["exact"][name] = list(recall(want, found))
            per["folded_all"][name] = list(recall(want, found, key=fold))
            if dia:
                per["folded_dia"][name] = list(recall(dia, found, key=fold))
        ship_inv += max(0, len(fs) - recall(want, fs)[0])
        cand_inv += max(0, len(fc) - recall(want, fc)[0])

    result = {"scored_photographs": scored, "shipped_reader": ship_id, "fine_tune_reader": cand_id,
              "invented": {"shipped": ship_inv, "fine_tune": cand_inv, "bar": 450},
              "metrics": {}}
    for m in metrics:
        iv = interval(ship_per[m], cand_per[m])
        result["metrics"][m] = {
            "shipped_per_photo": per_photo(ship_per[m]), "fine_tune_per_photo": per_photo(cand_per[m]),
            "shipped_pooled": pooled(ship_per[m]), "fine_tune_pooled": pooled(cand_per[m]),
            "paired": iv}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"scored {scored} photographs; shipped `{ship_id}` @0.9 vs fine-tune `{cand_id}` @0.94\n")
    print(f"invented (strict, exact): shipped {ship_inv}  fine-tune {cand_inv}  (bar ≤ 450) "
          f"{'HOLDS' if cand_inv <= 450 else 'FAILS'}\n")
    header = "%-12s %13s %16s %9s %18s %9s" % (
        "metric", "shipped/photo", "fine-tune/photo", "paired D", "95% interval", "up/down")
    print(header)
    for m in metrics:
        d = result["metrics"][m]; iv = d["paired"]
        ci = "%+.1f to %+.1f" % (iv["lo"], iv["hi"])
        ud = "%d/%d" % (iv["up"], iv["down"])
        print("%-12s %12.1f%% %15.1f%% %+8.1f %18s %9s" % (
            m, d["shipped_per_photo"], d["fine_tune_per_photo"], iv["mean_delta"], ci, ud))
    dec = result["metrics"]["folded_all"]["paired"]
    rise = dec["lo"] > 0
    print(f"\nDECIDER folded-all: rise bar (95% excludes zero) {'HOLDS' if rise else 'DOES NOT HOLD'} "
          f"(interval {dec['lo']:+.1f} to {dec['hi']:+.1f}); invented ≤450 "
          f"{'HOLDS' if cand_inv <= 450 else 'FAILS'}")
    print(f"SHIP on meaning metric: {'YES — both bars hold' if (rise and cand_inv <= 450) else 'NO'}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
