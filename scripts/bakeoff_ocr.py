#!/usr/bin/env python3
"""Step 1 of docs/OCR-ROADMAP.md: the engine bake-off, as one command on the Mac.

Four arms, each in its own process, each with its own reading cache:

    lilly      the shipped reader — EasyOCR CRAFT + the fine-tuned recogniser
    stock      LILLY_READER=stock — EasyOCR's own latin_g2, the control
    paddle-v6  LILLY_READER=paddle — PP-OCRv6_medium_det + PP-OCRv6_medium_rec
    paddle-v5  LILLY_PADDLE_VERSION=PP-OCRv5 — PP-OCRv5_server_det + latin_PP-OCRv5_mobile_rec

For each: training/evaluate_ocr.py on the 40 scored photographs through
app.ocr.scan, then training/score_crops.py on the human crops with the split
named, then one timed read. Results land in training/bakeoff/<arm>-*.json and
the report in training/RESULTS-ocr-bakeoff.md, where the pre-registered rule
(training/PREREGISTRATION.md, "v2 — read — bake-off") is applied verbatim.

Own processes because the readers are module-level singletons and torch and
paddle are happier apart. Own caches because evaluate_ocr.py's cache is stamped
with the reader identity now, and one shared file would be thrown away and
re-read four times over.

    pip install paddleocr paddlepaddle        # once, on the Mac; not in requirements.txt
    python3 scripts/bakeoff_ocr.py            # ~an hour: four arms x 40 photographs
    python3 scripts/bakeoff_ocr.py --arms lilly paddle-v5 --limit 3   # smoke
    python3 scripts/bakeoff_ocr.py --assemble-only                    # re-write the report

Not runnable from a Claude Code cloud session: its network policy refuses the
Commons photographs and the weight hosts. The Mac has both.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARMS = {
    "lilly": {},
    "stock": {"LILLY_READER": "stock"},
    "paddle-v6": {"LILLY_READER": "paddle", "LILLY_PADDLE_VERSION": "PP-OCRv6"},
    "paddle-v5": {"LILLY_READER": "paddle", "LILLY_PADDLE_VERSION": "PP-OCRv5"},
}
OUT = REPO_ROOT / "training" / "bakeoff"
REPORT = REPO_ROOT / "training" / "RESULTS-ocr-bakeoff.md"
PHOTOS = REPO_ROOT / "data" / "ocr" / "real-photos" / "scored"
SAMPLE = REPO_ROOT / "data" / "ocr" / "real-photos" / "scored-sample.txt"
LETTERS = "čćđšžČĆĐŠŽ"
# The shipped reader's numbers as published. The run must reproduce them; a
# shipped arm that does not is a changed code path, not a new result.
PUBLISHED = {"per_photo": 54.7, "invented": 180}


def cache_for(arm: str) -> Path:
    base = REPO_ROOT / "data" / "ocr" / "real-photos"
    return base / ("reader-output.json" if arm == "lilly" else f"reader-output-{arm}.json")


def commands(arm: str, limit: int, photo: Path) -> list:
    py = sys.executable
    limit_args = ["--limit", str(limit)] if limit else []
    return [
        [py, "training/evaluate_ocr.py", "--cache", str(cache_for(arm)),
         "--out", str(OUT / f"{arm}-photos.md"), "--json", str(OUT / f"{arm}-photos.json")] + limit_args,
        [py, "training/score_crops.py", "--json", str(OUT / f"{arm}-crops.json"),
         "--time", str(photo)] + limit_args,
    ]


def run_arm(arm: str, limit: int, photo: Path, dry: bool) -> None:
    env = dict(os.environ, **ARMS[arm])
    for cmd in commands(arm, limit, photo):
        shown = " ".join(f"{k}={v}" for k, v in ARMS[arm].items())
        prefix = f"{shown} " if shown else ""
        print(f"\n$ {prefix}{' '.join(cmd)}", flush=True)
        if dry:
            continue
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def timing_photo() -> Path:
    """One scored photograph to time on — the first of the sample that is on disk."""
    for name in SAMPLE.read_text(encoding="utf-8").split("\n"):
        if name.strip() and (PHOTOS / name.strip()).is_file():
            return PHOTOS / name.strip()
    raise SystemExit(f"no scored photograph on disk under {PHOTOS}; run "
                     "data/scripts/restore_scored_photos.py first")


# ----------------------------------------------------------------- the rule

def eligible(arm: dict, shipped: dict) -> bool:
    """Pre-registered: not worse on either row, better on at least one."""
    better_recall = arm["per_photo"] > shipped["per_photo"]
    same_recall = arm["per_photo"] >= shipped["per_photo"]
    fewer_invented = arm["invented"] < shipped["invented"]
    same_invented = arm["invented"] <= shipped["invented"]
    return (better_recall and same_invented) or (same_recall and fewer_invented)


def winner(results: dict) -> str:
    """Among eligible paddle arms: higher per-photo, then fewer invented, then v5."""
    shipped = results["lilly"]["photos"]
    ranked = []
    for arm in ("paddle-v6", "paddle-v5"):
        if arm in results and eligible(results[arm]["photos"], shipped):
            p = results[arm]["photos"]
            ranked.append((-p["per_photo"], p["invented"], 0 if arm == "paddle-v5" else 1, arm))
    return sorted(ranked)[0][3] if ranked else ""


def paired(arm: dict, shipped: dict, resamples: int = 10000, seed: int = 0) -> dict:
    """Per-photograph deltas and a paired bootstrap p. Reported; does not move the bar."""
    names = sorted(set(arm["per_photograph"]) & set(shipped["per_photograph"]))
    deltas = []
    for n in names:
        a_hit, a_need = arm["per_photograph"][n]
        s_hit, s_need = shipped["per_photograph"][n]
        if a_need and s_need:
            deltas.append(100 * (a_hit / a_need - s_hit / s_need))
    if not deltas:
        return {"n": 0, "mean_delta": 0.0, "p": 1.0, "up": 0, "down": 0}
    rng = random.Random(seed)
    k = len(deltas)
    means = []
    for _ in range(resamples):
        means.append(sum(rng.choice(deltas) for _ in range(k)) / k)
    le = sum(m <= 0 for m in means) / resamples
    ge = sum(m >= 0 for m in means) / resamples
    return {"n": k, "mean_delta": sum(deltas) / k, "p": min(1.0, 2 * min(le, ge)),
            "up": sum(d > 0 for d in deltas), "down": sum(d < 0 for d in deltas)}


def class_recall(photos: dict, lo: int, hi: int) -> tuple:
    """Mean per-photograph recall over photographs whose key holds lo..hi words."""
    sub = [(h, n) for h, n in photos["per_photograph"].values() if n and lo <= n <= hi]
    if not sub:
        return (None, 0)
    return (100 * sum(h / n for h, n in sub) / len(sub), len(sub))


def dictionary_check(rec_model: str) -> str:
    """Can the recogniser write our ten letters? Read off its downloaded config.

    Best effort: PaddleX keeps official models under ~/.paddlex/official_models.
    If no config with a character list is found, say so rather than guess.
    """
    root = Path.home() / ".paddlex" / "official_models" / rec_model
    if not root.is_dir():
        return "unverified — model directory not found"
    for cfg in sorted(list(root.rglob("*.yml")) + list(root.rglob("*.yaml"))):
        text = cfg.read_text(encoding="utf-8", errors="replace")
        if "character" not in text:
            continue
        chars = None
        try:
            import yaml
            data = yaml.safe_load(text)
            post = (data or {}).get("PostProcess", {})
            chars = set(post.get("character_dict") or post.get("character_list") or [])
        except Exception:
            chars = None
        if chars is None:
            lines = {ln.strip()[2:] for ln in text.splitlines() if ln.strip().startswith("- ")}
            chars = lines
        missing = [c for c in LETTERS if c not in chars]
        return "all ten letters present" if not missing else f"cannot write {''.join(missing)}"
    return "unverified — no character list in the model config"


# --------------------------------------------------------------- the report

def load_results(arms: list) -> dict:
    results = {}
    for arm in arms:
        photos = OUT / f"{arm}-photos.json"
        crops = OUT / f"{arm}-crops.json"
        if not photos.is_file() or not crops.is_file():
            continue
        results[arm] = {"photos": json.loads(photos.read_text(encoding="utf-8")),
                        "crops": json.loads(crops.read_text(encoding="utf-8"))}
    return results


def assemble(results: dict, limit: int) -> str:
    from training.score_crops import wilson

    if "lilly" not in results:
        raise SystemExit("no lilly arm results — the shipped reader is the bar and must be re-run")
    shipped = results["lilly"]["photos"]
    lines = ["# Engine bake-off — PaddleOCR untrained against the shipped reader", ""]
    if limit:
        lines += [f"**Smoke run with --limit {limit}. Not a result.**", ""]
    lines += [f"Run {time.strftime('%Y-%m-%d %H:%M')} by `scripts/bakeoff_ocr.py`, rule from "
              "`training/PREREGISTRATION.md`, \"v2 — read — bake-off\". Every arm through "
              "`app.ocr.scan` on the same 40 photographs and `app.ocr.read_regions` on the same "
              "crops; `training/bakeoff/<arm>-*.json` are the raw counts.", ""]

    repro = (abs(shipped["per_photo"] - PUBLISHED["per_photo"]) <= 0.05
             and shipped["invented"] == PUBLISHED["invented"])
    if not limit:
        lines += [("The shipped arm reproduces its published 54.7% / 180." if repro else
                   f"**The shipped arm does not reproduce its published numbers** "
                   f"({shipped['per_photo']:.1f}% / {shipped['invented']} against 54.7% / 180). "
                   "Something in the code path changed; find it before comparing."), ""]

    lines += ["## The 40 photographs", "",
              "| arm | reader | words per photograph | pooled | invented words | diacritic words | folded |",
              "|---|---|---|---|---|---|---|"]
    for arm, r in results.items():
        p = r["photos"]
        lines.append(f"| {arm} | `{r['crops'].get('reader', '?')}` | **{p['per_photo']:.1f}%** | "
                     f"{p['pooled']:.1f}% | {p['invented']} | {p['diacritic']:.1f}% | {p['folded']:.1f}% |")
    lines += ["", "## Signs against boards", "",
              "The product reads small signs and street names (docs/OCR-ROADMAP.md, decision 1). "
              "Reported beside the bar, which stays the mean over every photograph.", "",
              "| arm | signs, 1-5 words | short boards, 6-20 | long boards, 21+ |", "|---|---|---|---|"]
    for arm, r in results.items():
        cells = []
        for lo, hi in ((1, 5), (6, 20), (21, 10 ** 9)):
            v, n = class_recall(r["photos"], lo, hi)
            cells.append(f"{v:.1f}% (n={n})" if v is not None else "—")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired against the shipped reader, photograph by photograph", "",
              "Reported; the bar is the table above.", "",
              "| arm | photographs | mean Δ points | better on | worse on | bootstrap p |",
              "|---|---|---|---|---|---|"]
    for arm, r in results.items():
        if arm == "lilly":
            continue
        s = paired(r["photos"], shipped)
        lines.append(f"| {arm} | {s['n']} | {s['mean_delta']:+.1f} | {s['up']} | {s['down']} | {s['p']:.3f} |")

    lines += ["", "## The human crops, split named", "",
              "Only the held-out row compares readers; the shipped reader trained on the other.", "",
              "| arm | held out: exact | held out: folded | 95% (folded) | training-side: folded |",
              "|---|---|---|---|---|"]
    for arm, r in results.items():
        h = r["crops"]["halves"]
        ho, ts = h["held_out"], h["training_side"]
        lo, hi = ho["folded_ci"]
        lines.append(f"| {arm} | {ho['exact']}/{ho['n']} | {ho['folded']}/{ho['n']} | "
                     f"{lo:.0f}–{hi:.0f}% | {ts['folded']}/{ts['n']} |")

    lines += ["", "## Speed and letters", "", "| arm | seconds per photograph (CPU) | recogniser dictionary |", "|---|---|---|"]
    for arm, r in results.items():
        t = r["crops"].get("timing", {})
        secs = f"{t['seconds']:.1f} ({t['photo']}, {t['pixels'] / 1e6:.1f} MP)" if t else "not timed"
        if arm.startswith("paddle"):
            rec = r["crops"].get("reader", "").split("+")[-1].split(":")[0]
            dic = dictionary_check(rec) if rec else "unverified"
        else:
            dic = "all ten letters present (allowlist, checked at load)"
        lines.append(f"| {arm} | {secs} | {dic} |")

    lines += ["", "## Verdict, by the pre-registered rule", ""]
    if limit:
        lines.append("Smoke run: no verdict.")
    else:
        rows = []
        for arm in ("paddle-v6", "paddle-v5"):
            if arm in results:
                ok = eligible(results[arm]["photos"], shipped)
                rows.append(f"- {arm}: {'clears the bar' if ok else 'does not clear the bar'} "
                            f"({results[arm]['photos']['per_photo']:.1f}% / "
                            f"{results[arm]['photos']['invented']} invented against "
                            f"{shipped['per_photo']:.1f}% / {shipped['invented']})")
        lines += rows
        w = winner(results)
        lines += ["", (f"**Adopt {w}** as the app's reader, per the rule. The dictionary row above "
                       "is reported to the owner before the switch." if w else
                       "**No PaddleOCR arm clears the bar. The shipped reader stays.** The engine "
                       "question is closed with these numbers; next is docs/OCR-ROADMAP.md step 2.")]
    lines += ["", "---", "", "Generated by `scripts/bakeoff_ocr.py`."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- self-test

def self_test() -> int:
    from app.ocr import _paddle_results_to_regions
    from training.score_crops import wilson, fold
    import numpy as np

    fake = [{"rec_texts": ["Čaršija", " ", "kuća"], "rec_scores": [0.91, 0.2, 0.77],
             "rec_polys": np.array([[[10, 10], [60, 12], [58, 30], [9, 28]],
                                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                                    [[100, 100], [140, 100], [140, 120], [100, 120]]])}]
    regions = _paddle_results_to_regions(fake)
    assert [r[1] for r in regions] == ["Čaršija", "kuća"], regions
    assert regions[0][0] == [[9, 10], [60, 10], [60, 30], [9, 30]], regions[0][0]
    assert abs(regions[1][2] - 0.77) < 1e-9
    # dt_polys as the fallback, more than four points, no scores
    fake2 = [{"rec_texts": ["x"], "dt_polys": [[[0, 0], [5, 1], [9, 0], [9, 9], [0, 9]]]}]
    assert _paddle_results_to_regions(fake2) == [[[[0, 0], [9, 0], [9, 9], [0, 9]], "x", 1.0]]
    assert _paddle_results_to_regions([{"rec_texts": []}]) == []

    lo, hi = wilson(332, 737)
    assert 41 < lo < 45.1 < hi < 49, (lo, hi)
    assert wilson(0, 0) == (0.0, 0.0)
    assert fold("Čaršija") == "carsija" and fold("MEĐU") == "medu"

    shipped = {"per_photo": 54.7, "invented": 180}
    assert eligible({"per_photo": 54.8, "invented": 180}, shipped)
    assert eligible({"per_photo": 54.7, "invented": 179}, shipped)
    assert not eligible({"per_photo": 54.7, "invented": 180}, shipped)
    assert not eligible({"per_photo": 60.0, "invented": 181}, shipped)
    assert not eligible({"per_photo": 54.6, "invented": 100}, shipped)
    res = {"lilly": {"photos": shipped},
           "paddle-v6": {"photos": {"per_photo": 58.0, "invented": 150}},
           "paddle-v5": {"photos": {"per_photo": 58.0, "invented": 150}}}
    assert winner(res) == "paddle-v5", winner(res)
    res["paddle-v6"]["photos"]["per_photo"] = 58.1
    assert winner(res) == "paddle-v6"
    assert winner({"lilly": {"photos": shipped}}) == ""

    a = {"per_photograph": {f"p{i}": [i % 3 + 1, 3] for i in range(20)}}
    s = {"per_photograph": {f"p{i}": [1, 3] for i in range(20)}}
    st = paired(a, s, resamples=500)
    assert st["n"] == 20 and st["mean_delta"] > 0 and st["p"] < 0.05, st
    same = paired(s, s, resamples=200)
    assert same["mean_delta"] == 0 and same["up"] == 0 and same["down"] == 0
    v, n = class_recall({"per_photograph": {"a": [1, 2], "b": [0, 4], "c": [3, 30]}}, 1, 5)
    assert n == 2 and abs(v - 25.0) < 1e-9, (v, n)
    assert class_recall({"per_photograph": {"c": [3, 30]}}, 1, 5) == (None, 0)
    print("self-test ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--limit", type=int, help="first N photographs and crops per half — a smoke test")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    ap.add_argument("--assemble-only", action="store_true", help="rebuild the report from training/bakeoff/")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.assemble_only:
        photo = timing_photo() if not args.dry_run else Path("<first scored photograph>")
        for arm in args.arms:
            run_arm(arm, args.limit or 0, photo, args.dry_run)
        if args.dry_run:
            return 0
    results = load_results(args.arms)
    report = assemble(results, args.limit or 0)
    REPORT.write_text(report, encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(REPO_ROOT)}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
