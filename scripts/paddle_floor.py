#!/usr/bin/env python3
"""training/PREREGISTRATION.md, "PP-OCRv6 confidence floor": choose on the 40, decide on test-v2.

    python3 scripts/paddle_floor.py --sweep           # the 40, floors 0.5..0.9, own cache each
    python3 scripts/paddle_floor.py --decide          # test-v2 once, at the floor the sweep chose
    python3 scripts/paddle_floor.py --assemble-only   # rewrite the report from training/paddle-floor/

Floor 0 is the bake-off's paddle-v6 arm (training/bakeoff/paddle-v6-photos.json)
and is not re-run. The floor chosen is the highest whose words-per-photograph on
the 40 is still >= the shipped arm's there (training/bakeoff/lilly-photos.json,
the Mac's 54.5%); test-v2 never chooses. --decide refuses to run at any floor but
the chosen one, and refuses to run twice: one look, as pre-registered.

The test-v2 bar is the shipped reader's committed row, training/bakeoff/test-v2-
lilly.json (the Mac, 4 Sep: 34.6% / 2,071), strict invented count (owner decision
4). Rule as in the bake-off: not worse on either row, better on at least one.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.bakeoff_ocr import eligible, paired, class_recall  # noqa: E402

OUT = REPO_ROOT / "training" / "paddle-floor"
REPORT = REPO_ROOT / "training" / "RESULTS-ocr-paddle-floor.md"
REAL = REPO_ROOT / "data" / "ocr" / "real-photos"
FLOORS = (0.5, 0.6, 0.7, 0.8, 0.9)
ENV = {"LILLY_READER": "paddle", "LILLY_PADDLE_VERSION": "PP-OCRv6"}
FLOOR0 = REPO_ROOT / "training" / "bakeoff" / "paddle-v6-photos.json"
BAR40 = REPO_ROOT / "training" / "bakeoff" / "lilly-photos.json"
BAR_V2 = REPO_ROOT / "training" / "bakeoff" / "test-v2-lilly.json"
V2 = REAL / "test-v2"


def run_eval(floor: float, set_name: str, dry: bool) -> Path:
    tag = f"{set_name}-floor{floor:g}"
    out_json = OUT / f"{tag}.json"
    cmd = [sys.executable, "training/evaluate_ocr.py", "--json", str(out_json), "--out", str(OUT / f"{tag}.md"),
           "--cache", str(REAL / f"reader-output-paddle-v6-floor{floor:g}.json" if set_name == "the40"
                          else V2 / f"reader-output-paddle-v6-floor{floor:g}.json")]
    if set_name == "test-v2":
        cmd += ["--truth", str(V2 / "truth-v2.json"), "--photos", str(V2 / "photos"), "--sample", str(V2 / "sample.txt")]
    env = dict(os.environ, **ENV, LILLY_PADDLE_REC_THRESH=str(floor))
    print(f"\n$ LILLY_PADDLE_REC_THRESH={floor:g} {' '.join(cmd)}", flush=True)
    if not dry:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return out_json


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def choose(sweep: dict, bar: float):
    """Highest floor whose recall on the 40 is still >= the shipped arm's; else None."""
    ok = [f for f, r in sweep.items() if r["per_photo"] >= bar]
    return max(ok) if ok else None


def sweep_results() -> dict:
    out = {0.0: load(FLOOR0)}
    for f in FLOORS:
        p = OUT / f"the40-floor{f:g}.json"
        if p.is_file():
            out[f] = load(p)
    return out


def assemble(sweep: dict, bar40: float, chosen, decide: dict = None, bar_v2: dict = None) -> str:
    L = ["# PP-OCRv6 with a confidence floor — chosen on the 40, decided on test-v2", "",
         f"Run {time.strftime('%Y-%m-%d %H:%M')} by `scripts/paddle_floor.py`, pre-registered in "
         "`training/PREREGISTRATION.md`, \"PP-OCRv6 confidence floor\", before any floor was tried. One lever: "
         "`LILLY_PADDLE_REC_THRESH`, the recogniser confidence below which a region is dropped. Everything else "
         "is the bake-off's paddle-v6 arm.", "",
         "## The sweep, on the 40", "",
         f"Floor 0 is the bake-off's arm, not re-run. The floor chosen is the highest that keeps words per "
         f"photograph ≥ **{bar40:.1f}%**, the shipped arm's figure on the 40 under this stack "
         "(`training/bakeoff/lilly-photos.json`). test-v2 does not choose.", "",
         "| floor | words per photograph | pooled | invented words | diacritic words | folded | keeps the bar |",
         "|---|---|---|---|---|---|---|"]
    for f in sorted(sweep):
        r = sweep[f]
        L.append(f"| {f:g} | **{r['per_photo']:.1f}%** | {r['pooled']:.1f}% | {r['invented']} | "
                 f"{r['diacritic']:.1f}% | {r['folded']:.1f}% | {'yes' if r['per_photo'] >= bar40 else 'no'} |")
    L += ["", (f"**Chosen floor: {chosen:g}.**" if chosen is not None else
               "**No floor keeps the bar: confidence is not the separator. Nothing is run on test-v2.**"), ""]
    if decide is not None and bar_v2 is not None:
        d = decide
        L += [f"## The decision, on test-v2, at floor {chosen:g} — one run", "",
              "Bar: the shipped reader as scored on test-v2 on the Mac, `training/bakeoff/test-v2-lilly.json`. "
              "Strict invented count (owner decision 4). Rule: not worse on either row, better on at least one.", "",
              "| arm | words per photograph | pooled | invented words | diacritic words | folded |",
              "|---|---|---|---|---|---|",
              f"| shipped (`2010a2d4`, the Mac) | **{bar_v2['per_photo']:.1f}%** | {bar_v2['pooled']:.1f}% | "
              f"{bar_v2['invented']} | {bar_v2['diacritic']:.1f}% | {bar_v2['folded']:.1f}% |",
              f"| paddle-v6, floor {chosen:g} | **{d['per_photo']:.1f}%** | {d['pooled']:.1f}% | {d['invented']} | "
              f"{d['diacritic']:.1f}% | {d['folded']:.1f}% |", ""]
        s = paired(d, bar_v2)
        L += [f"Paired, photograph by photograph: {s['n']} photographs, mean Δ {s['mean_delta']:+.1f} points, "
              f"better on {s['up']}, worse on {s['down']}, bootstrap p = {s['p']:.3f}. Reported; the bar is the table.", "",
              "| | signs, 1–5 words | short boards, 6–20 | long boards, 21+ |", "|---|---|---|---|"]
        for label, r in (("shipped", bar_v2), (f"paddle-v6, floor {chosen:g}", d)):
            cells = []
            for lo, hi in ((1, 5), (6, 20), (21, 10 ** 9)):
                v, n = class_recall(r, lo, hi)
                cells.append(f"{v:.1f}% (n={n})" if v is not None else "—")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        ok = eligible(d, bar_v2)
        L += ["", "## Verdict, by the pre-registered rule", "",
              (f"**PP-OCRv6 at floor {chosen:g} clears the bar on test-v2** ({d['per_photo']:.1f}% / {d['invented']} "
               f"against {bar_v2['per_photo']:.1f}% / {bar_v2['invented']}). Adopt it — as a product change "
               "reviewed by the owner: default reader, requirements, the Docker image, the cv2 pinning of "
               "do-not-repeat 17." if ok else
               f"**PP-OCRv6 at floor {chosen:g} does not clear the bar on test-v2** ({d['per_photo']:.1f}% / "
               f"{d['invented']} against {bar_v2['per_photo']:.1f}% / {bar_v2['invented']}). Not adopted. "
               "The next lever is a new pre-registration; this one is spent.")]
    L += ["", "---", "", "Generated by `scripts/paddle_floor.py`. Raw counts in `training/paddle-floor/`."]
    return "\n".join(L) + "\n"


def self_test() -> int:
    sw = {0.0: {"per_photo": 67.7}, 0.5: {"per_photo": 60.0}, 0.7: {"per_photo": 55.0}, 0.9: {"per_photo": 40.0}}
    assert choose(sw, 54.5) == 0.7 and choose(sw, 70.0) is None and choose(sw, 40.0) == 0.9
    ph = {"per_photo": 60.0, "pooled": 66.0, "invented": 2000, "diacritic": 60.0, "folded": 80.0,
          "per_photograph": {"a": [1, 2], "b": [3, 10]}}
    bar = dict(ph, per_photo=34.6, invented=2071)
    text = assemble({0.0: dict(ph, per_photo=67.7, invented=2373), 0.7: ph}, 54.5, 0.7, ph, bar)
    assert "clears the bar" in text and "Chosen floor: 0.7" in text
    text = assemble({0.0: dict(ph, per_photo=67.7, invented=2373), 0.7: ph}, 54.5, 0.7, dict(ph, invented=2100), bar)
    assert "does not clear" in text
    assert "Nothing is run on test-v2" in assemble({0.0: ph}, 70.0, None)
    print("self-test ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    OUT.mkdir(parents=True, exist_ok=True)
    bar40 = load(BAR40)["per_photo"]
    if args.sweep:
        for f in FLOORS:
            if (OUT / f"the40-floor{f:g}.json").is_file():
                print(f"floor {f:g}: already measured, skipping")
                continue
            run_eval(f, "the40", args.dry_run)
    sweep = sweep_results()
    chosen = choose(sweep, bar40)
    decide = bar_v2 = None
    if args.decide:
        if chosen is None:
            raise SystemExit("no floor keeps the bar on the 40 — nothing is run on test-v2, as pre-registered")
        done = list(OUT.glob("test-v2-floor*.json"))
        if done and not (OUT / f"test-v2-floor{chosen:g}.json").is_file():
            raise SystemExit(f"test-v2 was already run at {done[0].name}; one look only, as pre-registered")
        if not (OUT / f"test-v2-floor{chosen:g}.json").is_file():
            run_eval(chosen, "test-v2", args.dry_run)
    p = OUT / f"test-v2-floor{chosen:g}.json" if chosen is not None else None
    if p is not None and p.is_file():
        decide, bar_v2 = load(p), load(BAR_V2)
    if args.dry_run:
        return 0
    report = assemble(sweep, bar40, chosen, decide, bar_v2)
    REPORT.write_text(report, encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(REPO_ROOT)}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
