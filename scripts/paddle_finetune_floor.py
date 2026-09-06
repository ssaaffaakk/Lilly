#!/usr/bin/env python3
"""Re-sweep the step-7 fine-tuned recogniser's confidence floor.

Pre-registered in training/PREREGISTRATION.md, "the fine-tuned recogniser's
confidence floor", written before any floor but 0.9 was read. Fixed: the
step-7 candidate weights, pointed at by LILLY_PADDLE_REC_DIR (no retraining).
The one free knob is LILLY_PADDLE_REC_THRESH. Two sets, two jobs, never crossed:

- Choose on the 40 (truth.json): sweep {0.5..0.9} and, above 0.9, steps of 0.02
  until the-40 words-per-photograph first falls below 67.0% (the shipped
  reader's the-40 figure at floor 0.9, training/paddle-floor/the40-floor0.9.json).
  f* = the highest swept floor whose the-40 words-per-photograph is still >= 67.0%.
- Decide on test-v2 (truth-v2.json), once, at f*. One look: test-v2 is read at
  exactly one floor; a second is refused.

    python3 scripts/paddle_finetune_floor.py --rec-dir models/kaggle-staging/read-paddle-candidate/rec

The floor-0.9 rows are seeded from step 7 (training/paddle-finetune/) so they
are not re-read. Everything else is read fresh, each floor in its own cache, so
the run resumes after a pause. This writes only the per-floor summaries under
training/paddle-finetune-floor/; the shipping decision is rescue_report.py
against the shipped configuration, run separately per the pre-registration.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "data" / "ocr" / "real-photos"
V2 = REAL / "test-v2"
OUT = REPO / "training" / "paddle-finetune-floor"
GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98]
THE40_BAR = 67.0  # shipped reader's the-40 words/photo at floor 0.9 (paddle-floor/the40-floor0.9.json)


def run_eval(rec_dir: Path, floor: float, set_name: str) -> dict:
    tag = f"{set_name}-floor{floor:g}"
    out_json = OUT / f"{tag}.json"
    cache = OUT / f"cache-{tag}.json"
    if out_json.is_file():  # seeded (0.9) or already done — no re-read
        return json.loads(out_json.read_text(encoding="utf-8"))
    cmd = [sys.executable, "training/evaluate_ocr.py",
           "--json", str(out_json), "--out", str(OUT / f"{tag}.md"), "--cache", str(cache)]
    if set_name == "test-v2":
        cmd += ["--truth", str(V2 / "truth-v2.json"), "--photos", str(V2 / "photos"),
                "--sample", str(V2 / "sample.txt")]
    env = dict(os.environ, LILLY_READER="paddle", LILLY_PADDLE_VERSION="PP-OCRv6",
               LILLY_PADDLE_REC_DIR=str(rec_dir), LILLY_PADDLE_REC_THRESH=f"{floor:g}",
               PADDLE_PDX_MODEL_SOURCE="huggingface")
    print(f"\n$ LILLY_PADDLE_REC_THRESH={floor:g} evaluate_ocr {set_name}", flush=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-dir", required=True)
    a = ap.parse_args()
    rec = Path(a.rec_dir).resolve()
    if not (rec / "inference.pdiparams").is_file():
        raise SystemExit(f"no inference.pdiparams in {rec}")
    OUT.mkdir(parents=True, exist_ok=True)

    # Choose on the 40.
    sweep = {}
    for f in GRID:
        r = run_eval(rec, f, "the40")
        sweep[f] = r
        print(f"the40 floor {f:g}: words/photo {r['per_photo']:.1f}%  invented {r['invented']}  "
              f"diacritic {r['diacritic']:.1f}%  folded {r['folded']:.1f}%", flush=True)
        if f > 0.9 and r["per_photo"] < THE40_BAR:
            break  # recall already below the bar; higher floors only drop further

    ok = [f for f, r in sweep.items() if r["per_photo"] >= THE40_BAR]
    fstar = max(ok) if ok else None
    print(f"\nchosen floor f* = {fstar} (highest keeping the-40 words/photo >= {THE40_BAR}%)", flush=True)
    if fstar is None:
        print("OUTCOME 4: no floor keeps the-40 words/photo >= 67.0% -> does not ship")
        print("FLOOR-RESWEEP DONE fstar=none")
        return 0

    # Decide on test-v2, once. If f* is 0.9 the candidate was already read there
    # in step 7 (same weights, same floor) — reuse that number rather than re-read.
    step7_v2 = REPO / "training" / "paddle-finetune" / "test-v2.json"
    if abs(fstar - 0.9) < 1e-9 and step7_v2.is_file() and not (OUT / "test-v2-floor0.9.json").is_file():
        (OUT / "test-v2-floor0.9.json").write_text(step7_v2.read_text(encoding="utf-8"), encoding="utf-8")
        print("f* = 0.9: reusing step 7's test-v2 read at this floor (same weights, same floor)")
    others = [p for p in OUT.glob("test-v2-floor*.json") if p.name != f"test-v2-floor{fstar:g}.json"]
    if others:
        raise SystemExit(f"test-v2 already read at {others[0].name}; one look only, as pre-registered")
    d = run_eval(rec, fstar, "test-v2")
    print(f"\ntest-v2 @ f*={fstar:g}: words/photo {d['per_photo']:.1f}%  invented {d['invented']}  "
          f"diacritic {d['diacritic']:.1f}%  folded {d['folded']:.1f}%", flush=True)
    print(f"FLOOR-RESWEEP DONE fstar={fstar:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
