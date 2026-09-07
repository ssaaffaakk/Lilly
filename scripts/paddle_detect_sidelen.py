#!/usr/bin/env python3
"""Sweep the detector's text_det_limit_side_len — step 8, higher-resolution detection.

Pre-registered in training/PREREGISTRATION.md, "the detector's small-type
misses: higher-resolution detection". Fixed: the shipped PP-OCRv6 medium
detector + medium recogniser at floor 0.9. The one free knob is
LILLY_PADDLE_DET_SIDE_LEN. Choose on the 40 (the side length that maximises
words-per-photograph subject to the-40 invented ≤ 65, the shipped detector's
count), decide once on test-v2 against the shipped configuration @0.9.

    python3 scripts/paddle_detect_sidelen.py

Inference only — no training. Each side length keeps its own cache, so a paused
run resumes. One look: test-v2 is read at exactly one side length (s*); a second
is refused.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / "data" / "ocr" / "real-photos"
V2 = REAL / "test-v2"
OUT = REPO / "training" / "detector-sidelen"
GRID = [960, 1280, 1600, 2048]
THE40_INVENTED_BAR = 65  # shipped detector's the-40 invented at floor 0.9


def run_eval(side: int, set_name: str) -> dict:
    tag = f"{set_name}-side{side}"
    out_json = OUT / f"{tag}.json"
    cache = OUT / f"cache-{tag}.json"
    if out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))
    cmd = [sys.executable, "training/evaluate_ocr.py", "--json", str(out_json),
           "--out", str(OUT / f"{tag}.md"), "--cache", str(cache)]
    if set_name == "test-v2":
        cmd += ["--truth", str(V2 / "truth-v2.json"), "--photos", str(V2 / "photos"),
                "--sample", str(V2 / "sample.txt")]
    env = dict(os.environ, LILLY_READER="paddle", LILLY_PADDLE_VERSION="PP-OCRv6",
               LILLY_PADDLE_REC_THRESH="0.9", LILLY_PADDLE_DET_SIDE_LEN=str(side),
               PADDLE_PDX_MODEL_SOURCE="huggingface")
    print(f"\n$ LILLY_PADDLE_DET_SIDE_LEN={side} evaluate_ocr {set_name}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=REPO, env=env, check=True)
    r = json.loads(out_json.read_text(encoding="utf-8"))
    r["_seconds"] = round(time.time() - t0, 1)
    return r


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sweep = {}
    for s in GRID:
        r = run_eval(s, "the40")
        sweep[s] = r
        secs = r.get("_seconds")
        tail = f"  {secs:.0f}s ({secs / 40:.0f}s/photo)" if secs else ""
        print(f"the40 side {s}: words/photo {r['per_photo']:.1f}%  invented {r['invented']}  "
              f"folded {r['folded']:.1f}%{tail}", flush=True)

    ok = [(s, r) for s, r in sweep.items() if r["invented"] <= THE40_INVENTED_BAR]
    sstar = max(ok, key=lambda sr: sr[1]["per_photo"])[0] if ok else None
    print(f"\nchosen side length s* = {sstar} "
          f"(max the-40 words/photo with invented ≤ {THE40_INVENTED_BAR})", flush=True)
    if sstar is None:
        print("OUTCOME: no side length keeps the-40 invented ≤ 65 — resolution buys recall "
              "with hallucination; does not ship")
        print("DETECTOR-SIDELEN DONE sstar=none")
        return 0

    others = [p for p in OUT.glob("test-v2-side*.json") if p.name != f"test-v2-side{sstar}.json"]
    if others:
        raise SystemExit(f"test-v2 already read at {others[0].name}; one look only, as pre-registered")
    d = run_eval(sstar, "test-v2")
    secs = d.get("_seconds")
    tail = f"  {secs / 280:.0f}s/photo" if secs else ""
    print(f"\ntest-v2 @ s*={sstar}: words/photo {d['per_photo']:.1f}%  invented {d['invented']}  "
          f"folded {d['folded']:.1f}%{tail}", flush=True)
    print(f"DETECTOR-SIDELEN DONE sstar={sstar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
