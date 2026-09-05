#!/usr/bin/env python3
"""Fetch the step-7 Kaggle run's output on the Mac and hand it to the cloud.

    .venv/bin/python3 training/fetch_paddle_run.py                 # status; downloads if COMPLETE
    HF_TOKEN=... .venv/bin/python3 training/fetch_paddle_run.py --upload   # ... then to the public dataset

The kernel `afaksrmeli/lilly-ocr-paddle` is private and the Kaggle token
lives only on the Mac, so this runs there. COMPLETE with a fresh
`lilly-read-paddle.zip` (> 1 MB) and `crop-gate.json` is the only success
(docs/kaggle-notebooks.md §0); ERROR and CANCEL are failures even if an old
zip is lying in Output. The download lands in models/kaggle-staging/
(gitignored). With --upload the zip, the crop gate and the tee go to the
public dataset `Safak11/lilly-ocr-paddle-runs` under a run tag, from where the
cloud scores the recogniser on test-v2 through LILLY_PADDLE_REC_DIR — the
pre-registered look. The weights are a candidate, not a release: nothing
here touches models/lilly/ or the app.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE = REPO_ROOT / ".venv" / "bin" / "kaggle"
SLUG = "afaksrmeli/lilly-ocr-paddle"
OUT = REPO_ROOT / "models" / "kaggle-staging" / "ocr-paddle-out"
DATASET = "Safak11/lilly-ocr-paddle-runs"


def kaggle(*args) -> str:
    r = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M", time.gmtime()), help="run tag in the dataset (default: UTC time)")
    ap.add_argument("--slug", default=SLUG)
    a = ap.parse_args()
    if not KAGGLE.is_file():
        raise SystemExit(f"{KAGGLE} missing — this runs on the Mac")
    status = kaggle("kernels", "status", a.slug)
    print(status.strip())
    m = re.search(r'"(\w+)"', status)
    state = (m.group(1) if m else status).lower()
    if state != "complete":
        if state in ("error", "cancelacknowledged", "cancelrequested", "cancel"):
            raise SystemExit(f"the run ended {state}: a failure, whatever is in Output. Read the log at "
                             f"https://www.kaggle.com/code/{a.slug} and fix the cause; do not install anything")
        print("not finished yet — run this again later")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.iterdir():
        f.unlink()
    kaggle("kernels", "output", a.slug, "-p", str(OUT))
    zip_path = OUT / "lilly-read-paddle.zip"
    gate = OUT / "crop-gate.json"
    if not zip_path.is_file() or zip_path.stat().st_size < 1_000_000:
        raise SystemExit("COMPLETE but no fresh lilly-read-paddle.zip in Output — the crop gate refused or the "
                         "kernel finished without a product. That is a failure; read the log")
    if not gate.is_file():
        raise SystemExit("lilly-read-paddle.zip without crop-gate.json — not this notebook's product")
    g = json.loads(gate.read_text(encoding="utf-8"))
    print(f"crop gate: n={g['n']} stock exact {g['stock_exact']} → fine-tuned {g['tuned_exact']} "
          f"(Δ {g['delta_exact_points']:+.1f} points, 95% {g['ci95'][0]:+.1f} to {g['ci95'][1]:+.1f}); "
          f"folded {g['stock_folded']} → {g['tuned_folded']}")
    print(f"downloaded to {OUT} ({zip_path.stat().st_size // 1_000_000} MB)")
    if not a.upload:
        print(f"upload with: HF_TOKEN=... {sys.executable} {Path(__file__).relative_to(REPO_ROOT)} --upload --tag {a.tag}")
        return 0
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not in the environment; export it in this shell (never in a file) and rerun")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.dataset if hasattr(a, "dataset") else DATASET, repo_type="dataset", private=False, exist_ok=True)
    for name in ("lilly-read-paddle.zip", "crop-gate.json", "stdout.txt", "experiment_log.json", "metrics.jsonl"):
        p = OUT / name
        if p.is_file():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{a.tag}/{name}", repo_id=DATASET, repo_type="dataset")
            print("uploaded", f"{a.tag}/{name}")
    print(f"done: https://huggingface.co/datasets/{DATASET}/tree/main/{a.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
