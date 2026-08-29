#!/usr/bin/env python3
"""Poll Kaggle speech + OCR. Do not trust API RUNNING.

This project has already watched a session finish in the UI while
`kernels status` stayed RUNNING. The thing that decides "done" is the
output zip, listed by `kernels files`, not the status string.

    python3 scripts/kaggle_poll.py          # crash check (both jobs)
    python3 scripts/kaggle_poll.py --done   # also treat output zips as finished

Exit: 0 still going, 1 crashed, 2 finished (API may still say RUNNING).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO / ".venv" / "bin" / "kaggle"
LOG = REPO / "kaggle-kernels-watch.log"
STATE = REPO / "kaggle-poll.state.json"
JOBS = {
    "speech": {
        "slug": "afaksrmeli/lilly-speech",
        "done_names": ("lilly-listen.zip", "lilly-listen-trained.zip"),
    },
    "ocr": {
        "slug": "afaksrmeli/lilly-ocr",
        "done_names": ("lilly-read.zip", "lilly-read-trained.zip"),
    },
}


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def kaggle(*args: str, timeout: int = 120) -> str:
    r = subprocess.run(
        [str(KAGGLE), *args],
        capture_output=True, text=True, timeout=timeout, cwd=REPO)
    return ((r.stdout or "") + (r.stderr or "")).strip()


def kind(line: str) -> str:
    for name in ("COMPLETE", "ERROR", "CANCEL", "QUEUED", "RUNNING"):
        if name in line:
            return name
    return "UNKNOWN"


def file_names(slug: str) -> list[str]:
    raw = kaggle("kernels", "files", slug, "-v", "--page-size", "200")
    names = []
    for row in raw.splitlines():
        if row.lower().startswith("name,") or not row.strip():
            continue
        names.append(row.split(",", 1)[0].strip().strip('"'))
    if not names:
        # table mode fallback
        for row in raw.splitlines():
            for part in row.replace("|", " ").split():
                if part.endswith(".zip"):
                    names.append(part)
    return names


def poll_job(name: str, spec: dict, probe_files: bool) -> dict:
    slug = spec["slug"]
    status = kaggle("kernels", "status", slug)
    st = kind(status)
    names = file_names(slug) if probe_files or st in ("RUNNING", "COMPLETE", "UNKNOWN") else []
    zips = [n for n in names if any(n.endswith(z) or n == z for z in spec["done_names"])]
    # kernels files sometimes returns the basename only
    zips += [n for n in names if Path(n).name in spec["done_names"]]
    zips = sorted(set(zips))
    actually_done = st == "COMPLETE" or bool(zips)
    crashed = st in ("ERROR", "CANCEL")
    api_lie = st == "RUNNING" and bool(zips)
    row = {
        "job": name,
        "slug": slug,
        "api": st,
        "status_line": status.replace("\n", " ")[:300],
        "output_zips": zips,
        "actually_done": actually_done,
        "crashed": crashed,
        "api_lie": api_lie,
    }
    extra = ""
    if api_lie:
        extra = " — API RUNNING but output zip exists (treat as DONE)"
    elif actually_done:
        extra = f" — done ({', '.join(zips) or 'COMPLETE'})"
    elif crashed:
        extra = " — CRASH"
    log(f"{name}: api={st}{extra}")
    if status:
        log(f"  {status}")
    if zips:
        log(f"  zips: {zips}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--done", action="store_true",
                    help="probe output files even when status looks live")
    args = ap.parse_args()

    rows = [poll_job(n, s, probe_files=args.done) for n, s in JOBS.items()]
    STATE.write_text(json.dumps({"when": ts(), "jobs": rows}, indent=2), encoding="utf-8")

    if any(r["crashed"] and not r["actually_done"] for r in rows):
        return 1
    if any(r["actually_done"] for r in rows):
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired:
        log("kaggle CLI timed out")
        raise SystemExit(0)
