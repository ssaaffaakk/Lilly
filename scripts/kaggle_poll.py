#!/usr/bin/env python3
"""Poll Kaggle speech + OCR. Do not trust API RUNNING.

v13–v15 finished in the UI while `kernels status` stayed RUNNING. A zip on
the files list is not enough either: it may be the *previous* version (this
pass saw an 853-byte lilly-read-trained.zip from an earlier run while v10
had only just been pushed).

A job is actually done when a *new, large* zip appears after this launch.

    python3 scripts/kaggle_poll.py          # crash check
    python3 scripts/kaggle_poll.py --done   # also decide finished-vs-API-lie

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
KAGGLE_BIN = REPO / ".venv" / "bin" / "kaggle"
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from log_paths import KAGGLE as KAGGLE_LOG  # noqa: E402

LOG = KAGGLE_LOG / "kernels-watch.log"
STATE = KAGGLE_LOG / "poll.state.json"
LAUNCH = KAGGLE_LOG / "launch.json"


def kaggle_user() -> str:
    token = Path.home() / ".kaggle" / "kaggle.json"
    return json.loads(token.read_text())["username"]


def jobs() -> dict:
    user = kaggle_user()
    return {
        "speech": {
            "slug": f"{user}/lilly-speech",
            "done_names": ("lilly-listen.zip", "lilly-listen-trained.zip",
                           "lilly-listen-half1.zip", "lilly-listen-half2.zip"),
            "min_bytes": 1_000_000,
        },
        "ocr": {
            "slug": f"{user}/lilly-ocr",
            "done_names": ("lilly-read.zip", "lilly-read-trained.zip"),
            "min_bytes": 100_000,
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
        [str(KAGGLE_BIN), *args],
        capture_output=True, text=True, timeout=timeout, cwd=REPO)
    return ((r.stdout or "") + (r.stderr or "")).strip()


def kind(line: str) -> str:
    for name in ("COMPLETE", "ERROR", "CANCEL", "QUEUED", "RUNNING"):
        if name in line:
            return name
    return "UNKNOWN"


def parse_kaggle_date(raw: str) -> datetime | None:
    raw = raw.strip().strip('"')
    if not raw:
        return None
    for fmt in (
        "%I:%M %p, %A %d %B %Y UTC",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def launched_at(job: str) -> datetime | None:
    if not LAUNCH.is_file():
        return None
    try:
        data = json.loads(LAUNCH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    stamp = (data.get(job) or {}).get("pushed")
    return parse_kaggle_date(stamp) if stamp else None


def list_files(slug: str) -> list[dict]:
    raw = kaggle("kernels", "files", slug, "-v", "--page-size", "200")
    rows = []
    for line in raw.splitlines():
        if not line.strip() or line.lower().startswith("name,") or line.startswith("Next Page"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        name = parts[0].strip().strip('"')
        try:
            size = int(parts[1].strip().strip('"'))
        except ValueError:
            size = 0
        created = parse_kaggle_date(",".join(parts[2:])) if len(parts) > 2 else None
        rows.append({"name": name, "size": size, "created": created})
    return rows


def fresh_zips(job: str, spec: dict, files: list[dict]) -> list[dict]:
    want = spec["done_names"]
    min_bytes = spec["min_bytes"]
    start = launched_at(job)
    found = []
    for f in files:
        base = Path(f["name"]).name
        if base not in want and f["name"] not in want:
            continue
        if f["size"] < min_bytes:
            continue
        if start is not None and f["created"] is not None and f["created"] < start:
            continue
        found.append(f)
    return found


def poll_job(name: str, spec: dict) -> dict:
    slug = spec["slug"]
    status = kaggle("kernels", "status", slug)
    st = kind(status)
    files = list_files(slug)
    zips = fresh_zips(name, spec, files)
    zip_names = [f"{Path(z['name']).name} {z['size']}B" for z in zips]
    # A job needs a fresh, large zip to count as done — COMPLETE with no zip
    # means the kernel finished without writing output, which is a failure.
    actually_done = st == "COMPLETE" and bool(zips)
    # ERROR and CANCEL are always failures, even when a zip was recovered from a
    # previous version. A CANCEL+zip is not success; it is a crashed run that
    # happened to leave old output on disk.
    crashed = st in ("ERROR", "CANCEL") or (st == "COMPLETE" and not bool(zips))
    api_lie = st == "RUNNING" and bool(zips)
    extra = ""
    if api_lie:
        extra = " — API RUNNING but a fresh zip exists (DONE)"
    elif actually_done:
        extra = f" — done ({', '.join(zip_names) or st})"
    elif crashed:
        extra = " — CRASH"
    log(f"{name}: api={st}{extra}")
    if status:
        log(f"  {status}")
    if zips:
        log(f"  zips: {zip_names}")
    tiny = [
        f"{Path(f['name']).name}:{f['size']}"
        for f in files
        if Path(f["name"]).name in spec["done_names"] and f["size"] < spec["min_bytes"]
    ]
    if tiny and not zips:
        log(f"  ignoring undersized leftover: {tiny}")
    return {
        "job": name,
        "slug": slug,
        "api": st,
        "status_line": status.replace("\n", " ")[:300],
        "output_zips": zip_names,
        "actually_done": actually_done,
        "crashed": crashed,
        "api_lie": api_lie,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--done", action="store_true",
                    help="watcher flag; files are always listed")
    ap.parse_args()

    rows = [poll_job(n, s) for n, s in jobs().items()]
    STATE.write_text(json.dumps({"when": ts(), "jobs": rows}, indent=2), encoding="utf-8")

    if any(r["crashed"] for r in rows):
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
