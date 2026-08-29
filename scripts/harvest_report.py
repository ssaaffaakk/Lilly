#!/usr/bin/env python3
"""Pass-7 harvest progress — for watchdog ticks and owner updates."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"
STATE = LOGS / "harvest-pass7.state.json"
LOG = LOGS / "harvest-pass7.log"

STEPS = [
    ("commons_cats", "Commons categories", 343, lambda: _count(REPO / "data/ocr/real-photos/commons-cats/CREDITS.tsv") - 1),
    ("osm_signs", "OSM sign text", 12000, lambda: _wc(REPO / "data/signs/sign-text.tsv")),
    ("synth_flat", "Synthetic crops", 50000, lambda: len(list((REPO / "data/ocr/synthetic").glob("*.png")))),
    ("commons_harvest", "Commons photos (text filter)", 500, lambda: _count(REPO / "data/ocr/real-photos/harvested/CREDITS.tsv") - 1),
    ("synth_photo", "Photo-style synthetic", 20000, lambda: len(list((REPO / "data/ocr/train").glob("photo*.png")))),
]


def _count(credits: Path) -> int:
    if not credits.exists():
        return 0
    return max(0, sum(1 for _ in credits.read_text(encoding="utf-8").splitlines()) - 1)


def _wc(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.read_text(encoding="utf-8").splitlines())


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def _running() -> dict:
    procs = subprocess.run(
        ["pgrep", "-fl", "harvest_pass7|harvest_watch|harvest_sign|generate_ocr|fetch_commons"],
        capture_output=True, text=True,
    ).stdout.strip()
    names = procs.splitlines() if procs else []
    return {
        "orchestrator": any("harvest_pass7" in n for n in names),
        "watchdog": any("harvest_watch" in n for n in names),
        "worker": any(any(x in n for x in ("harvest_sign", "generate_ocr", "fetch_commons")) for n in names),
        "detail": names[:3],
    }


def report() -> dict:
    state = _load_state()
    step_status = state.get("steps", {})
    lines = []
    step_pcts = []
    for key, label, target, counter in STEPS:
        have = max(0, counter())
        st = step_status.get(key, {}).get("status", "pending")
        if st == "done":
            pct = 100.0
        elif target > 0:
            pct = min(100.0, 100.0 * have / target)
        else:
            pct = 0.0
        step_pcts.append(pct)
        lines.append(f"  {label}: {have:,}/{target:,} ({pct:.0f}%) [{st}]")

    overall = sum(step_pcts) / len(step_pcts)
    run = _running()
    complete = LOG.exists() and "=== pipeline complete ===" in LOG.read_text(encoding="utf-8")
    crash = not complete and not (run["orchestrator"] or run["worker"])

    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_pct": round(overall, 1),
        "current_step": state.get("step", "?"),
        "complete": complete,
        "stalled": crash,
        "running": run,
        "lines": lines,
    }


def main() -> int:
    r = report()
    status = "COMPLETE" if r["complete"] else ("STALLED" if r["stalled"] else "RUNNING")
    print(f"HARVEST_STATUS overall={r['overall_pct']}% status={status} step={r['current_step']}")
    for line in r["lines"]:
        print(line)
    if r["stalled"]:
        print("  ⚠ pipeline stalled — watchdog should restart")
    elif r["running"]["worker"] or r["running"]["orchestrator"]:
        print("  ✓ no crash — worker active")
    print(
        f"AGENT_LOOP_TICK_harvest_progress "
        f'{{"overall":{r["overall_pct"]},"status":"{status}","step":"{r["current_step"]}"}}'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
