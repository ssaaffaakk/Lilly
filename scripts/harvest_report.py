#!/usr/bin/env python3
"""Pass-7 harvest progress — honest stall detection for watchdog ticks."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"
STATE = LOGS / "harvest-pass7.state.json"
LOG = LOGS / "harvest-pass7.log"
METRICS = LOGS / "harvest-metrics.json"

# No counter movement for this long while a step is "running" => stalled.
STALL_COUNTER_SECONDS = 600
# Heartbeat older than this with no log growth => stalled.
STALL_HEARTBEAT_SECONDS = 900
# Log grew or heartbeat touched within this window => active (not stalled).
ACTIVE_SECONDS = 300

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


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _age_seconds(ts: datetime | None) -> float | None:
    if ts is None:
        return None
    return (datetime.now(timezone.utc) - ts).total_seconds()


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def _load_metrics() -> dict:
    if METRICS.exists():
        return json.loads(METRICS.read_text(encoding="utf-8"))
    return {}


def _save_metrics(payload: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _running() -> dict:
    procs = subprocess.run(
        ["pgrep", "-fl", "harvest_pass7|harvest_watch|harvest_sign|generate_ocr|fetch_commons"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    names = procs.splitlines() if procs else []
    orch = sum(1 for n in names if "harvest_pass7" in n)
    return {
        "orchestrator": orch > 0,
        "orchestrator_count": orch,
        "watchdog": any("harvest_watch" in n for n in names),
        "worker": any(
            any(x in n for x in ("harvest_sign", "generate_ocr", "fetch_commons"))
            for n in names
        ),
        "detail": names[:5],
    }


def _step_counters() -> dict[str, int]:
    out: dict[str, int] = {}
    for key, _, _, counter in STEPS:
        out[key] = max(0, counter())
    return out


def _log_activity() -> dict:
    if not LOG.exists():
        return {"exists": False, "bytes": 0, "age_s": None}
    st = LOG.stat()
    return {
        "exists": True,
        "bytes": st.st_size,
        "age_s": _age_seconds(datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)),
    }


def _counter_delta(prev: dict, cur: dict) -> dict[str, int]:
    return {k: cur.get(k, 0) - prev.get(k, 0) for k in cur}


def report() -> dict:
    state = _load_state()
    prev = _load_metrics()
    step_status = state.get("steps", {})
    counters = _step_counters()
    run = _running()
    log_act = _log_activity()
    hb_age = _age_seconds(_parse_ts(state.get("heartbeat")))
    step = state.get("step", "?")
    step_st = step_status.get(step, {}).get("status", "pending")
    complete = LOG.exists() and "=== pipeline complete ===" in LOG.read_text(encoding="utf-8")

    lines = []
    step_pcts = []
    for key, label, target, counter in STEPS:
        have = counters[key]
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
    prev_counters = prev.get("counters", {})
    delta = _counter_delta(prev_counters, counters)
    any_progress = any(v > 0 for v in delta.values())
    log_grew = (
        prev.get("log_bytes") is not None
        and log_act["bytes"] > prev.get("log_bytes", 0)
    )
    log_recent = log_act.get("age_s") is not None and log_act["age_s"] <= ACTIVE_SECONDS
    hb_recent = hb_age is not None and hb_age <= ACTIVE_SECONDS
    active_signal = any_progress or log_grew or (log_recent and (run["worker"] or run["orchestrator"]))

    reasons: list[str] = []
    if complete:
        status = "COMPLETE"
    elif run["orchestrator_count"] > 1:
        status = "STALLED"
        reasons.append(f"{run['orchestrator_count']} orchestrators (duplicate restarts)")
    elif step_st == "failed":
        status = "FAILED"
        reasons.append(f"step {step} failed")
    elif not run["orchestrator"] and not run["worker"]:
        status = "STALLED"
        reasons.append("no orchestrator or worker process")
    elif step_st == "running" and not active_signal:
        if hb_age is not None and hb_age > STALL_HEARTBEAT_SECONDS:
            status = "STALLED"
            reasons.append(f"heartbeat stale ({hb_age:.0f}s)")
        elif prev.get("ts"):
            prev_age = _age_seconds(_parse_ts(prev.get("ts")))
            if prev_age is not None and prev_age >= STALL_COUNTER_SECONDS and not any_progress:
                status = "STALLED"
                reasons.append(f"no counter movement for {prev_age:.0f}s")
            else:
                status = "WAITING"
                reasons.append("process up but counters flat (Overpass slot wait?)")
        else:
            status = "WAITING"
            reasons.append("first sample — waiting for progress proof")
    else:
        status = "RUNNING"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _save_metrics(
        {
            "ts": now,
            "counters": counters,
            "log_bytes": log_act["bytes"],
            "status": status,
            "step": step,
        }
    )

    return {
        "ts": now,
        "overall_pct": round(overall, 1),
        "current_step": step,
        "complete": complete,
        "status": status,
        "reasons": reasons,
        "running": run,
        "delta": delta,
        "active_signal": active_signal,
        "heartbeat_age_s": hb_age,
        "log": log_act,
        "lines": lines,
    }


def main() -> int:
    r = report()
    print(
        f"HARVEST_STATUS overall={r['overall_pct']}% status={r['status']} "
        f"step={r['current_step']}"
    )
    for line in r["lines"]:
        print(line)
    if r["reasons"]:
        print(f"  why: {'; '.join(r['reasons'])}")
    if r["status"] == "RUNNING":
        moved = {k: v for k, v in r["delta"].items() if v}
        if moved:
            print(f"  ✓ progress since last tick: {moved}")
        else:
            print("  ✓ active (log/heartbeat fresh)")
    elif r["status"] == "WAITING":
        print("  ⏳ worker alive — no counter delta yet (long Overpass query?)")
    elif r["status"] == "STALLED":
        print("  ⚠ stalled — watchdog should restart (single instance only)")
    print(
        f"AGENT_LOOP_TICK_harvest_progress "
        f'{{"overall":{r["overall_pct"]},"status":"{r["status"]}",'
        f'"step":"{r["current_step"]}","active":{str(r["active_signal"]).lower()}}}'
    )
    if r["status"] == "COMPLETE":
        return 2
    if r["status"] in ("STALLED", "FAILED"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
