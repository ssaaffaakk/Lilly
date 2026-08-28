#!/usr/bin/env python3
"""Overnight v2 pipeline — speech finish → install → OCR launch.

Stages (stop after stage 5 — no measurement / HF / pass-2 launch):
  0  watch speech until COMPLETE or ERROR
  1  fetch + install listen
  2  smoke test (one clip)
  3  pass-2 decision (log only — do not launch pass 2 overnight)
  4  ensure OCR notebook + kaggle_train ocr job, commit/push, launch
  5  watch OCR until COMPLETE or ERROR, fetch read weights

On unrecoverable failure (Kaggle limit, auth, unfixable error): STOP and write
overnight-v2.report.md for the owner. No infinite retries.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
KAGGLE = REPO / ".venv" / "bin" / "kaggle"
LOG = REPO / "overnight-v2.log"
REPORT = REPO / "overnight-v2.report.md"
STATE = REPO / "overnight-v2.state.json"
SPEECH_SLUG = "afaksrmeli/lilly-speech"
OCR_SLUG = "afaksrmeli/lilly-ocr"
POLL = 300          # seconds between status checks
MAX_STATUS_FAILS = 6
MAX_RELAUNCH = 0    # no silent relaunch overnight unless we add resume notebook


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def stop(reason: str, detail: str) -> None:
    """Unrecoverable — write report and exit."""
    body = f"""# Overnight v2 — stopped

**When:** {ts()} UTC  
**Reason:** {reason}

## What happened

{detail}

## What to do

- Read `{LOG}` for the full trace.
- Check Kaggle: speech `{SPEECH_SLUG}`, OCR `{OCR_SLUG}`.
- Resume manually when awake; do not expect the pipeline to retry forever.
"""
    REPORT.write_text(body, encoding="utf-8")
    log(f"STOP: {reason}")
    log(f"report written -> {REPORT}")
    sys.exit(1)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"stage": 0}


def save_state(data: dict) -> None:
    STATE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, **kw)


def kernel_status(slug: str) -> str:
    for attempt in range(3):
        r = run([str(KAGGLE), "kernels", "status", slug])
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        time.sleep(10)
    return ""


def state_kind(line: str) -> str:
    if "COMPLETE" in line:
        return "COMPLETE"
    if "ERROR" in line:
        return "ERROR"
    if "RUNNING" in line:
        return "RUNNING"
    if "CANCEL" in line:
        return "CANCEL"
    return "UNKNOWN"


def watch_kernel(slug: str, label: str) -> None:
    fails = 0
    while True:
        line = kernel_status(slug)
        if not line:
            fails += 1
            log(f"{label}: status API failed ({fails}/{MAX_STATUS_FAILS})")
            if fails >= MAX_STATUS_FAILS:
                stop(
                    "Kaggle API unreachable",
                    f"Could not read status for `{slug}` after {MAX_STATUS_FAILS} tries. "
                    "Network or kaggle.json auth may be broken.",
                )
            time.sleep(POLL)
            continue
        fails = 0
        kind = state_kind(line)
        log(f"{label}: {line}")
        if kind == "COMPLETE":
            return
        if kind in ("ERROR", "CANCEL"):
            out = REPO / "models" / "kaggle-output" / f"{label}-error"
            out.mkdir(parents=True, exist_ok=True)
            run([str(KAGGLE), "kernels", "output", slug, "-p", str(out)])
            stop(
                f"{label} kernel {kind}",
                f"Kernel `{slug}` ended with {kind}.\n\nOutput fetched to `{out}`.\n\n"
                f"Last status: {line}\n\n"
                "Likely causes: 9h GPU limit, OOM, or notebook error. "
                "Inspect lilly-speech.log or Kaggle UI. Resume with a shorter run or pass-2 notebook.",
            )
        time.sleep(POLL)


def stage0_watch_speech() -> None:
    log("stage 0: watching speech")
    watch_kernel(SPEECH_SLUG, "speech")


def stage1_install_listen() -> None:
    log("stage 1: fetch + install listen")
    r = run([str(PY), "scripts/install_listen.py"])
    if r.returncode != 0:
        stop("listen install failed", r.stdout + r.stderr)
    log("stage 1 done")


def stage2_smoke() -> None:
    log("stage 2: smoke test")
    r = run([str(PY), "scripts/smoke_listen.py"])
    if r.returncode != 0:
        log("smoke test failed (non-fatal):\n" + r.stdout + r.stderr)
    else:
        log(r.stdout.strip() or "smoke ok")


def stage3_pass2_decision() -> None:
    log("stage 3: pass-2 decision (log only)")
    out = REPO / "models" / "kaggle-output"
    wer_before = wer_after = None
    for logpath in out.rglob("lilly-speech.log"):
        text = logpath.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"word error[^\d]*(\d+\.?\d*)%", text, re.I):
            if wer_before is None:
                wer_before = m.group(1)
            else:
                wer_after = m.group(1)
    note = REPO / "overnight-v2-pass2.txt"
    if wer_before and wer_after:
        improved = float(wer_after) < float(wer_before)
        note.write_text(
            f"before WER: {wer_before}%\nafter WER: {wer_after}%\n"
            f"pass 2 recommended: {improved}\n"
            "(not launched overnight — owner decides)\n",
            encoding="utf-8",
        )
        log(f"pass-2 note -> {note} (improved={improved})")
    else:
        note.write_text("Could not parse before/after WER from logs — decide pass 2 manually.\n")
        log("pass-2: WER not found in logs")


def stage4_launch_ocr() -> None:
    log("stage 4: launch OCR on Kaggle")
    nb = REPO / "training" / "Lilly_OCR_Kaggle.ipynb"
    if not nb.exists():
        stop("OCR notebook missing", f"Expected {nb}. Pipeline did not create it.")
    r = run([str(PY), "scripts/state.py"])
    if r.returncode != 0:
        stop("repo not clean/pushed", r.stdout + r.stderr + "\nCommit and push OCR notebook first.")
    r = run([str(PY), "scripts/kaggle_train.py", "ocr"])
    if r.returncode != 0:
        stop("OCR launch failed", r.stdout + r.stderr)
    log("stage 4 done — OCR running")


def stage5_watch_ocr_and_fetch() -> None:
    log("stage 5: watch OCR + fetch")
    watch_kernel(OCR_SLUG, "ocr")
    out = REPO / "models" / "kaggle-output"
    r = run([str(KAGGLE), "kernels", "output", OCR_SLUG, "-p", str(out)])
    if r.returncode != 0:
        stop("OCR fetch failed", r.stdout + r.stderr)
    zips = list(out.rglob("lilly-read.zip"))
    if not zips:
        log("OCR complete but no lilly-read.zip — check output manually")
    else:
        log(f"OCR artifact: {zips[0]}")
    log("stage 5 done — pipeline finished (steps 1-5 only)")


STAGES = [
    stage0_watch_speech,
    stage1_install_listen,
    stage2_smoke,
    stage3_pass2_decision,
    stage4_launch_ocr,
    stage5_watch_ocr_and_fetch,
]


def main() -> int:
    LOG.touch(exist_ok=True)
    log("overnight v2 pipeline starting")
    st = load_state()
    start = st.get("stage", 0)
    for i, fn in enumerate(STAGES):
        if i < start:
            continue
        fn()
        st["stage"] = i + 1
        save_state(st)
    REPORT.write_text(
        f"# Overnight v2 — finished\n\n**When:** {ts()} UTC\n\n"
        "Stages 0-5 complete. Speech installed; OCR run finished and fetched.\n"
        "No pass-2, no measurement evening, no HF publish (by design).\n",
        encoding="utf-8",
    )
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
