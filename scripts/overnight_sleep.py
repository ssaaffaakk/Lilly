#!/usr/bin/env python3
"""While the owner sleeps: keep the Mac awake, fetch speech, launch OCR after harvest.

    .venv/bin/python3 scripts/harvest_detach.py --role sleep \\
        --log logs/training/overnight-sleep.nohup -- \\
        .venv/bin/python3 scripts/overnight_sleep.py

Does not Mac-benchmark. Does not relaunch a refused OCR recipe in a loop.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from log_paths import HARVEST, KAGGLE, TRAINING  # noqa: E402

PY = REPO / ".venv" / "bin" / "python3"
KAGGLE_BIN = REPO / ".venv" / "bin" / "kaggle"
LOG = TRAINING / "overnight-sleep.log"
STATE = TRAINING / "overnight-sleep.state.json"
REPORT = REPO / "overnight-sleep.report.md"
POLL = 180


def kaggle_user() -> str:
    return json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


SPEECH_SLUG = f"{kaggle_user()}/lilly-speech"
OCR_SLUG = f"{kaggle_user()}/lilly-ocr"


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(data: dict) -> None:
    STATE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=check)


def write_report(title: str, body: str) -> None:
    REPORT.write_text(f"# {title}\n\n**When:** {ts()} UTC\n\n{body}\n", encoding="utf-8")
    log(f"report -> {REPORT}")


def speech_ready() -> bool:
    r = run([str(PY), "scripts/kaggle_poll.py", "--done"])
    log((r.stdout or "").strip()[-800:])
    data = json.loads((KAGGLE / "poll.state.json").read_text(encoding="utf-8"))
    for job in data.get("jobs", []):
        if job.get("job") == "speech":
            if job.get("crashed"):
                raise SystemExit("speech kernel crashed — see kaggle_poll log")
            return bool(job.get("actually_done") or job.get("api_lie"))
    return False


def harvest_complete() -> bool:
    r = run([str(PY), "scripts/harvest_report.py"])
    log((r.stdout or "").split("AGENT_LOOP")[0].strip())
    return r.returncode == 2


def fetch_speech(st: dict) -> None:
    log("speech done — fetch + install")
    r = run([str(PY), "scripts/install_listen.py"])
    log((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        write_report("Overnight — speech install failed", r.stdout + r.stderr)
        raise SystemExit(1)
    out = REPO / "models" / "kaggle-output"
    wer = []
    for path in out.rglob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        wer += re.findall(r"(\d+\.\d+)\s*%\s*WER|WER[^\d]*(\d+\.\d+)\s*%", text, re.I)
    lines = ["# Speech v16 — Kaggle result\n",
             f"Fetched {ts()}. Numbers from the Kaggle log only. No Mac eval.\n"]
    if wer:
        lines.append(f"Parsed WER matches: {wer[:8]}\n")
    results = REPO / "training" / "RESULTS-speech-v16.md"
    results.write_text("".join(lines), encoding="utf-8")
    log(f"wrote {results}")
    st["speech_installed"] = True
    save_state(st)


def launch_ocr(st: dict) -> None:
    raise SystemExit(
        "OCR pass-8 is not auto-launched after harvest. Speech holds the GPU; "
        "pass-7c already refused. Launch later with: python3 scripts/kaggle_train.py ocr")


def fetch_ocr(st: dict) -> None:
    log("OCR done — fetch")
    r = run([str(PY), "scripts/kaggle_train.py", "ocr", "--fetch"])
    log((r.stdout or "") + (r.stderr or ""))
    out = REPO / "models" / "kaggle-output"
    zips = list(out.rglob("lilly-read.zip"))
    ship = [z for z in zips if z.stat().st_size > 100_000]
    body = ["OCR pass-7 fetched.\n"]
    if ship:
        dest = REPO / "models" / "lilly"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ship[0]) as zf:
            zf.extractall(REPO)
        body.append(f"Installed `{ship[0].name}` over repo root (gate passed).\n")
        log(f"installed {ship[0]}")
    else:
        body.append("No `lilly-read.zip` > 100k — gate refused. Do not install "
                    "`lilly-read-trained.zip`.\n")
        log("OCR complete, not shippable")
    (REPO / "training" / "RESULTS-ocr-pass7.md").write_text(
        "# OCR pass-7 — Kaggle result\n\n"
        f"Fetched {ts()}. Numbers from the Kaggle log only.\n\n" + "".join(body),
        encoding="utf-8",
    )
    st["ocr_fetched"] = True
    save_state(st)
    write_report("Overnight — finished", "".join(body))


def ocr_ready() -> bool:
    line = run([str(KAGGLE_BIN), "kernels", "status", OCR_SLUG]).stdout
    log(line.strip())
    return "COMPLETE" in line or "ERROR" in line or "CANCEL" in line


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("overnight sleep watcher starting")
    st = load_state()
    while True:
        try:
            if not st.get("speech_installed"):
                if speech_ready():
                    fetch_speech(st)
            if not st.get("ocr_launched"):
                if harvest_complete():
                    launch_ocr(st)
            if st.get("ocr_launched") and not st.get("ocr_fetched"):
                if ocr_ready():
                    fetch_ocr(st)
                    return 0
            if st.get("speech_installed") and st.get("ocr_fetched"):
                return 0
        except SystemExit:
            raise
        except Exception as exc:
            log(f"tick error: {exc!r}")
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
