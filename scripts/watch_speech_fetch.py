#!/usr/bin/env python3
"""Pull speech zips the moment they exist. Do not trust `kernels status`.

v13–v16 stayed RUNNING in the API after the UI was done, or died during
AFTER WER so the zip listed in Output 404'd. Last night nobody pulled
`lilly-listen-trained.zip` while the worker still had it.

This loop downloads only `lilly-listen*.zip` into models/kaggle-output/speech-keep/
(later fetches cannot wipe that). COMPLETE is not required. A zip while the
API still says RUNNING is done.

    python3 scripts/harvest_detach.py --role speech-fetch \\
        --log logs/kaggle/speech-fetch.nohup -- \\
        .venv/bin/python3 scripts/watch_speech_fetch.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from log_paths import KAGGLE as KAGGLE_LOG  # noqa: E402

KAGGLE_BIN = REPO / ".venv" / "bin" / "kaggle"
LOG = KAGGLE_LOG / "speech-fetch.log"
STATE = KAGGLE_LOG / "speech-fetch.state.json"
REPORT = KAGGLE_LOG / "speech-fetch.report.md"
LOCK = KAGGLE_LOG / "speech-fetch.lock"
DEST = REPO / "models" / "kaggle-output"
KEEP = DEST / "speech-keep"
POLL = 300
MIN_ZIP = 1_000_000
ZIP_PATTERN = r"lilly-listen.*\.zip"
END_TRIES = 8
WANT = ("lilly-listen.zip", "lilly-listen-trained.zip",
        "lilly-listen-half1.zip", "lilly-listen-half2.zip")
BACKOFF_429 = 300


def kaggle_user() -> str:
    return json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


def slug() -> str:
    return f"{kaggle_user()}/lilly-speech"


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


def take_lock() -> None:
    if LOCK.exists():
        try:
            other = int(LOCK.read_text().split()[0])
        except (ValueError, IndexError):
            other = 0
        if other and other != os.getpid():
            try:
                os.kill(other, 0)
            except OSError:
                log(f"stale lock pid {other} — taking over")
            else:
                log(f"already running pid {other}, not starting a second watcher")
                raise SystemExit(0)
    LOCK.write_text(f"{os.getpid()} {ts()}\n")


def drop_lock() -> None:
    try:
        if LOCK.exists() and LOCK.read_text().split()[0] == str(os.getpid()):
            LOCK.unlink()
    except OSError:
        pass


def kind(line: str) -> str:
    for name in ("COMPLETE", "ERROR", "CANCEL", "QUEUED", "RUNNING"):
        if name in line:
            return name
    return "UNKNOWN"


def status_line() -> str:
    r = subprocess.run(
        [str(KAGGLE_BIN), "kernels", "status", slug()],
        capture_output=True, text=True, timeout=120, cwd=REPO)
    return ((r.stdout or "") + (r.stderr or "")).strip()


def kept() -> dict[str, int]:
    sizes: dict[str, int] = {}
    if not KEEP.is_dir():
        return sizes
    for name in WANT:
        paths = [p for p in KEEP.rglob(name) if p.is_file()]
        if paths:
            sizes[name] = max(p.stat().st_size for p in paths)
    return sizes


def fetch_zips(*, dest: Path | None = None) -> dict[str, int]:
    """Zip-only pull. Empty match exits 0; do not treat that as failure."""
    dest = dest or KEEP
    dest.mkdir(parents=True, exist_ok=True)
    before = kept() if dest == KEEP else {}
    r = subprocess.run(
        [str(KAGGLE_BIN), "kernels", "output", slug(),
         "-p", str(dest), "--file-pattern", ZIP_PATTERN],
        capture_output=True, text=True, timeout=7200, cwd=REPO)
    combined = (r.stdout or "") + (r.stderr or "")
    tail = combined.strip()[-800:]
    if tail:
        log(tail.replace("\n", " | "))
    if "429" in combined or "Too Many Requests" in combined:
        log(f"429 from kernels output — backing off {BACKOFF_429}s")
        time.sleep(BACKOFF_429)
    after = kept() if dest == KEEP else {
        p.name: p.stat().st_size for p in dest.glob("lilly-listen*.zip") if p.is_file()
    }
    if dest == KEEP:
        for name, size in after.items():
            if size >= MIN_ZIP and size != before.get(name):
                log(f"kept {name} ({size:,} bytes)")
        save_state({"when": ts(), "api": load_state().get("api"), "zips": after,
                    "pid": os.getpid()})
    return after


def fetch_full() -> None:
    """Same zip-only pull into DEST. Never list the whole Output tree."""
    DEST.mkdir(parents=True, exist_ok=True)
    log(f"$ kaggle kernels output {slug()} -p {DEST} --file-pattern {ZIP_PATTERN}")
    fetch_zips(dest=DEST)


def alarm(msg: str) -> None:
    """Mac is caffeinated; sound + banner so the download is not silent."""
    log(f"ALARM {msg}")
    note = json.dumps(msg)
    subprocess.run(
        ["osascript", "-e",
         f"display notification {note} with title \"Lilly speech\" sound name \"Sosumi\""],
        capture_output=True, timeout=30)
    sound = Path("/System/Library/Sounds/Sosumi.aiff")
    if sound.is_file():
        subprocess.run(["afplay", str(sound)], capture_output=True, timeout=15)
    subprocess.run(["say", "Lilly speech download finished"], capture_output=True, timeout=30)


def write_report(title: str, body: str) -> None:
    REPORT.write_text(f"# {title}\n\n**When:** {ts()} UTC\n\n{body}\n", encoding="utf-8")
    log(f"report -> {REPORT}")


def usable(zips: dict[str, int], name: str) -> bool:
    return zips.get(name, 0) >= MIN_ZIP


def finish(title: str, body: str, code: int, *, already_grabbed: bool = False) -> int:
    if not already_grabbed:
        fetch_full()
        alarm(title)
    write_report(title, body)
    drop_lock()
    return code


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    take_lock()
    log(f"speech-fetch watcher starting — {slug()}")
    log("will pull lilly-listen*.zip as soon as they exist; status RUNNING is not 'not done'")
    ended = 0
    grabbed = False
    while True:
        try:
            line = status_line()
            api = kind(line)
            log(f"api={api} {line}")
            zips = fetch_zips()
            have_listen = usable(zips, "lilly-listen.zip")
            have_trained = usable(zips, "lilly-listen-trained.zip")
            have_half1 = usable(zips, "lilly-listen-half1.zip")
            have_any = any(usable(zips, name) for name in WANT)
            if have_any and not grabbed:
                sizes = ", ".join(f"{n} {s:,}B" for n, s in zips.items() if s >= MIN_ZIP)
                log(f"zip on disk ({sizes}) — zip-only copy to DEST, do not wait for COMPLETE")
                fetch_full()
                alarm("Lilly speech zip downloaded")
                grabbed = True
            if have_listen:
                sizes = ", ".join(f"{n} {s:,}B" for n, s in zips.items())
                return finish(
                    "Speech fetch — packaged listener saved",
                    f"API was `{api}` (may still say RUNNING).\n\nKept: {sizes}\n\n"
                    f"Zips: `{KEEP}`\nFull output: `{DEST}`\n"
                    "Install when you want: `python3 scripts/install_listen.py`\n"
                    "(that command will use the local zip, not Kaggle).",
                    0, already_grabbed=grabbed)
            if have_half1:
                sizes = ", ".join(f"{n} {s:,}B" for n, s in zips.items())
                return finish(
                    "Speech fetch — half 1 adapter zip saved",
                    f"API was `{api}`.\n\nKept: {sizes}\n\n"
                    f"Zips: `{KEEP}`\n"
                    "This is the Trainer checkpoint, not the app listener.\n"
                    "After this kernel COMPLETEs, launch half 2:\n"
                    "`python3 scripts/kaggle_train.py speech-half2`\n",
                    0, already_grabbed=grabbed)
            if api in ("ERROR", "CANCEL", "COMPLETE"):
                ended += 1
                if have_trained or grabbed:
                    return finish(
                        "Speech fetch — trained zip saved (no listen.zip yet)",
                        f"API `{api}` after {ended} end-state polls.\n\n"
                        f"Files in `{KEEP}` and `{DEST}`.\n",
                        0, already_grabbed=grabbed)
                if ended >= END_TRIES:
                    return finish(
                        "Speech fetch — kernel ended, no zip",
                        f"API `{api}` and no zip ≥ {MIN_ZIP} bytes after "
                        f"{END_TRIES} pulls.\n\nLast status: {line}",
                        1, already_grabbed=grabbed)
                log(f"end state {api} but no zip yet — retry {ended}/{END_TRIES}")
                time.sleep(120)
                continue
            ended = 0
        except SystemExit:
            drop_lock()
            raise
        except Exception as exc:
            log(f"tick error: {exc!r}")
        time.sleep(POLL)


def backup_alarms() -> int:
    """Clock alarms at 03:50 and 04:10 local — pull even if the watcher stalled."""
    tz = ZoneInfo("Europe/Berlin")
    now = datetime.now(tz)
    times = [
        datetime(2026, 8, 31, 3, 50, tzinfo=tz),
        datetime(2026, 8, 31, 4, 10, tzinfo=tz),
    ]
    log(f"backup alarm sleeper starting (now {now.isoformat()})")
    for when in times:
        wait = (when - datetime.now(tz)).total_seconds()
        if wait > 0:
            log(f"sleeping {wait:.0f}s until {when.isoformat()}")
            time.sleep(wait)
        else:
            log(f"{when.isoformat()} already passed — firing now")
        alarm(f"Lilly speech download alarm {when.strftime('%H:%M')}")
        zips = fetch_zips()
        api = kind(status_line())
        log(f"backup tick api={api} zips={zips}")
        if any(usable(zips, name) for name in WANT) or api != "RUNNING":
            fetch_full()
    return 0


if __name__ == "__main__":
    if "--backup-alarms" in sys.argv:
        raise SystemExit(backup_alarms())
    try:
        raise SystemExit(main())
    finally:
        drop_lock()
