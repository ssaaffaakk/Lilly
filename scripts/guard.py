#!/usr/bin/env python3
"""Refuse to start a heavy job when the machine has no room for it.

On 27 August this machine kernel-panicked. The log read
`Compressor Info: 100% of compressed pages limit (BAD)` with fifteen swapfiles
and a watchdog that had heard nothing for 94 seconds. Five of this project's
Python jobs were running at once on 8 GB: a photograph harvest with two workers
each holding its own EasyOCR model, a crop cutter holding a third, a speech
evaluation holding Whisper, and an audio download decoding into memory.

Nothing in the project stopped that. Each job is reasonable alone and none of
them knows the others exist, so the only thing standing between a normal
afternoon and a hard reboot was somebody remembering. That is not a safeguard.

Heavy entry points call `claim(gigabytes, name)` before loading a model. It
measures what is actually free, counts the project's other running jobs, and
exits with a message naming them rather than starting a sixth.

    from scripts.guard import claim
    claim(1.5, "photo harvest")

    python3 scripts/guard.py            # what is running, and what is free
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Leave this much for the operating system, the editor and the browser. Below
# it macOS is already compressing and swapping, which is where the panic began.
RESERVE_GB = 2.5


def free_gb() -> float:
    """Memory that could be handed out now, without swapping for it."""
    if sys.platform == "darwin":
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        size = int(re.search(r"page size of (\d+)", out).group(1))
        pages = {}
        for line in out.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip().rstrip(".")
            if value.isdigit():
                pages[key.strip()] = int(value)
        if "Pages free" not in pages:
            raise RuntimeError(f"could not read vm_stat; got keys {sorted(pages)}")
        usable = (pages["Pages free"]
                  + pages.get("Pages inactive", 0)
                  + pages.get("Pages speculative", 0))
        return usable * size / 1024 ** 3
    # Kaggle and other Linux hosts have no vm_stat. MemAvailable is the same
    # question: how much could a new allocation get without pushing into swap?
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            kb = int(line.split()[1])
            return kb / 1024 ** 2
    raise RuntimeError("could not read MemAvailable from /proc/meminfo")


def running_jobs() -> list:
    """This project's other Python jobs, by pid and script name."""
    out = subprocess.run(["ps", "-eo", "pid,rss,command"],
                         capture_output=True, text=True).stdout
    jobs = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss, command = int(parts[0]), int(parts[1]), parts[2]
        if pid == os.getpid() or "guard.py" in command:
            continue
        if str(REPO_ROOT) not in command and ".venv/bin/python" not in command:
            continue
        # Match on the executable, not on the whole command line. A job started
        # as `zsh -c '.venv/bin/python training/train_ocr.py'` mentions python
        # inside the shell's own argument, so the wrapper matched too and one
        # real run was reported as two — the phantom named after whatever the
        # fallback below happened to find first, which read as a username.
        words = command.split()
        executable = Path(words[0]).name.lower() if words else ""
        if "python" not in executable:
            continue
        script = next((w for w in words if w.endswith(".py")), command[:40])
        # ps reports rss in kilobytes. Dividing by 1024**2 gives gigabytes, and
        # the result was printed under an "MB" heading: a 607 MB process read as
        # "0.6 MB" in the one message whose whole job is to name what to stop.
        jobs.append((pid, rss / 1024, Path(script).name))
    return jobs


def claim(gigabytes: float, name: str, force_env: str = "LILLY_IGNORE_GUARD") -> None:
    """Stop here unless there is room for `gigabytes` on top of what is running.

    Stacking jobs is what panicked this machine on 27 August. One serial job
    on 8 GB is the designed path: require only the model, not model + 2.5 GB
    reserve (that bar is ~3.9 GB free and never clears while the editor is open).
    A second Lilly job still pays the reserve, so two EasyOCR copies cannot start.
    """
    if os.environ.get(force_env):
        return
    free = free_gb()
    jobs = running_jobs()
    required = gigabytes + RESERVE_GB if jobs else gigabytes
    if free >= required:
        return
    lines = [f"not enough memory to start {name}: it needs about "
             f"{gigabytes:.1f} GB and {free:.1f} GB is free, with "
             f"{RESERVE_GB:.1f} GB reserved for everything else."]
    if jobs:
        lines.append("\nalready running:")
        lines += [f"  {pid:>6}  {rss:6.0f} MB  {script}" for pid, rss, script in jobs]
        lines.append("\nStop one of those and try again. Running them together is "
                     "what panicked this machine on 27 August.")
    else:
        lines.append("\nNothing of this project's is running, so the memory is "
                     "elsewhere — close what you can, or set "
                     f"{force_env}=1 if you are sure.")
    raise SystemExit("\n".join(lines))


def main() -> int:
    free = free_gb()
    print(f"{free:.1f} GB free, {RESERVE_GB:.1f} GB reserved")
    jobs = running_jobs()
    if not jobs:
        print("no Lilly jobs running")
        return 0
    print(f"\n{len(jobs)} job(s) running:")
    for pid, rss, script in jobs:
        print(f"  {pid:>6}  {rss:6.0f} MB  {script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def workers_for(gigabytes_each: float, requested: int, name: str = "workers") -> int:
    """How many copies of a model actually fit, whatever was asked for.

    `claim` is checked once, at startup, and a pool that spawns two workers each
    loading its own EasyOCR reader commits three times what the parent measured.
    That is the specific shape of the 27 August panic: `--workers 2` on a photo
    harvest, while a crop cutter and a speech evaluation held models of their
    own.

    Returns at least one — refusing to run at all is worse than running slowly,
    and `claim` has already established there is room for the first.
    """
    free = free_gb()
    room = int(max(0.0, free - RESERVE_GB) // gigabytes_each)
    allowed = max(1, min(requested, room))
    if allowed < requested:
        print(f"  {requested} {name} asked for, {allowed} fit in "
              f"{free:.1f} GB free — using {allowed}", file=sys.stderr)
    return allowed
