#!/usr/bin/env python3
"""Run a command in a new POSIX session so parent-shell death cannot kill it.

Cursor/agent `nohup cmd &` keeps the child's pgid. When that shell's process
group is torn down, harvest dies with no FAIL line. Double-fork + setsid
moves the job to session 1 (or a new session leader).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    log_path = REPO / "logs" / "harvest-detach.nohup"
    role = "job"
    args = argv[:]
    while args:
        if args[0] == "--log" and len(args) >= 2:
            log_path = Path(args[1])
            args = args[2:]
        elif args[0] == "--role" and len(args) >= 2:
            role = args[1]
            args = args[2:]
        elif args[0] == "--":
            args = args[1:]
            break
        else:
            break
    if not args:
        print("usage: harvest_detach.py [--log PATH] -- CMD [ARGS...]", file=sys.stderr)
        return 2

    os.chdir(REPO)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if os.fork() > 0:
        return 0

    os.setsid()
    if os.fork() > 0:
        os._exit(0)

    os.environ["LILLY_HARVEST_DETACHED"] = "1"
    os.environ["LILLY_HARVEST_ROLE"] = role
    os.environ["LILLY_HARVEST_PGID"] = str(os.getpgid(0))

    with log_path.open("ab", buffering=0) as logf:
        os.dup2(logf.fileno(), 1)
        os.dup2(logf.fileno(), 2)

    cmd = args
    caff = shutil.which("caffeinate")
    if caff:
        cmd = [caff, "-i"] + args
    os.execvp(cmd[0], cmd)
    return 127


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
