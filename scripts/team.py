#!/usr/bin/env python3
"""The team's shared task list.

Work sits here and is claimed, not handed out. That distinction matters: a task
list lets a teammate pick up what it is free for, while assignment makes one
agent decide another's scope — which is the project owner's call, not a
teammate's.

    python3 scripts/team.py list                    # what is there, who has it
    python3 scripts/team.py claim ocr-photos lilly-b4
    python3 scripts/team.py done ocr-photos "800 images, 41% diacritic loss"
    python3 scripts/team.py block ocr-photos "needs a GPU"
    python3 scripts/team.py add "title" --files a.py --accept "what proves it"

Claiming is atomic: two teammates reaching for the same task at the same moment
cannot both get it, because the claim is an exclusive file creation rather than
a read-then-write on the list. Without that, both would read "unclaimed", both
would write their name, and one would silently lose.

A task with no acceptance criteria is not a task, it is a wish — `add` refuses
one. That is the difference between a board that coordinates work and a board
that only looks like it does.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD = REPO_ROOT / ".agents" / "tasks.json"
CLAIMS = REPO_ROOT / ".agents" / "claims"
STATES = ("open", "claimed", "blocked", "done")


def load() -> dict:
    if BOARD.exists():
        return json.loads(BOARD.read_text(encoding="utf-8"))
    return {"tasks": []}


def save(board: dict) -> None:
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    BOARD.write_text(json.dumps(board, ensure_ascii=False, indent=1), encoding="utf-8")


def find(board: dict, task_id: str) -> dict:
    for task in board["tasks"]:
        if task["id"] == task_id:
            return task
    raise SystemExit(f"no task called {task_id!r} — run: team.py list")


def stamp() -> str:
    return time.strftime("%H:%M")


def cmd_list(args) -> int:
    board = load()
    if not board["tasks"]:
        print("nothing on the board")
        return 0
    width = max(len(t["id"]) for t in board["tasks"])
    for state in STATES:
        rows = [t for t in board["tasks"] if t["state"] == state]
        if not rows:
            continue
        print(f"\n{state.upper()}")
        for t in rows:
            who = f"  [{t['owner']}]" if t.get("owner") else ""
            print(f"  {t['id']:<{width}}  {t['title']}{who}")
            if state == "open":
                print(f"  {'':<{width}}  proves it: {t['accept']}")
            if t.get("note"):
                print(f"  {'':<{width}}  → {t['note']}")
            if t.get("files"):
                print(f"  {'':<{width}}  touches: {', '.join(t['files'])}")
    return 0


def cmd_claim(args) -> int:
    board = load()
    task = find(board, args.id)
    if task["state"] == "done":
        raise SystemExit(f"{args.id} is already done")

    # Atomic: whoever creates this file first owns the task. A read-then-write
    # on the list would let two teammates both see "open" and both write.
    CLAIMS.mkdir(parents=True, exist_ok=True)
    marker = CLAIMS / f"{args.id}.claim"
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = marker.read_text(encoding="utf-8").strip()
        if holder == args.who:
            print(f"{args.id} is already yours")
            return 0
        raise SystemExit(f"{args.id} was claimed by {holder} first")
    with os.fdopen(fd, "w") as f:
        f.write(args.who)

    task.update(state="claimed", owner=args.who, note=f"claimed {stamp()}")
    save(board)
    print(f"{args.id} → {args.who}\n  proves it: {task['accept']}")
    if task.get("files"):
        print(f"  yours to touch: {', '.join(task['files'])}")
    return 0


def cmd_done(args) -> int:
    board = load()
    task = find(board, args.id)
    if not args.evidence.strip():
        raise SystemExit("say what proves it — a task closed without evidence is "
                         "a claim, not a result")
    task.update(state="done", note=f"{stamp()} — {args.evidence}")
    save(board)
    print(f"{args.id} done: {args.evidence}")
    return 0


def cmd_block(args) -> int:
    board = load()
    task = find(board, args.id)
    task.update(state="blocked", note=f"{stamp()} — {args.reason}")
    save(board)
    print(f"{args.id} blocked: {args.reason}")
    return 0


def cmd_add(args) -> int:
    if not args.accept:
        raise SystemExit("--accept is required: what would prove this is finished? "
                         "A task nobody can check off is a wish.")
    board = load()
    task_id = args.id or args.title.lower().replace(" ", "-")[:24]
    if any(t["id"] == task_id for t in board["tasks"]):
        raise SystemExit(f"{task_id} already exists")
    board["tasks"].append({
        "id": task_id, "title": args.title, "state": "open", "owner": "",
        "accept": args.accept, "files": args.files or [], "note": "",
    })
    save(board)
    print(f"added {task_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(fn=cmd_list)

    p = sub.add_parser("claim"); p.add_argument("id"); p.add_argument("who")
    p.set_defaults(fn=cmd_claim)

    p = sub.add_parser("done"); p.add_argument("id"); p.add_argument("evidence")
    p.set_defaults(fn=cmd_done)

    p = sub.add_parser("block"); p.add_argument("id"); p.add_argument("reason")
    p.set_defaults(fn=cmd_block)

    p = sub.add_parser("add"); p.add_argument("title")
    p.add_argument("--id"); p.add_argument("--accept", required=True)
    p.add_argument("--files", nargs="*")
    p.set_defaults(fn=cmd_add)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
