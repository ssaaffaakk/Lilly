#!/usr/bin/env python3
"""Refresh the team room from what is actually happening.

The room showed hand-written state, which goes stale the moment anything moves
and — worse — can say a run is going after it has died. This reads the live
sources instead: the task board, the evaluation log, the Kaggle run, and the
processes on this machine.

    python3 scripts/room.py            # rewrite the room's state block
    python3 scripts/room.py --print    # show what it found, change nothing

The crew's descriptions stay hand-written: what a teammate found today is a
judgement, not a reading. Everything that can be measured is measured.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOM = Path("/private/tmp/claude-501/-Users-safaksurmeli-Desktop-Lilly"
            "/13a83bf0-7b4c-4272-b678-1aa83ed5313c/scratchpad/lilly-ekibi.html")
BOARD = REPO_ROOT / ".agents" / "tasks.json"
KAGGLE = REPO_ROOT / ".venv" / "bin" / "kaggle"


def shell(*cmd, timeout=90) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def running(pattern: str) -> bool:
    return bool(shell("pgrep", "-f", pattern).strip())


def eval_progress(name: str = "eval-full.log") -> tuple:
    """How far a scoring run has got, from its own log."""
    log = REPO_ROOT / name
    if not log.exists():
        return None, None
    text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    marks = re.findall(r"(\d+)/(\d+)", text)
    if not marks:
        return None, None
    done, total = int(marks[-1][0]), int(marks[-1][1])
    return done, total


def kaggle_state(slug: str) -> str:
    out = shell(str(KAGGLE), "kernels", "status", slug)
    for state in ("COMPLETE", "ERROR", "RUNNING", "QUEUED", "CANCEL"):
        if state in out:
            return state
    return "UNKNOWN"


def build_runs() -> list:
    runs = []

    state = kaggle_state("afaksrmeli/lilly-speech")
    live = state in ("RUNNING", "QUEUED")
    runs.append({
        "name": "Ses eğitimi", "where": "Kaggle T4",
        "state": "live" if live else ("off" if state == "ERROR" else "wait"),
        "what": "Whisper ince ayarı — önce ölç, eğit, tekrar ölç",
        "pct": None,
        "label": ["Kaggle GPU", state.lower()],
    })

    done, total = eval_progress()
    alive = running("training/evaluate.py")
    if done and total:
        pct = round(100 * done / total)
        label = [f"{done} / {total} cümle", f"%{pct}"]
    else:
        pct, label = None, ["—", "başlamadı" if not alive else "hazırlanıyor"]
    runs.append({
        "name": "Çeviri ölçümü", "where": "yerel",
        "state": "live" if alive else ("off" if done and done >= (total or 1) else "wait"),
        "what": "2.009 görülmemiş cümle, taban ve ince ayarlı model",
        "pct": pct, "label": label,
    })

    # The scoring that actually decides what gets published: both builds run
    # through app/translate.py, sentence splitting included. The older run above
    # feeds rows to the model whole, which is not what the app does — and the
    # difference is not small, so the room has to show which one is speaking.
    done, total = eval_progress("app-path-eval.log")
    alive = running("app_path_eval.py")
    if done and total:
        pct = round(100 * done / total)
        label = [f"{done} / {total} cümle", f"%{pct}"]
    else:
        pct, label = None, ["—", "hazırlanıyor" if alive else "beklemede"]
    runs.append({
        "name": "Ürün yolundan ölçüm", "where": "yerel",
        "state": "live" if alive else "wait",
        "what": "Taban ve Lilly, ikisi de int8 — uygulamanın gerçek yolundan",
        "pct": pct, "label": label,
    })

    ocr = running("training/train_ocr.py")
    runs.append({
        "name": "Foto okuma eğitimi", "where": "yerel",
        "state": "live" if ocr else "wait",
        "what": "Gerçekçi fotoğraflar hazır olunca",
        "pct": None,
        "label": ["sırada", "koşuyor" if ocr else "bekliyor"],
    })
    return runs


def build_board() -> list:
    if not BOARD.exists():
        return []
    tasks = json.loads(BOARD.read_text(encoding="utf-8"))["tasks"]
    order = {"claimed": 0, "blocked": 1, "open": 2, "done": 3}
    tasks.sort(key=lambda t: order.get(t["state"], 4))
    return [{"id": t["id"], "state": t["state"], "who": t.get("owner", ""),
             "title": t["title"], "accept": t["accept"]} for t in tasks]


def build_now(runs, board) -> str:
    claimed = sum(1 for t in board if t["state"] == "claimed")
    done = sum(1 for t in board if t["state"] == "done")
    live = [r["name"].lower() for r in runs if r["state"] == "live"]
    what = " ve ".join(live) + " koşuyor" if live else "şu an koşan iş yok"
    return (f"{what.capitalize()}. Pano: {len(board)} iş, {claimed}'i alınmış"
            + (f", {done}'i bitmiş." if done else "."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args()

    if not ROOM.exists():
        print(f"no room at {ROOM}", file=sys.stderr)
        return 1

    html = ROOM.read_text(encoding="utf-8")
    match = re.search(r'(<script type="application/json" id="state">)(.*?)(</script>)',
                      html, re.S)
    if not match:
        print("the room has no state block", file=sys.stderr)
        return 1

    state = json.loads(match.group(2))
    state["runs"] = build_runs()
    state["board"] = build_board()
    state["now"] = build_now(state["runs"], state["board"])

    if args.show:
        print(state["now"])
        for r in state["runs"]:
            print(f"  {r['state']:<5} {r['name']:<22} {r['label'][0]}  {r['label'][1]}")
        for t in state["board"]:
            print(f"  {t['state']:<8} {t['id']:<14} {t['who'] or '—'}")
        return 0

    fresh = json.dumps(state, ensure_ascii=False, indent=1)
    ROOM.write_text(html[:match.start(2)] + "\n" + fresh + "\n" + html[match.end(2):],
                    encoding="utf-8")
    print(state["now"])
    print(f"refreshed {ROOM.name} — republish it to show the change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
