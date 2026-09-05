#!/usr/bin/env python3
"""Split a drawn set's photographs into blind transcription batches.

    python3 training/transcribe/make_batches.py --set test-v2 --pass a --size 20 [--offset 7]
    python3 training/transcribe/make_batches.py --set test-v2 --pass a --missing

Writes <work>/pass-<a|b>/batch-NN.todo.json, each a list of
{"filename", "photo"} for photographs on disk that no earlier batch of this
pass already holds, and prints the batch files. One transcriber (a person or
a vision agent that never sees the reader's output, see BRIEF.md) takes one
batch and saves batch-NN.json beside it. Run again after more photographs
arrive: only unassigned ones are batched.

--offset rotates the draw order so pass b groups photographs differently from
pass a (a batch boundary is the one thing two transcribers could share).
--missing ignores the todo files and re-batches every drawn photograph that no
finished batch-NN.json of the pass has an entry for — the recovery after a
transcriber was cut off half way.

<work> defaults to data/ocr/real-photos/<set>/transcribe, which git ignores;
only the merged pass-<x>.json is committed (merge.py).
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))
from transcription_pass import set_dir, drawn, photos_dir  # noqa: E402


def finished(pass_dir: Path) -> set:
    """Filenames with an entry in a finished batch-NN.json (drafts and parts are not answers)."""
    done = set()
    for f in pass_dir.glob("batch-*.json"):
        if re.fullmatch(r"batch-\d\d\.json", f.name):
            done |= {e.get("filename") for e in json.loads(f.read_text(encoding="utf-8"))}
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="name under data/ocr/real-photos/, e.g. test-v2")
    ap.add_argument("--pass", dest="pass_", required=True, choices=["a", "b"])
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="rotate the order so pass b groups photographs differently")
    ap.add_argument("--missing", action="store_true",
                    help="re-batch the photographs that no finished batch-NN.json of this pass holds an entry for")
    ap.add_argument("--work", type=Path, help="batch directory root (default <set>/transcribe)")
    a = ap.parse_args()

    d = set_dir(a.set)
    pdir = photos_dir(a.set, d)
    names = drawn(d)
    names = names[a.offset:] + names[:a.offset]
    out = (a.work or d / "transcribe") / f"pass-{a.pass_}"
    out.mkdir(parents=True, exist_ok=True)

    assigned = set()
    for f in sorted(out.glob("batch-*.todo.json")):
        assigned |= {e["filename"] for e in json.loads(f.read_text(encoding="utf-8"))}
    if a.missing:
        assigned = finished(out)
    todo = [n for n in names if n not in assigned and (pdir / n).is_file()]
    start = len(list(out.glob("batch-*.todo.json")))
    for i in range(0, len(todo), a.size):
        batch = [{"filename": n, "photo": str(pdir / n)} for n in todo[i:i + a.size]]
        p = out / f"batch-{start + i // a.size:02d}.todo.json"
        p.write_text(json.dumps(batch, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(p, len(batch))
    off_disk = sum(1 for n in names if not (pdir / n).is_file())
    print(f"{len(todo)} newly batched; {len(assigned) + len(todo)} assigned of {len(names)}; "
          f"{off_disk} not on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
