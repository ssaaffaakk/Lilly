#!/usr/bin/env python3
"""Merge one pass's batch answers into data/ocr/real-photos/<set>/pass-<x>.json.

    python3 training/transcribe/merge.py --set test-v2 --pass a [--partial]

Reads <work>/pass-<x>/batch-NN.json (the filled answers; drafts, parts and
todo files are skipped), keeps sample.txt order, and reports what is missing.
Refuses to write if any drawn photograph has no entry or one appears twice,
unless --partial. The written file is what `transcription_pass.py check` and
`pair` read; it is the file that gets committed.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))
from transcription_pass import set_dir, drawn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="name under data/ocr/real-photos/, e.g. test-v2")
    ap.add_argument("--pass", dest="pass_", required=True, choices=["a", "b"])
    ap.add_argument("--partial", action="store_true", help="write what there is even if photographs are missing")
    ap.add_argument("--work", type=Path, help="batch directory root (default <set>/transcribe)")
    a = ap.parse_args()

    d = set_dir(a.set)
    names = drawn(d)
    pass_dir = (a.work or d / "transcribe") / f"pass-{a.pass_}"
    entries, bad = {}, []
    for f in sorted(pass_dir.glob("batch-*.json")):
        if not re.fullmatch(r"batch-\d\d\.json", f.name):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            bad.append(f"{f.name}: {exc}")
            continue
        for e in data:
            fn = e.get("filename")
            if fn in entries:
                bad.append(f"{fn} appears in two batches")
            entries[fn] = e
    for line in bad:
        print("  " + line)
    missing = [n for n in names if n not in entries]
    extra = [n for n in entries if n not in names]
    print(f"{len(entries)} entries, {len(missing)} missing, {len(extra)} unknown filenames")
    for n in missing[:10]:
        print("  missing:", n)
    for n in extra[:10]:
        print("  unknown:", n)
    if (missing or extra or bad) and not a.partial:
        return 1
    out = d / f"pass-{a.pass_}.json"
    merged = [entries[n] for n in names if n in entries]
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    clear = sum(1 for e in merged for ln in e.get("lines", []) if ln.get("legibility") == "clear")
    opened = sum(1 for e in merged if e.get("opened"))
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(merged)} entries, {opened} opened, {clear} clear lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
