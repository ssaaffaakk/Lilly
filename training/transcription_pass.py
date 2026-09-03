#!/usr/bin/env python3
"""The blind two-pass transcription behind every photograph answer key.

`training/build_truth.py` turns two independent transcriptions into a key by
keeping only the words both passes saw. That only means something if the two
passes were blind: neither saw the other's work, neither saw the reader's
output, and each looked at every photograph in the draw. This script makes the
blindness checkable and the hand-off mechanical.

    python3 training/transcription_pass.py sheet --set test-v2 --pass a
        writes test-v2/pass-a.todo.json: one entry per drawn photograph, empty

    (a person, or a vision agent that never sees the reader's output, fills in
     lines and legibility for every entry and saves it as pass-a.json;
     pass b is a different person or agent, same procedure, no peeking)

    python3 training/transcription_pass.py check --set test-v2
        refuses if either pass skipped a photograph, invented a filename, left
        the empty template in place, or looks like the reader's own output

    python3 training/transcription_pass.py pair --set test-v2
        writes test-v2/pair.json in the shape build_truth.py reads, then:
        python3 training/build_truth.py --result data/ocr/real-photos/test-v2/pair.json \\
            --out data/ocr/real-photos/test-v2/truth-v2.json

Entry shape, the same one the 40 were transcribed in:

    {"filename": "Mostar_signs.JPG", "opened": true,
     "lines": [{"text": "Dubrovnik", "legibility": "clear"},
               {"text": "Sara...", "legibility": "unclear"}]}

`legibility` is `clear`, `unclear` or `none`. Only `clear` lines enter the
key; an `unclear` line records that a person saw text there and could not
read it, which is a detector question, not a reading error. A photograph with
no text has `lines: []` — and is still `opened: true`, because a photograph
nobody looked at is a hole, not a blank.

Rules the transcriber follows, written here so both passes follow the same:
- Transcribe what is on the sign, letter for letter, diacritics included.
- One line per line of text on the sign; do not merge signs.
- Do not correct spelling, expand abbreviations or translate.
- Do not run any OCR on the photograph to "help". The reader is what is being
  measured; a key it helped write measures agreement with itself.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL = REPO_ROOT / "data" / "ocr" / "real-photos"
LEGIBILITY = {"clear", "unclear", "none"}


def set_dir(name: str) -> Path:
    d = REAL / name
    if not (d / "sample.txt").is_file():
        raise SystemExit(f"{d / 'sample.txt'} is missing — draw the set first")
    return d


def drawn(d: Path) -> list:
    return [ln.strip() for ln in (d / "sample.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]


def photos_dir(name: str, d: Path) -> Path:
    # test-v2 keeps its own photos/; test-mly reads the harvest directory
    return d / "photos" if (d / "photos").is_dir() else REAL / "mapillary"


def cmd_sheet(args) -> int:
    d = set_dir(args.set)
    names = drawn(d)
    pdir = photos_dir(args.set, d)
    missing = [n for n in names if not (pdir / n).is_file()]
    if missing:
        print(f"note: {len(missing)} of {len(names)} photographs are not under {pdir} yet "
              f"(first: {missing[0]}); the sheet lists them anyway", file=sys.stderr)
    todo = [{"filename": n, "photo": str((pdir / n).relative_to(REPO_ROOT)), "opened": False, "lines": []}
            for n in names]
    out = d / f"pass-{args.pass_}.todo.json"
    out.write_text(json.dumps(todo, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}: {len(todo)} photographs to transcribe, blind")
    return 0


def load_pass(d: Path, which: str) -> list:
    path = d / f"pass-{which}.json"
    if not path.is_file():
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} is missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "entries" in data:
        data = data["entries"]
    if not isinstance(data, list):
        raise SystemExit(f"{path.name}: expected a list of entries")
    return data


def problems(entries: list, names: list, label: str) -> list:
    out = []
    seen = {}
    for e in entries:
        fn = e.get("filename")
        if fn not in names:
            out.append(f"{label}: unknown filename {fn!r}")
            continue
        if fn in seen:
            out.append(f"{label}: {fn} appears twice")
        seen[fn] = e
        if not e.get("opened"):
            out.append(f"{label}: {fn} not marked opened")
        for ln in e.get("lines", []):
            if not isinstance(ln, dict) or "text" not in ln:
                out.append(f"{label}: {fn} has a malformed line {ln!r}")
                continue
            if ln.get("legibility") not in LEGIBILITY:
                out.append(f"{label}: {fn} line {ln.get('text')!r} has legibility "
                           f"{ln.get('legibility')!r}, want one of {sorted(LEGIBILITY)}")
            if ln.get("legibility") == "clear" and not ln["text"].strip():
                out.append(f"{label}: {fn} has an empty clear line")
    for n in names:
        if n not in seen:
            out.append(f"{label}: {n} was never transcribed — a hole, not a blank")
    return out


def cmd_check(args) -> int:
    d = set_dir(args.set)
    names = drawn(d)
    a, b = load_pass(d, "a"), load_pass(d, "b")
    issues = problems(a, names, "pass a") + problems(b, names, "pass b")
    if a == b:
        issues.append("pass a and pass b are identical — one of them was copied, not transcribed")
    # A pass that reproduces the reader's cached output word for word is not blind.
    for cache in d.glob("reader-output*.json"):
        readings = json.loads(cache.read_text(encoding="utf-8")).get("readings", {})
        for label, entries in (("pass a", a), ("pass b", b)):
            same = 0
            for e in entries:
                got = " ".join(ln["text"] for ln in e.get("lines", []) if isinstance(ln, dict)).split()
                mine = readings.get(e.get("filename"), "").split()
                if got and got == mine:
                    same += 1
            if same and same >= max(3, len(entries) // 10):
                issues.append(f"{label}: {same} photographs match {cache.name} word for word — "
                              "that is the reader's output, not a transcription")
    for line in issues[:40]:
        print("  " + line)
    if issues:
        print(f"\n{len(issues)} problem(s). Fix the pass files; do not edit around them.")
        return 1
    clear = lambda es: sum(1 for e in es for ln in e.get("lines", []) if ln.get("legibility") == "clear")
    print(f"both passes cover all {len(names)} photographs; clear lines: a={clear(a)}, b={clear(b)}")
    return 0


def cmd_pair(args) -> int:
    if cmd_check(args):
        return 1
    d = set_dir(args.set)
    a, b = load_pass(d, "a"), load_pass(d, "b")
    out = d / "pair.json"
    out.write_text(json.dumps({"a": a, "b": b}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)} — now: python3 training/build_truth.py --result {out.relative_to(REPO_ROOT)} "
          f"--out {(d / ('truth-' + args.set.replace('test-', '') + '.json')).relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet"); s.add_argument("--set", required=True); s.add_argument("--pass", dest="pass_", choices=["a", "b"], required=True)
    c = sub.add_parser("check"); c.add_argument("--set", required=True)
    p = sub.add_parser("pair"); p.add_argument("--set", required=True)
    args = ap.parse_args()
    return {"sheet": cmd_sheet, "check": cmd_check, "pair": cmd_pair}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
