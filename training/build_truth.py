#!/usr/bin/env python3
"""Turn two independent transcriptions into one answer key.

Two people were shown the same photographs and neither saw the other's work or
the reader's output. Where they agree, the text is on the sign. Where they
disagree, either the photograph is genuinely ambiguous or one of them made a
mistake, and in both cases it is not safe to hold the machine to it.

So the key is the intersection, at word level rather than line level — the two
often break a sign into lines differently while reading the same words, and
line-level matching would throw away agreement that is really there.

Only lines both marked "clear" count. A line a person could not read is not a
line a reader can be marked wrong for missing, and including it would push the
score down for the wrong reason.

    python3 training/build_truth.py --result <workflow-output.json>

The agreement rate it prints is worth as much as the key itself: it is the
ceiling on how precise any score built from this can be.
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data/ocr/real-photos/truth.json"
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def words(lines) -> Counter:
    text = " ".join(lines)
    return Counter(w.lower() for w in
                   WORD.findall(unicodedata.normalize("NFC", text)))


def clear_lines(photo: dict) -> list:
    return [ln["text"] for ln in photo.get("lines", [])
            if ln.get("legibility") == "clear" and ln.get("text", "").strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    # The task file wraps the workflow's return value in a envelope; a
    # hand-saved copy of just the return value should work too.
    data = json.loads(args.result.read_text(encoding="utf-8", errors="replace"))
    while "a" not in data and "result" in data:
        data = data["result"]
    if "a" not in data or "b" not in data:
        raise SystemExit(f"{args.result} has no pair of transcription passes")

    def by_name(rows):
        out = {}
        for p in rows:
            if p.get("opened") and p.get("filename"):
                out[p["filename"]] = p
        return out

    a, b = by_name(data["a"]), by_name(data["b"])
    shared = sorted(set(a) & set(b))
    print(f"{len(a)} from one reader, {len(b)} from the other, "
          f"{len(shared)} photographs both opened\n")

    photos, agreed_total, seen_total = {}, 0, 0
    no_text = 0
    for name in shared:
        wa, wb = words(clear_lines(a[name])), words(clear_lines(b[name]))
        both = wa & wb                      # multiset intersection
        union = wa | wb
        agreed_total += sum(both.values())
        seen_total += sum(union.values())
        if not sum(both.values()):
            no_text += 1
            photos[name] = {"lines": [], "agreed_words": 0,
                            "seen_by_one": sum(union.values())}
            continue
        photos[name] = {
            "lines": [" ".join(w for w, n in sorted(both.items()) for _ in range(n))],
            "agreed_words": sum(both.values()),
            "seen_by_one": sum(union.values()),
        }
        print(f"  {name[:52]:<52} {sum(both.values()):>3} agreed "
              f"of {sum(union.values()):>3}")

    rate = 100 * agreed_total / seen_total if seen_total else 0
    print(f"\n{agreed_total:,} words both readers saw, of {seen_total:,} either saw"
          f"  —  {rate:.0f}% agreement")
    print(f"{no_text} photographs carry no text the two agree on")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"photos": photos,
         "agreement": {"agreed": agreed_total, "either": seen_total,
                       "rate": round(rate, 1)}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
