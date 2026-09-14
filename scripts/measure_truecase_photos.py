#!/usr/bin/env python3
"""Does LILLY_TRUECASE help on real photographs? — the photograph bar.

`training/RESULTS-truecase.md` measured the restorer on uppercased FLORES
sentences and left the honest bar open in its own words: "the honest bar for the
product is a score on photographs, not uppercased FLORES." This is that bar,
built the only way it can be without English references for the signs: as an A/B
the owner can read, not a chrF2 the machine can.

It changes nothing. It reuses the shipped reader's own cached output on the 40
Commons photographs (`data/ocr/real-photos/reader-output-paddle-v6.json`, PP-OCRv6
at the 0.9 floor, the reader `app/ocr.py` serves) and drives the real `bs-en`
`app.translate.Engine` twice per photo: once with LILLY_TRUECASE off, once on.
That is the product path — `translate_photo` feeds the whole OCR text to
`translate()`, and `translate()` is where the flag lives — so the two columns are
exactly what a user gets with the flag down and up, not a reimplementation of it.

Output: `training/RESULTS-truecase-photos.md`, every photo whose translation the
flag changes, source and both renderings side by side, for the pre-registered
owner judgement the flag's default-on is gated behind.

    .venv/bin/python scripts/measure_truecase_photos.py
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

READINGS = REPO / "data/ocr/real-photos/reader-output-paddle-v6.json"
OUT = REPO / "training/RESULTS-truecase-photos.md"


def translate_both(engine, text, restore):
    """(off, on) — the product's output with the flag down and up.

    The flag is read from the environment inside translate(); toggling it here
    drives the real decision rather than a copy of it. `restore` is the pure
    restorer, called only to report what the flag did to the source.
    """
    os.environ["LILLY_TRUECASE"] = "0"
    off = engine.translate(text, truncate=True)
    os.environ["LILLY_TRUECASE"] = "1"
    on = engine.translate(text, truncate=True)
    os.environ["LILLY_TRUECASE"] = "0"
    restored, changed = restore(text)
    return off, on, restored, changed


def main():
    from app.translate import get_engine
    from app.truecase import restore_sentence_case

    blob = json.loads(READINGS.read_text())
    reader_fp = blob.get("reader", "?")
    readings = blob["readings"]
    engine = get_engine("bs-en")

    rows = []
    touched = 0        # the flag changed the source text
    changed_out = 0    # the flag changed the translation
    for photo in sorted(readings):
        text = (readings[photo] or "").strip()
        if not text:
            continue
        off, on, restored, src_changed = translate_both(engine, text, restore_sentence_case)
        if src_changed:
            touched += 1
        if off != on:
            changed_out += 1
        rows.append({"photo": photo, "src": text, "restored": restored,
                     "src_changed": src_changed, "off": off, "on": on})

    with_text = len(rows)
    diffs = [r for r in rows if r["off"] != r["on"]]

    def block(label, s):
        return f"**{label}**\n\n```\n{s}\n```\n"

    lines = [
        "# RESULTS — truecase on real photographs — 14 September 2026", "",
        f"The photograph bar `training/RESULTS-truecase.md` left open: does "
        f"`LILLY_TRUECASE` help on real signs, not uppercased FLORES? No English "
        f"reference exists for these signs, so this is an A/B for the owner to "
        f"read, not a chrF2. Nothing here trains or changes the served build.", "",
        f"- Reader: PP-OCRv6 cached output, fingerprint `{reader_fp}` "
        f"(`data/ocr/real-photos/reader-output-paddle-v6.json`), the reader "
        f"`app/ocr.py` serves.",
        f"- Engine: `bs-en` `app.translate.Engine`, the camera's default "
        f"direction (`Lilly.translate_photo`).",
        f"- Method: each photo's whole OCR text through `translate()` twice, "
        f"`LILLY_TRUECASE` off then on. This is the product path; the flag lives "
        f"inside `translate()`.", "",
        "## What the flag did", "",
        f"| | count | of 40 |",
        f"|---|---|---|",
        f"| photographs with OCR text | {with_text} | {100*with_text//40}% |",
        f"| source the flag recased | {touched} | {100*touched//40}% |",
        f"| **translation the flag changed** | **{len(diffs)}** | "
        f"**{100*len(diffs)//40}%** |", "",
        f"The recaser only fires on predominantly-uppercase text (≥8 letters, "
        f"≥80% upper), so on the {with_text - touched} photographs whose OCR was "
        f"not shouted it did nothing and the two columns are identical. The "
        f"{len(diffs)} below are where a user would see a different answer.", "",
        "---", "",
    ]

    for r in diffs:
        lines.append(f"### {r['photo']}")
        lines.append("")
        lines.append(block("Sign (OCR, as read)", r["src"]))
        if r["src_changed"]:
            lines.append(block("Source after recasing", r["restored"]))
        lines.append(block("Translation — flag OFF (shipped today)", r["off"]))
        lines.append(block("Translation — flag ON (candidate)", r["on"]))
        lines.append("---")
        lines.append("")

    lines += [
        "## The judgement this bar needs", "",
        "For each pair above: is the flag-ON English a better rendering of the "
        "sign than flag-OFF? Turning the flag on by default is a product change; "
        "it ships only if ON wins clearly across these, per the pre-registration "
        "`training/RESULTS-truecase.md` points to. This file is the evidence for "
        "that call, not the call itself.", "",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{with_text} photos with text, flag recased {touched}, "
          f"changed the translation on {len(diffs)}")
    print(f"written to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
