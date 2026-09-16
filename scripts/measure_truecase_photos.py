#!/usr/bin/env python3
"""Does the photograph-path truecaser help on real signs? — the photograph bar.

`training/RESULTS-truecase.md` measured the restorer on uppercased FLORES
sentences and left the honest bar open in its own words: "the honest bar for the
product is a score on photographs, not uppercased FLORES." This is that bar,
built the only way it can be without English references for the signs: as an A/B
for the owner (or a blind pass) to read, not a chrF2 the machine can.

It changes nothing. It reuses the shipped reader's own cached output on the 40
Commons photographs (`data/ocr/real-photos/reader-output-paddle-v6.json`, PP-OCRv6
at the 0.9 floor, the reader `app/ocr.py` serves) and drives the real `bs-en`
`app.translate.Engine` twice per photo:

    OFF  the whole OCR text straight through translate()   (shipped today)
    ON   the photograph path with LILLY_TRUECASE on:
         app.truecase.restore_photo_text line by line -- a line is recased only
         when it is shouted AND app.detect calls it Bosnian, so an English
         caption, a brand or a calm line is left alone -- then translate()

ON is exactly what `Lilly.translate_photo` produces with the flag up: the flag
now lives on the photograph path, not inside translate(), and it recases per
line rather than the whole blob. The two columns are the product with the flag
down and up, not a reimplementation of it.

Output: `training/RESULTS-truecase-photos.md`, every photo whose translation the
flag changes, source and both renderings side by side, each with a blank verdict
for a blind pass to fill -- the pre-registered owner judgement the flag's
default-on is gated behind. This file is the evidence, not the call.

    .venv/bin/python scripts/measure_truecase_photos.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

READINGS = REPO / "data/ocr/real-photos/reader-output-paddle-v6.json"
OUT = REPO / "training/RESULTS-truecase-photos.md"


def annotate_lines(text, is_upper, detect):
    """Per line: what the photograph restorer did and why. -> list of dicts.

    Mirrors app.truecase.restore_photo_text's own decision, so the report says
    exactly which lines were recased, which were kept as a foreign caption, and
    which were too calm to touch.
    """
    notes = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if not is_upper(line):
            notes.append((line, "calm — left as read"))
        elif detect(line) == "bs":
            notes.append((line, "recased (shouted, Bosnian)"))
        else:
            notes.append((line, "kept — read as English"))
    return notes


def main():
    from app.translate import get_engine
    from app.truecase import restore_photo_text, is_predominantly_upper
    from app.detect import detect_language

    blob = json.loads(READINGS.read_text())
    reader_fp = blob.get("reader", "?")
    readings = blob["readings"]
    engine = get_engine("bs-en")

    rows = []
    touched = 0        # the flag recased at least one line of the source
    kept_english = 0   # the flag deliberately left a shouted English line alone
    for photo in sorted(readings):
        text = (readings[photo] or "").strip()
        if not text:
            continue
        recased, src_changed = restore_photo_text(text, "bs")
        off = engine.translate(text, truncate=True)
        on = engine.translate(recased, truncate=True)
        notes = annotate_lines(text, is_predominantly_upper, detect_language)
        if src_changed:
            touched += 1
        if any(n[1].startswith("kept") for n in notes):
            kept_english += 1
        rows.append({"photo": photo, "src": text, "recased": recased,
                     "src_changed": src_changed, "off": off, "on": on,
                     "notes": notes})

    with_text = len(rows)
    diffs = [r for r in rows if r["off"] != r["on"]]

    def block(label, s):
        return f"**{label}**\n\n```\n{s}\n```\n"

    lines = [
        "# RESULTS — truecase on real photographs — 16 September 2026", "",
        "The photograph bar `training/RESULTS-truecase.md` left open: does the "
        "truecaser help on real signs, not uppercased FLORES? No English "
        "reference exists for these signs, so this is an A/B for a blind pass or "
        "the owner to read, not a chrF2. Nothing here trains or changes the "
        "served build.", "",
        "Re-measured 16 Sep after the restorer was re-scoped: it now lives on the "
        "photograph path (`Lilly.translate_photo`), not inside `translate()`, and "
        "runs line by line — a line is recased only when it is shouted **and** "
        "`app.detect` reads it as Bosnian, so an English caption or a brand on the "
        "same sign is left alone. The earlier version recased the whole blob "
        "through `translate()` and touched typed text too.", "",
        f"- Reader: PP-OCRv6 cached output, fingerprint `{reader_fp}` "
        f"(`data/ocr/real-photos/reader-output-paddle-v6.json`), the reader "
        f"`app/ocr.py` serves.",
        f"- Engine: `bs-en` `app.translate.Engine`, the camera's default "
        f"direction (`Lilly.translate_photo`).",
        f"- Method: OFF = whole OCR text through `translate()` (shipped today). "
        f"ON = `restore_photo_text` line by line, then `translate()` — the flag "
        f"up on the photograph path.", "",
        "## What the flag did", "",
        f"| | count | of 40 |",
        f"|---|---|---|",
        f"| photographs with OCR text | {with_text} | {100*with_text//40}% |",
        f"| source the flag recased | {touched} | {100*touched//40}% |",
        f"| had a shouted English line kept back | {kept_english} | "
        f"{100*kept_english//40}% |",
        f"| **translation the flag changed** | **{len(diffs)}** | "
        f"**{100*len(diffs)//40}%** |", "",
        f"The recaser fires only on predominantly-uppercase lines (≥8 letters, "
        f"≥80% upper) that `app.detect` reads as Bosnian. The {len(diffs)} below "
        f"are where a user would see a different answer; label each **better / "
        f"same / worse** against OFF.", "",
        "---", "",
    ]

    for r in diffs:
        lines.append(f"### {r['photo']}")
        lines.append("")
        lines.append(block("Sign (OCR, as read)", r["src"]))
        lines.append("**What the restorer did, line by line**")
        lines.append("")
        for line, note in r["notes"]:
            lines.append(f"- `{line}` — {note}")
        lines.append("")
        if r["src_changed"]:
            lines.append(block("Source after recasing (what ON translates)", r["recased"]))
        lines.append(block("Translation — flag OFF (shipped today)", r["off"]))
        lines.append(block("Translation — flag ON (candidate)", r["on"]))
        lines.append("**Verdict (blind): better / same / worse — _____**")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines += [
        "## The judgement this bar needs", "",
        "For each pair above, fill the blind verdict: is the flag-ON English a "
        "better rendering of the sign than flag-OFF, the same, or worse? Turning "
        "the flag on by default is a product change; per the pre-registration it "
        "ships only on a clear majority of **better** with no serious "
        "regressions. Tally the verdicts here before the call — this file is the "
        "evidence, not the call itself.", "",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{with_text} photos with text, flag recased {touched}, "
          f"kept {kept_english} English lines, changed the translation on {len(diffs)}")
    print(f"written to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
