#!/usr/bin/env python3
"""Does the photograph-path truecaser help on real signs? — the photograph bar.

`training/RESULTS-truecase.md` measured the restorer on uppercased FLORES and
left the honest bar open in its own words: "the honest bar for the product is a
score on photographs, not uppercased FLORES." This is that bar, built the only
way it can be without English references for the signs: a blind, randomised A/B
for the owner (or a blind pass) to read, not a chrF2.

It reads the 40 Commons photographs with the SHIPPED reader (PP-OCRv6 at the 0.9
floor, `app.ocr.scan`, the exact text `Lilly.translate_photo` feeds the
translator) and drives the real `bs-en` engine twice per photo:

    OFF  the whole OCR text straight through translate()   (shipped today)
    ON   the photograph path with LILLY_TRUECASE on:
         app.truecase.restore_photo_text line by line, then translate()

Guards, so a wrong or leaky measurement cannot slip through:

  * cv2 fail-stop. The shipped OCR numbers hold only under OpenCV 4.10.0
    (do-not-repeat 17); the script refuses any other cv2 and records the version.
  * reader fail-stop. The A/B must run on the shipped reader's floor-0.9 text,
    not the floor-off bake-off arm. The cache is stamped with
    app.ocr.reader_identity() and refused unless it equals the live identity and
    that identity carries `rec>=0.9`. Identity is the stable discriminator here.
  * product-path parity (--verify-parity). scan() feeds translate_photo and
    scan_regions()[0] feeds translate_photo_regions; the run asserts they are
    identical on all 40 so both product paths are measured, not just one.
  * blind output. The two renderings print in a per-photo random, balanced order
    with no OFF/ON or candidate/shipped label; the mapping and per-line detail go
    to a key that is NOT given to the evaluator. Only the key's SHA-256 is
    committed (the .gitignore keeps the key out of Git, not off the disk), so the
    eventual reveal is tamper-evident, not that the file is unreachable.

Provenance, not just metadata: the cache records the cv2 version, the live
reader_identity() and the generation timestamp captured while the scan actually
ran under the cv2 gate, so the stamp is written by the reading, not appended to
a cache made some other run.

Outputs:
  training/RESULTS-truecase-photos.md      the blind A/B form (what a pass reads)
  training/truecase-photos-key.json        the sealed key (git-ignored; local only)
  training/truecase-photos-key.sha256      the committed commitment to that key

    .venv/bin/python scripts/measure_truecase_photos.py --verify-parity
"""
import argparse
import datetime
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PHOTOS = REPO / "data/ocr/real-photos/scored"
CACHE = REPO / "data/ocr/real-photos/reader-output-floor0.9.json"
OUT = REPO / "training/RESULTS-truecase-photos.md"
KEY = REPO / "training/truecase-photos-key.json"
KEY_SHA = REPO / "training/truecase-photos-key.sha256"
SEED = 20260916
CV2_PIN = "4.10.0"

GATE = (
    "Default-on ships only if, on the changed photographs, the candidate is "
    "**better in at least two thirds** of them AND there is **no serious "
    "regression**. Serious regression = a place / person / brand name corrupted, "
    "a new repetition or hallucination, or a correct English line broken.")


def preflight() -> str:
    """cv2 must be the pinned 4.10.0 or the shipped OCR numbers do not hold."""
    import cv2
    if cv2.__version__ != CV2_PIN:
        raise SystemExit(
            f"refusing: cv2 is {cv2.__version__}, not the pinned {CV2_PIN} "
            f"(do-not-repeat 17); the shipped reader's output drifts otherwise.")
    return cv2.__version__


def photo_files():
    return sorted([p for p in PHOTOS.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def shipped_readings(cv2_version, do_parity):
    """Floor-0.9 OCR text per photo from the shipped reader. Fail-stop guarded."""
    from app.ocr import reader_identity, scan
    live = reader_identity()
    if "rec>=0.9" not in live:
        raise SystemExit(
            f"refusing: the shipped reader must be PP-OCRv6 at floor 0.9, but "
            f"reader_identity() is {live!r} (the floor-off bake-off arm is not "
            f"the product).")

    # Only reuse a cache whose provenance says it was made under this exact
    # reader + cv2 and carries a generation stamp; anything else is re-read, so a
    # patched or foreign cache can never masquerade as this run's product input.
    blob = None
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if (cached.get("reader_identity") == live and cached.get("cv2") == cv2_version
                and cached.get("generated_at")):
            blob = cached
    if blob is None:
        photos = photo_files()
        if not photos:
            raise SystemExit(f"no photographs under {PHOTOS}")
        generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        readings, t0 = {}, time.time()
        for i, p in enumerate(photos, 1):
            readings[p.name] = (scan(str(p)) or "").strip()
            print(f"  read {i}/{len(photos)} {p.name} ({time.time()-t0:.0f}s)", flush=True)
        blob = {"reader_identity": live, "cv2": cv2_version,
                "generated_at": generated_at, "readings": readings, "scan_parity": None}
        CACHE.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")

    if do_parity:
        blob["scan_parity"] = _verify_parity(blob["readings"])
        CACHE.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    return blob


def _verify_parity(readings) -> str:
    """scan_regions()[0] == scan() text on all 40 — both product paths measured."""
    from app.ocr import scan_regions
    photos = photo_files()
    mismatches = []
    for p in photos:
        rt = (scan_regions(str(p))[0] or "").strip()
        if rt != (readings.get(p.name) or "").strip():
            mismatches.append(p.name)
    if mismatches:
        raise SystemExit(
            f"scan()/scan_regions() disagree on {len(mismatches)} photos: "
            f"{mismatches[:5]} — the region path and the whole-text path do not "
            f"measure the same input; fix before trusting the A/B.")
    return f"{len(photos)}/{len(photos)} identical"


def annotate_lines(text, is_upper, detect):
    notes = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if not is_upper(line):
            notes.append([line, "calm — left as read"])
        elif detect(line) == "bs":
            notes.append([line, "recased (shouted, Bosnian)"])
        else:
            notes.append([line, "kept — read as a foreign line"])
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-parity", action="store_true",
                    help="run scan_regions() on all 40 and assert it equals scan()")
    args = ap.parse_args()

    cv2_version = preflight()
    from app.translate import get_engine
    from app.truecase import restore_photo_text, is_predominantly_upper
    from app.detect import detect_language

    blob = shipped_readings(cv2_version, args.verify_parity)
    readings, reader_id, parity = blob["readings"], blob["reader_identity"], blob["scan_parity"]
    generated_at = blob.get("generated_at", "?")
    print(f"reader {reader_id} | cv2 {cv2_version} | generated {generated_at} | parity {parity}")
    engine = get_engine("bs-en")

    rows = []
    for photo in sorted(readings):
        text = (readings[photo] or "").strip()
        if not text:
            continue
        recased, src_changed = restore_photo_text(text, "bs")
        off = engine.translate(text, truncate=True)
        on = engine.translate(recased, truncate=True)
        rows.append({"photo": photo, "src": text, "recased": recased,
                     "src_changed": src_changed, "off": off, "on": on,
                     "notes": annotate_lines(text, is_predominantly_upper, detect_language)})

    with_text = len(rows)
    diffs = [r for r in rows if r["off"] != r["on"]]

    # Balanced, shuffled A/B so slot A carries no tell even on a handful of photos.
    rnd = random.Random(SEED)
    n = len(diffs)
    assign = ["on"] * (n // 2) + ["off"] * (n - n // 2)
    rnd.shuffle(assign)
    key_rows = []
    for r, a_is in zip(diffs, assign):
        a_is_on = a_is == "on"
        r["A"] = r["on"] if a_is_on else r["off"]
        r["B"] = r["off"] if a_is_on else r["on"]
        key_rows.append({"photo": r["photo"], "A_is": "on" if a_is_on else "off",
                         "off": r["off"], "on": r["on"], "recased": r["recased"],
                         "src_changed": r["src_changed"], "notes": r["notes"]})

    def block(label, s):
        return f"**{label}**\n\n```\n{s}\n```\n"

    lines = [
        "# RESULTS — truecase on real photographs (blind A/B) — 16 September 2026", "",
        "Does the photograph-path truecaser help on real signs? No English "
        "reference exists for these signs, so this is a **blind, randomised A/B** "
        "for a pass or the owner to read — not a chrF2. Nothing here trains or "
        "changes the served build.", "",
        f"- Reader: **shipped** PP-OCRv6 at floor 0.9 (`{reader_id}`), cv2 "
        f"`{cv2_version}`, OCR read fresh this run `{generated_at}` under the cv2 "
        f"gate, product-path parity `{parity}` (`scan()` == `scan_regions()[0]` on "
        f"all 40 — both photo paths measured).",
        f"- Engine: `bs-en` `app.translate.Engine`, the camera's default direction.",
        f"- Each photo shows the OCR text and two renderings, **A** and **B**, in "
        f"random balanced order. One is today's output, one is the candidate. The "
        f"mapping is **not given to the evaluator** — only its SHA-256 is committed "
        f"(`truecase-photos-key.sha256`); the key "
        f"(`training/truecase-photos-key.json`) is revealed and checked against "
        f"that hash after the verdicts are locked.", "",
        f"**Pre-registered gate (locked before these numbers existed).** {GATE}", "",
        f"{len(diffs)} of {with_text} photographs with text render differently; "
        f"label each and note any serious regression.", "",
        "---", "",
    ]
    for r in diffs:
        lines.append(f"### {r['photo']}")
        lines.append("")
        lines.append(block("Sign (OCR, as read)", r["src"]))
        lines.append(block("Rendering A", r["A"]))
        lines.append(block("Rendering B", r["B"]))
        lines.append("**Verdict: A better / B better / same — _____**")
        lines.append("")
        lines.append("**Serious regression? (name corrupted / new repetition or "
                     "hallucination / a correct English line broken) — _____**")
        lines.append("")
        lines.append("---")
        lines.append("")
    lines += [
        "## After the verdicts are locked", "",
        "Fill every A/B and regression line above and commit them. Then reveal "
        "`training/truecase-photos-key.json`, check its SHA-256 against "
        "`truecase-photos-key.sha256`, unblind, and tally against the gate. This "
        "file is the evidence; the default-on decision is the tally.", "",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    key_json = json.dumps({
        "reader_identity": reader_id, "cv2": cv2_version, "generated_at": generated_at,
        "scan_parity": parity, "seed": SEED, "gate": GATE,
        "photos_with_text": with_text, "changed": len(diffs), "mapping": key_rows,
    }, ensure_ascii=False, indent=2)
    KEY.write_text(key_json, encoding="utf-8")
    sha = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
    KEY_SHA.write_text(f"{sha}  {KEY.name}\n", encoding="utf-8")

    print(f"{with_text} photos with text, changed the translation on {len(diffs)}")
    print(f"blind form -> {OUT.relative_to(REPO)}")
    print(f"sealed key -> {KEY.relative_to(REPO)} (git-ignored)")
    print(f"key SHA-256 -> {KEY_SHA.relative_to(REPO)}: {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
