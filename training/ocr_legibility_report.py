#!/usr/bin/env python3
"""Report shipped-path OCR recall/invention by frozen photo legibility cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from training.evaluate_ocr import load_truth, recall, words


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    truth = load_truth(args.truth)
    with args.manifest.open(encoding="utf-8", newline="") as fh:
        manifest = {r["file"]: r for r in csv.DictReader(fh, delimiter="\t")}
    cached = json.loads(args.cache.read_text(encoding="utf-8"))
    readings = cached.get("readings", {})
    if set(truth) != set(manifest) or set(truth) != set(readings) or len(truth) != 40:
        raise SystemExit(
            f"OCR coverage refused: truth={len(truth)} manifest={len(manifest)} readings={len(readings)}")
    cohorts = ("clean", "noisy_but_understandable", "human_unintelligible", "no_text")
    totals = {c: {"photographs": 0, "found": 0, "words": 0, "invented": 0}
              for c in cohorts}
    per_photo = []
    for name in sorted(truth):
        cohort = manifest[name]["cohort"]
        wanted = words(" ".join(truth[name]["lines"]))
        found_words = words(readings[name]["text"])
        hit, need = recall(wanted, found_words)
        invented = max(0, len(found_words) - hit)
        item = totals[cohort]
        item["photographs"] += 1; item["found"] += hit; item["words"] += need
        item["invented"] += invented
        per_photo.append({"file": name, "cohort": cohort, "found": hit, "words": need,
                          "invented": invented, "prediction": readings[name]["text"]})
    for item in totals.values():
        item["recall"] = 100 * item["found"] / item["words"] if item["words"] else None

    def aggregate(names):
        parts = [totals[name] for name in names]
        out = {key: sum(p[key] for p in parts)
               for key in ("photographs", "found", "words", "invented")}
        out["recall"] = 100 * out["found"] / out["words"] if out["words"] else None
        return out

    result = {
        "schema": 1,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "reader": cached.get("reader"), "by_cohort": totals,
        "legible_overall": aggregate(cohorts[:2]),
        "all_text_photos": aggregate(cohorts[:3]), "all_40": aggregate(cohorts),
        "per_photograph": per_photo,
    }
    lines = ["# Clean OCR evaluation — frozen human-legibility cohorts", "",
             f"Manifest SHA-256: `{result['manifest_sha256']}`. Reader context: `{result['reader']}`.",
             "", "| cohort | photos | found / words | recall | invented words |",
             "|---|---:|---:|---:|---:|"]
    for cohort in cohorts:
        item = totals[cohort]
        rate = "—" if item["recall"] is None else f"{item['recall']:.1f}%"
        lines.append(f"| {cohort} | {item['photographs']} | {item['found']} / "
                     f"{item['words']} | {rate} | {item['invented']} |")
    for label, key in (("**legible overall**", "legible_overall"),
                       ("all text photos", "all_text_photos")):
        item = result[key]
        lines.append(f"| {label} | {item['photographs']} | {item['found']} / "
                     f"{item['words']} | **{item['recall']:.1f}%** | {item['invented']} |")
    lines.append("")
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
