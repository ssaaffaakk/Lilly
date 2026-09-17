#!/usr/bin/env python3
"""Report product-path WER by the frozen speech-legibility cohorts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate_speech import edits, normalise
from training.speech_bench import clip_key, fingerprint
from training.train_speech import read_tsv


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", type=Path, action="append", required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as fh:
        manifest = list(csv.DictReader(fh, delimiter="\t"))
    index = {(r["audio_sha256"], r["transcript_sha256"]): r for r in manifest}
    rows = read_tsv(args.data)
    if len(rows) != 925 or len(manifest) != 925 or len(index) != 925:
        raise SystemExit(
            f"speech coverage refused: rows={len(rows)} manifest={len(manifest)} unique={len(index)}")
    cases = []
    for clip, reference in rows:
        path = Path(clip)
        ident = (digest(path.read_bytes()), digest(reference.encode("utf-8")))
        if ident not in index:
            raise SystemExit(f"downloaded FLEURS row is not in frozen manifest: {path}")
        cases.append({"clip": path, "reference": reference,
                      "cohort": index[ident]["cohort"], "identity": ident[0]})
    if len({c["identity"] for c in cases}) != 925:
        raise SystemExit("downloaded FLEURS audio has duplicate identities")

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    results = {"schema": 1, "n_clips": 925,
               "manifest_sha256": digest(args.manifest.read_bytes()), "listeners": {}}
    cohorts = ("clean", "noisy_but_understandable", "human_unintelligible")
    for model in args.model:
        fp = fingerprint(model)[:16]
        have = cache.get(f"{fp}:bs", {})
        stats = {c: [0, 0, 0] for c in cohorts}
        per_clip = []
        for case in cases:
            key = clip_key(case)
            if key not in have:
                raise SystemExit(f"cache misses {key} for listener {fp}")
            hypothesis = have[key]
            wrong = edits(normalise(case["reference"]), normalise(hypothesis))
            n_words = len(normalise(case["reference"]))
            bucket = stats[case["cohort"]]
            bucket[0] += wrong; bucket[1] += n_words; bucket[2] += 1
            per_clip.append({"audio_sha256": case["identity"], "cohort": case["cohort"],
                             "reference": case["reference"], "hypothesis": hypothesis,
                             "wrong": wrong, "words": n_words})

        def rate(parts):
            wrong, n_words, n_clips = parts
            return {"clips": n_clips, "wrong": wrong, "words": n_words,
                    "wer": (100 * wrong / n_words if n_words else None)}

        by_cohort = {cohort: rate(stats[cohort]) for cohort in cohorts}
        all_parts = [sum(x[i] for x in stats.values()) for i in range(3)]
        legible_parts = [sum(stats[c][i] for c in cohorts[:2]) for i in range(3)]
        results["listeners"][model.name] = {
            "fingerprint": fp, "by_cohort": by_cohort,
            "legible_overall": rate(legible_parts), "all": rate(all_parts),
            "predictions": per_clip,
        }

    lines = ["# Clean speech evaluation — WER by frozen legibility cohort", "",
             f"Manifest SHA-256: `{results['manifest_sha256']}`. Product decode path; 925/925 clips.",
             "", "| listener | cohort | clips | errors / words | WER |",
             "|---|---|---:|---:|---:|"]
    for label, listener in results["listeners"].items():
        for cohort, item in listener["by_cohort"].items():
            wer = "—" if item["wer"] is None else f"{item['wer']:.2f}%"
            lines.append(f"| {label} `{listener['fingerprint']}` | {cohort} | "
                         f"{item['clips']} | {item['wrong']} / {item['words']} | {wer} |")
        item = listener["all"]
        lines.append(f"| {label} | **all** | {item['clips']} | {item['wrong']} / "
                     f"{item['words']} | **{item['wer']:.2f}%** |")
    lines += ["", "The empty noisy and unintelligible rows remain explicit; this clean "
              "FLEURS split is not padded with synthetic corruption.", ""]
    args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
