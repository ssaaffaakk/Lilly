#!/usr/bin/env python3
"""Turn the blind-labelled crops into PaddleOCR recognition lists, split by source photograph.

    python3 training/paddle_rec_data.py --root /kaggle/temp/train_data
        # expects <root>/crops/*.png (+ labels-human.tsv) and <root>/crops2/*.png (+ labels-human.tsv)
        # writes <root>/train_list.txt, <root>/val_list.txt, <root>/split-report.json
    python3 training/paddle_rec_data.py --root data/ocr --allow-missing      # dry run without the PNGs

Pre-registered in training/PREREGISTRATION.md ("step 7, PP-OCRv6 recogniser
fine-tune"): Cyrillic labels are dropped (the dictionary has none), labels
longer than MAX_TEXT_LENGTH are dropped (the head is built for 25), __NOTEXT__
and __UNSURE__ are dropped, and a crop is validation when
blake2b(source photograph, 4 bytes) % 10 == 0 — by photograph, never by crop,
so no board contributes lines to both sides. The report says which sources
fell where, so nobody can quietly re-roll the split.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

MAX_TEXT_LENGTH = 25
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
SETS = ("crops", "crops2")


def source_of(png: str) -> str:
    return re.sub(r"_\d{3}\.png$", "", png)


def bucket(source: str) -> int:
    return int(hashlib.blake2b(source.encode("utf-8"), digest_size=4).hexdigest(), 16) % 10


def load(root: Path, allow_missing: bool):
    rows, dropped = [], Counter()
    for sub in SETS:
        labels = root / sub / "labels-human.tsv"
        if not labels.is_file():
            raise SystemExit(f"{labels} is missing")
        for line in labels.read_text(encoding="utf-8").splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].endswith(".png"):
                continue
            png, text = parts[0], parts[1].strip()
            if not text or text.startswith("__"):
                dropped["no text or special"] += 1
                continue
            if CYRILLIC.search(text):
                dropped["cyrillic"] += 1
                continue
            if len(text) > MAX_TEXT_LENGTH:
                dropped[f"longer than {MAX_TEXT_LENGTH}"] += 1
                continue
            path = root / sub / png
            if not path.is_file():
                if not allow_missing:
                    raise SystemExit(f"crop missing on disk: {path} — the dataset did not land")
                dropped["png missing (allowed)"] += 1
            rows.append((f"{sub}/{png}", text, source_of(png)))
    return rows, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="directory holding crops/ and crops2/")
    ap.add_argument("--allow-missing", action="store_true", help="dry run: list rows even if PNGs are absent")
    ap.add_argument("--min-train", type=int, default=1000, help="refuse to write fewer training lines than this")
    a = ap.parse_args()
    rows, dropped = load(a.root, a.allow_missing)
    train = [(p, t) for p, t, s in rows if bucket(s) != 0]
    valid = [(p, t) for p, t, s in rows if bucket(s) == 0]
    src_train = Counter(s for p, t, s in rows if bucket(s) != 0)
    src_valid = Counter(s for p, t, s in rows if bucket(s) == 0)
    if src_train.keys() & src_valid.keys():
        raise SystemExit("a source photograph is on both sides — the split rule is broken")
    report = {"kept": len(rows), "dropped": dict(dropped),
              "train": {"crops": len(train), "photographs": len(src_train), "top": src_train.most_common(5)},
              "valid": {"crops": len(valid), "photographs": len(src_valid), "top": src_valid.most_common(5)},
              "rule": "valid iff blake2b(source, 4 bytes) % 10 == 0", "max_text_length": MAX_TEXT_LENGTH}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if len(train) < a.min_train:
        raise SystemExit(f"only {len(train)} training lines; the pre-registration expects about 1,132")
    if not valid:
        raise SystemExit("no validation crops")
    (a.root / "train_list.txt").write_text("".join(f"{p}\t{t}\n" for p, t in train), encoding="utf-8")
    (a.root / "val_list.txt").write_text("".join(f"{p}\t{t}\n" for p, t in valid), encoding="utf-8")
    (a.root / "split-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.root / 'train_list.txt'} ({len(train)}), {a.root / 'val_list.txt'} ({len(valid)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
