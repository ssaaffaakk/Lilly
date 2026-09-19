#!/usr/bin/env python3
"""Zip the local datasets the paper uses, and optionally put them on Hugging Face.

Git holds the lists, answer keys and credits. It does not hold the photographs
or the cleaned parallel TSVs — they are too large, and .gitignore says so.
This packs those bytes so a clone can score without scraping Commons.

    .venv/bin/python scripts/pack_datasets.py            # zip into models/kaggle-staging/lilly-data/
    .venv/bin/python scripts/pack_datasets.py --upload   # zip, then public dataset Safak11/lilly-data

Mapillary's 20,240 frames stay on Kaggle (`afaksrmeli/lilly-mapillary-photos`).
They are not a train set. FLORES is Meta's; this does not re-host it.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "models" / "kaggle-staging" / "lilly-data"
DATASET = "Safak11/lilly-data"
SKIP_NAMES = {"dataset-metadata.json", ".DS_Store"}
IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def zip_files(dest: Path, members: list[tuple[Path, str]], stored: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(dest, "w", compression) as z:
        for src, name in members:
            if not src.is_file():
                raise SystemExit(f"missing {src}")
            z.write(src, name)
    mb = dest.stat().st_size / 1e6
    print(f"wrote {dest.relative_to(REPO)} — {len(members)} files, {mb:.0f} MB", flush=True)


def translation_members() -> list[tuple[Path, str]]:
    clean = REPO / "data" / "clean"
    names = ("train.tsv", "valid.tsv", "test.tsv", "ntrex-holdout.tsv")
    return [(clean / n, n) for n in names]


def commons40_members() -> list[tuple[Path, str]]:
    root = REPO / "models" / "kaggle-staging" / "lilly-ocr-commons-40"
    if not root.is_dir():
        raise SystemExit(f"missing {root} — the 40 Commons originals live there on this machine")
    out = []
    for p in sorted(root.iterdir()):
        if p.name in SKIP_NAMES or p.name == "by-sha1" or not p.is_file():
            continue
        out.append((p, p.name))
    if len(out) < 40:
        raise SystemExit(f"{root} has {len(out)} files; expected the 40 photographs plus the sha1 map")
    return out


def test_v2_members() -> list[tuple[Path, str]]:
    root = REPO / "data" / "ocr" / "real-photos" / "test-v2"
    photos = root / "photos"
    if not photos.is_dir():
        raise SystemExit(f"missing {photos}")
    out = []
    for p in sorted(photos.iterdir()):
        if p.suffix in IMAGE_SUFFIX:
            out.append((p, f"photos/{p.name}"))
    for name in ("CREDITS.tsv", "sample.txt", "truth-v2.json", "pass-a.json", "pass-b.json"):
        out.append((root / name, name))
    return out


def harvest_members() -> list[tuple[Path, str]]:
    root = REPO / "data" / "ocr" / "real-photos" / "harvested"
    out = [(root / "CREDITS.tsv", "CREDITS.tsv")]
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in IMAGE_SUFFIX:
            out.append((p, f"photos/{p.name}"))
    return out


def crops_members() -> list[tuple[Path, str]]:
    root = REPO / "data" / "ocr" / "crops"
    out = [(root / "labels-human.tsv", "labels-human.tsv")]
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() == ".png":
            out.append((p, f"crops/{p.name}"))
    return out


PACKS = (
    ("translation-clean.zip", translation_members, False),
    ("ocr-the-40.zip", commons40_members, True),
    ("ocr-test-v2.zip", test_v2_members, True),
    ("ocr-harvest.zip", harvest_members, True),
    ("ocr-crops.zip", crops_members, True),
)

CARD = """---
license: other
task_categories:
  - translation
  - image-to-text
language:
  - bs
  - en
pretty_name: Lilly evaluation and training bytes
---

# Lilly data (the bytes git does not hold)

Code, answer keys, credits and results stay in
[ssaaffaakk/Lilly](https://github.com/ssaaffaakk/Lilly). This dataset is the
photographs and the cleaned parallel sentences those files name.

| File | What it is |
|---|---|
| `translation-clean.zip` | `train.tsv` (313,612 pairs), `valid.tsv`, `test.tsv`, `ntrex-holdout.tsv` — WikiMatrix / SETIMES / TED2020 / Tatoeba after the project's filters. The base model already saw this material (`docs/WHITE-PAPER.md` §3.1). |
| `ocr-the-40.zip` | The 40 Wikimedia Commons photographs the product OCR number (67.0%) was measured on, plus `commons40-sha1.json`. |
| `ocr-test-v2.zip` | test-v2: 280 Commons photographs (132 with text), credits, the two blind passes, the answer key. |
| `ocr-harvest.zip` | Harvested Commons sign photographs and `CREDITS.tsv`. |
| `ocr-crops.zip` | Crops cut from the harvest, with `labels-human.tsv`. |

Photograph licences are per-file in each archive's `CREDITS.tsv` (Wikimedia Commons).
The parallel sentences come from OPUS (SETIMES, WikiMatrix, TED2020, Tatoeba) and
NTREX-128; credit those sources, not this page.

Not here: FLORES-200 (fetch from Meta), FLEURS (fetch from Google), the 20,240
Mapillary frames (`afaksrmeli/lilly-mapillary-photos` on Kaggle — not a train set).
"""


def pack() -> list[Path]:
    STAGE.mkdir(parents=True, exist_ok=True)
    written = []
    for name, members, stored in PACKS:
        dest = STAGE / name
        zip_files(dest, members(), stored)
        written.append(dest)
    (STAGE / "README.md").write_text(CARD, encoding="utf-8")
    return written


def upload(paths: list[Path]) -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        try:
            from huggingface_hub import get_token
            token = (get_token() or "").strip()
        except Exception:
            token = ""
    if not token:
        raise SystemExit("no token: run .venv/bin/hf auth login, or set HF_TOKEN")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(DATASET, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=str(STAGE / "README.md"), path_in_repo="README.md",
                    repo_id=DATASET, repo_type="dataset")
    for path in paths:
        print(f"uploading {path.name} …", flush=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                        repo_id=DATASET, repo_type="dataset")
    print(f"uploaded to https://huggingface.co/datasets/{DATASET}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    paths = pack()
    if args.upload:
        upload(paths)
    else:
        print(f"upload with: {sys.executable} {Path(__file__).relative_to(REPO)} --upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
