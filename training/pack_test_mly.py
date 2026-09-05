#!/usr/bin/env python3
"""Pack the drawn test-mly photographs so the cloud can transcribe them.

    .venv/bin/python3 training/pack_test_mly.py            # zip only
    HF_TOKEN=... .venv/bin/python3 training/pack_test_mly.py --upload   # zip, then a public HF dataset

The 240 photographs `training/build_test_mly.py` drew exist only on the Mac
(`data/ocr/real-photos/mapillary/`, 20,240 files, CC BY-SA 4.0 from
Mapillary). The two blind transcription passes run on the cloud with
`training/transcribe/`, which needs the pixels. This zips exactly the files
in `test-mly/sample.txt` together with `CREDITS.tsv` (attribution, the
licence's condition) and `sample.txt`, into `models/kaggle-staging/`
(gitignored), and with --upload puts the zip and the credits file in the
public dataset repo `Safak11/lilly-test-mly-photos`. Nothing here trains
anything; the photographs are a test set and stay one.
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST = REPO_ROOT / "data" / "ocr" / "real-photos" / "test-mly"
PHOTOS = REPO_ROOT / "data" / "ocr" / "real-photos" / "mapillary"
OUT = REPO_ROOT / "models" / "kaggle-staging" / "test-mly-photos.zip"
DATASET = "Safak11/lilly-test-mly-photos"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upload", action="store_true", help="upload the zip and CREDITS.tsv to the public dataset repo")
    ap.add_argument("--dataset", default=DATASET)
    a = ap.parse_args()
    names = [l.strip() for l in (TEST / "sample.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [n for n in names if not (PHOTOS / n).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(names)} drawn photographs are not in {PHOTOS}: {missing[:3]} — "
                         "this runs on the Mac, where they are")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:   # JPEGs do not compress; STORED keeps it fast
        for n in names:
            z.write(PHOTOS / n, f"photos/{n}")
        z.write(TEST / "CREDITS.tsv", "CREDITS.tsv")
        z.write(TEST / "sample.txt", "sample.txt")
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} — {len(names)} photographs, {mb:.0f} MB")
    if not a.upload:
        print(f"upload with: HF_TOKEN=... {sys.executable} {Path(__file__).relative_to(REPO_ROOT)} --upload")
        return 0
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not in the environment; export it in this shell (never in a file) and rerun")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.dataset, repo_type="dataset", private=False, exist_ok=True)
    readme = ("---\nlicense: cc-by-sa-4.0\n---\n# Lilly test-mly photographs\n\n"
              f"{len(names)} street-level photographs drawn by `training/build_test_mly.py` from Mapillary "
              "(CC BY-SA 4.0). Attribution per photograph: `CREDITS.tsv` (image_id, city, creator). "
              "A held-out test set for the Lilly reader (github.com/ssaaffaakk/Lilly); nothing trains on it.\n")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=a.dataset, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(TEST / "CREDITS.tsv"), path_in_repo="CREDITS.tsv", repo_id=a.dataset, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(OUT), path_in_repo="test-mly-photos.zip", repo_id=a.dataset, repo_type="dataset")
    print(f"uploaded to https://huggingface.co/datasets/{a.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
