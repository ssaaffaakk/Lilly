#!/usr/bin/env python3
"""Fetch the 40 original Commons photos and verify their frozen identities."""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path


SKIP_SOURCE_NAMES = {"dataset-metadata.json", "commons40-sha1.json"}


def index_local_source(local_source: Path) -> dict[str, Path]:
    """Map SHA-1 → file. Kaggle's zip step can rename Unicode filenames."""
    index: dict[str, Path] = {}
    for path in sorted(local_source.rglob("*")):
        if not path.is_file() or path.name in SKIP_SOURCE_NAMES:
            continue
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        index.setdefault(digest, path)
    return index


def fetch_bytes(row: dict, local_source: Path | None,
                sha_index: dict[str, Path] | None = None) -> bytes:
    """Return the photo's bytes, from an attached dataset dir or from Commons.

    A Kaggle datacenter IP downloading 40 multi-megabyte originals from Commons
    is throttled hard (HTTP 429, "contact noc@wikimedia.org") — so the run
    attaches the same originals as a frozen dataset and reads them here instead.
    The manifest verification below is identical either way, so a mismatch still
    fail-stops and the "local" path is provably the same bytes as Commons.

    Identity on the attached dataset is commons_sha1, not the zip entry name:
    v1 of lilly-ocr-commons-40 lost Međugorje_Banner.jpg under the Unicode
    filename and ERROR'd the Read eval at 9/40.
    """
    if local_source is not None:
        source = local_source / row["file"]
        if source.is_file():
            return source.read_bytes()
        sha_copy = local_source / "by-sha1" / row["commons_sha1"]
        if sha_copy.is_file():
            return sha_copy.read_bytes()
        if sha_index is None:
            sha_index = index_local_source(local_source)
        hashed = sha_index.get(row["commons_sha1"])
        if hashed is None:
            raise SystemExit(
                f"{row['file']}: not in attached photo dataset {local_source} "
                f"(no file named that, and no file with commons_sha1 "
                f"{row['commons_sha1']})")
        return hashed.read_bytes()
    request = urllib.request.Request(row["original_url"], headers={"User-Agent": "Lilly/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 4:
                wait = 30 * (2 ** attempt)
                print(f"  HTTP 429 on {row['file']}, waiting {wait}s (attempt {attempt+1})", flush=True)
                time.sleep(wait)
            else:
                raise
    raise SystemExit(f"{row['file']}: exhausted retries downloading from Commons")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local-source", type=Path, default=None,
                    help="Copy each photo from this directory (an attached Kaggle "
                         "dataset) instead of downloading from Commons; still "
                         "verified byte-for-byte against the manifest.")
    args = ap.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if len(rows) != 40 or len({r["file"] for r in rows}) != 40:
        raise SystemExit(f"OCR manifest is not exactly 40 unique photographs: {len(rows)}")
    args.out.mkdir(parents=True, exist_ok=True)
    sha_index = index_local_source(args.local_source) if args.local_source else None
    from PIL import Image
    for index, row in enumerate(rows, 1):
        path = args.out / row["file"]
        data = fetch_bytes(row, args.local_source, sha_index)
        got_sha1 = hashlib.sha1(data).hexdigest()
        if len(data) != int(row["bytes"]) or got_sha1 != row["commons_sha1"]:
            raise SystemExit(f"{row['file']}: Commons bytes changed or download is partial")
        path.write_bytes(data)
        with Image.open(path) as image:
            if image.size != (int(row["width"]), int(row["height"])):
                raise SystemExit(f"{row['file']}: dimensions {image.size} do not match manifest")
        print(f"  {index}/40 {row['file']} {len(data)} bytes {got_sha1}", flush=True)
        if args.local_source is None and index < len(rows):
            time.sleep(1)
    if {p.name for p in args.out.iterdir() if p.is_file()} != {r["file"] for r in rows}:
        raise SystemExit("OCR download directory has missing or extra files")
    source_kind = "attached-dataset" if args.local_source else "Commons"
    print(f"40/40 original Commons photographs verified ({source_kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
