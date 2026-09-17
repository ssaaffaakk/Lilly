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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if len(rows) != 40 or len({r["file"] for r in rows}) != 40:
        raise SystemExit(f"OCR manifest is not exactly 40 unique photographs: {len(rows)}")
    args.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for index, row in enumerate(rows, 1):
        path = args.out / row["file"]
        request = urllib.request.Request(row["original_url"], headers={"User-Agent": "Lilly/1.0"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 4:
                    wait = 30 * (2 ** attempt)
                    print(f"  HTTP 429 on {row['file']}, waiting {wait}s (attempt {attempt+1})", flush=True)
                    time.sleep(wait)
                else:
                    raise
        if index < len(rows):
            time.sleep(1)
        got_sha1 = hashlib.sha1(data).hexdigest()
        if len(data) != int(row["bytes"]) or got_sha1 != row["commons_sha1"]:
            raise SystemExit(f"{row['file']}: Commons bytes changed or download is partial")
        path.write_bytes(data)
        with Image.open(path) as image:
            if image.size != (int(row["width"]), int(row["height"])):
                raise SystemExit(f"{row['file']}: dimensions {image.size} do not match manifest")
        print(f"  {index}/40 {row['file']} {len(data)} bytes {got_sha1}", flush=True)
    if {p.name for p in args.out.iterdir() if p.is_file()} != {r["file"] for r in rows}:
        raise SystemExit("OCR download directory has missing or extra files")
    print("40/40 original Commons photographs verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
