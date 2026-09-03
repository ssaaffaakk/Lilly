#!/usr/bin/env python3
"""Fetch the test-v2 photographs onto this machine, at the size the 40 were read at.

    python3 data/scripts/fetch_test_v2.py           # fetch what is missing
    python3 data/scripts/fetch_test_v2.py --check   # report only

`training/build_test_v2.py` froze the draw in `test-v2/sample.txt` from files
in git; this turns the names into pixels. Runs on the Mac (or anywhere that can
reach Commons) — a Claude Code cloud session cannot.

Every photograph is fetched as the 1280px rendering, the same size
`restore_scored_photos.py` restores the 40 at, so the two sets are read at one
resolution. Kept photographs have that URL recorded in `staged.json`; dropped
ones were unlinked by the harvester and are resolved by title through the
Commons imageinfo API, fifty at a time, which also re-reads the licence.

A photograph whose licence is not CC0, public domain, CC BY or CC BY-SA, or
whose file is not an image, is refused and logged, not substituted: the set is
"the draw, minus what could not be fetched", and `fetch-log.tsv` says which.
`CREDITS.tsv` is the attribution record and is committed; the photographs are
not.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from training.build_test_v2 import ALLOWED_LICENCES, licence_family, title_of  # noqa: E402

TEST = REPO_ROOT / "data" / "ocr" / "real-photos" / "test-v2"
PHOTOS = TEST / "photos"
API = "https://commons.wikimedia.org/w/api.php"
UA = "Lilly-OCR-eval/1.0 (https://github.com/ssaaffaakk/Lilly; test-v2 fetch)"
WIDTH = 1280


def get(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as exc:                      # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def imageinfo(titles: list) -> dict:
    """title -> imageinfo dict, for up to 50 titles."""
    params = {"action": "query", "prop": "imageinfo", "titles": "|".join(titles),
              "iiprop": "url|mime|extmetadata", "iiurlwidth": str(WIDTH), "format": "json"}
    data = json.loads(get(API + "?" + urllib.parse.urlencode(params)).decode("utf-8"))
    out = {}
    for page in data.get("query", {}).get("pages", {}).values():
        if page.get("missing") or not page.get("imageinfo"):
            continue
        out[title_of(page["title"])] = page["imageinfo"][0]
    return out


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    names = [ln.strip() for ln in (TEST / "sample.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    with (TEST / "pool.tsv").open(encoding="utf-8") as fh:
        pool = {r["file"]: r for r in csv.DictReader(fh, delimiter="\t")}
    missing = [n for n in names if not (PHOTOS / n).is_file()]
    print(f"{len(names)} drawn, {len(names) - len(missing)} on disk, {len(missing)} to fetch")
    if args.check or not missing:
        return 0

    PHOTOS.mkdir(parents=True, exist_ok=True)
    credits_path, log_path = TEST / "CREDITS.tsv", TEST / "fetch-log.tsv"
    credits = {}
    if credits_path.exists():
        with credits_path.open(encoding="utf-8") as fh:
            credits = {r["file"]: r for r in csv.DictReader(fh, delimiter="\t")}
    log = []

    # Resolve by title, fifty at a time, for everything without a stored URL.
    need_lookup = [n for n in missing if not pool[n]["screen_url"]]
    info = {}
    for i in range(0, len(need_lookup), 50):
        batch = [pool[n]["key"] for n in need_lookup[i:i + 50]]
        info.update(imageinfo(batch))
        time.sleep(0.5)

    fetched = 0
    for n in missing:
        row = pool[n]
        title = title_of(row["key"])
        meta = info.get(title, {})
        url = row["screen_url"] or meta.get("thumburl", "")
        licence = licence_family(row["licence"] or strip_html(
            meta.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")))
        mime = meta.get("mime", "image/jpeg" if row["screen_url"] else "")
        reason = ("no imageinfo" if not url else
                  f"licence:{licence or 'none'}" if licence not in ALLOWED_LICENCES else
                  f"mime:{mime}" if not mime.startswith("image/") or mime == "image/svg+xml" else "")
        if reason:
            log.append((n, "refused", reason))
            print(f"  refused {n}: {reason}")
            continue
        try:
            (PHOTOS / n).write_bytes(get(url))
        except Exception as exc:                      # noqa: BLE001
            log.append((n, "failed", str(exc)[:80]))
            print(f"  failed  {n}: {exc}")
            continue
        fetched += 1
        credits[n] = {"file": n, "key": row["key"],
                      "page_url": row["page_url"] or f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(row['key'].replace(' ', '_'))}",
                      "screen_url": url, "licence": licence,
                      "attribution": row["attribution"] or strip_html(
                          meta.get("extmetadata", {}).get("Artist", {}).get("value", ""))}
        print(f"  {fetched:3d}/{len(missing)} {n}")
        time.sleep(0.3)

    with credits_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "key", "page_url", "screen_url", "licence", "attribution"],
                           delimiter="\t")
        w.writeheader()
        for n in names:
            if n in credits:
                w.writerow(credits[n])
    with log_path.open("a", encoding="utf-8") as fh:
        for entry in log:
            fh.write("\t".join(entry) + "\n")
    print(f"\nfetched {fetched}; refused or failed {len(log)} (see {log_path.name}); "
          f"{sum((PHOTOS / n).is_file() for n in names)} of {len(names)} on disk")
    print("commit CREDITS.tsv and fetch-log.tsv; the photographs stay out of git")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
