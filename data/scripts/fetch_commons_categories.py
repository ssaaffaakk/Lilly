#!/usr/bin/env python3
"""Pull the sign-bearing Commons categories the keyword crawl missed.

Six category trees, verified by API count during the 28 Aug source hunt, all
carrying exactly our distribution: road signs, street signs, city-limit signs,
monument signs, Cyrillic inscriptions, Wikivoyage banners. A few hundred images
-- not the thousands Flickr will bring, but every one is a sign.

    python3 data/scripts/fetch_commons_categories.py

Writes to data/ocr/real-photos/commons-cats/ with a CREDITS.tsv, same etiquette
as the harvester: serial, rate-limited, User-Agent set. Skips files already on
disk, so re-running is cheap. Descends subcategories to DEPTH levels.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "ocr" / "real-photos" / "commons-cats"
API = "https://commons.wikimedia.org/w/api.php"
UA = "Lilly/fetch_commons_categories (surmeliisafak@gmail.com)"
GAP = 1.2          # seconds between calls — Wikimedia etiquette
DEPTH = 3
SEEDS = [
    "Category:Road signs in Bosnia and Herzegovina",
    "Category:Street signs in Bosnia and Herzegovina",
    "Category:City limit signs in Bosnia and Herzegovina",
    "Category:Signs for National Monuments of Bosnia and Herzegovina",
    "Category:Cyrillic inscriptions in Bosnia and Herzegovina",
    "Category:Wikivoyage banners of Bosnia and Herzegovina",
]
MIN_WIDTH = 800    # same bar the harvester uses


def call(params: dict) -> dict:
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    time.sleep(GAP)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def members(cat: str, depth: int, seen: set) -> list:
    """File titles under cat, descending subcats to `depth`."""
    if cat in seen or depth < 0:
        return []
    seen.add(cat)
    files, cont = [], {}
    while True:
        d = call({"action": "query", "list": "categorymembers", "cmtitle": cat,
                  "cmtype": "file|subcat", "cmlimit": "500", **cont})
        for m in d["query"]["categorymembers"]:
            if m["title"].startswith("Category:"):
                files += members(m["title"], depth - 1, seen)
            else:
                files.append(m["title"])
        cont = d.get("continue") or {}
        if not cont:
            return files


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    credits = OUT / "CREDITS.tsv"
    have = set()
    if credits.exists():
        have = {l.split("\t")[0] for l in credits.read_text(encoding="utf-8").splitlines()}
    else:
        credits.write_text("file\tsource_title\tlicense\tartist\turl\n", encoding="utf-8")

    seen_cats: set = set()
    titles = []
    for seed in SEEDS:
        got = members(seed, DEPTH, seen_cats)
        print(f"{seed.split(':',1)[1]}: {len(got)} files")
        titles += got
    titles = sorted(set(titles))
    print(f"\n{len(titles)} unique files across the trees")

    kept = skipped = 0
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = call({"action": "query", "prop": "imageinfo", "titles": "|".join(chunk),
                  "iiprop": "url|size|extmetadata", "iiurlwidth": "1280"})
        for page in d["query"]["pages"].values():
            info = (page.get("imageinfo") or [{}])[0]
            if not info or info.get("width", 0) < MIN_WIDTH:
                skipped += 1
                continue
            meta = info.get("extmetadata", {})
            lic = (meta.get("LicenseShortName") or {}).get("value", "")
            if "NC" in lic or "ND" in lic:
                skipped += 1          # the project's license bar
                continue
            name = page["title"].replace("File:", "").replace(" ", "_")
            name = "".join(c for c in name if c not in '/\\:')
            target = OUT / name
            if name in have or target.exists():
                continue
            url = info.get("thumburl") or info["url"]
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            time.sleep(GAP)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    target.write_bytes(r.read())
            except Exception as exc:
                print(f"  failed {name[:50]}: {exc}", file=sys.stderr)
                continue
            artist = (meta.get("Artist") or {}).get("value", "")[:120]
            with open(credits, "a", encoding="utf-8") as f:
                f.write(f"{name}\t{page['title']}\t{lic}\t{artist}\t{info['url']}\n")
            kept += 1
            if kept % 25 == 0:
                print(f"  {kept} downloaded...", flush=True)
    print(f"\ndone: {kept} new files, {skipped} skipped (small or NC/ND) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
