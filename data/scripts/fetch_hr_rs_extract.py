#!/usr/bin/env python3
"""Download Croatia and Serbia OSM extracts and append signage points to points.tsv.

    python3 data/scripts/fetch_hr_rs_extract.py

Appends to data/signs/points.tsv (same format as fetch_osm_extract.py).
"""
import sys, time, urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse PBF parsing helpers from the Bosnia script
from fetch_osm_extract import (
    _blobs, _nodes, is_signage, AGENT
)

OSM_DIR = REPO_ROOT / "data" / "osm"
POINTS_CACHE = REPO_ROOT / "data" / "signs" / "points.tsv"

COUNTRIES = {
    "hr": {
        "url": "https://download.geofabrik.de/europe/croatia-latest.osm.pbf",
        "cities": {
            "zagreb":    (45.77, 15.87, 45.87, 16.05),
            "split":     (43.49, 16.40, 43.54, 16.48),
            "dubrovnik": (42.63, 18.08, 42.66, 18.12),
            "rijeka":    (45.32, 14.42, 45.36, 14.48),
            "osijek":    (45.54, 18.67, 45.57, 18.72),
        },
    },
    "rs": {
        "url": "https://download.geofabrik.de/europe/serbia-latest.osm.pbf",
        "cities": {
            "beograd":  (44.78, 20.39, 44.85, 20.52),
            "novi_sad": (45.24, 19.82, 45.27, 19.87),
            "nis":      (43.30, 21.87, 43.36, 21.93),
        },
    },
}


def download_pbf(url: str) -> Path:
    name = url.split("/")[-1]
    pbf = OSM_DIR / name
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    if pbf.exists():
        print(f"{name}: already on disk ({pbf.stat().st_size // 1048576} MB)")
        return pbf
    print(f"downloading {url}", flush=True)
    started = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    part = pbf.with_suffix(".pbf.part")
    with urllib.request.urlopen(req, timeout=300) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done % (20 << 20) < (1 << 20):
                    share = f"/{total // 1048576}" if total else ""
                    print(f"  {done // 1048576}{share} MB", flush=True)
    part.rename(pbf)
    print(f"{done // 1048576} MB in {time.time()-started:.0f}s", flush=True)
    return pbf


def city_of(lat: float, lon: float, cities: dict) -> str:
    for name, (s, w, n, e) in cities.items():
        if s <= lat <= n and w <= lon <= e:
            return name
    return "other"


def extract_points(pbf: Path, cities: dict) -> list:
    rows = []
    scanned = blocks = 0
    for kind, payload in _blobs(pbf):
        if kind != "OSMData":
            continue
        blocks += 1
        for lat, lon, tags in _nodes(payload):
            scanned += 1
            if is_signage(tags):
                c = city_of(lat, lon, cities)
                if c != "other":
                    name = " ".join(tags["name"].split())
                    rows.append((c, lat, lon, name))
        if blocks % 50 == 0:
            print(f"  {blocks} blocks, {scanned:,} nodes, {len(rows):,} with signage",
                  flush=True)
    print(f"  total: {scanned:,} nodes in {blocks} blocks, {len(rows)} signage hits",
          flush=True)
    return rows


def main():
    existing = set()
    if POINTS_CACHE.exists():
        for line in POINTS_CACHE.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                existing.add((parts[0], parts[1][:8], parts[2][:8]))

    new_rows = []
    for code, info in COUNTRIES.items():
        print(f"\n=== {code.upper()} ===")
        pbf = download_pbf(info["url"])
        rows = extract_points(pbf, info["cities"])
        print(f"  {code}: {len(rows)} signage points in target cities")
        for city, lat, lon, name in rows:
            key = (city, f"{lat:.6f}"[:8], f"{lon:.6f}"[:8])
            if key not in existing:
                new_rows.append(f"{city}\t{lat:.7f}\t{lon:.7f}\t{name}")
                existing.add(key)

    if new_rows:
        with open(POINTS_CACHE, "a", encoding="utf-8") as f:
            f.write("\n".join(new_rows) + "\n")
        print(f"\nAppended {len(new_rows)} new points to {POINTS_CACHE}")
    else:
        print("\nNo new points to add (already up to date)")

    for code, info in COUNTRIES.items():
        for city in info["cities"]:
            count = sum(1 for l in POINTS_CACHE.read_text().splitlines()
                        if l.startswith(city + "\t"))
            print(f"  {city}: {count} points")


if __name__ == "__main__":
    main()
