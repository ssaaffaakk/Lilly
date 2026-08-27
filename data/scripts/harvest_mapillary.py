#!/usr/bin/env python3
"""Collect street-level photographs of Bosnian signage from Mapillary.

Everything Lilly's reader has been trained and scored on is synthesised, and
three real photographs were enough to find a fault four thousand synthetic ones
never showed: on a bilingual sign the Cyrillic half comes back as Latin
lookalikes rather than being skipped. Real photographs are the only thing that
finds that class of problem.

Mapillary is street-level imagery people uploaded under CC-BY-SA, with an API
meant for exactly this. That matters because this model is published: imagery
scraped from a viewer that forbids it would poison the release.

    export MAPILLARY_TOKEN='MLY|...'          # from mapillary.com/dashboard/developers
    python3 data/scripts/harvest_mapillary.py --city sarajevo --limit 400

The token is read from the environment and never written anywhere — not to a
file, not to the log, not into the credits.

Where it looks is the part worth explaining. Sweeping a city's streets returns
mostly road surface and parked cars. Instead it asks OpenStreetMap where the
shops, streets and stops actually are — every one of those has a sign on it —
and then asks Mapillary for imagery within fifty metres of each. Aiming at
signage beats filtering for it afterwards.

Those coordinates come from a file, not a service. Run

    python3 data/scripts/fetch_osm_extract.py

once and it writes every named shop, amenity and bus stop in the country to
data/signs/points.tsv from Geofabrik's nightly country extract. This script
reads that file and never touches the network for coordinates again.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "data" / "ocr" / "real-photos" / "mapillary"
GRAPH = "https://graph.mapillary.com"
# The file is the route now; Overpass is only what is left if nobody has run
# fetch_osm_extract.py yet. Mirrors, because when it is reached at all it is
# unreliable: all four were refusing connections for an afternoon, which is what
# made a live service the wrong place to keep coordinates that never move.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
# Fifteen seconds, not ninety. A healthy Overpass answers a bbox query in two or
# three; one that has not replied in fifteen is overloaded, and waiting out the
# other seventy-five only to be handed a timeout costs ten minutes across seven
# cities and four mirrors. Fail fast and let the file be the answer.
OVERPASS_TIMEOUT = 15
# Written by fetch_osm_extract.py from Geofabrik's country extract, and the only
# thing consulted when it exists. Shop and street coordinates are static; asking
# a live service for them every run was the actual fault, not the timeouts.
POINTS_CACHE = REPO_ROOT / "data" / "signs" / "points.tsv"
AGENT = "LillyBosnianOCR/0.1 (research; github.com/ssaaffaakk/Lilly)"

# Bosnian cities, as bounding boxes. Not only Sarajevo: signage differs across
# the country, and the Republika Srpska cities carry Cyrillic, which is where
# the fault the synthetic set never produced actually lives.
CITIES = {
    "sarajevo":  (43.82, 18.28, 43.92, 18.48),
    "mostar":    (43.32, 17.77, 43.38, 17.84),
    "tuzla":     (44.51, 18.63, 44.58, 18.73),
    "zenica":    (44.17, 17.87, 44.24, 17.95),
    "banjaluka": (44.74, 17.14, 44.81, 17.24),
    "bihac":     (44.79, 15.83, 44.84, 15.90),
    "travnik":   (44.21, 17.63, 44.25, 17.69),
}


def fetch(url: str, token: str = None, binary: bool = False):
    headers = {"User-Agent": AGENT}
    if token:
        headers["Authorization"] = f"OAuth {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read() if binary else json.load(response)


def cached_points(city: str) -> list:
    if not POINTS_CACHE.exists():
        return []
    rows = []
    for line in POINTS_CACHE.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 4 and parts[0] == city:
            rows.append((float(parts[1]), float(parts[2]), parts[3]))
    return rows


def remember_points(city: str, points: list) -> None:
    POINTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    fresh = not POINTS_CACHE.exists()
    with open(POINTS_CACHE, "a", encoding="utf-8") as f:
        if fresh:
            f.write("city\tlat\tlon\tname\n")
        for lat, lon, name in points:
            f.write(f"{city}\t{lat}\t{lon}\t{name}\n")


def signage_points(city: str, bbox, limit: int) -> list:
    """Where the signs are: from the file if it exists, from Overpass if it does not.

    A shop, a named street, a bus stop and a restaurant all carry text on a
    board. Asking for those coordinates and requesting imagery nearby finds far
    more signage per download than sweeping a city would.

    The file wins whenever it is present, and a city with no rows in it is an
    answer too — not a reason to go asking. That is the point: these coordinates
    are static, so one network round trip in the project's lifetime is one more
    than strictly needed, and every run after fetch_osm_extract.py is offline.
    """
    if POINTS_CACHE.exists():
        remembered = cached_points(city)
        print(f"{city}: {len(remembered):,} signage points from the cache", flush=True)
        return remembered

    print(f"{city}: no {POINTS_CACHE.name}; falling back to Overpass. Run "
          f"data/scripts/fetch_osm_extract.py to stop needing it.", flush=True)
    south, west, north, east = bbox
    query = f"""[out:json][timeout:{OVERPASS_TIMEOUT}];
      (node({south},{west},{north},{east})["name"]["shop"];
       node({south},{west},{north},{east})["name"]["amenity"];
       node({south},{west},{north},{east})["name"]["tourism"];
       node({south},{west},{north},{east})["name"]["highway"="bus_stop"];);
      out center {limit};"""
    for mirror in OVERPASS_MIRRORS:
        try:
            request = urllib.request.Request(
                mirror, data=urllib.parse.urlencode({"data": query}).encode(),
                headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=OVERPASS_TIMEOUT) as response:
                data = json.load(response)
        except Exception as exc:
            print(f"  {urllib.parse.urlparse(mirror).netloc}: {exc}", flush=True)
            continue
        points = [(e["lat"], e["lon"], e.get("tags", {}).get("name", ""))
                  for e in data.get("elements", []) if e.get("lat") and e.get("lon")]
        if points:
            remember_points(city, points)
            print(f"{city}: {len(points):,} signage points from "
                  f"{urllib.parse.urlparse(mirror).netloc}", flush=True)
            return points
    print(f"{city}: no Overpass mirror answered — sweeping the map instead. That "
          f"returns more road surface and fewer signs, so expect a lower keep rate.",
          flush=True)
    return []


def sweep_tiles(bbox, token: str, per_tile: int) -> list:
    """Fallback when nothing can say where the signs are: walk the map in tiles.

    Mapillary caps a bounding box at 0.01 degrees square, so a city has to be
    covered as a grid. Less efficient than aiming at known signage — most frames
    are road and parked cars — but it works with no second service involved.
    """
    south, west, north, east = bbox
    found, step = [], 0.009
    lat = south
    while lat < north:
        lon = west
        while lon < east:
            url = (f"{GRAPH}/images?fields=id,thumb_1024_url,is_pano,creator"
                   f"&bbox={lon},{lat},{lon + step},{lat + step}&limit={per_tile}")
            try:
                found += fetch(url, token).get("data", [])
            except Exception:
                pass
            lon += step
            time.sleep(0.2)
        lat += step
    return found


_reported = set()


def images_near(lat: float, lon: float, token: str, per_point: int) -> list:
    """Mapillary's radius search — up to fifty metres, its own maximum.

    Failures are reported once each rather than swallowed. An earlier version
    caught everything and returned an empty list, so a run that found nothing
    looked exactly like a run over an area with no coverage: the folder appeared,
    the credits file got its header, and not one image arrived with no way to
    tell whether the token, the fields or the coverage was the problem.
    """
    url = (f"{GRAPH}/images?fields=id,thumb_1024_url,is_pano"
           f"&lat={lat}&lng={lon}&radius=50&limit={per_point}")
    try:
        answer = fetch(url, token)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        key = f"http{exc.code}"
        if key not in _reported:
            _reported.add(key)
            print(f"  Mapillary returned {exc.code}: {body}", flush=True)
        return []
    except Exception as exc:
        if "other" not in _reported:
            _reported.add("other")
            print(f"  Mapillary request failed: {exc}", flush=True)
        return []
    if "error" in answer:
        if "api" not in _reported:
            _reported.add("api")
            print(f"  Mapillary said: {answer['error'].get('message')}", flush=True)
        return []
    return answer.get("data", [])


def has_readable_text(path: Path) -> tuple:
    """Whether this photograph actually shows words, and whether any are Bosnian.

    Most street imagery is road, sky and parked cars. Downloading is cheap and
    keeping is not: a set padded with photographs carrying no text teaches the
    reader nothing and makes every count on it meaningless.
    """
    from app.ocr import read_regions, BOSNIAN_LETTERS

    try:
        regions = read_regions(str(path))
    except Exception:
        return False, False
    words = [text for _, text, confidence in regions
             if len(text.strip()) >= 3 and confidence > 0.3]
    bosnian = any(c in "".join(words) for c in BOSNIAN_LETTERS)
    return len(words) >= 2, bosnian


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="sarajevo",
                        help="one of: " + ", ".join(CITIES) + ", or 'all'")
    parser.add_argument("--limit", type=int, default=300,
                        help="how many photographs to keep")
    parser.add_argument("--per-point", type=int, default=2,
                        help="images requested near each signage point")
    parser.add_argument("--measure", action="store_true",
                        help="download a few and report what the detector sees, "
                             "filtering nothing")
    parser.add_argument("--probe", action="store_true",
                        help="test the API against one known point and stop")
    args = parser.parse_args()

    token = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not token:
        print("MAPILLARY_TOKEN is not set. Get a client token from "
              "https://www.mapillary.com/dashboard/developers, then:\n"
              "    export MAPILLARY_TOKEN='MLY|...'", file=sys.stderr)
        return 1

    if args.measure:
        # Download a handful and report what the detector actually sees, without
        # filtering anything out. The first harvest kept nothing, and the reason
        # was invisible: the images arrived and every one was deleted by a
        # threshold set from intuition rather than from this distribution.
        from app.ocr import read_regions, BOSNIAN_LETTERS

        sample = OUT / "measure"
        sample.mkdir(parents=True, exist_ok=True)
        points = signage_points(args.city, CITIES[args.city], 400)[:args.limit]
        print(f"looking at {len(points)} points near signage", flush=True)
        seen, texts = [], []
        for lat, lon, near in points:
            for image in images_near(lat, lon, token, 1):
                url = image.get("thumb_1024_url")
                if not url:
                    continue
                path = sample / f"mly_{image['id']}.jpg"
                try:
                    path.write_bytes(fetch(url, binary=True))
                    regions = read_regions(str(path))
                except Exception as exc:
                    print(f"  {image['id']}: {exc}")
                    continue
                words = [(t, c) for _, t, c in regions if len(t.strip()) >= 3]
                strong = [t for t, c in words if c > 0.3]
                bosnian = any(ch in "".join(strong) for ch in BOSNIAN_LETTERS)
                seen.append((len(regions), len(words), len(strong), bosnian))
                texts.append((image["id"], " | ".join(strong)[:120]))
                print(f"  {image['id']}: {len(regions)} regions, "
                      f"{len(words)} of 3+ chars, {len(strong)} above 0.3"
                      f"{'  [Bosnian letters]' if bosnian else ''}"
                      f"   {' | '.join(strong[:3])[:60]}", flush=True)
                time.sleep(0.3)
        # Written to a file as well as printed, because whoever reads this is
        # often not the person who ran it — pasting a terminal back through a
        # terminal mangles it, and a report on disk can just be read.
        report = OUT / "measure-report.tsv"
        with open(report, "w", encoding="utf-8") as f:
            f.write("image\tregions\tthree_plus\tconfident\tbosnian\ttext\n")
            for row, (image_id, text) in zip(seen, texts):
                f.write(f"{image_id}\t{row[0]}\t{row[1]}\t{row[2]}\t"
                        f"{int(row[3])}\t{text}\n")
        print(f"\nwritten to {report}")
        if seen:
            print(f"{len(seen)} images looked at, kept in {sample}")
            for name, index in (("any region", 0), ("3+ chars", 1), ("above 0.3", 2)):
                counts = sorted(row[index] for row in seen)
                mid = counts[len(counts) // 2]
                print(f"  {name:<12} median {mid}, "
                      f"{sum(1 for c in counts if c >= 2)} of {len(counts)} have 2 or more")
            print(f"  carrying Bosnian letters: {sum(1 for r in seen if r[3])}")
        return 0

    if args.probe:
        # One request, one point, everything printed. A harvest that returns
        # nothing has three possible causes — a rejected token, no coverage, or
        # a field the API will not accept — and they are indistinguishable from
        # the outside. This makes them distinguishable in one run instead of
        # three.
        lat, lon = 43.8563, 18.4131          # Skenderija, central Sarajevo
        print(f"probing {lat},{lon} with a token of {len(token)} characters "
              f"starting {token[:4]}")
        for fields in ("id,thumb_1024_url,is_pano",
                       "id,thumb_1024_url,captured_at,is_pano,creator",
                       "id"):
            url = (f"{GRAPH}/images?fields={fields}"
                   f"&lat={lat}&lng={lon}&radius=50&limit=3")
            try:
                answer = fetch(url, token)
                rows = answer.get("data", [])
                if "error" in answer:
                    print(f"  fields={fields}\n    API error: {answer['error']}")
                else:
                    print(f"  fields={fields}\n    {len(rows)} images")
                    for row in rows[:2]:
                        print(f"      {row.get('id')} "
                              f"{(row.get('thumb_1024_url') or '')[:55]}")
            except urllib.error.HTTPError as exc:
                print(f"  fields={fields}\n    HTTP {exc.code}: "
                      f"{exc.read().decode('utf-8', 'replace')[:250]}")
            except Exception as exc:
                print(f"  fields={fields}\n    failed: {exc}")
        # And a bounding box, in case the radius search is the part that is off.
        url = (f"{GRAPH}/images?fields=id&bbox={lon - 0.004},{lat - 0.004},"
               f"{lon + 0.004},{lat + 0.004}&limit=3")
        try:
            rows = fetch(url, token).get("data", [])
            print(f"  bbox search\n    {len(rows)} images")
        except Exception as exc:
            print(f"  bbox search\n    failed: {exc}")
        return 0

    cities = list(CITIES) if args.city == "all" else [args.city]
    for name in cities:
        if name not in CITIES:
            print(f"unknown city {name!r}; known: {', '.join(CITIES)}", file=sys.stderr)
            return 1

    OUT.mkdir(parents=True, exist_ok=True)
    credits_path = OUT / "CREDITS.tsv"
    already = set()
    if credits_path.exists():
        already = {line.split("\t")[0] for line in
                   credits_path.read_text(encoding="utf-8").splitlines()[1:]}
    credits = open(credits_path, "a", encoding="utf-8")
    if not already:
        credits.write("file\timage_id\tcity\tnear\tlicence\tcreator\tbosnian\n")

    kept = skipped_no_text = 0
    with_bosnian = 0
    for city in cities:
        if kept >= args.limit:
            break
        # Either route ends in the same list of candidate images, so the
        # download-and-filter below does not need to know which one was used.
        points = signage_points(city, CITIES[city], 400)
        if points:
            candidates = ((near, image) for lat, lon, near in points
                          for image in images_near(lat, lon, token, args.per_point))
        else:
            candidates = (("", image)
                          for image in sweep_tiles(CITIES[city], token, args.per_point))

        for near, image in candidates:
            if kept >= args.limit:
                break
            image_id = str(image.get("id"))
            name = f"mly_{image_id}.jpg"
            if name in already or (OUT / name).exists():
                continue
            url = image.get("thumb_1024_url")
            if not url:
                continue
            try:
                (OUT / name).write_bytes(fetch(url, binary=True))
            except Exception:
                continue
            readable, bosnian = has_readable_text(OUT / name)
            if not readable:
                (OUT / name).unlink(missing_ok=True)
                skipped_no_text += 1
                continue
            kept += 1
            with_bosnian += bosnian
            credits.write(f"{name}\t{image_id}\t{city}\t{near}\t"
                          f"CC-BY-SA 4.0\t{image.get('creator', '')}"
                          f"\t{int(bosnian)}\n")
            credits.flush()
            if kept % 25 == 0:
                print(f"  kept {kept}, dropped {skipped_no_text} with no readable "
                      f"text, {with_bosnian} carrying Bosnian letters", flush=True)
            time.sleep(0.3)          # a shared server, used politely

    credits.close()
    if kept == 0:
        print("\nnothing was kept. In order of likelihood: the token was rejected "
              "(any Mapillary error above says so), the points had no imagery "
              "within fifty metres, or every frame was dropped for showing no "
              "readable text — the counts below tell which.", file=sys.stderr)
    print(f"\nkept {kept} photographs in {OUT}")
    print(f"dropped {skipped_no_text} that showed no readable text")
    print(f"{with_bosnian} of the kept ones carry c, c, d, s or z with diacritics")
    print(f"attribution in {credits_path.name} — CC-BY-SA requires it on redistribution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
