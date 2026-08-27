#!/usr/bin/env python3
"""Where Bosnia's signs are, from a file on disk instead of a live service.

harvest_mapillary.py needs shop, street and bus-stop coordinates to aim the
photo search at. It was asking Overpass for them, which is the wrong shape of
dependency: those coordinates have not moved in years, and Overpass is a free
shared service that goes down. It was refusing on all four mirrors for an entire
afternoon, and a harvest that cannot run because someone else's server is having
a bad day is a harvest that does not run.

Geofabrik publishes the whole country as one file, rebuilt nightly. Downloaded
once, every later query is local and the network is out of the loop for good.

    python3 data/scripts/fetch_osm_extract.py

Writes data/signs/points.tsv — city, lat, lon, name — the format
harvest_mapillary.cached_points() already reads, so that script needs no new
parsing code, only permission to trust the file.

WHY THE FILE IS READ BY HAND
This reads the .osm.pbf with nothing but zlib and struct. Two other routes were
considered and are closed:

  * pyosmium/esy-osmfile would decode it in three lines, but neither is in
    .venv and adding a compiled dependency to pull a list of coordinates that
    changes once a year is a bad trade.
  * Geofabrik used to publish the same extract as .osm.bz2 XML, which the
    standard library can stream with bz2 + ElementTree. They have retired it:
    every date returns 404 and the europe index carries no bz2 links at all.

So the file is decoded directly. PBF is protobuf, and the slice of protobuf
needed here is small — varints, length-delimited submessages, zlib blobs — and
it is a published, frozen format. Only nodes are decoded; ways and relations are
skipped without being read, because a sign hangs at a point.
"""
import argparse
import hashlib
import struct
import sys
import time
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harvest_mapillary import CITIES, POINTS_CACHE      # noqa: E402  the bboxes, single source

OSM_DIR = REPO_ROOT / "data" / "osm"
URL = "https://download.geofabrik.de/europe/bosnia-herzegovina-latest.osm.pbf"
PBF = OSM_DIR / "bosnia-herzegovina-latest.osm.pbf"
AGENT = "LillyBosnianOCR/0.1 (research; github.com/ssaaffaakk/Lilly)"

# A node carries a sign if it is a shop, something with an amenity board, a
# tourism listing, or a bus stop — and if it has a name, because an unnamed
# bench is a point with no text on it. Same four tags harvest_mapillary.py asked
# Overpass for, kept identical so the two routes return the same population.
SIGN_KEYS = ("shop", "amenity", "tourism")
SIGN_PAIRS = (("highway", "bus_stop"),)

# Bosnia and Herzegovina's own bounding box, used to catch a coordinate decode
# that has gone wrong. Nanodegree arithmetic fails loudly if it fails at all: a
# wrong granularity or a missed delta puts points in the Atlantic, not two
# streets over.
COUNTRY_BBOX = (42.5, 15.7, 45.3, 19.7)

# PBF caps an uncompressed blob at 32 MiB by spec. Anything larger is a corrupt
# length field, and acting on it would mean handing zlib an arbitrary number.
MAX_BLOB = 32 << 20


# ---------------------------------------------------------------- protobuf

def _varint(buf: bytes, i: int):
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _signed(value: int) -> int:
    """int64 on the wire is two's complement in a 10-byte varint, not zigzag."""
    return value - (1 << 64) if value >= (1 << 63) else value


def _zigzag(value: int) -> int:
    """sint64 — what dense node ids and deltas use."""
    return (value >> 1) ^ -(value & 1)


def _fields(buf: bytes):
    """Walk one protobuf message: (field number, value) for the wire types PBF uses.

    Varints come back as ints, length-delimited fields as bytes. PBF never uses
    the fixed-width or group wire types, so an unknown one means the offsets
    have drifted and every byte after it is nonsense — worth stopping for.
    """
    i, end = 0, len(buf)
    while i < end:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
            yield field, value
        elif wire == 2:
            length, i = _varint(buf, i)
            yield field, buf[i:i + length]
            i += length
        elif wire == 5:
            i += 4
        elif wire == 1:
            i += 8
        else:
            raise ValueError(f"wire type {wire} at byte {i}: stream is out of step")


def _packed(buf: bytes) -> list:
    out, i, end = [], 0, len(buf)
    while i < end:
        value, i = _varint(buf, i)
        out.append(value)
    return out


# ---------------------------------------------------------------- pbf file

def _blobs(path: Path):
    """Yield (type, payload) per blob. The payload is decompressed, nothing more.

    A .osm.pbf is a flat run of blobs, each prefixed by a four-byte big-endian
    header length. Streaming it this way means peak memory is one blob, not the
    160 MB file.
    """
    with open(path, "rb") as handle:
        while True:
            prefix = handle.read(4)
            if len(prefix) < 4:
                return
            header = handle.read(struct.unpack(">I", prefix)[0])
            kind, datasize = None, 0
            for field, value in _fields(header):
                if field == 1:
                    kind = value.decode("utf-8")
                elif field == 3:
                    datasize = value
            if not 0 < datasize <= MAX_BLOB:
                raise ValueError(f"blob claims {datasize} bytes; file is damaged")
            yield kind, _payload(handle.read(datasize))


def _payload(blob: bytes) -> bytes:
    raw = None
    for field, value in _fields(blob):
        if field == 1:
            raw = value                      # stored uncompressed
        elif field == 3:
            return zlib.decompress(value)    # what Geofabrik writes
        elif field in (4, 6, 7):
            raise ValueError("blob uses lzma/lz4/zstd; only zlib is handled here")
    if raw is None:
        raise ValueError("blob carries no data")
    return raw


def _header(payload: bytes) -> dict:
    """The OSMHeader blob: what the file requires of a reader, and how old it is."""
    info = {"required": [], "timestamp": None, "bbox": None}
    box = {}
    for field, value in _fields(payload):
        if field == 4:
            info["required"].append(value.decode("utf-8"))
        elif field == 32:
            info["timestamp"] = _signed(value)
        elif field == 1:
            for sub, raw in _fields(value):
                if sub in (1, 2, 3, 4):
                    # sint64, so zigzag — unlike the replication timestamp above,
                    # which is a plain int64. Reading these as int64 silently
                    # doubles every one of them.
                    box[sub] = _zigzag(raw) / 1e9
    if len(box) == 4:
        info["bbox"] = (box[4], box[1], box[3], box[2])   # south, west, north, east
    return info


def _nodes(payload: bytes):
    """Yield (lat, lon, tags) for every node in one PrimitiveBlock.

    Coordinates are stored as integers so they survive being delta-encoded;
    granularity and the offsets turn them back into degrees. Both node encodings
    are handled: Geofabrik writes DenseNodes, but a plain Node group in a file
    from another producer would otherwise be dropped in silence.
    """
    strings, groups = [], []
    granularity, lat_off, lon_off = 100, 0, 0
    for field, value in _fields(payload):
        if field == 1:
            strings = [s for sub, s in _fields(value) if sub == 1]
        elif field == 2:
            groups.append(value)
        elif field == 17:
            granularity = value
        elif field == 19:
            lat_off = _signed(value)
        elif field == 20:
            lon_off = _signed(value)

    def degrees(lat: int, lon: int):
        return ((lat_off + granularity * lat) / 1e9,
                (lon_off + granularity * lon) / 1e9)

    for group in groups:
        for field, value in _fields(group):
            if field == 2:
                yield from _dense(value, strings, degrees)
            elif field == 1:
                yield _plain(value, strings, degrees)
            # fields 3, 4 and 5 are ways, relations and changesets: a sign hangs
            # at a point, so they are stepped over without being decoded.


def _dense(buf: bytes, strings: list, degrees):
    ids = lats = lons = keys_vals = ()
    for field, value in _fields(buf):
        if field == 8:
            lats = _packed(value)
        elif field == 9:
            lons = _packed(value)
        elif field == 10:
            keys_vals = _packed(value)
        elif field == 1:
            ids = _packed(value)

    # Every column is delta-encoded against the row above, so they only mean
    # anything read in order and from the start of the block.
    lat = lon = 0
    cursor = 0
    for index in range(len(ids)):
        lat += _zigzag(lats[index])
        lon += _zigzag(lons[index])
        tags = {}
        # keys_vals is one flat run for the whole block: key, value, key,
        # value, then a zero that ends this node and starts the next.
        while cursor < len(keys_vals) and keys_vals[cursor]:
            key = strings[keys_vals[cursor]].decode("utf-8", "replace")
            val = strings[keys_vals[cursor + 1]].decode("utf-8", "replace")
            tags[key] = val
            cursor += 2
        cursor += 1
        yield (*degrees(lat, lon), tags)


def _plain(buf: bytes, strings: list, degrees):
    keys, vals, lat, lon = [], [], 0, 0
    for field, value in _fields(buf):
        if field == 2:
            keys = _packed(value)
        elif field == 3:
            vals = _packed(value)
        elif field == 8:
            lat = _zigzag(value)
        elif field == 9:
            lon = _zigzag(value)
    tags = {strings[k].decode("utf-8", "replace"): strings[v].decode("utf-8", "replace")
            for k, v in zip(keys, vals)}
    return (*degrees(lat, lon), tags)


# ---------------------------------------------------------------- download

def download(force: bool = False) -> bool:
    """Fetch the extract once. Returns True if the network was used.

    The checksum is saved beside the file so a later run can confirm the
    download finished without asking Geofabrik anything — the whole point is
    that the second run needs no network at all.
    """
    digest_file = PBF.with_suffix(PBF.suffix + ".md5")
    if PBF.exists() and not force:
        size = PBF.stat().st_size
        if digest_file.exists():
            want = digest_file.read_text(encoding="utf-8").split()[0]
            got = _md5(PBF)
            if got != want:
                print(f"{PBF.name} is {size / 1048576:.0f} MB but its checksum does "
                      f"not match; re-run with --refresh", file=sys.stderr)
                return False
            print(f"{PBF.name}: {size / 1048576:.0f} MB, checksum verified, "
                  f"no download needed", flush=True)
        else:
            print(f"{PBF.name}: {size / 1048576:.0f} MB already here", flush=True)
        return False

    OSM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}", flush=True)
    started = time.time()
    request = urllib.request.Request(URL, headers={"User-Agent": AGENT})
    part = PBF.with_suffix(PBF.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        modified = response.headers.get("Last-Modified", "unknown")
        done = 0
        with open(part, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                # One line per 10 MB. Enough to show a stall on a 160 MB file
                # without turning the log into a progress bar nobody can read.
                if done % (10 << 20) < (1 << 20):
                    share = f" of {total / 1048576:.0f}" if total else ""
                    print(f"  {done / 1048576:.0f}{share} MB "
                          f"({done / 1048576 / max(time.time() - started, 0.1):.1f} MB/s)",
                          flush=True)
    if total and done != total:
        part.unlink(missing_ok=True)
        print(f"got {done} bytes, expected {total}: download was cut short",
              file=sys.stderr)
        return False

    # Named only once it is complete, so an interrupted run cannot leave
    # something that looks like a finished extract.
    part.rename(PBF)
    print(f"{done / 1048576:.0f} MB in {time.time() - started:.0f}s, "
          f"published {modified}", flush=True)

    try:
        with urllib.request.urlopen(
                urllib.request.Request(URL + ".md5", headers={"User-Agent": AGENT}),
                timeout=30) as response:
            published = response.read().decode("utf-8")
        got = _md5(PBF)
        if got != published.split()[0]:
            PBF.unlink(missing_ok=True)
            print(f"checksum mismatch: got {got}, Geofabrik says "
                  f"{published.split()[0]}", file=sys.stderr)
            return False
        digest_file.write_text(published, encoding="utf-8")
        print(f"checksum verified: {got}", flush=True)
    except OSError as exc:
        # Not fatal. The file downloaded to its full advertised length; the
        # checksum is a second opinion, and the parse below will not survive a
        # corrupt file anyway.
        print(f"could not fetch the published checksum ({exc}); "
              f"relying on the byte count", flush=True)
    return True


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- extract

def city_of(lat: float, lon: float) -> str:
    for name, (south, west, north, east) in CITIES.items():
        if south <= lat <= north and west <= lon <= east:
            return name
    # Kept, not dropped. A sign in Trebinje reads the same as a sign in Mostar,
    # and the seven bboxes cover a few per cent of the country.
    return "other"


def is_signage(tags: dict) -> bool:
    if not tags.get("name", "").strip():
        return False
    return (any(k in tags for k in SIGN_KEYS)
            or any(tags.get(k) == v for k, v in SIGN_PAIRS))


def extract() -> dict:
    """Read the extract once and collect every named signage node."""
    started = time.time()
    found, scanned, blocks = [], 0, 0
    for kind, payload in _blobs(PBF):
        if kind == "OSMHeader":
            info = _header(payload)
            unknown = set(info["required"]) - {"OsmSchema-V0.6", "DenseNodes"}
            if unknown:
                raise ValueError(f"file requires {sorted(unknown)}, which this "
                                 f"reader does not implement")
            if info["timestamp"]:
                age = datetime.fromtimestamp(info["timestamp"], timezone.utc)
                print(f"extract is OSM as of {age:%Y-%m-%d %H:%M} UTC", flush=True)
            if info["bbox"]:
                south, west, north, east = info["bbox"]
                # Checked, not just printed. The header bbox is the one chance to
                # find out the wrong country was downloaded, or that the decode
                # is off, before spending a minute reading 23 million nodes.
                if (north < COUNTRY_BBOX[0] or south > COUNTRY_BBOX[2]
                        or east < COUNTRY_BBOX[1] or west > COUNTRY_BBOX[3]):
                    raise ValueError(f"extract covers {info['bbox']}, which does not "
                                     f"overlap Bosnia {COUNTRY_BBOX}")
                print(f"covers {south:.2f},{west:.2f} to {north:.2f},{east:.2f}",
                      flush=True)
            continue
        if kind != "OSMData":
            continue
        blocks += 1
        for lat, lon, tags in _nodes(payload):
            scanned += 1
            if is_signage(tags):
                # Tabs and newlines would split a row into the wrong number of
                # columns and cached_points() drops any row that is not four.
                name = " ".join(tags["name"].split())
                found.append((city_of(lat, lon), lat, lon, name))
        if blocks % 50 == 0:
            print(f"  {blocks} blocks, {scanned:,} nodes, {len(found):,} with signage",
                  flush=True)

    south, west, north, east = COUNTRY_BBOX
    stray = [p for p in found if not (south <= p[1] <= north and west <= p[2] <= east)]
    if stray:
        raise ValueError(f"{len(stray)} points landed outside Bosnia, e.g. "
                         f"{stray[0]} — the coordinate decode is wrong")

    print(f"read {scanned:,} nodes in {blocks} blocks in {time.time() - started:.0f}s",
          flush=True)
    return {"points": found, "scanned": scanned}


def write_points(points: list) -> None:
    POINTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    order = {name: i for i, name in enumerate(CITIES)}
    # Grouped by city, in the order the bboxes are declared, so the file can be
    # read by eye. cached_points() scans the whole file regardless.
    points = sorted(points, key=lambda p: (order.get(p[0], len(order)), p[3]))
    with open(POINTS_CACHE, "w", encoding="utf-8") as out:
        out.write("city\tlat\tlon\tname\n")
        for city, lat, lon, name in points:
            out.write(f"{city}\t{lat}\t{lon}\t{name}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="download the extract again even if it is already here")
    args = parser.parse_args()

    if not download(args.refresh) and not PBF.exists():
        return 1

    result = extract()
    points = result["points"]
    if not points:
        print("no signage nodes found — the filter or the parse is wrong",
              file=sys.stderr)
        return 1
    write_points(points)

    counts = {}
    for city, _, _, _ in points:
        counts[city] = counts.get(city, 0) + 1
    print(f"\n{len(points):,} signage points from {result['scanned']:,} nodes "
          f"-> {POINTS_CACHE}")
    for city in list(CITIES) + ["other"]:
        if counts.get(city):
            print(f"  {city:<10} {counts[city]:>7,}")
    diacritic = sum(any(c in p[3] for c in "čćđšžČĆĐŠŽ") for p in points)
    print(f"{diacritic:,} names carry a Bosnian diacritic "
          f"({diacritic / len(points) * 100:.0f}%)")
    print("harvest_mapillary.py reads this file and will not call Overpass again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
