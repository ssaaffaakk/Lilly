#!/usr/bin/env python3
"""Collect real Bosnian sign text from OpenStreetMap.

The photo reader practises on text it will never be shown. Its training labels
come from the news corpus — data/scripts/generate_ocr_data.py samples words out
of data/clean/train.tsv, and generate_ocr_photos.py reuses that same loader — so
every label is corpus vocabulary: inflected common nouns like "obogaćene",
"rješavanju", "također". What a phone is actually pointed at in Bosnia is a
different vocabulary: proper names, brands, abbreviations, street plates
("Ulica E. Šehovića", "OŠ Hasan Kikić", "Apoteke Sarajevo"). The reader has had
no practice on the strings it will be asked to read, and that gap is a plain
train/serve mismatch.

This script fetches text that is literally painted on Bosnian signage — shop and
amenity names, street names, transit stop names — from OpenStreetMap via the
Overpass API, across both entities rather than Sarajevo alone.

    python3 data/scripts/harvest_sign_text.py
    python3 data/scripts/harvest_sign_text.py --cap 300      # quick sample
    python3 data/scripts/harvest_sign_text.py --no-cache     # ignore cached JSON
    python3 data/scripts/harvest_sign_text.py --sample 30    # print rows to read

Writes data/signs/sign-text.tsv with columns:
    text            the sign string, NFC-normalised, whitespace collapsed
    kind            shop | amenity | street | stop | sign
    city            which city bbox it was first seen in
    script          latin | cyrillic
    has_diacritic   1 if it contains č ć đ š ž (either case), else 0

It measures rather than assumes. Every discarded row is counted under a named
reason and printed with an example, so a filter that is quietly eating good data
shows up instead of hiding; and every run compares the harvest against the
synthetic OCR labels already in data/ocr on both diacritic rate and length,
which is the comparison that says whether this data is worth adding at all.

Data source and licence
-----------------------
Data is © OpenStreetMap contributors, licensed under the Open Database Licence
(ODbL) 1.0 — https://opendatacommons.org/licenses/odbl/1-0/

ODbL allows commercial use and redistribution, which is why this source was
chosen over a scrape of somebody's website. It carries two obligations. Source
must be attributed: anything shipping this file, or a database derived from it,
must carry "© OpenStreetMap contributors". And share-alike applies to a derived
*database*: redistributing sign-text.tsv, or a modified version of it, means
redistributing it under ODbL too. A model trained on this text is a produced
work, not a derived database, so weights are not forced under ODbL — but the
attribution still travels with the corpus.

Overpass is a free service run on donated hardware, and it will say so when you
are asking too much: the first version of this script paused a flat three
seconds between requests, collected a 429 on six of its first seven cities, and
was then refused at the TCP level for twenty minutes. So it now asks the
server's own /api/status when it may go again and waits that long, falls back to
public mirrors when one instance is overloaded, caps each response, and caches
raw replies in the system temp directory — never in the repo — so a re-run costs
the server nothing at all. Queries are clipped to the BiH admin area, because a
bounding box around a border town does not stop at the border.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, OrderedDict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
OUT_FILE = DATA_DIR / "signs" / "sign-text.tsv"
SYNTHETIC_LABELS = DATA_DIR / "ocr" / "train" / "gt.txt"

# The main instance first, public mirrors after. Mirrors are not a way to dodge
# a rate limit — the slot wait below is still honoured on each — they are what
# you fall back to when one instance is saturated, which during this harvest was
# the normal case rather than the exception: overpass-api.de refused connections
# outright for over twenty minutes and kumi answered 500/502 throughout.
# Only overpass-api.de serves /api/status; the others get the fixed pause.
# Every one of these is checked for Bosnian coverage before it is trusted.
ENDPOINTS = ("https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter")

# An Overpass instance can be up, answer 200, and still be the wrong database.
# overpass.osm.ch was in this list until it silently returned {"elements": []}
# for eleven consecutive Bosnian cities: it serves a Switzerland-only extract,
# so every query outside Switzerland is a valid, empty, useless answer. No
# status code catches that. Each endpoint is therefore probed once against a
# block of central Sarajevo known to hold data, and any instance that answers
# empty is dropped from the rotation for the rest of the run.
COVERAGE_PROBE = ('[out:json][timeout:25];'
                  'node(43.8540,18.4200,43.8620,18.4320)["amenity"];out tags 5;')
_COVERAGE_CHECKED = {}      # endpoint -> (verdict or None if it errored, when)
# How long to take an unreachable instance's word for it before probing again.
# Five minutes is shorter than the outage seen here (overpass-api.de refused for
# a little over twenty) but long enough that a run does not re-probe per query.
PROBE_RETRY_SECONDS = 300
# Measured, not guessed: an earlier version of this script paused a flat 3s
# between requests and was rate-limited on six of its first seven cities, then
# refused outright at the TCP level for several minutes. Overpass rations by
# slot, not by wall clock, so the fixed pause is only the floor — ask_slot_wait()
# below reads the server's own /api/status and waits as long as it says to.
PAUSE_SECONDS = 12.0
REQUEST_TIMEOUT = 180
# Server-side query budget. Kept under Overpass's own default so the server
# aborts on its own terms with a readable error instead of dropping the socket.
QUERY_TIMEOUT = 120

# Bosnia and Herzegovina is not one city and not one script community. Sarajevo
# alone would bias the harvest toward Federation naming and miss the Cyrillic
# question entirely, so this covers both entities plus Brčko District: the
# Federation cantonal centres, the larger Republika Srpska towns, and the
# Herzegovina south. Boxes are drawn around each built-up core rather than the
# municipal boundary — signage density is what matters, and a municipal box
# would spend the response cap on farm tracks. Istočno Sarajevo's box stops at
# 43.82 where Sarajevo's begins so the two do not overlap and double-count.
CITIES = OrderedDict([
    # (south, west, north, east)
    ("Sarajevo",          (43.82, 18.28, 43.92, 18.48)),
    ("Istočno Sarajevo",  (43.75, 18.36, 43.82, 18.46)),
    ("Banja Luka",        (44.72, 17.14, 44.82, 17.25)),
    ("Mostar",            (43.30, 17.75, 43.39, 17.86)),
    ("Tuzla",             (44.50, 18.62, 44.58, 18.73)),
    ("Zenica",            (44.16, 17.86, 44.24, 17.96)),
    ("Bihać",             (44.77, 15.82, 44.85, 15.92)),
    ("Travnik",           (44.20, 17.62, 44.27, 17.70)),
    ("Brčko",             (44.83, 18.75, 44.92, 18.85)),
    ("Bijeljina",         (44.72, 19.17, 44.80, 19.27)),
    ("Prijedor",          (44.94, 16.66, 45.02, 16.76)),
    ("Doboj",             (44.71, 18.05, 44.78, 18.14)),
    ("Trebinje",          (42.68, 18.31, 42.75, 18.39)),
    ("Goražde",           (43.63, 18.94, 43.70, 19.02)),
    ("Livno",             (43.79, 16.97, 43.86, 17.05)),
    ("Zvornik",           (44.35, 19.06, 44.42, 19.14)),
    ("Cazin",             (44.94, 15.90, 45.01, 15.98)),
])

# OSM tag -> the kind column. shop and craft are both "a business with its name
# over the door"; amenity, tourism and office are all "an institution with a
# plate on it". Collapsing them keeps the column to the four kinds the reader's
# failure modes actually differ between, plus `sign` for painted road text.
POI_TAGS = {"shop": "shop", "craft": "shop",
            "amenity": "amenity", "tourism": "amenity", "office": "amenity"}
TRANSPORT_TAGS = {"public_transport": "stop", "railway": "stop"}

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# The five Bosnian letters the reader keeps losing, precomposed. A name typed
# in decomposed form is "c" + a combining caron and would match none of these,
# so normalise() runs NFC before anything looks at this set.
DIACRITICS = set("čćđšžČĆĐŠŽ")
# Latin-script letters we expect on a Bosnian sign: ASCII plus the Latin-1 and
# Latin Extended-A ranges, which cover č ć đ š ž and the borrowed ü ö é of
# foreign brand names. Anything outside this and outside Cyrillic is a script we
# are not collecting for.
LATIN_OK = re.compile(r"^[ -ɏ‐-‟′″ ]*$")

# OSM's `name` is free text and mappers put non-names in it. These are the
# literal values seen in the raw harvest that are tag artifacts, not signage.
JUNK_VALUES = {"yes", "no", "none", "null", "unknown", "n/a", "na", "?", "-",
               "unnamed", "noname", "fixme", "tbd", "test"}
# A guard rail, not a filter. The measured length distribution of the harvest is
# already short — p90 is 24 characters, p99 is 42, p99.9 is 67 — and reading by
# hand every one of the 17 names longer than 55 characters showed the long band
# is mostly genuine institutional plates ("Ministarstvo za pitanja branitelja
# Hercegovačko-neretvanske županije") rather than junk. Cutting aggressively
# there would discard real signage, so this sits above all but the single
# longest observed name (94 chars) and exists only to stop a pathological value.
# The separator rules below are what actually removes mapper prose.
MAX_CHARS = 80
# One character cannot teach a letter shape in context and is usually a mapping
# stub ("A", "1"). Two is a real sign ("BH", "OŠ").
MIN_CHARS = 2
# Measured separators, not guessed ones. A pipe and a semicolon never appear on
# a sign; in this harvest they appear only in mapper-formatted prose ("Advokat |
# … | Zenica") and in a name accidentally joined to itself
# ("Vokabula;Vokabula - …"), 4 names in 6,149.
#
# Two patterns are deliberately NOT filtered, because measuring them showed a
# filter would cost more than it saves:
#   * a trailing full stop looks like sentence punctuation, but 83 of the names
#     ending in one are the Bosnian company suffix "d.o.o." — rejecting on it
#     would throw away legitimate shopfronts.
#   * a parenthesis often marks a mapper's aside ("Adi-Ad (elektro-vodo
#     materijal)"), but just as often marks a genuine second name that is really
#     on the plaque ("Džindijska (Huseina Čauša) džamija"). At 101 names, 1.06%
#     of the harvest, the band is roughly half real, so it is left in and
#     flagged here rather than cut blind. Anything training on this file that
#     cares should strip parentheticals itself.
PROSE_SEPARATORS = (" | ", ";")


def overpass_query(bbox: tuple, group: str, cap: int) -> str:
    """Build one Overpass QL request for a bbox, clipped to Bosnia.

    Two groups per city rather than one combined query: `out tags N` caps the
    whole result set, so streets — which arrive as one element per way segment
    and can run to thousands in a city — would eat the entire cap and starve the
    shop names. Separate caps keep the mix balanced.

    Every filter is also clipped to the BiH admin area. A rectangle drawn round
    a border town does not stop at the border: the plain Zvornik box reached
    across the Drina and returned Мали Зворник, which is in Serbia — 9 of its 36
    amenities were the wrong country — and the Brčko box crosses the Sava into
    Croatia. Clipping to the country is more robust than hand-shrinking boxes,
    because it stays correct when someone edits the list below.
    """
    box = "{:.4f},{:.4f},{:.4f},{:.4f}".format(*bbox)
    where = f"(area.ba)({box})"
    lines = []
    if group == "poi":
        for tag in POI_TAGS:
            # Both nodes and ways: a kiosk is a point, a supermarket is usually
            # drawn as a building outline, and both have the same sign on them.
            lines.append(f'node{where}["name"]["{tag}"];')
            lines.append(f'way{where}["name"]["{tag}"];')
    elif group == "transport":
        lines.append(f'way{where}["name"]["highway"];')
        for tag in TRANSPORT_TAGS:
            lines.append(f'node{where}["name"]["{tag}"];')
        # traffic_sign holds the sign's text on the minority of nodes that have
        # it; ask for it separately since those nodes often carry no `name`.
        lines.append(f'node{where}["traffic_sign"];')
    else:
        raise ValueError(group)
    body = "".join(lines)
    return (f'[out:json][timeout:{QUERY_TIMEOUT}];'
            f'area["ISO3166-1"="BA"][admin_level=2]->.ba;'
            f"({body});out tags {cap};")


UA = {"User-Agent": "lilly-sign-harvest/1.0 (Bosnian OCR training data)"}


def ask_slot_wait(endpoint: str) -> float:
    """Seconds the server says to wait before it will take another query.

    /api/status is the documented way to ask instead of probing with real
    queries and collecting 429s. It answers either "N slots available now" or
    "Slot available after: <timestamp>, in N seconds"; the second form is a
    direct instruction and is what keeps this script from being throttled.
    A server that does not answer is not an excuse to charge ahead — the caller
    still falls back to PAUSE_SECONDS.
    """
    status_url = endpoint.replace("/api/interpreter", "/api/status")
    try:
        req = urllib.request.Request(status_url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return -1.0
    if re.search(r"\bslots? available now", text):
        return 0.0
    waits = [int(n) for n in re.findall(r"in (\d+) seconds", text)]
    # +2s of slack: the server's countdown and our clock are not the same clock.
    return min(waits) + 2.0 if waits else 0.0


def post(endpoint: str, query: str, timeout: int) -> str:
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def serves_bosnia(endpoint: str) -> bool:
    """Does this instance actually hold Bosnian data?

    A positive answer is remembered for the whole run — coverage does not change
    under us. A probe that *errors* is not an answer about coverage at all, so
    the endpoint stays in the rotation and the real request decides; but the
    attempt is remembered for PROBE_RETRY_SECONDS so that a mirror which is down
    for twenty minutes is not probed again on every single query. Without that
    the script sent two doomed requests per city to servers it already knew were
    refusing it, which is precisely the behaviour that got it blocked.
    """
    cached = _COVERAGE_CHECKED.get(endpoint)
    if cached is not None:
        verdict, checked_at = cached
        if verdict is not None or time.monotonic() - checked_at < PROBE_RETRY_SECONDS:
            return True if verdict is None else verdict
    host = endpoint.split("/")[2]
    try:
        elements = json.loads(post(endpoint, COVERAGE_PROBE, 60)).get("elements", [])
    except Exception as err:
        _COVERAGE_CHECKED[endpoint] = (None, time.monotonic())
        print(f"    {host}: coverage probe failed ({err}), will still try",
              flush=True)
        return True
    ok = len(elements) > 0
    _COVERAGE_CHECKED[endpoint] = (ok, time.monotonic())
    print(f"    {host}: coverage probe {'ok' if ok else 'EMPTY — dropping'} "
          f"({len(elements)} elements in central Sarajevo)", flush=True)
    return ok


def fetch(query: str, cache_dir: Path, use_cache: bool) -> list:
    """POST one query, with cache, slot-gating and backoff.

    Tries each endpoint in turn and, on each, waits for the slot the server says
    is free before spending it. Returns the elements list.
    """
    key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
    cached = cache_dir / f"{key}.json"
    if use_cache and cached.exists():
        return json.loads(cached.read_text("utf-8")).get("elements", [])

    raw, reason = None, "unknown"
    # Each round tries every mirror once, then sleeps. Rotating within the round
    # rather than sleeping between mirrors matters because the usual failure is
    # one instance being busy, not both — and a refusal comes back
    # instantly, so a full round costs nothing when a mirror is simply down.
    for attempt in range(3):
        for endpoint in ENDPOINTS:
            host = endpoint.split("/")[2]
            if not serves_bosnia(endpoint):
                continue
            slot = ask_slot_wait(endpoint)
            if slot > 0:
                print(f"    {host}: slot in {slot:.0f}s, waiting", flush=True)
                time.sleep(min(slot, 180))
            try:
                raw = post(endpoint, query, REQUEST_TIMEOUT)
                break
            except urllib.error.HTTPError as err:
                # 429 (no free slot) and any 5xx (the instance is broken or
                # overloaded — kumi served a steady diet of 500 and 502 during
                # this harvest) both mean "this server, later or never". Move to
                # the next mirror. A 4xx other than 429 means the query itself
                # is malformed, which is our bug and must not be retried away.
                if err.code != 429 and err.code < 500:
                    raise
                reason = f"{host} HTTP {err.code}"
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                # Connection refused lands here. It is what an instance does
                # once it has decided a client is too pushy, so treat it as the
                # harshest possible "back off", not as a permanently dead host.
                reason = f"{host} {err}"
            except Exception as err:
                raise
        if raw is not None:
            break
        if attempt == 2:
            raise RuntimeError(f"no Overpass instance answered: {reason}")
        wait = 60 * (attempt + 1)
        print(f"    all mirrors refused ({reason}), backing off {wait}s",
              flush=True)
        time.sleep(wait)

    if raw is None:
        return []
    elements = json.loads(raw).get("elements", [])
    # An empty answer is the exact signature of the wrong-database failure above,
    # and it is cheap to re-ask, so it never gets frozen into the cache. Only a
    # response with something in it is worth keeping.
    if elements:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(raw, "utf-8")
    else:
        print("    empty response — not cached", flush=True)
    return elements


def kind_of(tags: dict) -> str:
    """Which of the five sign kinds this element is, or '' if none."""
    # Checked before the POI tags because a traffic_sign node is a sign whether
    # or not it also carries a name — city_limit nodes hold the town-boundary
    # plate ("Mostar", and bilingual ones like "Кијево Kijevo") in `name`, and
    # matching on the absence of a name would throw exactly those away.
    if "traffic_sign" in tags:
        return "sign"
    for tag, kind in POI_TAGS.items():
        if tag in tags:
            return kind
    for tag, kind in TRANSPORT_TAGS.items():
        if tag in tags:
            return kind
    if "highway" in tags:
        return "street"
    return ""


def sign_text_of(tags: dict) -> str:
    """The string a camera would read off this element.

    The bare `traffic_sign` value is a code, not words — "maxspeed", "city_limit",
    "DE:253" — so it is never the text. The words live in traffic_sign:text when
    a mapper typed them, and otherwise in `name` on the city-limit plates. Most
    traffic_sign nodes have neither and drop out as empty, which is the honest
    outcome: OSM simply does not record what those signs say.
    """
    if "traffic_sign" in tags:
        return tags.get("traffic_sign:text") or tags.get("name", "")
    return tags.get("name", "")


def normalise(text: str) -> str:
    """NFC and single spaces.

    NFC matters twice over: a decomposed "č" is two codepoints, which would both
    dodge the diacritic count and give the reader a label it has no class for.
    OSM names also carry stray newlines and doubled spaces from copy-paste.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def reject_reason(text: str) -> str:
    """Name why this string is not sign text, or '' if it is."""
    if not text:
        return "empty"
    if text.lower() in JUNK_VALUES:
        return "tag-artifact"
    if not any(ch.isalpha() for ch in text):
        return "no-letters"          # house numbers, coordinates, bare refs
    if len(text) < MIN_CHARS:
        return "too-short"
    if len(text) > MAX_CHARS:
        return "too-long"
    if any(sep in text for sep in PROSE_SEPARATORS):
        return "mapper-prose"
    if CYRILLIC.search(text):
        return ""                    # kept, flagged in the script column
    if not LATIN_OK.match(text):
        return "other-script"        # Arabic, Greek, CJK restaurant names
    return ""


def harvest(cap: int, cache_dir: Path, use_cache: bool,
            resume: bool = False, progress_path: Path | None = None,
            out_path: Path | None = None) -> tuple:
    """Query every city. Returns (rows, rejects) with rows deduped on text."""
    progress_path = progress_path or OUT_FILE.parent / ".osm-harvest.progress.json"
    out_path = out_path or OUT_FILE
    seen: dict = {}
    completed: set[str] = set()

    if resume and out_path.exists():
        for line in out_path.read_text("utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 5:
                seen[parts[0]] = {
                    "text": parts[0],
                    "kind": parts[1],
                    "city": parts[2],
                    "script": parts[3],
                    "has_diacritic": int(parts[4]),
                }
    if resume and progress_path.exists():
        try:
            completed = set(json.loads(progress_path.read_text("utf-8")).get("completed_cities", []))
        except (json.JSONDecodeError, OSError):
            completed = set()
    if seen:
        print(f"  resume: {len(seen):,} rows loaded, {len(completed)} cities done", flush=True)

    rejects = Counter()
    reject_samples: dict = {}
    raw_total = 0

    for city, bbox in CITIES.items():
        if city in completed:
            print(f"  {city:<18} skip (checkpoint)", flush=True)
            continue
        city_kept = 0
        for group in ("poi", "transport"):
            query = overpass_query(bbox, group, cap)
            was_cached = use_cache and (
                cache_dir / f"{hashlib.sha1(query.encode()).hexdigest()[:16]}.json").exists()
            elements = fetch(query, cache_dir, use_cache)
            raw_total += len(elements)
            for element in elements:
                tags = element.get("tags", {})
                kind = kind_of(tags)
                if not kind:
                    rejects["no-kind"] += 1
                    continue
                text = normalise(sign_text_of(tags))
                reason = reject_reason(text)
                if reason:
                    rejects[reason] += 1
                    reject_samples.setdefault(reason, []).append(text[:60])
                    continue
                if text in seen:
                    rejects["duplicate"] += 1
                    continue
                seen[text] = {
                    "text": text,
                    "kind": kind,
                    "city": city,
                    "script": "cyrillic" if CYRILLIC.search(text) else "latin",
                    "has_diacritic": int(any(ch in DIACRITICS for ch in text)),
                }
                city_kept += 1
            if not was_cached:
                time.sleep(PAUSE_SECONDS)
        completed.add(city)
        write_tsv(list(seen.values()), out_path)
        progress_path.write_text(
            json.dumps({
                "completed_cities": sorted(completed),
                "row_count": len(seen),
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  {city:<18} +{city_kept:>5} new  (checkpoint {len(seen):,} total)", flush=True)

    print(f"\nraw elements returned: {raw_total:,}")
    return list(seen.values()), (rejects, reject_samples)


def diacritic_rate(texts) -> float:
    texts = list(texts)
    if not texts:
        return 0.0
    hits = sum(1 for t in texts if any(ch in DIACRITICS for ch in t))
    return 100.0 * hits / len(texts)


def describe_lengths(texts) -> dict:
    words = sorted(len(t.split()) for t in texts)
    chars = sorted(len(t) for t in texts)
    if not words:
        return {}
    def pct(seq, p):
        return seq[min(len(seq) - 1, int(len(seq) * p))]
    return {"n": len(words),
            "words_mean": sum(words) / len(words),
            "words_median": pct(words, 0.5),
            "words_p90": pct(words, 0.9),
            "words_max": words[-1],
            "chars_mean": sum(chars) / len(chars),
            "chars_median": pct(chars, 0.5)}


def load_synthetic_labels() -> list:
    """The reader's current training text, for the comparison in --report."""
    if not SYNTHETIC_LABELS.exists():
        return []
    texts = []
    for line in SYNTHETIC_LABELS.read_text("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip():
            texts.append(parts[1].strip())
    return texts


def report(rows: list, rejects: tuple) -> None:
    counts, samples = rejects
    texts = [r["text"] for r in rows]

    print(f"\nunique sign texts kept: {len(rows):,}")
    print("\ndropped:")
    for reason, n in counts.most_common():
        example = samples.get(reason, [""])[0]
        tail = f"   e.g. {example!r}" if example else ""
        print(f"  {reason:<14} {n:>6,}{tail}")

    print("\nby kind:")
    for kind, n in Counter(r["kind"] for r in rows).most_common():
        print(f"  {kind:<10} {n:>6,}  ({100*n/len(rows):.1f}%)")

    print("\nby city:")
    for city, n in Counter(r["city"] for r in rows).most_common():
        print(f"  {city:<18} {n:>6,}")

    print("\nby script:")
    for script, n in Counter(r["script"] for r in rows).most_common():
        print(f"  {script:<10} {n:>6,}  ({100*n/len(rows):.1f}%)")

    real_rate = diacritic_rate(texts)
    print(f"\ndiacritic rate (real signs): {real_rate:.1f}%")

    synthetic = load_synthetic_labels()
    if synthetic:
        syn_rate = diacritic_rate(synthetic)
        print(f"diacritic rate (synthetic, {SYNTHETIC_LABELS.relative_to(DATA_DIR.parent)}): "
              f"{syn_rate:.1f}%  over {len(synthetic):,} labels")
        print(f"  -> real signs are {real_rate - syn_rate:+.1f} points "
              f"{'richer' if real_rate > syn_rate else 'poorer'} in č ć đ š ž")

        real_len = describe_lengths(texts)
        syn_len = describe_lengths(synthetic)
        print("\nlength:")
        print(f"  real      {real_len['words_mean']:.2f} words mean, "
              f"median {real_len['words_median']}, p90 {real_len['words_p90']}, "
              f"max {real_len['words_max']}; {real_len['chars_mean']:.1f} chars mean")
        print(f"  synthetic {syn_len['words_mean']:.2f} words mean, "
              f"median {syn_len['words_median']}, p90 {syn_len['words_p90']}, "
              f"max {syn_len['words_max']}; {syn_len['chars_mean']:.1f} chars mean")
    else:
        print(f"(no synthetic labels at {SYNTHETIC_LABELS} to compare against)")


def write_tsv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted by kind then text so a diff between two runs is readable; the
    # trainer that consumes this will shuffle anyway.
    rows = sorted(rows, key=lambda r: (r["kind"], r["text"]))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("text\tkind\tcity\tscript\thas_diacritic\n")
        for r in rows:
            handle.write(f"{r['text']}\t{r['kind']}\t{r['city']}\t"
                         f"{r['script']}\t{r['has_diacritic']}\n")


def main() -> int:
    def _on_signal(signum, _frame):
        raise SystemExit(128 + signum)

    for _sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _on_signal)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cap", type=int, default=1200,
                        help="max elements per city per group (default 1200: "
                             "enough to fill a city centre, small enough that "
                             "one query stays inside Overpass's budget)")
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    parser.add_argument("--no-cache", action="store_true",
                        help="re-query Overpass instead of reusing cached JSON")
    parser.add_argument("--sample", type=int, default=0,
                        help="print N random kept texts for hand-checking")
    parser.add_argument("--resume", action="store_true",
                        help="skip cities in checkpoint; append to existing TSV")
    args = parser.parse_args()

    cache_dir = Path(tempfile.gettempdir()) / "lilly-overpass-cache"
    print(f"querying {len(CITIES)} cities, cap {args.cap} per group, "
          f"cache {cache_dir}\n", flush=True)

    try:
        rows, rejects = harvest(args.cap, cache_dir, not args.no_cache,
                                resume=args.resume, out_path=args.out)
    except urllib.error.HTTPError as err:
        print(f"Overpass refused: HTTP {err.code} {err.reason}", file=sys.stderr)
        return 1
    if not rows:
        print("no sign text collected", file=sys.stderr)
        return 1

    report(rows, rejects)
    write_tsv(rows, args.out)
    print(f"\nwrote {args.out}  ({len(rows):,} rows)")
    print("© OpenStreetMap contributors, ODbL 1.0 — attribution travels with "
          "this file")

    if args.sample:
        import random
        random.seed(7)
        print(f"\n{args.sample} random rows for hand-checking:")
        for r in random.sample(rows, min(args.sample, len(rows))):
            print(f"  [{r['kind']:<7} {r['city']:<16}] {r['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
