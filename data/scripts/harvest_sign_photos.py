#!/usr/bin/env python3
"""Collect freely-licensed photographs of Bosnian signage, and throw away the
ones with nothing to read on them.

WHY THIS EXISTS
---------------
The reader was trained on 4,000 *synthesised* photos — text rendered onto
backgrounds with blur, skew and uneven light applied — and reads 75.0% of the
words in them. What it reads on a photograph actually taken in Bosnia is
unknown. Three real photographs were looked at by hand, and three were already
enough to expose a defect class the synthetic set cannot contain: on a sign
carrying both alphabets the Latin-only recogniser forces the Cyrillic through
its Latin character set and emits rubbish ("Лубровник" -> "Lybpobhuk"). Three
photos is an anecdote. This fetches a few hundred.

THE LICENCE RULE, WHICH IS NOT NEGOTIABLE
-----------------------------------------
The model and its data are published openly on Hugging Face, so every image
here is redistributed. "It was on the internet" is not a licence. A file is
downloaded only when its own metadata names one of:

    CC0 / public domain    no attribution obligation, but we record one anyway
    CC BY                  attribution obligatory on redistribution
    CC BY-SA               attribution obligatory, and share-alike

Everything else is refused, including licences that are free in other senses:
GFDL-only, "Attribution" (the bare custom template), FAL, OGL. Not because
they are unusable, but because each one carries obligations of its own and the
gate has to be a list of things checked rather than a list of things excluded.
`licence_of()` is deny-by-default: a file whose licence field it does not
recognise is dropped and counted, never downloaded "to look at later".

That is why the sources are Wikimedia Commons and Openverse and nothing else.
Both attach machine-readable licence metadata to every result. Google Images,
Street View, Instagram and ordinary web pages do not, and a photograph found
there is unusable however good it is.

CREDITS.tsv is the redistribution licence being honoured, not a log. One row
per kept file, carrying the licence, the licence URL, the author as the source
names them, and the file's own description page — enough for a downstream user
to attribute it without coming back here.

THE FILTER THAT MAKES THE WHOLE THING WORTH DOING
-------------------------------------------------
Most photographs of Sarajevo are of buildings, bridges, mountains and mosques,
and carry no readable text at all. A folder of those is worth nothing here: it
would look like 400 real photos and measure nothing. So every candidate is
downloaded to a staging area, read with the project's own detector, and kept
only if there is text on it (see SCREEN_* below for the thresholds and where
each number came from). Landscapes are deleted, not shipped.

The second half of that filter is a photograph test, because the first half
cannot be one: Category:Road signs in Bosnia and Herzegovina is mostly
*drawings* of road signs, and a drawing of a sign passes a text filter
perfectly while being exactly the thing this folder exists to stop standing in
for a real photograph. Flat colour is what gives them away — see
DRAWING_MAX_COLOURS.

TWO BIASES THIS PROCESS HAS, STATED UP FRONT
--------------------------------------------
1. Candidates are screened in priority order (SIGN_WORDS): files whose name or
   category mentions a sign, a street, a shop go first, because reading one
   candidate costs ~9 s of CPU and the budget is finite. So the kept set
   over-represents photographs whose uploader thought to name them after the
   sign in them, and under-represents signs caught incidentally in a street
   scene. It is a sampling bias, not a labelling error.
2. The screen is the same detector the reader uses. It therefore keeps photos
   whose text this detector can find, which is not the same as photos with
   text on them. Signs it cannot see at all are silently absent — precisely
   the hardest cases. A set built this way puts a floor under the real error
   rate, not a fair estimate of it.

Usage:
    python3 data/scripts/harvest_sign_photos.py                 # the full run
    python3 data/scripts/harvest_sign_photos.py --survey        # discover only
    python3 data/scripts/harvest_sign_photos.py --target 300
    python3 data/scripts/harvest_sign_photos.py --report        # counts so far

Everything lands in data/ocr/real-photos/harvested/ (gitignored, like the rest
of data/ocr). A run is resumable: every candidate's verdict is appended to
.state/screened.tsv, and a second run skips whatever is already decided.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import requests

# Refuse to start if the machine has no room. Five of these ran at once on
# 27 August and the kernel panicked: 100% of the compressor limit, fifteen
# swapfiles, watchdog silent for 94 seconds. Each job is reasonable alone and
# none of them knew the others existed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.guard import claim
# claim() is in main(), not here. multiprocessing spawn re-imports this file
# in every reader; import-time claim saw the parent as "already running"
# and the workers died at ~19 MB, so screening never started.

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "data" / "ocr" / "real-photos" / "harvested"
STATE_DIR = OUT_DIR / ".state"
STAGING_DIR = STATE_DIR / "staging"
CANDIDATES_PATH = STATE_DIR / "candidates.json"
STAGED_PATH = STATE_DIR / "staged.json"
CRAWL_STATE_PATH = STATE_DIR / "crawl-state.json"
SCREENED_PATH = STATE_DIR / "screened.tsv"
CREDITS_PATH = OUT_DIR / "CREDITS.tsv"

# Wikimedia returns 403 without one, and asks that it identify the client and
# give a way to reach whoever runs it. This is a shared server being used for
# free; the sleeps below are the other half of that bargain.
USER_AGENT = "LillyBosnianOCR/0.1 (research; github.com/ssaaffaakk/Lilly)"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"

# Measured, not guessed. At 0.3 s between calls Commons stopped answering after
# about ten requests; at 1.0 s it answered for roughly twenty and then returned
# 429 with `Retry-After: 51` and a pointer to the rate-limit policy. That is an
# hourly quota for unauthenticated clients, so the sustainable gap is seconds,
# not milliseconds, and no amount of retrying buys a faster crawl. 6 s is the
# starting gap; Fetcher raises its own floor every time a 429 arrives, so the
# script finds the real limit instead of being told it.
#
# upload.wikimedia.org — where the image bytes come from — is a different
# service with a limit of its own, hence the separate gap. It is looser than
# the API's but it is not absent: at 0.4 s roughly one image in ten came back
# 429. 1.2 s is the floor, and Fetcher.download widens from there the same way.
API_GAP_SECONDS = 6.0
DOWNLOAD_GAP_SECONDS = 1.2
# upload.wikimedia.org rate-limits requests for *original* files per address
# and says so with a 429 and Retry-After (600 s, measured 3 Sep 2026: "please
# ... instead use thumbnail images in sizes listed"). Standard thumbnails are
# exempt; a photograph narrower than 1280px has no thumbnail, so its rendering
# is the original and meets the limit. Waiting the window out is what the
# message asks for, and a wait does not spend one of the download attempts.
MAX_429_WAITS = 24                                   # four hours per file at 600 s
MAX_429_WAIT_SECONDS = 900
MAX_API_TRIES = 6

# ---------------------------------------------------------------- discovery

# The seed categories. Commons category pages hold few files directly; the
# photographs live in subcategories, hence the recursion.
SEED_CATEGORIES = [
    ("Category:Signs in Bosnia and Herzegovina", ""),
    ("Category:Road signs in Bosnia and Herzegovina", ""),
    ("Category:Shops in Bosnia and Herzegovina", ""),
    ("Category:Streets in Bosnia and Herzegovina", ""),
    ("Category:Sarajevo", "Sarajevo"),
    ("Category:Mostar", "Mostar"),
    ("Category:Banja Luka", "Banja Luka"),
    ("Category:Tuzla", "Tuzla"),
    ("Category:Zenica", "Zenica"),
    ("Category:Bihać", "Bihać"),
    ("Category:Travnik", "Travnik"),
]

# Deeper than this the tree stops being about Bosnia: Category:Sarajevo reaches
# "Category:People of Sarajevo" -> "...by name" -> a portrait of a footballer
# in four hops. Three keeps the crawl inside the place.
MAX_CATEGORY_DEPTH = 3

OPENVERSE_QUERIES = [
    "Sarajevo street sign", "Mostar sign", "Bosnia road sign",
    "Bosnia street name", "Sarajevo shop", "Banja Luka street",
    "Bosnia and Herzegovina signpost", "Sarajevo Baščaršija",
    "Tuzla street", "Zenica street", "Bosnia memorial plaque",
    "Sarajevo restaurant menu", "Mostar Stari Most sign",
]

# Used only to order the screening queue, never to accept or reject. Bosnian
# and English both, because uploaders use either.
SIGN_WORDS = (
    "sign", "signs", "signage", "signpost", "signboard", "billboard", "banner",
    "plaque", "plate", "board", "poster", "notice", "label", "menu", "graffiti",
    "street", "road", "avenue", "square", "crossing", "direction", "entrance",
    "shop", "store", "market", "bazaar", "restaurant", "cafe", "kiosk", "hotel",
    "tabla", "natpis", "ploca", "ploča", "znak", "putokaz", "tablica", "naziv",
    "ulica", "ulice", "trg", "cesta", "put", "saobracaj", "saobraćaj", "raskrsnica",
    "prodavnica", "radnja", "pijaca", "jelovnik", "kafana", "čaršija", "carsija",
    "spomen", "memorial", "museum", "muzej", "station", "stanica", "bus", "tram",
)

# Vector road-sign diagrams are the bulk of Category:Road signs and would sail
# through the text filter while being drawings, not photographs — exactly the
# thing this folder exists to stop substituting for. Same for animations and
# multi-page documents.
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

# Under this the file is a crop, a thumbnail or an icon, not a photograph of a
# scene — and the word crops cut out of it later would be a few pixels tall.
#
# Two numbers rather than one shortest side, because a shortest-side rule is
# wrong about panoramas: Sarajevo_Panorama.jpg is 2086x400, plenty of picture,
# and a 480 px shortest-side rule threw it out as "too small" while letting
# nothing else through that it should have. Area catches the genuinely tiny
# (a 386x466 crop of a cigarette packet is 0.18 MP); the shortest side stops an
# area rule from admitting a 4000x60 letterbox strip.
MIN_PIXELS = 400_000
MIN_SHORT_SIDE = 240

# ------------------------------------------------------------------ licence

# Matched against extmetadata's LicenseShortName, which is the field Commons
# renders on the file page ("CC BY-SA 4.0", "CC0", "Public domain"). The
# machine-readable `License` field is checked too, because PD files often carry
# a template name there ("pd-old-70") and a prose short name.
_CC0 = re.compile(r"^cc0\b|^cc[- ]zero\b")
_CC_BY_SA = re.compile(r"^cc[- ]by[- ]sa[- ]?\d")
_CC_BY = re.compile(r"^cc[- ]by[- ]?\d")
_PD = re.compile(r"^pd(\b|-)|^public domain\b")


def licence_of(short_name: str, machine_name: str) -> str | None:
    """One of CC0 / CC BY / CC BY-SA / PD, or None meaning: do not download.

    Deny by default. An unrecognised licence string is a file we have not
    established the right to redistribute, and this dataset gets published.
    """
    for field in (short_name, machine_name):
        text = re.sub(r"\s+", " ", field or "").strip().lower()
        if not text:
            continue
        # Order matters: "cc by-sa 4.0" also matches the plain-BY pattern's
        # prefix on some spellings, so share-alike is tested first.
        if _CC0.match(text):
            return "CC0"
        if _CC_BY_SA.match(text):
            return "CC BY-SA"
        if _CC_BY.match(text):
            return "CC BY"
        if _PD.match(text):
            return "PD"
    return None


_TAG = re.compile(r"<[^>]+>")


def plain(value: str) -> str:
    """Commons returns Artist and Credit as HTML. TSV wants neither tags nor tabs."""
    text = html.unescape(_TAG.sub(" ", value or ""))
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------ screening

# Thresholds, and where each one came from.
#
# The detector finds "text" on brickwork, foliage, window frames and roof tiles
# all day long; what separates those from writing is that the recogniser cannot
# make a confident word of them. Confidence, not box count, is the real filter.
#
# 0.50 was read off the three photographs already in ../commons, which are
# genuine signs and all have to pass. Their confident regions sit at 0.72-1.00
# and their junk regions at 0.02-0.44 — 'BOSIIEIHERCEGOIIEE' scores 0.02, the
# word 'project' on the same sign scores 0.72. Nothing in that set lands
# between 0.44 and 0.72, so the threshold is in a gap rather than on a slope.
# The sharpest case is Mostar_signs.JPG, which clears the bar with exactly two
# regions ('Dubrovnik' 0.89, 'Sarajevo' 0.99) — so 2 is the count that keeps
# the hardest of the three known-good photographs, not a round number.
SCREEN_MIN_CONFIDENCE = 0.50   # a region below this is not a word we can read
SCREEN_MIN_REGIONS = 2         # one confident region is a house number
SCREEN_MIN_LETTERS = 3         # and at least one of them has to be a word,
                               # not a date or a street number
BOSNIAN_DIACRITICS = set("čćđšžČĆĐŠŽ")
DRAWING_MAX_COLOURS = 5_000

# The screen reads a 1200 px rendering downscaled to 1 MP, not the original.
# Two reasons, both about cost rather than quality:
#
#   bandwidth — pulling a 12 MB original to decide whether to keep it costs
#   Wikimedia forty times what the decision needs. Keepers get their original
#   afterwards, and only keepers.
#
#   CPU — this reader has no GPU here. Measured on the three photographs in
#   ../commons, per image: 20.7 s at 2 MP (app/ocr.py's own working size),
#   8.9 s at 1 MP, 3.9 s at 0.6 MP. All three pass the filter at all three
#   sizes, but at 0.6 MP the Kovači street-history panel loses 18 of its 22
#   confident regions — the body text stops being readable, and that is the
#   small-and-distant text this collection is partly for. 1 MP is where the
#   cost halves twice over without that starting to happen.
#
# It makes the screen slightly weaker than the app's own read, so a photo with
# only small text on it can be dropped where the app would have read it. That
# is a false drop, not a false keep: it costs recall, never licence or quality.
SCREEN_WIDTH = 1200
SCREEN_PIXELS = 1_000_000
KEEP_WIDTH = 2560


def screen(path: Path) -> dict:
    """Read a staged image and decide whether it has a sign on it."""
    import numpy as np
    from PIL import Image

    from app import ocr

    try:
        # load_within_limits first, for its refusal of images too big to decode
        # at all; the second resize is this script's own cost cap on top.
        image = ocr.load_within_limits(str(path))
        height, width = image.shape[:2]
        if width * height > SCREEN_PIXELS:
            scale = (SCREEN_PIXELS / (width * height)) ** 0.5
            image = np.asarray(Image.fromarray(image).resize(
                (max(int(width * scale), 1), max(int(height * scale), 1))))
    except Exception as exc:
        return {"verdict": "drop", "reason": f"unreadable: {type(exc).__name__}",
                "regions": 0, "diacritics": 0, "text": ""}

    # Before the reader, and cheaper than it: is this a photograph at all?
    # Category:Road signs in Bosnia and Herzegovina is mostly *drawings* of road
    # signs, and a drawing of a sign would sail through a text filter while
    # being the one thing this folder exists to stop standing in for a real
    # photograph. SVG is refused earlier by mime type; the PNG and JPEG
    # renderings of the same diagrams are not, and this is what catches those.
    #
    # Measured at 1 MP: the seven real photographs to hand carry 70k-283k
    # distinct colours (lowest: 69,902). A flat-shaded sign diagram carries 65.
    # Anything JPEG-compressed picks up thousands of ringing artefacts, so the
    # line goes an order of magnitude above a drawing and an order of magnitude
    # below the sparsest real photograph, where nothing has to be decided
    # finely.
    packed = (image[:, :, 0].astype("int32") << 16 |
              image[:, :, 1].astype("int32") << 8 | image[:, :, 2])
    colours = int(np.unique(packed).size)
    if colours < DRAWING_MAX_COLOURS:
        return {"verdict": "drop", "reason": f"flat colour ({colours} shades): "
                                             f"a drawing, not a photograph",
                "regions": 0, "diacritics": 0, "text": ""}

    regions = ocr.read_regions(image)
    solid = [(text or "").strip() for _, text, conf in regions
             if conf >= SCREEN_MIN_CONFIDENCE and (text or "").strip()]
    wordy = [t for t in solid if sum(c.isalpha() for c in t) >= SCREEN_MIN_LETTERS]
    joined = " | ".join(solid)
    diacritics = sum(1 for t in solid for c in t if c in BOSNIAN_DIACRITICS)

    if len(solid) < SCREEN_MIN_REGIONS:
        reason = f"only {len(solid)} confident region(s)"
    elif not wordy:
        reason = f"no region with {SCREEN_MIN_LETTERS}+ letters"
    else:
        return {"verdict": "keep", "reason": "", "regions": len(solid),
                "diacritics": diacritics, "text": joined}
    return {"verdict": "drop", "reason": reason, "regions": len(solid),
            "diacritics": diacritics, "text": joined}


# ------------------------------------------------------------------- network

class Fetcher:
    """Serial, rate-limited, and it backs off when told to.

    One session, one request at a time, a gap between every pair. Wikimedia's
    etiquette asks for exactly this, and a harvester that ignores it gets the
    project blocked rather than the run finished faster.
    """

    def __init__(self, verbose: bool = True):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.last_call = 0.0
        self.gap = API_GAP_SECONDS
        self.download_gap = DOWNLOAD_GAP_SECONDS
        self.throttled = 0
        self.verbose = verbose

    def _wait(self, gap: float) -> None:
        due = self.last_call + gap - time.time()
        if due > 0:
            time.sleep(due)
        self.last_call = time.time()

    def api(self, url: str, params: dict) -> dict | None:
        for attempt in range(MAX_API_TRIES):
            self._wait(self.gap)
            try:
                response = self.session.get(url, params=params, timeout=60)
                if response.status_code == 200:
                    # Widening is instant, narrowing is slow and never goes
                    # below the floor. Without the decay one 429 — including
                    # one caused by something else on this machine talking to
                    # the same API — halves the crawl rate for the rest of the
                    # run, and the crawl is already the long pole.
                    self.gap = max(API_GAP_SECONDS, self.gap * 0.98)
                    return response.json()
                delay = float(response.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
                if response.status_code == 429:
                    # Being told off once means the standing gap was wrong, not
                    # that this one call was unlucky. Widen it a lot.
                    self.throttled += 1
                    self.gap = min(self.gap * 1.5, 30.0)
            except Exception:
                delay = 5 * 2 ** attempt
            if self.verbose:
                print(f"    backing off {delay:.0f}s (gap now {self.gap:.1f}s)", flush=True)
            time.sleep(delay)
        return None

    def download(self, url: str, dest: Path) -> bool:
        """Fetch image bytes, with its own throttle separate from the API's.

        upload.wikimedia.org has a rate limit of its own and enforces it the
        same way: measured, 0.4 s between images produced a 429 on about one
        request in ten, each one costing fifteen seconds of retries and ending
        as a permanent "download failed" for a file that was perfectly fine.
        Same adaptive gap as the API, separate number, because the two limits
        are separate.
        """
        attempt, waits = 0, 0
        while attempt < 4:
            self._wait(self.download_gap)
            delay = 5 * 2 ** attempt
            throttled = False
            try:
                response = self.session.get(url, timeout=120, stream=True)
                if response.status_code == 200:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, "wb") as handle:
                        for chunk in response.iter_content(65536):
                            handle.write(chunk)
                    self.download_gap = max(DOWNLOAD_GAP_SECONDS,
                                            self.download_gap * 0.98)
                    return True
                delay = float(response.headers.get("Retry-After") or 0) or delay
                if response.status_code == 429:
                    self.throttled += 1
                    self.download_gap = min(self.download_gap * 1.5, 8.0)
                    throttled = waits < MAX_429_WAITS
            except Exception:
                pass
            # A connection that dies mid-stream leaves a truncated file behind.
            # Left there it would be screened, half-read, and possibly kept.
            dest.unlink(missing_ok=True)
            if throttled:
                # Told to wait: wait, and come back with all four attempts.
                waits += 1
                delay = min(delay, MAX_429_WAIT_SECONDS)
                if self.verbose:
                    print(f"    429 on download; waiting {delay:.0f}s "
                          f"({waits}/{MAX_429_WAITS})", flush=True)
                time.sleep(delay)
                continue
            attempt += 1
            if attempt < 4:
                time.sleep(delay)
        return False


def commons_api(fetcher: Fetcher, **params) -> dict | None:
    # maxlag is the courtesy flag: when Commons' databases are lagging it tells
    # the server to refuse us rather than add to the pile.
    query = {"action": "query", "format": "json", "formatversion": 2, "maxlag": 5}
    query.update(params)
    return fetcher.api(COMMONS_API, query)


def crawl_categories(fetcher: Fetcher, max_calls: int) -> dict[str, dict]:
    """Walk the seed categories to MAX_CATEGORY_DEPTH and collect file titles.

    Commons categories are a graph, not a tree — Sarajevo and Streets in Bosnia
    both reach Streets in Sarajevo, and some pairs point at each other.
    `seen_categories` is what stops that becoming an infinite walk.

    Breadth-first on purpose, and the queue starts with every seed rather than
    draining one city at a time: an API call costs six seconds, so a crawl that
    runs out of budget should have run out of it at an even depth across all
    eleven seeds, not two levels deep into Sarajevo and nowhere else.

    Files and subcategories come back from one call each ("file|subcat"), not
    two. At this request cost, halving the call count halves the crawl.
    """
    # The crawl is the expensive half of this script — an hour of calls nobody
    # can make go faster. It is checkpointed every 25 categories and resumed
    # from the checkpoint, so a kill, a crash or a machine going to sleep costs
    # minutes instead of the hour. The queue and the visited set are saved with
    # the candidates: without those, resuming would restart the walk.
    if CRAWL_STATE_PATH.exists():
        state = json.loads(CRAWL_STATE_PATH.read_text(encoding="utf-8"))
        seen_categories = set(state["seen"])
        candidates = state["candidates"]
        queue = [tuple(item) for item in state["queue"]]
        calls = 0
        print(f"  resuming the crawl: {len(seen_categories)} categories done, "
              f"{len(candidates)} files, {len(queue)} queued")
    else:
        seen_categories = set()
        candidates = {}
        queue = [(title, city, 0, title) for title, city in SEED_CATEGORIES]
        calls = 0

    def checkpoint():
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        CRAWL_STATE_PATH.write_text(json.dumps(
            {"seen": sorted(seen_categories), "candidates": candidates,
             "queue": queue}, ensure_ascii=False), encoding="utf-8")

    while queue and calls < max_calls:
        title, city, depth, path = queue.pop(0)
        if title in seen_categories:
            continue
        seen_categories.add(title)

        cont = None
        while calls < max_calls:
            params = {"list": "categorymembers", "cmtitle": title,
                      "cmtype": "file|subcat", "cmlimit": 500}
            if cont:
                params["cmcontinue"] = cont
            data = commons_api(fetcher, **params)
            calls += 1
            if not data or "query" not in data:
                break
            for member in data["query"]["categorymembers"]:
                name = member["title"]
                if member["ns"] == 6:                       # ns 6 is File:
                    if name not in candidates:
                        candidates[name] = {"source": "commons", "title": name,
                                            "city": city, "found_under": path}
                elif depth + 1 <= MAX_CATEGORY_DEPTH:
                    queue.append((name, city, depth + 1, f"{path} > {name}"))
            cont = data.get("continue", {}).get("cmcontinue")
            if not cont:
                break
        print(f"  [{calls:4d} calls, {len(seen_categories):4d} cats, "
              f"{len(candidates):5d} files] {title}", flush=True)
        if len(seen_categories) % 25 == 0:
            checkpoint()
    checkpoint()
    if queue:
        print(f"  crawl budget spent with {len(queue)} categories unvisited")
    return candidates


def search_openverse(fetcher: Fetcher) -> dict[str, dict]:
    """Openverse, filtered to licences that permit commercial use and modification.

    That filter is Openverse's own and it is not trusted on its own — every
    result still goes through licence_of() below. It only narrows what comes
    back, so the gate has less to refuse.

    One page per query rather than two, and the whole search gives up after
    three queries in a row come back empty-handed. Openverse's anonymous quota
    is much tighter than Commons': measured, a handful of calls exhausts it,
    after which every query burns the full retry ladder — 5+10+20+40+80+160 s —
    to learn nothing. Thirteen queries of that is an hour of sleeping. Openverse
    is the second source here, not the one the collection depends on; when it
    is shut, the right move is to notice and stop asking.
    """
    found: dict[str, dict] = {}
    empty_in_a_row = 0
    for query in OPENVERSE_QUERIES:
        if empty_in_a_row >= 3:
            print("  openverse is not answering; skipping the rest", flush=True)
            break
        for page in (1,):
            data = fetcher.api(OPENVERSE_API, {
                "q": query, "license_type": "commercial,modification",
                "page_size": 50, "page": page})
            if not data:
                empty_in_a_row += 1
                break
            empty_in_a_row = 0
            for item in data.get("results", []):
                url = item.get("url") or ""
                # Openverse indexes Commons, so most hits are files the crawl
                # already has. Keyed on the upload URL's basename, which is the
                # Commons title for those.
                key = "openverse:" + url
                if not url or key in found:
                    continue
                found[key] = {"source": "openverse", "title": item.get("title") or "",
                              "city": "", "found_under": f"openverse: {query}",
                              "item": item}
            if not data.get("page_count") or page >= data["page_count"]:
                break
        print(f"  [{len(found):4d}] openverse: {query}", flush=True)
    return found


# --------------------------------------------------------- priority and place

# The four sign-first seed categories are not a city, so files found through
# them start with no city. Most still name one somewhere — in the file title
# or in a subcategory on the way down — and a city is worth having in
# CREDITS.tsv: it is what tells a later reader whether this collection is four
# hundred photographs of Bosnia or four hundred photographs of Sarajevo.
CITIES = ["Sarajevo", "Mostar", "Banja Luka", "Tuzla", "Zenica", "Bihać", "Bihac",
          "Travnik", "Jajce", "Trebinje", "Bijeljina", "Doboj", "Prijedor",
          "Brčko", "Brcko", "Konjic", "Višegrad", "Visegrad", "Stolac", "Neum",
          "Livno", "Goražde", "Gorazde", "Srebrenica", "Počitelj", "Pocitelj"]


def city_of(entry: dict) -> str:
    if entry.get("city"):
        return entry["city"]
    haystack = f"{entry.get('title', '')} {entry.get('found_under', '')}"
    for city in CITIES:
        if city.lower() in haystack.lower():
            return city.replace("Bihac", "Bihać").replace("Brcko", "Brčko")
    return ""


def priority(entry: dict) -> int:
    haystack = f"{entry.get('title','')} {entry.get('found_under','')}".lower()
    haystack = unicodedata.normalize("NFKD", haystack)
    name = (entry.get("title") or "").lower()
    score = 0
    if any(word in name for word in SIGN_WORDS):
        score += 2
    if any(word in haystack for word in SIGN_WORDS):
        score += 1
    return score


# ----------------------------------------------------------------- filenames

_UNSAFE = re.compile(r"[^\w.\- ]", re.UNICODE)


def safe_name(title: str, fallback: str, prefix: str = "") -> str:
    name = title.split(":", 1)[-1] if title.lower().startswith("file:") else title
    name = _UNSAFE.sub("_", name).strip().replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    return (prefix + name[:120]) if name else (prefix + fallback)


def unused_name(name: str, taken: set) -> str:
    """Commons titles are unique; Openverse titles are not, and the two share
    this folder. A collision would overwrite a kept file while its credit row —
    naming a different photograph's licence and author — stayed behind."""
    stem, dot, ext = name.rpartition(".")
    candidate, n = name, 1
    while (candidate in taken or (OUT_DIR / candidate).exists()
           or (STAGING_DIR / candidate).exists()):
        n += 1
        candidate = f"{stem}-{n}{dot}{ext}" if dot else f"{name}-{n}"
    return candidate


# --------------------------------------------------------------------- state

def load_screened() -> dict[str, dict]:
    if not SCREENED_PATH.exists():
        return {}
    rows = {}
    with open(SCREENED_PATH, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["key"]] = row
    return rows


# `text` is what the screen actually read. It is the evidence for the verdict:
# without it, checking whether the filter is keeping signs or keeping brickwork
# means running the reader over the whole folder again.
SCREENED_FIELDS = ["key", "verdict", "reason", "licence", "regions", "diacritics",
                   "file", "text"]


def record(row: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    new = not SCREENED_PATH.exists()
    with open(SCREENED_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCREENED_FIELDS, delimiter="\t",
                                extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


CREDITS_FIELDS = ["file", "source_url", "license", "license_url", "attribution",
                  "city", "text_regions", "diacritics", "restrictions"]


def credit(row: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not CREDITS_PATH.exists()
    with open(CREDITS_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CREDITS_FIELDS, delimiter="\t",
                                extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


# ------------------------------------------------------------------ the run

def commons_metadata(fetcher: Fetcher, titles: list[str], width: int) -> dict[str, dict]:
    """imageinfo for up to 50 titles at a time — url, size, mime and licence."""
    data = commons_api(
        fetcher, prop="imageinfo", titles="|".join(titles),
        iiprop="url|size|mime|extmetadata", iiurlwidth=width,
        iiextmetadatafilter="LicenseShortName|License|LicenseUrl|Artist|Credit|"
                            "AttributionRequired|Restrictions")
    out: dict[str, dict] = {}
    if not data or "query" not in data:
        return out
    for page in data["query"].get("pages", []):
        if page.get("missing") or not page.get("imageinfo"):
            continue
        out[page["title"]] = page["imageinfo"][0]
    return out


def describe_commons(info: dict) -> dict:
    meta = info.get("extmetadata", {})

    def field(name):
        return plain(meta.get(name, {}).get("value", ""))

    licence = licence_of(field("LicenseShortName"), field("License"))
    artist = field("Artist") or field("Credit") or "unknown"
    return {
        "licence": licence,
        "licence_label": field("LicenseShortName") or field("License"),
        "licence_url": field("LicenseUrl"),
        "attribution": artist,
        "restrictions": field("Restrictions"),
        "page_url": info.get("descriptionurl", ""),
        "screen_url": info.get("thumburl") or info.get("url"),
        "keep_url": info.get("url"),
        "size": info.get("size") or 0,
        "width": info.get("width") or 0,
        "height": info.get("height") or 0,
        "mime": info.get("mime", ""),
    }


OPENVERSE_LICENCES = {"cc0": "CC0", "pdm": "PD", "by": "CC BY", "by-sa": "CC BY-SA"}


def describe_openverse(item: dict) -> dict:
    code = (item.get("license") or "").lower()
    version = item.get("license_version") or ""
    licence = OPENVERSE_LICENCES.get(code)
    label = {"CC0": "CC0", "PD": "Public domain"}.get(
        licence, f"CC {code.upper()} {version}".strip()) if licence else code
    creator = item.get("creator") or "unknown"
    return {
        "licence": licence,
        "licence_label": label,
        "licence_url": item.get("license_url") or "",
        "attribution": plain(item.get("attribution") or f"{creator} ({item.get('source','')})"),
        "restrictions": "",
        "page_url": item.get("foreign_landing_url") or item.get("url") or "",
        "screen_url": item.get("url") or "",
        "keep_url": item.get("url") or "",
        "size": 0,
        "width": item.get("width") or 0,
        "height": item.get("height") or 0,
        "mime": f"image/{(item.get('filetype') or 'jpeg').lower().replace('jpg','jpeg')}",
    }


def discover(fetcher: Fetcher, max_crawl_calls: int,
             with_openverse: bool = True) -> list[dict]:
    if CANDIDATES_PATH.exists():
        entries = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
        print(f"reusing {len(entries)} candidates from {CANDIDATES_PATH}")
        # Re-derived rather than trusted: the cached file may predate the
        # current city and priority rules, and a crawl costs half an hour to
        # repeat while these cost nothing.
        for entry in entries:
            entry["city"] = city_of(entry)
            entry["priority"] = priority(entry)
        entries.sort(key=lambda e: -e["priority"])
        return entries

    print("crawling Wikimedia Commons categories...")
    commons = crawl_categories(fetcher, max_crawl_calls)
    print("searching Openverse...")
    openverse = search_openverse(fetcher) if with_openverse else {}

    entries = list(commons.values()) + list(openverse.values())
    for entry in entries:
        entry["key"] = (entry["title"] if entry["source"] == "commons"
                        else "openverse:" + (entry["item"].get("url") or ""))
        entry["priority"] = priority(entry)
        entry["city"] = city_of(entry)
    # Highest priority first; within a band, a fixed shuffle rather than the
    # order the crawl happened to produce, so a run that stops early is not all
    # one city, one category or one uploader. Fixed seed: the same candidate
    # list twice, so a resumed run continues rather than starts somewhere new.
    random.Random(20260827).shuffle(entries)
    entries.sort(key=lambda e: -e["priority"])

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    print(f"{len(entries)} candidates -> {CANDIDATES_PATH}")
    return entries


# ------------------------------------------------------------------- staging
#
# The run is three phases rather than one loop, because the two costs are on
# different machines. Talking to Commons is capped at roughly one call every
# six seconds by their rate limiter and uses no CPU; reading an image costs
# ~9 s of CPU and no network. Interleaved, each spends its time waiting for the
# other. Split, the screening phase can run several readers at once while the
# network phases run flat out.

def load_staged() -> dict[str, dict]:
    if STAGED_PATH.exists():
        return json.loads(STAGED_PATH.read_text(encoding="utf-8"))
    return {}


def save_staged(staged: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STAGED_PATH.write_text(json.dumps(staged, ensure_ascii=False, indent=1),
                           encoding="utf-8")


def stage(fetcher: Fetcher, entries: list[dict], decided: dict, want: int,
          batch: int = 50) -> dict[str, dict]:
    """Fetch metadata, apply the licence gate, download what survives it.

    Nothing is downloaded before its licence is known. The gate runs on the
    imageinfo response, so a file with an unrecognised licence costs us one
    row in the log and no bytes at all.
    """
    staged = load_staged()
    pending = [e for e in entries
               if e["key"] not in decided and e["key"] not in
               {v["key"] for v in staged.values()}]
    print(f"staging: {len(staged)} already staged, {len(pending)} candidates left")

    for start in range(0, len(pending), batch):
        if len(staged) >= want:
            break
        chunk = pending[start:start + batch]
        titles = [e["title"] for e in chunk if e["source"] == "commons"]
        info = commons_metadata(fetcher, titles, SCREEN_WIDTH) if titles else {}

        for entry in chunk:
            if len(staged) >= want:
                break
            key = entry["key"]
            if entry["source"] == "commons":
                raw = info.get(entry["title"])
                if not raw:
                    record({"key": key, "verdict": "skip", "reason": "no imageinfo"})
                    decided[key] = {"verdict": "skip"}
                    continue
                meta = describe_commons(raw)
            else:
                meta = describe_openverse(entry["item"])

            # --- the licence gate, before a single byte is downloaded ---
            if not meta["licence"]:
                record({"key": key, "verdict": "refused",
                        "reason": f"licence not on the list: {meta['licence_label']!r}"})
                decided[key] = {"verdict": "refused"}
                continue
            if meta["mime"] not in ALLOWED_MIME:
                record({"key": key, "verdict": "skip",
                        "reason": f"not a photo format: {meta['mime']}"})
                decided[key] = {"verdict": "skip"}
                continue
            if meta["width"] and meta["height"] and (
                    meta["width"] * meta["height"] < MIN_PIXELS
                    or min(meta["width"], meta["height"]) < MIN_SHORT_SIDE):
                record({"key": key, "verdict": "skip",
                        "reason": f"too small: {meta['width']}x{meta['height']}"})
                decided[key] = {"verdict": "skip"}
                continue

            name = safe_name(entry["title"], f"image_{start:05d}",
                             "" if entry["source"] == "commons" else "ov_")
            if "." not in name[-6:]:
                name += "." + meta["mime"].split("/")[-1].replace("jpeg", "jpg")
            name = unused_name(name, set(staged))
            if not fetcher.download(meta["screen_url"], STAGING_DIR / name):
                record({"key": key, "verdict": "skip", "reason": "download failed"})
                decided[key] = {"verdict": "skip"}
                continue

            staged[name] = {"key": key, "meta": meta, "city": entry.get("city", ""),
                            "source": entry["source"]}
            if len(staged) % 25 == 0:
                save_staged(staged)
                print(f"  staged {len(staged)}/{want}", flush=True)
    save_staged(staged)
    print(f"staged {len(staged)} images for screening")
    return staged


# ------------------------------------------------------------ screening phase

def _worker_init(workers: int) -> None:
    # Each reader gets its share of the cores and no more. Left alone, torch
    # takes every core in every process, and two readers then spend their time
    # taking the same cores off each other rather than reading.
    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // max(workers, 1)))
    except Exception:
        pass
    from app import ocr
    ocr.get_reader()          # pay the 9 s model load once per worker, not per image


def _screen_worker(job: tuple[str, str]) -> tuple[str, dict]:
    # The path travels with the job rather than being read off a module global:
    # workers are spawned, not forked, so they re-import this file and would not
    # see a caller's staging directory.
    name, path = job
    # One unreadable file must not take the pool down and lose every verdict
    # that has not been written yet. A crash is a drop with its reason recorded.
    try:
        return name, screen(Path(path))
    except Exception as exc:
        return name, {"verdict": "drop", "reason": f"reader failed: {exc!r}"[:120],
                      "regions": 0, "diacritics": 0, "text": ""}


def screen_staged(staged: dict, decided: dict, target: int, max_screen: int,
                  workers: int, staging: Path = None) -> int:
    from multiprocessing import get_context

    staging = staging or STAGING_DIR
    todo = [(name, str(staging / name)) for name, item in staged.items()
            if item["key"] not in decided]
    kept = sum(1 for row in decided.values() if row.get("verdict") == "keep")
    screened = 0
    if not todo:
        return kept
    # Each worker loads its own reader, so the pool commits `workers` times what
    # claim() measured at startup. Asking again here is what stops two of them
    # from being started on a machine with room for one.
    from scripts.guard import workers_for
    workers = workers_for(1.4, workers, "readers")
    print(f"screening {len(todo)} staged images on {workers} reader(s)...")

    context = get_context("spawn")
    with context.Pool(workers, initializer=_worker_init, initargs=(workers,)) as pool:
        for name, result in pool.imap_unordered(_screen_worker, todo, chunksize=1):
            item = staged[name]
            screened += 1
            row = {"key": item["key"], "verdict": result["verdict"],
                   "reason": result["reason"], "licence": item["meta"]["licence"],
                   "regions": result["regions"], "diacritics": result["diacritics"],
                   "file": name, "text": result["text"][:300]}
            record(row)
            decided[item["key"]] = row
            if result["verdict"] == "keep":
                kept += 1
                mark = " *" if result["diacritics"] else ""
                print(f"  KEEP  {name[:56]:56s} {result['regions']:3d} regions{mark}"
                      f"  [{kept}/{target}]", flush=True)
            else:
                (staging / name).unlink(missing_ok=True)
                print(f"  drop  {name[:56]:56s} {result['reason']}", flush=True)
            if kept >= target or screened >= max_screen:
                print("  reached the target; stopping the readers")
                pool.terminate()
                break
    return kept


# ------------------------------------------------------------ collecting phase

def collect(fetcher: Fetcher, staged: dict, decided: dict, batch: int = 50) -> None:
    """Fetch a keeping-size copy of everything that passed, and credit it.

    A keeper needs more resolution than the screen used: training/
    prepare_ocr_data.py cuts word crops out of the file as it sits, not out of
    the 1 MP the screen worked at, so shipping the screening rendering would
    throw away the detail the crops are made of.

    It does not need the original either. The pilot's four keepers came to
    11 MB, one of them a 7 MB 12 MP JPEG, which extrapolates to about 800 MB
    for a full harvest — for pixels nothing downstream looks at, since the
    reader caps itself at 2 MP. So: a 2560 px rendering, which is four times
    the screening width and still well past what the reader uses, and the
    original only when Commons has nothing smaller to give.

    The renderings are asked for fifty titles at a time. One call per keeper
    would be three hundred rate-limited round trips — half an hour of waiting
    to save the same bytes.
    """
    credited = set()
    if CREDITS_PATH.exists():
        with open(CREDITS_PATH, newline="", encoding="utf-8") as handle:
            credited = {row["file"] for row in csv.DictReader(handle, delimiter="\t")}

    wanted = [(name, item) for name, item in staged.items()
              if decided.get(item["key"], {}).get("verdict") == "keep"
              and name not in credited]
    print(f"collecting {len(wanted)} images...")

    for start in range(0, len(wanted), batch):
        chunk = wanted[start:start + batch]
        titles = [item["key"] for _, item in chunk if item["source"] == "commons"]
        bigger = commons_metadata(fetcher, titles, KEEP_WIDTH) if titles else {}

        for name, item in chunk:
            meta = item["meta"]
            row = decided[item["key"]]
            final = OUT_DIR / name
            url = (bigger.get(item["key"], {}).get("thumburl")
                   if item["source"] == "commons" else None) or meta["keep_url"]
            if not fetcher.download(url, final):
                staged_copy = STAGING_DIR / name
                if staged_copy.exists():
                    staged_copy.replace(final)   # the screening rendering will do
                else:
                    print(f"  lost   {name}")
                    continue
            credit({"file": name, "source_url": meta["page_url"],
                    "license": meta["licence_label"] or meta["licence"],
                    "license_url": meta["licence_url"],
                    "attribution": meta["attribution"], "city": item.get("city", ""),
                    "text_regions": row["regions"], "diacritics": row["diacritics"],
                    "restrictions": meta["restrictions"]})
            (STAGING_DIR / name).unlink(missing_ok=True)
        print(f"  collected {min(start + batch, len(wanted))}/{len(wanted)}", flush=True)
    print(f"collected into {OUT_DIR}")


def run(target: int, max_screen: int, workers: int, max_crawl_calls: int,
        with_openverse: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher()
    entries = discover(fetcher, max_crawl_calls, with_openverse)

    decided = load_screened()
    kept = sum(1 for row in decided.values() if row["verdict"] == "keep")
    print(f"already decided: {len(decided)} ({kept} kept)")

    # Stage more than the target, because most candidates will be thrown out.
    # The multiplier is a guess before the first run and is corrected by the
    # measured pass rate on the next one; over-staging costs thumbnails, and
    # under-staging costs another rate-limited pass over the API.
    staged = stage(fetcher, entries, decided, want=max_screen)
    kept = screen_staged(staged, decided, target, max_screen, workers)
    collect(fetcher, staged, decided)

    # Whatever was staged but never read stays a candidate: its verdict row was
    # never written, so the next run re-stages and reads it.
    leftover = [name for name, item in staged.items()
                if item["key"] not in decided]
    for name in leftover:
        (STAGING_DIR / name).unlink(missing_ok=True)
        del staged[name]
    save_staged({name: item for name, item in staged.items()
                 if decided.get(item["key"], {}).get("verdict") == "keep"})

    # The staging area has to end up empty, not just unused. Downstream,
    # training/prepare_ocr_data.py walks --photos with rglob, so a screening
    # rendering left behind under .state/ would be cropped and labelled as
    # though it were one of the photographs that passed.
    for stray in STAGING_DIR.glob("*"):
        stray.unlink(missing_ok=True)
    STAGING_DIR.rmdir()
    if fetcher.throttled:
        print(f"(Commons throttled us {fetcher.throttled} time(s); "
              f"final gap {fetcher.gap:.1f}s)")
    report()


def report() -> None:
    done = load_screened()
    verdicts = Counter(row["verdict"] for row in done.values())
    print("\n--- screening ---")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict:9s} {count}")

    kept = [row for row in done.values() if row["verdict"] == "keep"]
    print(f"\n--- licences of the {len(kept)} kept ---")
    for licence, count in Counter(r["licence"] for r in kept).most_common():
        print(f"  {licence or '(BLANK — MUST BE ZERO)':9s} {count}")
    blank = sum(1 for r in kept if not r["licence"])
    print(f"  unclear licences among kept: {blank}")

    with_diacritics = sum(1 for r in kept if int(r["diacritics"] or 0) > 0)
    print(f"\ndiacritics (č ć đ š ž) found on: {with_diacritics} of {len(kept)}")

    files = [p for p in OUT_DIR.glob("*") if p.is_file() and p.name != "CREDITS.tsv"]
    total = sum(p.stat().st_size for p in files)
    print(f"on disk: {len(files)} files, {total / 1024 / 1024:.1f} MB")

    # The checks above read the run's own verdict log, which is the same thing
    # that decided to keep these files — so on its own it proves nothing. These
    # read the shipped folder instead, and re-derive the licence from the string
    # written into CREDITS.tsv rather than trusting the verdict recorded beside
    # it. A licence this cannot re-recognise is a file that must not ship.
    if CREDITS_PATH.exists():
        with open(CREDITS_PATH, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        missing = [r["file"] for r in rows
                   if not r["license"] or not r["attribution"] or not r["source_url"]]
        unrecognised = [r["file"] for r in rows if not licence_of(r["license"], "")]
        orphans = [r["file"] for r in rows if not (OUT_DIR / r["file"]).exists()]
        uncredited = [p.name for p in files if p.name not in {r["file"] for r in rows}]
        print(f"\nCREDITS.tsv: {len(rows)} rows")
        print(f"  rows missing licence, attribution or source page: {len(missing)}")
        print(f"  rows whose licence string is not one of the four: {len(unrecognised)}")
        print(f"  rows naming a file that is not here:              {len(orphans)}")
        print(f"  files here with no row:                           {len(uncredited)}")
        for label, names in (("missing", missing), ("unrecognised", unrecognised),
                             ("orphan", orphans), ("uncredited", uncredited)):
            if names:
                print(f"  {label}: " + ", ".join(names[:10]))
        by_licence = Counter(r["license"].split(",")[0] for r in rows)
        for label, count in by_licence.most_common():
            print(f"  {count:5d}  {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", type=int, default=400,
                        help="stop once this many images have passed the filter")
    parser.add_argument("--max-screen", type=int, default=1400,
                        help="stop after reading this many candidates, kept or not")
    parser.add_argument("--workers", type=int, default=2,
                        help="readers to run at once (each one wants ~1 GB)")
    parser.add_argument("--max-crawl-calls", type=int, default=220,
                        help="ceiling on Commons category calls; at ~6 s each "
                             "this is what bounds how long discovery takes")
    parser.add_argument("--no-openverse", action="store_true",
                        help="Commons only; use when Openverse is refusing calls")
    parser.add_argument("--survey", action="store_true",
                        help="discover candidates and stop")
    parser.add_argument("--report", action="store_true",
                        help="print the counts for what is already on disk")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        return 0
    if args.survey:
        entries = discover(Fetcher(), args.max_crawl_calls,
                           not args.no_openverse)
        bands = Counter(e["priority"] for e in entries)
        print("priority bands:", dict(sorted(bands.items(), reverse=True)))
        return 0

    claim(1.4, "photo harvest")
    run(args.target, args.max_screen, args.workers, args.max_crawl_calls,
        not args.no_openverse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
