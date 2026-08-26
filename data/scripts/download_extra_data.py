#!/usr/bin/env python3
"""Download Bosnian-English pairs the base translation model was never trained on.

WHY THIS EXISTS
---------------
The base model was trained on `opusTCv20210807` — a snapshot of OPUS taken on
7 August 2021. Everything download_data.py fetches (SETIMES v2, WikiMatrix v1,
TED2020 v1) is inside that snapshot, so fine-tuning on it mostly re-teaches the
model sentences it already knows. This script fetches material that is provably
outside it.

HOW "PROVABLY OUTSIDE IT" WAS ESTABLISHED
-----------------------------------------
The snapshot ships a file listing the origin of every single training pair:

    https://object.pouta.csc.fi/Tatoeba-Challenge-v2021-08-07/eng-hbs.tar
      -> data/release/v2021-08-07/eng-hbs/train.id.gz   (byte offset 4,087,258,112)

All 159,611,854 rows of it were read. The base model's Bosnian side comes from
exactly 44 corpora:

    OpenSubtitles-v2018 82,298,975   CCMatrix-v1     44,465,043
    ParaCrawl-v8        10,315,942   CCAligned-v1    10,151,407
    XLEnt-v1.1           3,638,099   DGT-v2019        2,076,197
    WikiMatrix-v1        1,004,991   JW300-v1c          989,285
    TildeMODEL-v2018       711,350   GNOME-v1           701,802
    SETIMES-v2             556,459   QED-v2.0a          483,867
    TED2020-v1             459,916   GoURMET-v1         322,247
    Tanzil-v1              240,594   KDE4-v2            226,445
    Mozilla-l10n-v1        206,618   bible-uedin-v1     119,932
    wikimedia-v20210402    115,410   hrenWaC-v1          97,241
    TedTalks-v1             84,399   Ubuntu-v14.10       82,650
    EuroPat-v2              73,502   MontenegrinSubs-v1  55,335
    EUbookshop-v2            8,706   KDEdoc-v1               24
    + 20 ELRA-W*/ELRC_* collections, each under 21,000

A source counts as unseen here only if it is absent from that list AND was
published after 7 August 2021. Both conditions, not one.

THE CATCH, STATED PLAINLY
-------------------------
Absence from the list proves the base never saw that *corpus*. It does not
prove it never saw those *sentences*: CCMatrix (44M), CCAligned (10M) and
ParaCrawl (10M) are all in the list, and they mine the same web pages that
later crawls mine again. So for the web-mined sources below, treat "unseen"
as "unseen as a corpus, with measured-but-incomplete evidence at the sentence
level" — the measurement is reported per source under EVIDENCE.

Usage:
    python3 download_extra_data.py                  # the two verified-clean sources
    python3 download_extra_data.py --list           # show every source and its evidence
    python3 download_extra_data.py --source hplt    # add the large web-mined one
    python3 download_extra_data.py --source nllb    # add the largest, noisiest one

Output goes to data/extra/, NOT data/raw/. That is deliberate: clean_data.py
globs data/raw/*/ and re-splits everything with a fixed seed, so a new corpus
landing there would silently move the test set and make every number measured
so far incomparable. Instead this writes

    data/extra/<Source>/<Source>.bs        one sentence per line
    data/extra/<Source>/<Source>.en        line N matches line N of the .bs
    data/extra/extra-train.tsv             corpus \t bosnian \t english

extra-train.tsv is in train.tsv's exact format and is already deduplicated
against train.tsv, valid.tsv, test.tsv and the FLORES benchmark, so it can be
appended to train.tsv directly:

    cat data/extra/extra-train.tsv >> data/clean/train.tsv
"""
import argparse
import io
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
EXTRA_DIR = DATA_DIR / "extra"
CLEAN_DIR = DATA_DIR / "clean"
FLORES_DIR = DATA_DIR / "flores"

OPUS = "https://object.pouta.csc.fi/OPUS-{corpus}/{version}/moses/bs-en.txt.zip"
NTREX = ("https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/"
         "NTREX-128/{name}")

# Corpora that ARE in the 44-corpus list above and can be fetched cheaply. Rows
# of a candidate that also appear here are rows the base model demonstrably saw,
# so they are dropped rather than counted as new.
SEEN_REFERENCE = [("wikimedia", "v20210402"), ("CCAligned", "v1")]


class Source:
    """One candidate, with the evidence that justifies downloading it."""

    def __init__(self, key, title, pairs, license_, published, evidence,
                 quality, default=False, heavy_mb=0, fetch=None):
        self.key, self.title, self.pairs = key, title, pairs
        self.license, self.published = license_, published
        self.evidence, self.quality = evidence, quality
        self.default, self.heavy_mb, self.fetch = default, heavy_mb, fetch


# --------------------------------------------------------------------------
# The sources. Every number below was measured, not estimated; the method is
# named next to it so it can be re-run and disagreed with.
# --------------------------------------------------------------------------
SOURCES = [
    Source(
        key="wikimedia", title="wikimedia v20260327 (Wikipedia content translations)",
        pairs="57,865 raw -> 44,771 new after removing the 2021 release",
        license_="CC-BY-SA 4.0 / GFDL (Wikipedia's own terms). Commercial use and "
                 "redistribution allowed; share-alike and attribution required.",
        published="2026-03-27",
        evidence="The list contains wikimedia-v20210402 (115,410 pairs) and nothing "
                 "newer. Measured: 10,568 of the 2026 release's rows are byte-identical "
                 "to rows of v20210402 and are dropped here; the other 44,771 are text "
                 "Wikipedia gained after the snapshot.",
        quality="Best of everything tested. 87.2% of dialect-marked lines carry the "
                "Bosnian ijekavian form, against 93.8% for FLORES and 94.1% for NTREX "
                "(both human-translated Bosnian) and 91-93% for the corpora already in "
                "use — i.e. as Bosnian as what we already train on, and far above the "
                "web-mined sources. 12 random new pairs read by hand: 11 correctly "
                "aligned, 1 reference-list junk.",
        default=True,
        fetch=lambda: opus_pairs("wikimedia", "v20260327"),
    ),
    Source(
        key="ntrex", title="NTREX-128 Bosnian reference (Microsoft)",
        pairs="1,997",
        license_="CC-BY-SA 4.0. Commercial use and redistribution allowed with "
                 "attribution and share-alike.",
        published="2022-11-24",
        evidence="Not an OPUS corpus at all, so it cannot be in the list — and it did "
                 "not exist until 15 months after the snapshot. The Bosnian side was "
                 "commissioned by Microsoft in 2022; the English side is WMT "
                 "newstest2019 source text, which appears in none of the 44 corpora.",
        quality="Human translation, not mining, and it doubles as the calibration "
                "control for the dialect filter here: 94.1% Bosnian-form, 1.65% false "
                "ekavian. Pairs read by hand are clean ijekavian Bosnian ('prijedloga', "
                "'promijeni'). News domain. Small, but the only professionally "
                "translated new material found.",
        default=True,
        fetch=lambda: ntrex_pairs(),
    ),
    Source(
        key="elrc-health", title="ELRC-3047 Wikipedia health (EU Language Resource Coordination)",
        pairs="205",
        license_="CC-BY 4.0 (ELRC-SHARE). Commercial use and redistribution allowed "
                 "with attribution.",
        published="ELRC-SHARE collection, added to OPUS after the snapshot",
        evidence="The list contains ELRC_2922, ELRC_2923, ELRC_3382 and ELRC_416. It "
                 "does not contain ELRC-3047. Measured: 0 of its 205 pairs match the "
                 "seen-corpus reference set.",
        quality="Tiny. 27 of 205 pairs (13%) already appear in our train.tsv via "
                "WikiMatrix and are dropped. Health domain, human-curated. Included "
                "for completeness — it will not move a metric.",
        default=False,
        fetch=lambda: opus_pairs("ELRC-3047-wikipedia_health", "v1"),
    ),
    Source(
        key="hplt", title="HPLT v3 bs-en (web bitext, High-Performance Language Technologies)",
        pairs="6,822,086",
        license_="CC0 on the packaging ONLY. Verbatim from hplt-project.org: 'We do "
                 "not own any of the text from which these text data has been "
                 "extracted.' No licence is granted to the underlying web text, and "
                 "the project runs a takedown procedure. Commercial use is left to the "
                 "user under the EU Copyright Directive / GDPR — this is NOT a clean "
                 "commercial grant, it is the standard web-crawl disclaimer.",
        published="v1 2023, v2 2025, v3 2025-11 (arXiv:2511.01066)",
        evidence="No HPLT version appears in the list, and none existed in 2021. But "
                 "HPLT is mined from Internet Archive and Common Crawl material, and "
                 "CCMatrix + CCAligned + ParaCrawl (65M pairs together) mine the same "
                 "web and ARE in the list. Measured on a 200,000-line sample against a "
                 "202k-pair reference of corpora the base did see: 0.05% exact-pair "
                 "overlap. That reference covers only 0.13% of the base's training "
                 "data, so 0.05% is a weak lower bound, not a clean bill of health.",
        quality="Big, and only partly Bosnian. 72.0% of dialect-marked lines use the "
                "ijekavian form against 93.8% for FLORES, with an ekavian rate of 3.67% "
                "against a 1.7% false-positive floor — so a real, if minority, share of "
                "what the language identifier labelled 'bs' is Serbian. Sampled pairs "
                "included a Bosnian news sentence aligned to an English site navigation "
                "bar. Note v2 is markedly worse than v3 on this measure (42.7% "
                "Bosnian-form, 11.88% ekavian); v3 is the one wired up here.",
        default=False, heavy_mb=627,
        fetch=lambda: opus_pairs("HPLT", "v3"),
    ),
    Source(
        key="nllb", title="NLLB v1 bs-en (Meta's mined bitext)",
        pairs="79,334,034",
        license_="ODC-BY 1.0 on the dataset. Commercial use and redistribution "
                 "allowed with attribution. NOTE: this is the DATA licence and it is "
                 "not the NLLB-200 model's CC-BY-NC licence — the project's standing "
                 "ban on NLLB is about the model weights and does not apply here.",
        published="2022-07",
        evidence="Absent from the list; published 11 months after the snapshot. Same "
                 "caveat as HPLT and worse: it is LASER-mined from Common Crawl, and "
                 "the head of the file is Quran text that duplicates Tanzil-v1, which "
                 "IS in the list (240,594 pairs). Measured on 200,000 lines: 0.02% "
                 "exact-pair overlap with the seen reference — again a weak bound.",
        quality="Noisiest of everything tested. 40.9% Bosnian-form against 93.8% for "
                "FLORES, with 11.96% ekavian against a 1.7% floor — predominantly "
                "Serbian text filed under 'bs'. 12.9% of pairs have a character-length "
                "ratio outside 0.5-2.0x and 0.55% are exact duplicates. Only worth "
                "touching with aggressive filtering.",
        default=False, heavy_mb=4870,
        fetch=lambda: opus_pairs("NLLB", "v1"),
    ),
    Source(
        key="translatewiki", title="translatewiki v2026-07-01 (software interface strings)",
        pairs="22,596 raw -> 18,920 unique",
        license_="Mixed free licences per upstream project (translatewiki.net "
                 "relicenses contributions; MediaWiki strings are GPL-2.0+/CC-BY-SA). "
                 "Redistribution allowed; the exact terms vary per string.",
        published="2026-07-01",
        evidence="Absent from the list. The list's software-localisation entries are "
                 "GNOME-v1, KDE4-v2, Mozilla-l10n-v1, Ubuntu-v14.10 and KDEdoc-v1 — "
                 "translatewiki is not among them.",
        quality="Cleanest dialect signal measured (93.9% Bosnian-form) but almost none "
                "of it is sentences: 'Napravi'/'Create', 'U redu'/'Okay'. 16.3% are "
                "exact duplicates and 1.5% are untranslated. Listed for honesty; "
                "training a sentence translator on button labels is unlikely to help "
                "and may teach it to emit fragments.",
        default=False,
        fetch=lambda: opus_pairs("translatewiki", "v2026-07-01"),
    ),
]
BY_KEY = {s.key: s for s in SOURCES}

# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def get(url: str, note: str = "", attempts: int = 3) -> bytes:
    """Fetch a URL, retrying transient failures.

    Without the retry a single timed-out socket makes a source report "0 rows
    read, 0 kept" and the run still ends with a summary table, which reads as
    "there was nothing there" rather than "the download broke".
    """
    if note:
        print(f"    {note}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "lilly-translator"})
    for attempt in range(1, attempts + 1):
        try:
            # without a timeout one stalled socket hangs an unattended run for hours
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retry anything, then give up loudly
            if attempt == attempts:
                raise
            print(f"    attempt {attempt}/{attempts} failed ({exc}), retrying…",
                  flush=True)
    raise RuntimeError("unreachable")


def opus_pairs(corpus: str, version: str):
    """Yield (bs, en) from an OPUS moses zip, streamed so a 4.9 GB one fits."""
    url = OPUS.format(corpus=corpus, version=version)
    blob = get(url, f"downloading {url}")
    print(f"    got {len(blob) / 1048576:.0f} MB", flush=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        bs_name = next(n for n in zf.namelist() if n.endswith(".bs"))
        en_name = next(n for n in zf.namelist() if n.endswith(".en"))
        with zf.open(bs_name) as fb, zf.open(en_name) as fe:
            tb = io.TextIOWrapper(fb, encoding="utf-8", errors="replace")
            te = io.TextIOWrapper(fe, encoding="utf-8", errors="replace")
            for bs, en in zip(tb, te):
                yield bs, en


def ntrex_pairs():
    bs = get(NTREX.format(name="newstest2019-ref.bos.txt"),
             "downloading NTREX-128 Bosnian reference").decode("utf-8")
    en = get(NTREX.format(name="newstest2019-src.eng.txt")).decode("utf-8")
    bs_lines, en_lines = bs.splitlines(), en.splitlines()
    if len(bs_lines) != len(en_lines):
        raise RuntimeError(f"NTREX sides disagree: {len(bs_lines)} vs {len(en_lines)}")
    yield from zip(bs_lines, en_lines)


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
WS = re.compile(r"\s+")

# Wikipedia carries its citation markers into both sides but renumbers them
# ([1][2] on one side, [6][7] on the other) while the sentence itself is a
# perfect translation. Strip them before comparing figures, or the number check
# throws away good data: measured on wikimedia v20260327, not stripping them
# flags 29.2% of pairs, stripping them flags 12.6%, and hand-reading 6 of the
# 9,583 pairs that difference covers found all 6 correctly aligned.
BRACKET_REF = re.compile(r"\[[^\]]{0,40}\]")
# Two digits or more: years and quantities. Bare single digits disagree far too
# often for innocent reasons (English spells "two", Bosnian writes "2").
BIG_NUMBER = re.compile(r"\d{2,}")

# Reflexes of Proto-Slavic yat. Bosnian and Croatian took the ijekavian form
# (mlijeko, vrijeme, djeca); Serbian took the ekavian one (mleko, vreme, deca).
# A line carrying only ekavian forms is Serbian text that a language identifier
# filed under "bs" — the commonest defect in web-mined South Slavic data, and
# invisible to every other filter here.
#
# Anchored at word start, and deliberately NOT containing "reka" or "sveta":
# those match inside "rekao"/"rekavši" ("said") and "sveti" ("holy"), which are
# ordinary Bosnian. With them in, this filter threw out 210 of NTREX's 1,997
# professionally translated Bosnian sentences. Calibration on text known to be
# Bosnian — FLORES 1.84% and NTREX 1.65% flagged as ekavian — puts the false
# positive rate of the list below at under 2%.
IJEKAVIAN = re.compile(
    r"\b(?:vrijem|mlijek|dijel|djec|vjer|mjest|mjesec|cvijet|lijep|rijek|"
    r"svijet|tijel|bijel|prije|poslije|htjel|vrijedn|uvijek|ovdje|gdje|"
    r"zvijezd|snijeg|nedjelj|razumij|dvije|obavijest|izvje[sš]t|primjer|"
    r"sjed|vje[cč]|cijel|cijen)")
EKAVIAN = re.compile(
    r"\b(?:vremen|mlek|deca|deci|decu|dece|de[cč]j|mesto|mestu|mesta|mesec|"
    r"cvet|lepo|lepa|telo|tela|belo|bela|posle|hteo|htela|vredn|uvek|ovde|"
    r"gde|zvezd|sneg|nedelj|razume|primer|obave[sš]t|izve[sš]t|dve|ceo |"
    r"cela|cena|cene|seo |sede)")


def normalize(text: str) -> str:
    return WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def key_of(bs: str, en: str):
    return (bs.lower(), en.lower())


def keep(bs: str, en: str, drops: Counter, dialect: str) -> bool:
    """Same filters clean_data.py applies, plus two this data needs."""
    if not bs or not en:
        drops["empty"] += 1
        return False
    bs_words, en_words = len(bs.split()), len(en.split())
    if bs_words > 250 or en_words > 250:
        drops["too_long"] += 1
        return False
    if bs_words >= 4 and en_words >= 4:
        if max(bs_words, en_words) / min(bs_words, en_words) > 2.5:
            drops["length_ratio"] += 1
            return False
    if CYRILLIC.search(bs):
        drops["cyrillic"] += 1
        return False
    for text in (bs, en):
        if len(LETTERS.findall(text)) < 0.3 * len(text):
            drops["mostly_symbols"] += 1
            return False
    if bs.lower() == en.lower():
        drops["untranslated"] += 1
        return False
    # Numbers are the cheapest misalignment detector there is: a translation
    # keeps its figures. Ten pairs this flags, read by hand, were all genuine
    # wreckage — reference-list fragments and template errors, not sentences.
    if bs_words >= 4:
        bs_nums = set(BIG_NUMBER.findall(BRACKET_REF.sub(" ", bs)))
        en_nums = set(BIG_NUMBER.findall(BRACKET_REF.sub(" ", en)))
        if bs_nums ^ en_nums:
            drops["digits_disagree"] += 1
            return False
    if dialect != "off":
        low = bs.lower()
        has_ije = bool(IJEKAVIAN.search(low))
        has_eka = bool(EKAVIAN.search(low))
        if has_eka and not has_ije:
            drops["serbian_form"] += 1
            return False
        if dialect == "strict" and not has_ije:
            drops["no_bosnian_marker"] += 1
            return False
    return True


# --------------------------------------------------------------------------
# what must not be duplicated
# --------------------------------------------------------------------------


def load_existing():
    """Everything a new pair must not be: our splits, FLORES, and the corpora
    the base model provably saw."""
    ours, flores = set(), set()

    for split in ("train", "valid", "test"):
        path = CLEAN_DIR / f"{split}.tsv"
        if not path.exists():
            print(f"  note: {path} missing, cannot deduplicate against it")
            continue
        n = 0
        for line in open(path, encoding="utf-8", errors="replace"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                ours.add(key_of(normalize(parts[1]), normalize(parts[2])))
                n += 1
        print(f"  {path.name}: {n:,} pairs")

    # FLORES is the honest benchmark. A pair leaking from it into training
    # would quietly invalidate every score the project reports.
    for split in ("devtest", "dev"):
        fb, fe = FLORES_DIR / f"{split}.bs", FLORES_DIR / f"{split}.en"
        if fb.exists() and fe.exists():
            for side in (fb, fe):
                for line in open(side, encoding="utf-8", errors="replace"):
                    flores.add(normalize(line).lower())
    if flores:
        print(f"  FLORES benchmark: {len(flores):,} protected sentences")
    else:
        print("  note: data/flores/ missing — run download_flores.py, otherwise "
              "benchmark sentences could leak into training")

    seen = set()
    for corpus, version in SEEN_REFERENCE:
        try:
            n = 0
            for bs, en in opus_pairs(corpus, version):
                seen.add(key_of(normalize(bs), normalize(en)))
                n += 1
            print(f"  {corpus}-{version}: {n:,} pairs the base model demonstrably saw")
        except Exception as exc:  # noqa: BLE001 - a missing reference weakens the
            # check but must not stop the run; say so loudly instead of silently
            # letting seen-by-base rows through.
            print(f"  WARNING: could not fetch {corpus}-{version} ({exc}). "
                  f"Rows the base already saw may survive.", file=sys.stderr)
    return ours, flores, seen


# --------------------------------------------------------------------------


def harvest(source, ours, flores, seen, dialect, limit):
    drops = Counter()
    kept, raw = [], 0
    within = set()
    for bs, en in source.fetch():
        raw += 1
        bs, en = normalize(bs), normalize(en)
        if not keep(bs, en, drops, dialect):
            continue
        k = key_of(bs, en)
        if k in within:
            drops["duplicate_in_source"] += 1
            continue
        within.add(k)
        if k in ours:
            drops["already_in_our_splits"] += 1
            continue
        if k in seen:
            drops["base_model_already_saw_it"] += 1
            continue
        if bs.lower() in flores or en.lower() in flores:
            drops["would_leak_flores"] += 1
            continue
        kept.append((bs, en))
        if limit and len(kept) >= limit:
            print(f"    stopping at --limit {limit:,}; "
                  f"{raw:,} rows read of {source.pairs}")
            break
    return raw, kept, drops


def write_out(name, pairs):
    out_dir = EXTRA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.bs", "w", encoding="utf-8") as fb, \
         open(out_dir / f"{name}.en", "w", encoding="utf-8") as fe:
        for bs, en in pairs:
            fb.write(bs + "\n")
            fe.write(en + "\n")
    return out_dir


def show_list():
    for s in SOURCES:
        flag = " [downloaded by default]" if s.default else ""
        heavy = f" [{s.heavy_mb:,} MB download]" if s.heavy_mb else ""
        print(f"\n=== {s.key}{flag}{heavy}")
        print(f"  {s.title}")
        print(f"  pairs      : {s.pairs}")
        print(f"  published  : {s.published}")
        print(f"  licence    : {s.license}")
        print(f"  EVIDENCE   : {s.evidence}")
        print(f"  QUALITY    : {s.quality}")
    print("\nEvery figure above was measured against the base model's own training "
          "manifest\n(159,611,854 rows, 44 corpora) and against samples pulled from "
          "each candidate.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="print every source with its licence and evidence, download nothing")
    ap.add_argument("--source", action="append", metavar="KEY", default=None,
                    help=f"download this source instead of the defaults; repeatable. "
                         f"one of: {', '.join(BY_KEY)}")
    ap.add_argument("--all", action="store_true",
                    help="every source, including the 5.5 GB of web-mined ones")
    ap.add_argument("--dialect", choices=("drop-serbian", "strict", "off"),
                    default="drop-serbian",
                    help="drop-serbian (default): remove lines whose only dialect "
                         "markers are Serbian. strict: keep only lines with an "
                         "explicit Bosnian/Croatian marker (high precision, discards "
                         "most lines). off: no dialect filtering")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N kept pairs per source (for the huge ones)")
    args = ap.parse_args()

    if args.list:
        show_list()
        return 0

    if args.all:
        chosen = list(SOURCES)
    elif args.source:
        unknown = [k for k in args.source if k not in BY_KEY]
        if unknown:
            print(f"unknown source(s): {', '.join(unknown)}\n"
                  f"available: {', '.join(BY_KEY)}", file=sys.stderr)
            return 2
        chosen = [BY_KEY[k] for k in args.source]
    else:
        chosen = [s for s in SOURCES if s.default]

    heavy = sum(s.heavy_mb for s in chosen)
    if heavy:
        print(f"NOTE: this will download about {heavy:,} MB. The web-mined sources "
              f"are large and,\nby the measurements in --list, substantially Serbian "
              f"rather than Bosnian.\n")

    print("Building the do-not-duplicate set:")
    ours, flores, seen = load_existing()
    print()

    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    combined, report = [], []
    for source in chosen:
        print(f"[{source.key}] {source.title}")
        try:
            raw, pairs, drops = harvest(source, ours, flores, seen,
                                        args.dialect, args.limit)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            print(f"    FAILED: {exc}\n", file=sys.stderr)
            report.append((source.key, 0, 0, "FAILED"))
            continue
        name = source.key
        out_dir = write_out(name, pairs)
        for bs, en in pairs:
            combined.append((name, bs, en))
            ours.add(key_of(bs, en))  # later sources must not repeat earlier ones
        detail = ", ".join(f"{k}={v:,}" for k, v in drops.most_common()) or "nothing dropped"
        print(f"    {raw:,} rows read -> {len(pairs):,} kept")
        print(f"    dropped: {detail}")
        print(f"    wrote {out_dir.relative_to(DATA_DIR.parent)}/{name}.bs and .en\n")
        report.append((name, raw, len(pairs), detail))

    if not combined:
        print("Nothing new was kept.")
        return 1

    out = EXTRA_DIR / "extra-train.tsv"
    with open(out, "w", encoding="utf-8") as fh:
        for name, bs, en in combined:
            fh.write(f"{name}\t{bs}\t{en}\n")

    print("=" * 72)
    for name, raw, kept, _ in report:
        print(f"  {name:16} {raw:>12,} read  {kept:>10,} kept")
    print(f"\n  {len(combined):,} new pairs -> {out.relative_to(DATA_DIR.parent)}")
    print("\nThese are deduplicated against train/valid/test and FLORES, so append "
          "them to\nthe training split without re-running clean_data.py (which would "
          "reshuffle the\ntest set and invalidate every score measured so far):")
    print(f"\n    cat {out.relative_to(DATA_DIR.parent)} >> data/clean/train.tsv\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
