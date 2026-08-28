#!/usr/bin/env python3
"""Download extra transcribed speech to widen Lilly's listening beyond FLEURS bs_ba.

data/speech holds 3,091 Bosnian training clips (9.99 h) and the speech model
sits at 35.5% WER on them -- the weakest part of the project. There is no
Bosnian speech corpus at meaningful scale to fix that with, so this script
brings in the closest thing that exists: Croatian.

Croatian, not "South Slavic". That distinction is the whole point of this
script and it was measured, not assumed. See MEASURED below.

    python3 data/scripts/download_extra_speech.py               # all three sources
    python3 data/scripts/download_extra_speech.py --source fleurs_hr
    python3 data/scripts/download_extra_speech.py --hours 20     # smaller pull
    python3 data/scripts/download_extra_speech.py --verify-only  # re-run the gates

Writes data/speech-extra/<source>/train/*.wav plus
data/speech-extra/<source>/train.tsv -- the exact "path<TAB>text" format
data/speech/train.tsv uses, with paths relative to the TSV, which is how
training/train_speech.py's read_tsv() resolves them.

--------------------------------------------------------------------------
MEASURED (2026-08-27, by dialect_report() below -- re-run it and disagree)
--------------------------------------------------------------------------
Two numbers per corpus, both from the IJEKAVIAN/EKAVIAN regexes in
download_extra_data.py, so they are directly comparable to the calibration
that file already carries:

  A = share of lines NOT flagged ekavian  (ekavian marker present, ijekavian
      marker absent). Real Bosnian prose scores 98-99.7%.
  B = among lines carrying any dialect marker, the share carrying the
      ijekavian one. FLORES bs 93.8%, NTREX bs 94.1% -- and because Bosnian
      reference text itself only reaches the mid-90s here, ~93-96% is the
      ceiling any proxy can be asked for. Higher is not "better than
      Bosnian", it is Croatian being more uniformly ijekavian than Bosnian is.

  our data/speech/train.tsv (FLEURS bs_ba)   A 98.06%   B 93.7%   <- control
  classla/ParlaSpeech-HR                     A 99.33%   B 97.7%   TAKEN
  facebook/voxpopuli hr                      A 98.77%   B 96.3%   TAKEN
  google/fleurs hr_hr                        A 98.37%   B 94.4%   TAKEN
  facebook/voxpopuli sl (Slovenian)          A 87.11%   B  0.7%   REJECTED
  classla/ParlaSpeech-RS (Serbian)           A 77.62%   B  0.0%   REJECTED
  google/fleurs sr_rs (Serbian)              -- Cyrillic --       REJECTED

The control line is the important one: it says the measurement is calibrated,
because it reproduces download_extra_data.py's published FLORES figures on
text this repo already holds.

Why the three Serbian/Slovenian sources are not here, since an earlier version
of this script offered two of them:

  * google/fleurs sr_rs is written in CYRILLIC ("Судија је рекао Блејку да је
    'скоро неизбежно'"). data/speech/train.tsv is Latin throughout, and
    download_extra_data.py's own keep() drops Cyrillic outright. Feeding it to
    a Whisper fine-tune would teach the decoder a second script for the same
    sounds. B reads 0.0% only because Latin regexes cannot match Cyrillic --
    that is a "not applicable", not a score.
  * classla/ParlaSpeech-RS is Latin, and genuinely ekavian: 22.4% of its lines
    are flagged, against 1.9% for our Bosnian, and of 179 dialect-marked lines
    in the sample, ZERO carried the ijekavian form. Serbian says vreme, mleko,
    deca where Bosnian says vrijeme, mlijeko, djeca. An ASR model trained on it
    learns to emit the ekavian spelling, which is a word error on every yat
    reflex in the Bosnian test set. The previous version of this file offered
    it as --lang sr; that was the wrong call and this is the measurement that
    says so.
  * facebook/voxpopuli sl is Slovenian -- a different language, not a Bosnian
    variant ("V tem duhu se zahvaljujem poročevalcem v senci").

Mozilla Common Voice is not here either, and not for want of trying. Checked
2026-08-27, by fetching the repos rather than trusting the docs:
  * huggingface.co/datasets/mozilla-foundation/common_voice_17_0 (and 13_0)
    are public and NOT gated, but contain exactly two files: .gitattributes
    and README.md. The README says data moved to Mozilla Data Collective
    effective October 2025. The audio is gone from Hugging Face; there is no
    endpoint to fix, no auth to add. common_voice_16_1 and 11_0 404 entirely.
  * Even with access it would not reach this script's target: Croatian is not
    a Common Voice language at all, and Serbian has ~12 h (arXiv:2409.15397,
    the ParlaSpeech paper, makes both points).
  * The community mirrors on the Hub are per-language forks of other
    languages (Bengali, Urdu, Kinyarwanda...); none carries bs, hr or sr.

Two lookalike "Bosnian speech" datasets are also not new data, and are worth
naming so nobody re-finds them: shunyalabs/bosnian-speech-dataset has exactly
3,091/400/925 clips -- byte-for-byte the split counts of google/fleurs bs_ba,
i.e. a re-upload of what download_speech_data.py already fetches. And
Speech-data/Bosnian-Speech-Dataset advertises 169 h but ships one sample MP3
behind a CC BY-NC-ND licence: a storefront, not a corpus.

--------------------------------------------------------------------------
TRANSCRIPT PROVENANCE -- none of this is machine-transcribed, but read on
--------------------------------------------------------------------------
All three carry human-written text, which is the reason these three and not,
say, a pile of YouTube auto-captions:

  * fleurs_hr    read speech: the speaker was given the sentence to read, so
                 the transcript is exact by construction.
  * voxpopuli_hr the European Parliament's official record, with an
                 is_gold_transcript flag on the subset they verified.
  * parlaspeech  the ParlaMint official record of the Croatian Sabor. ASR
                 (Wav2Vec2-XLS-R) appears in their pipeline ONLY to locate
                 where in the audio each sentence of the official record
                 falls -- the words come from the record, not the model
                 (arXiv:2409.15397).

The honest caveat is on the last one, and it applies to any parliamentary
corpus: an official record is edited for readability, so it is not always a
literal transcript of the disfluencies and repetitions in the audio. The
ParlaSpeech authors filter on exactly this, dropping segments whose character
error rate against the aligned ASR output reaches 10%; their Croatian yield
was 74%. So what survives is human text within <10% CER of what was actually
said -- excellent for a corpus of this size, but not the word-perfect match
that fleurs_hr's read speech gives you. That is part of why fleurs_hr is kept
at its full 11.5 h despite being the most expensive source per byte.

--------------------------------------------------------------------------
LICENCES -- they differ, and one of them has a condition
--------------------------------------------------------------------------
  google/fleurs hr_hr      CC BY 4.0     attribution
  facebook/voxpopuli hr    CC0 1.0 AND "other" -- NOT "no conditions", see below
  classla/ParlaSpeech-HR   CC BY-SA 4.0  attribution + SHARE-ALIKE

This file used to say voxpopuli was "CC0 1.0, no conditions". That was read off
the headline licence tag and it is wrong. Checked 2026-08-28 by fetching the
dataset card rather than trusting it: the Hugging Face card carries TWO licence
tags, `cc0-1.0` and `other`, and its Licensing Information section says "The
dataset is distributed under CC0 license, see also European Parliament's legal
notice for the raw data." The legal notice it points at
(europarl.europa.eu/legal-notice/en) says reuse of European Union material is
authorised "for personal use or for further non-commercial or commercial
dissemination, provided that the entire item is reproduced and the source is
acknowledged", with the acknowledgement in the form "(c) European Union,
[year(s)] - Source: European Parliament", and the user undertakes "not to delete
or change the indications of the author or the source".

So the transcription layer is CC0 and the underlying recordings carry an
attribution condition. Not a blocker -- attribution is cheap and CREDITS.tsv
already carries per-source rows -- but "no conditions" was a claim nobody had
checked, and the difference between CC0 and CC0-plus-attribution is exactly the
kind of thing that is discovered after publication rather than before. The
"entire item is reproduced" clause is about redistributing EP material, which
training weights is not, but that reading is the OWNER's to make and not this
file's.

Share-alike is the one to think about before shipping: anything you
distribute that was built from ParlaSpeech-HR is expected to carry the same
licence. That is why each source keeps its own directory instead of being
poured into one pool -- so you can still tell, later, which clips came from
where, and drop just that one if the licence becomes inconvenient:

    python3 data/scripts/download_extra_speech.py --source fleurs_hr voxpopuli_hr

--------------------------------------------------------------------------
WHAT THIS IS NOT
--------------------------------------------------------------------------
Everything written here goes into a *train* split. There is deliberately no
extra valid or test split: valid and test stay real Bosnian
(data/speech/valid.tsv, data/speech/test.tsv). A Croatian test set sitting
next to a Bosnian one is a benchmark waiting to be quoted as if it measured
Bosnian, and this project has already been bitten once by a metric born with
leakage. Train on the mix, score on the real thing:

    python3 data/scripts/download_extra_speech.py
    python3 training/train_speech.py --data data/speech-extra/train.tsv \\
        --valid data/speech/valid.tsv

data/speech-extra/train.tsv is written with ABSOLUTE paths and already
includes data/speech/train.tsv's own 3,091 Bosnian clips, so it is the one
file to point --data at. The per-source TSVs next to it use relative paths to
match data/speech/train.tsv's format exactly. Note that `cat`-ing relative-path
TSVs together into /tmp silently breaks every path in them, because read_tsv()
resolves relative to the TSV's own directory -- hence the absolute-path
aggregate rather than a suggestion to cat.

--------------------------------------------------------------------------
LEAKAGE
--------------------------------------------------------------------------
Nothing is written until every candidate line has been checked against
data/speech/test.tsv and data/speech/valid.tsv three ways: exact match after
normalisation, exact match after folding the yat reflex (so Croatian
"vrijeme" and Bosnian "vreme" collide instead of sliding past each other),
and token-Jaccard near-duplicate search. The yat fold matters specifically
because FLEURS bs_ba and hr_hr are both built from FLORES: the same source
sentences translated into both languages. Different splits, so they should
not meet -- but "should not" is not a measurement, and folded matching is
what turns it into one. Hits are dropped and counted, not warned about.
"""
import argparse
import io
import json
import re
import sys
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
SPEECH_DIR = REPO_ROOT / "data" / "speech"
OUT_DIR = REPO_ROOT / "data" / "speech-extra"
LISTING = "https://datasets-server.huggingface.co/parquet?dataset={repo}&config={config}"
SAMPLE_RATE = 16_000

# The dialect regexes live in download_extra_data.py and are calibrated there
# against FLORES and NTREX. Importing rather than copying keeps this script's
# numbers comparable to that file's published ones forever.
sys.path.insert(0, str(SCRIPTS_DIR))
from download_extra_data import IJEKAVIAN, EKAVIAN  # noqa: E402

# Refuse to start if the machine has no room. Five of these ran at once on
# 27 August and the kernel panicked: 100% of the compressor limit, fifteen
# swapfiles, watchdog silent for 94 seconds. Each job is reasonable alone and
# none of them knew the others existed.
#
# Claimed in main(), where the harvest starts, and not here at import. At import
# it also blocked --verify-only, which downloads nothing, decodes one clip per
# source and re-runs the leakage gate on text: a check that costs nothing was
# refused by the memory guard on the machine it was written to protect. The same
# mistake was in training/train_speech.py and training/evaluate_speech.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.guard import claim  # noqa: E402

SOURCES = {
    "fleurs_hr": dict(
        repo="google/fleurs", config="hr_hr", split="train",
        text_cols=("raw_transcription", "transcription"),
        license="CC BY 4.0",
        note="Croatian half of the same corpus family as our Bosnian data -- "
             "read speech, studio-clean, identical pipeline.",
        default_hours=12.0,   # = all 3,461 train clips (~11.5 h); ~224 MB/h -> 2.5 GB
    ),
    "voxpopuli_hr": dict(
        repo="facebook/voxpopuli", config="hr", split="train",
        text_cols=("raw_text", "normalized_text"),
        license="CC0 1.0 + EP legal notice (attribution)",
        note="European Parliament floor speeches, official human record. "
             "Spontaneous speech, which read-speech FLEURS has none of.",
        default_hours=8.0,    # ~172 MB/h upstream -> ~1.4 GB streamed
    ),
    "parlaspeech_hr": dict(
        repo="classla/ParlaSpeech-HR", config="default", split="train",
        text_cols=("text",), duration_col="audio_length",
        license="CC BY-SA 4.0 (SHARE-ALIKE)",
        note="Croatian Sabor, aligned to the official ParlaMint record. "
             "Human-written transcripts; ASR was used only to locate each "
             "sentence in the audio, never to produce the words.",
        default_hours=20.0,   # ~56 MB/h upstream -> ~1.1 GB streamed. Cheapest
                              # hours per byte of the three, and the best
                              # ijekavica score, so it carries the most weight.
    ),
}

# ---------------------------------------------------------------------------
# text normalisation, for leakage checking
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
# Proto-Slavic yat: Bosnian/Croatian ijekavian "ije"/"je" against ekavian "e".
# Folding both to "e" makes vrijeme/vreme, mlijeko/mleko, djeca/deca collide,
# which is what lets a leakage check see through a bs/hr translation pair.
_YAT = re.compile(r"ije|je")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def fold(text: str) -> str:
    return _YAT.sub("e", norm(text))


def clean_text(text: str) -> str:
    """Collapse whitespace, preserving casing and punctuation.

    Not cosmetic: train_speech.py's read_tsv() keeps a row only when the line
    splits into EXACTLY two tab-separated fields, so a single stray tab inside
    a parliamentary transcript would silently delete that clip from training
    with no error anywhere. A newline would corrupt the TSV outright. Casing
    and punctuation stay, for the same reason download_speech_data.py keeps
    raw_transcription: this text reaches the translator and the reader.
    """
    return _WS.sub(" ", (text or "")).strip()


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def list_shards(repo: str, config: str, split: str) -> list:
    url = LISTING.format(repo=repo.replace("/", "%2F"), config=config)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    files = [f for f in data.get("parquet_files", []) if f["split"] == split]
    if not files:
        raise SystemExit(
            f"no '{split}' shards listed for {repo}/{config}. The dataset moved "
            f"or was emptied -- check https://huggingface.co/datasets/{repo} "
            f"before assuming this is a network problem.")
    return files


class RemoteParquet(io.RawIOBase):
    """A read-only seekable file over HTTP Range, so pyarrow can read a remote
    parquet shard a row group at a time.

    This exists because of a measurement. The link here tops out near 1 MB/s
    (measured single-stream and 4-way parallel -- four streams together moved no
    more than one, so it is the pipe, not the client), which makes transferred
    bytes the entire cost of this script. Downloading whole shards the way
    download_speech_data.py does would mean pulling a 2.1 GB VoxPopuli shard to
    keep a third of it, and a 398 MB ParlaSpeech shard to keep 2% of it.

    Parquet is columnar and row-group addressed, so a range-reading file object
    turns "give me 8 hours" into roughly 8 hours' worth of transfer and lets the
    loop stop the moment the budget is met. Each read_row_group() is one large
    contiguous range request, so this is not death by a thousand round trips.
    """

    def __init__(self, url: str):
        self.url, self._pos, self.bytes_read = url, 0, 0
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "lilly-translator"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") == "none":
                raise SystemExit(f"{url} refuses range requests")

    def seek(self, off, whence=0):
        self._pos = (off if whence == 0
                     else self._pos + off if whence == 1 else self.size + off)
        return self._pos

    def tell(self):
        return self._pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        if n <= 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + n, self.size) - 1
        for attempt in range(1, 6):
            try:
                req = urllib.request.Request(self.url, headers={
                    "Range": f"bytes={self._pos}-{end}",
                    "User-Agent": "lilly-translator"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except Exception as exc:
                if attempt == 5:
                    raise
                print(f"        range read retry {attempt} "
                      f"({type(exc).__name__}: {exc})", flush=True)
                time.sleep(3 * attempt)
        self._pos += len(data)
        self.bytes_read += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def decode_clip(raw: bytes) -> np.ndarray | None:
    """Decode whatever container the source used; return float32 mono @ 16 kHz."""
    try:
        audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        return None
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, SAMPLE_RATE, rate).astype("float32")
    return audio


# ---------------------------------------------------------------------------
# the leakage gate
# ---------------------------------------------------------------------------
class LeakGate:
    """Refuses any line that resembles something in data/speech/{test,valid}.tsv.

    Three independent tests, because one is not enough for two languages this
    close: exact, yat-folded exact, and token-Jaccard near-duplicate. The
    inverted index keeps the near-dup test linear enough to run on every
    candidate rather than on a sample.
    """

    def __init__(self, jaccard: float = 0.60):
        self.jaccard = jaccard
        self.exact, self.folded = set(), set()
        self.tokens, self.index = [], defaultdict(list)
        self.counts = {"exact": 0, "folded": 0, "near": 0, "cross_source": 0}
        self.worst = []          # (score, candidate, matched held-out line)
        # Cross-source de-duplication only. NOT within-source: FLEURS and the
        # parliamentary corpora both record the same sentence from several
        # speakers (fleurs hr_hr is 1.56x, our own data/speech/train.tsv is
        # 2.09x), and those repeats are different audio -- exactly the acoustic
        # variety a 35.5% WER model is short of. Dropping them threw away 40%
        # of FLEURS' clips on the first run. train.tsv keeps its repeats; so do
        # we.
        self.seen = set()        # transcripts taken by EARLIER sources
        self.current = set()     # transcripts taken by the source in flight

        held = []
        for name in ("test", "valid"):
            path = SPEECH_DIR / f"{name}.tsv"
            if not path.exists():
                raise SystemExit(
                    f"{path} is missing -- refusing to run without something to "
                    f"check leakage against. Run download_speech_data.py first.")
            for line in path.open(encoding="utf-8"):
                if "\t" in line:
                    held.append(line.split("\t", 1)[1].strip())
        for text in held:
            self.exact.add(norm(text))
            self.folded.add(fold(text))
            toks = set(fold(text).split())
            i = len(self.tokens)
            self.tokens.append((toks, text))
            for t in toks:
                self.index[t].append(i)
        self.n_held = len(held)
        print(f"  leakage gate armed on {self.n_held:,} held-out Bosnian lines "
              f"(test.tsv + valid.tsv)")

    def rejects(self, text: str) -> bool:
        n, f = norm(text), fold(text)
        if not n:
            return True
        if n in self.seen or f in self.seen:
            self.counts["cross_source"] += 1
            return True
        if n in self.exact:
            self.counts["exact"] += 1
            return True
        if f in self.folded:
            self.counts["folded"] += 1
            return True
        toks = set(f.split())
        if toks:
            hits = defaultdict(int)
            for t in toks:
                for i in self.index.get(t, ()):
                    hits[i] += 1
            for i, shared in hits.items():
                other, raw = self.tokens[i]
                j = shared / len(toks | other)
                if j >= self.jaccard:
                    self.counts["near"] += 1
                    self.worst.append((j, text, raw))
                    return True
                if j >= 0.40:
                    self.worst.append((j, text, raw))
        self.current.add(n)
        self.current.add(f)
        return False

    def start_source(self) -> None:
        self.current = set()

    def finish_source(self) -> None:
        self.seen |= self.current
        self.current = set()

    def report(self) -> None:
        total = sum(self.counts.values())
        print(f"\n  leakage gate: {total:,} candidate line(s) refused "
              f"({self.counts['exact']} exact, {self.counts['folded']} yat-folded, "
              f"{self.counts['near']} near-duplicate, "
              f"{self.counts['cross_source']} already taken by another source)")
        near = sorted((w for w in self.worst), key=lambda w: -w[0])[:3]
        if near:
            print("  closest surviving/refused pairs against held-out Bosnian:")
            for j, cand, raw in near:
                print(f"    jaccard {j:.2f}\n      extra: {cand[:88]}\n      held: {raw[:88]}")
        else:
            print("  no candidate reached even jaccard 0.40 against held-out Bosnian")


# ---------------------------------------------------------------------------
# dialect measurement
# ---------------------------------------------------------------------------
def dialect_report(lines: list) -> dict:
    n = len(lines)
    flagged = marked = ije = 0
    for text in lines:
        low = text.lower()
        i, e = bool(IJEKAVIAN.search(low)), bool(EKAVIAN.search(low))
        if e and not i:
            flagged += 1
        if i or e:
            marked += 1
            ije += bool(i)
    return dict(n=n, flagged=flagged, marked=marked,
                A=100 * (1 - flagged / n) if n else 0.0,
                B=100 * ije / marked if marked else 0.0)


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
def harvest(key: str, hours: float, min_sec: float, max_sec: float,
            gate: "LeakGate", max_shards: int) -> dict | None:
    """Pull clips until the hours budget is met, writing each one as it decodes.

    Deliberately NOT "collect into a list, then write". 12 h of 16 kHz float32
    audio is 2.76 GB of live numpy on a machine with 8 GB, and the budgets here
    add up to ~40 h. Clips go to disk the moment they are decoded; what stays in
    memory is one Arrow batch plus the transcripts, which are a few MB.
    """
    cfg = SOURCES[key]
    shards = list_shards(cfg["repo"], cfg["config"], cfg["split"])
    total_mb = sum(f["size"] for f in shards) / 1048576
    print(f"\n{key}: {cfg['repo']} ({cfg['config']}/{cfg['split']}), "
          f"{len(shards)} shard(s), {total_mb:,.0f} MB upstream")
    print(f"  licence: {cfg['license']}")
    print(f"  target: {hours:.1f} h of {min_sec:.0f}-{max_sec:.0f}s clips")

    import pyarrow.parquet as pq

    out = OUT_DIR / key
    clips_dir = out / "train"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for stale in clips_dir.glob("*.wav"):
        stale.unlink()

    gate.start_source()
    want = hours * 3600
    n, secs, transferred = 0, 0.0, 0
    texts = []
    dropped = {"short_or_long": 0, "undecodable": 0, "empty": 0}
    t0 = time.time()
    tsv = out / "train.tsv"

    with open(tsv, "w", encoding="utf-8") as fh:
        for shard_i, entry in enumerate(shards):
            if secs >= want or shard_i >= max_shards:
                break
            try:
                handle = RemoteParquet(entry["url"])
                reader = pq.ParquetFile(handle)
            except Exception as exc:
                print(f"      shard {shard_i + 1} unreadable, skipping: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            names = reader.schema_arrow.names
            text_col = next((c for c in cfg["text_cols"] if c in names), None)
            if text_col is None:
                raise SystemExit(
                    f"no transcript column in {names} (looked for "
                    f"{cfg['text_cols']}) -- {cfg['repo']}'s schema changed")
            dur_col = cfg.get("duration_col")
            if dur_col and dur_col not in names:
                dur_col = None
            cols = [c for c in ("audio", text_col, dur_col) if c]
            print(f"      shard {shard_i + 1}/{len(shards)}: "
                  f"{reader.metadata.num_rows:,} rows, "
                  f"{reader.num_row_groups} row group(s), streaming", flush=True)

            last_report = 0.0
            # batch_size 32 keeps the Python-object copy small; Arrow still
            # decodes a row group at a time, which is the real memory floor.
            for batch in reader.iter_batches(batch_size=32, columns=cols):
                if secs >= want:
                    break
                for row in batch.to_pylist():
                    if secs >= want:
                        break
                    said = clean_text(row.get(text_col))
                    if not said:
                        dropped["empty"] += 1
                        continue
                    # cheap rejects before the expensive audio decode
                    hinted = row.get(dur_col) if dur_col else None
                    if hinted is not None and not (min_sec <= hinted <= max_sec):
                        dropped["short_or_long"] += 1
                        continue
                    if gate.rejects(said):
                        continue
                    raw = (row.get("audio") or {}).get("bytes")
                    if not raw:
                        dropped["undecodable"] += 1
                        continue
                    audio = decode_clip(raw)
                    if audio is None:
                        dropped["undecodable"] += 1
                        continue
                    actual = len(audio) / SAMPLE_RATE
                    if not (min_sec <= actual <= max_sec):
                        dropped["short_or_long"] += 1
                        continue
                    n += 1
                    name = f"{n:05d}.wav"
                    # PCM_16, not FLOAT: train_speech.py's load_audio() passes
                    # dtype="float32" so soundfile converts on read either way,
                    # and 32-bit floats would double 40 h of audio from ~4.6 GB
                    # to ~9.2 GB for no gain anywhere in the pipeline.
                    sf.write(clips_dir / name, audio, SAMPLE_RATE, subtype="PCM_16")
                    fh.write(f"train/{name}\t{said}\n")
                    texts.append(said)
                    secs += actual
                    del audio
                # Flush every batch. The previous version flushed every half
                # hour of audio, and when the run was interrupted 36 wavs had
                # been written whose transcripts were still sitting in the
                # buffer -- clips on disk that no TSV line pointed at.
                fh.flush()
                if secs - last_report >= 1800:      # progress line every half hour
                    last_report = secs
                    mb = (transferred + handle.bytes_read) / 1048576
                    print(f"        {n:,} clips, {secs / 3600:.2f}/{hours:.1f} h, "
                          f"{mb:,.0f} MB pulled, "
                          f"{mb / max(time.time() - t0, 1):.2f} MB/s", flush=True)
            transferred += handle.bytes_read

    gate.finish_source()
    if n == 0:
        print(f"  {key}: no usable clips.", file=sys.stderr)
        return None

    size_mb = sum(p.stat().st_size for p in clips_dir.glob("*.wav")) / 1048576
    d = dialect_report(texts)
    meta = dict(source=key, repo=cfg["repo"], config=cfg["config"],
                license=cfg["license"], clips=n, hours=round(secs / 3600, 3),
                disk_mb=round(size_mb, 1), transferred_mb=round(transferred / 1048576, 1),
                ijekavica_A=round(d["A"], 2), ijekavica_B=round(d["B"], 1),
                dialect_marked_lines=d["marked"])
    (out / "SOURCE.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  dropped on the way: {dropped}")
    print(f"  wrote {n:,} clips / {secs / 3600:.2f} h / {size_mb:,.0f} MB on disk "
          f"(pulled {transferred / 1048576:,.0f} MB) -> {tsv}")
    print(f"  ijekavica: A {d['A']:.2f}%  B {d['B']:.1f}% "
          f"(on {d['marked']:,} dialect-marked lines)")
    return meta


def reconcile_partial(key: str) -> dict | None:
    """Make a source left behind by an interrupted run usable, or clear it.

    A run that dies mid-source leaves clips and a train.tsv but no SOURCE.json.
    The TSV is the source of truth -- every line in it was flushed after its wav
    was written -- so any wav beyond the last TSV line is an orphan with no
    transcript, and goes. What remains is a smaller but entirely valid corpus,
    which matters when the bytes cost an hour to fetch and re-downloading them
    buys nothing.
    """
    out = OUT_DIR / key
    tsv = out / "train.tsv"
    if not tsv.exists() or (out / "SOURCE.json").exists():
        return None

    rows = [l for l in tsv.read_text(encoding="utf-8").splitlines()
            if l.count("\t") == 1 and l.split("\t")[1].strip()]
    keep = {r.split("\t")[0] for r in rows}
    orphans = [p for p in (out / "train").glob("*.wav")
               if f"train/{p.name}" not in keep]
    for p in orphans:
        p.unlink()
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    secs = sum(sf.info(str(out / r.split("\t")[0])).duration for r in rows)
    size_mb = sum(p.stat().st_size for p in (out / "train").glob("*.wav")) / 1048576
    d = dialect_report([r.split("\t")[1] for r in rows])
    meta = dict(source=key, repo=SOURCES[key]["repo"], config=SOURCES[key]["config"],
                license=SOURCES[key]["license"], clips=len(rows),
                hours=round(secs / 3600, 3), disk_mb=round(size_mb, 1),
                transferred_mb=None, partial=True,
                ijekavica_A=round(d["A"], 2), ijekavica_B=round(d["B"], 1),
                dialect_marked_lines=d["marked"])
    (out / "SOURCE.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  {key}: recovered an interrupted run -- {len(rows):,} clips / "
          f"{secs / 3600:.2f} h kept, {len(orphans)} orphaned wav(s) removed")
    return meta


def write_aggregate(metas: list) -> Path:
    """One absolute-path TSV: the Bosnian train split plus every extra source.

    Absolute, because read_tsv() resolves relative paths against the TSV's own
    directory -- so a relative-path TSV that pools clips from several trees
    cannot exist, and `cat`-ing the per-source ones elsewhere would break them.
    """
    agg = OUT_DIR / "train.tsv"
    n = 0
    with open(agg, "w", encoding="utf-8") as out:
        base = SPEECH_DIR / "train.tsv"
        if base.exists():
            for line in base.open(encoding="utf-8"):
                if "\t" not in line:
                    continue
                rel, text = line.rstrip("\n").split("\t", 1)
                out.write(f"{(SPEECH_DIR / rel).resolve()}\t{text}\n")
                n += 1
        for meta in metas:
            src = OUT_DIR / meta["source"]
            for line in (src / "train.tsv").open(encoding="utf-8"):
                if "\t" not in line:
                    continue
                rel, text = line.rstrip("\n").split("\t", 1)
                out.write(f"{(src / rel).resolve()}\t{text}\n")
                n += 1
    print(f"\naggregate: {n:,} rows -> {agg}")
    return agg


# ---------------------------------------------------------------------------
# verification -- run through the trainer's own loader, not a lookalike
# ---------------------------------------------------------------------------
def verify(metas: list) -> bool:
    """Prove the output is loadable by importing train_speech.py's real reader.

    Re-implementing "read the TSV, open the wav" here would only prove that
    this file agrees with itself. Importing read_tsv/load_audio proves the
    thing that actually matters: the trainer can open what we wrote.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    sys.path.insert(0, str(REPO_ROOT / "training"))
    try:
        from train_speech import read_tsv, load_audio, SAMPLE_RATE as TRAIN_SR
    except Exception as exc:
        print(f"  could not import training/train_speech.py: {exc}")
        return False
    print(f"  using training/train_speech.py's own read_tsv() and load_audio(), "
          f"SAMPLE_RATE={TRAIN_SR}")

    ok = True
    agg = OUT_DIR / "train.tsv"
    rows = read_tsv(agg)
    print(f"  read_tsv({agg.name}) -> {len(rows):,} rows")
    missing = [p for p, _ in rows[:: max(1, len(rows) // 400)] if not Path(p).exists()]
    if missing:
        print(f"  MISSING {len(missing)} file(s), e.g. {missing[0]}")
        ok = False
    else:
        print(f"  spot-checked {len(rows[:: max(1, len(rows) // 400)]):,} sampled "
              f"paths: all present on disk")

    for meta in metas:
        tsv = OUT_DIR / meta["source"] / "train.tsv"
        rows = read_tsv(tsv)
        path, text = rows[0]
        info = sf.info(str(path))
        audio = load_audio(Path(path))
        print(f"\n  --- {meta['source']}: opened {Path(path).name} ---")
        print(f"      sf.info : {info.samplerate} Hz, {info.channels} channel(s), "
              f"{info.subtype}, {info.frames:,} frames, {info.duration:.2f} s")
        print(f"      load_audio(): shape {audio.shape}, dtype {audio.dtype}, "
              f"{len(audio) / TRAIN_SR:.2f} s, "
              f"peak {float(np.abs(audio).max()):.3f}, "
              f"RMS {float(np.sqrt((audio ** 2).mean())):.4f}")
        print(f"      transcript : {text[:96]}")
        if info.samplerate != TRAIN_SR or info.channels != 1:
            print("      FAIL: not 16 kHz mono")
            ok = False
        if audio.ndim != 1 or audio.dtype != np.float32:
            print("      FAIL: loader did not return 1-D float32")
            ok = False
        if float(np.abs(audio).max()) == 0.0:
            print("      FAIL: clip is digital silence")
            ok = False

    # the leakage claim, re-measured from what is actually on disk
    print("\n  --- leakage, re-measured from the written TSVs ---")
    held_n, held_f = set(), set()
    for name in ("test", "valid"):
        for line in (SPEECH_DIR / f"{name}.tsv").open(encoding="utf-8"):
            if "\t" in line:
                t = line.split("\t", 1)[1].strip()
                held_n.add(norm(t))
                held_f.add(fold(t))
    for meta in metas:
        lines = [l.split("\t", 1)[1].strip()
                 for l in (OUT_DIR / meta["source"] / "train.tsv").open(encoding="utf-8")
                 if "\t" in l]
        e = sum(1 for t in lines if norm(t) in held_n)
        f_ = sum(1 for t in lines if fold(t) in held_f)
        print(f"      {meta['source']}: {len(lines):,} lines, "
              f"{e} exact and {f_} yat-folded matches against "
              f"{len(held_n):,} held-out Bosnian lines")
        if e or f_:
            ok = False
    print("\n  RESULT:", "PASS" if ok else "FAIL")
    return ok


def summarise(metas: list) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'source':<16}{'clips':>8}{'hours':>8}{'disk MB':>10}"
          f"{'A%':>8}{'B%':>7}  licence")
    tc = th = td = 0
    for m in metas:
        print(f"{m['source']:<16}{m['clips']:>8,}{m['hours']:>8.2f}{m['disk_mb']:>10,.0f}"
              f"{m['ijekavica_A']:>8.2f}{m['ijekavica_B']:>7.1f}  {m['license']}")
        tc += m["clips"]; th += m["hours"]; td += m["disk_mb"]
    print(f"{'TOTAL':<16}{tc:>8,}{th:>8.2f}{td:>10,.0f}")
    print(f"\nagainst what we had: data/speech/train.tsv is 3,091 clips / 9.99 h")
    if th:
        print(f"  -> {tc / 3091:.1f}x the clips, {th / 9.99:.1f}x the hours")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", default=sorted(SOURCES),
                    choices=sorted(SOURCES),
                    help="which sources to pull (default: all three)")
    ap.add_argument("--hours", type=float, default=None,
                    help="hours per source; overrides the per-source budgets "
                         "(12 + 8 + 20, ~40 h total)")
    ap.add_argument("--min-sec", type=float, default=2.0)
    ap.add_argument("--max-sec", type=float, default=25.0,
                    help="Whisper's window is 30 s; our Bosnian clips average "
                         "11.6 s and reach 25 s (default 25.0)")
    ap.add_argument("--max-shards", type=int, default=8,
                    help="stop after this many shards per source even if the "
                         "hours budget is unmet (default 8)")
    ap.add_argument("--jaccard", type=float, default=0.60,
                    help="near-duplicate threshold for the leakage gate")
    ap.add_argument("--force", action="store_true",
                    help="re-download a source that already has a train.tsv")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip downloading; re-run verification on what is on disk")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        metas = []
        for key in sorted(SOURCES):
            p = OUT_DIR / key / "SOURCE.json"
            if p.exists():
                metas.append(json.loads(p.read_text(encoding="utf-8")))
        if not metas:
            print("nothing on disk to verify.", file=sys.stderr)
            return 1
        summarise(metas)
        return 0 if verify(metas) else 1

    claim(0.8, "speech download")

    gate = LeakGate(args.jaccard)
    metas = []
    for key in args.source:
        if not args.force:
            reconcile_partial(key)
        existing = OUT_DIR / key / "SOURCE.json"
        if existing.exists() and not args.force:
            print(f"\n{key}: already downloaded, keeping it (use --force to redo)")
            meta = json.loads(existing.read_text(encoding="utf-8"))
            # keep the gate aware of it, so later sources don't duplicate it
            for line in (OUT_DIR / key / "train.tsv").open(encoding="utf-8"):
                if "\t" in line:
                    t = line.split("\t", 1)[1].strip()
                    gate.seen.add(norm(t)); gate.seen.add(fold(t))
            metas.append(meta)
            continue
        hours = args.hours if args.hours is not None else SOURCES[key]["default_hours"]
        meta = harvest(key, hours, args.min_sec, args.max_sec, gate, args.max_shards)
        if meta is None:
            continue
        metas.append(meta)

    if not metas:
        print("nothing written.", file=sys.stderr)
        return 1

    gate.report()
    write_aggregate(metas)
    summarise(metas)
    return 0 if verify(metas) else 1


if __name__ == "__main__":
    raise SystemExit(main())
