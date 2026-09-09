#!/usr/bin/env python3
"""Who speaks how much in ParlaSpeech-HR, without downloading the audio.

The dataset is 179 GB in 326 parquet shards; the voice line needs a few
speakers' worth of clean segments, chosen by a rule written down before any
training. DuckDB reads only the metadata columns of every shard over HTTP
range requests, in parallel -- about five seconds a shard where pyarrow over a
Python file object took minutes -- and one pass returns every row's speaker,
gender, length, id and whether its transcript is clean. Written to one JSON:
every speaker's total and clean hours, the shard URLs and sizes the scan saw
(so the box can check it fetches the same files), and for the largest
speakers the exact segments (shard, row, id, seconds).

Clean, for a voice: 3-20 s long (20 s is the batch-memory cap from the FLEURS
line), a non-empty normalised transcript of at most 300 characters, no digit
(the speaker said a number in words the phonemizer cannot be trusted to
reproduce), no bracket (a parliamentary comment). Rows whose speaker is "-"
(unattributed) are counted and never selected.

    python3 data/scripts/parlaspeech_speakers.py                 # all 326 shards, ~30 min
    python3 data/scripts/parlaspeech_speakers.py --shards 3      # a look
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from download_extra_speech import list_shards  # noqa: E402

REPO, CONFIG, SPLIT = "classla/ParlaSpeech-HR", "default", "train"
OUT = SCRIPTS_DIR.parents[1] / "data" / "speech-extra" / "parlaspeech-hr-speakers.json"
MIN_SEC, MAX_SEC, MAX_CHARS = 3.0, 20.0, 300
CLEAN_SQL = ("audio_length BETWEEN {lo} AND {hi} AND length(text_normalised) BETWEEN 1 AND {chars} "
             "AND NOT regexp_matches(text_normalised, '[0-9\\[\\]()]')")
TOP = 40   # speakers whose segments are listed
UNKNOWN = "-"


CHUNK = 10        # shards per query: 8 threads over 326 files at once drew HTTP 429 at shard 35
CACHE = OUT.parent / ".parla-scan"


def scan(urls: list, threads: int = 4) -> list:
    """Every row's metadata, shard chunk by shard chunk, cached per chunk so a
    rate limit or a dropped link costs one chunk, not the half hour before it."""
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET threads TO {threads};")
    con.execute("SET http_retries = 6; SET http_retry_wait_ms = 3000; SET http_retry_backoff = 2;")
    clean = CLEAN_SQL.format(lo=MIN_SEC, hi=MAX_SEC, chars=MAX_CHARS)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(urls), CHUNK):
        part = urls[start:start + CHUNK]
        cached = CACHE / f"chunk-{start:04d}-{len(part)}.json"
        if cached.is_file():
            rows.extend(tuple(r) for r in json.loads(cached.read_text(encoding="utf-8")))
            continue
        for attempt in range(1, 7):
            try:
                got = con.execute(f"""
                    SELECT filename, file_row_number, id, audio_length, speaker_name, speaker_gender,
                           ({clean}) AS clean
                    FROM read_parquet($urls, filename=true, file_row_number=true)
                """, {"urls": part}).fetchall()
                break
            except duckdb.HTTPException as exc:
                if attempt == 6:
                    raise
                wait = 30 * attempt
                print(f"  shards {start}-{start + len(part) - 1}: {str(exc)[:80]} -- waiting {wait} s", flush=True)
                time.sleep(wait)
        cached.write_text(json.dumps(got), encoding="utf-8")
        rows.extend(got)
        print(f"  shards {start + len(part)}/{len(urls)}: {len(rows):,} rows so far", flush=True)
    return rows


def build(rows: list, shards: list) -> dict:
    index = {e["url"]: i for i, e in enumerate(shards)}
    total, clean_secs, n_clean = collections.Counter(), collections.Counter(), collections.Counter()
    gender, segments = {}, collections.defaultdict(list)
    for filename, row, seg_id, secs, spk, gen, clean in rows:
        spk = spk or UNKNOWN
        secs = float(secs or 0.0)
        total[spk] += secs
        gender.setdefault(spk, gen)
        if clean:
            clean_secs[spk] += secs
            n_clean[spk] += 1
            segments[spk].append((index[filename], int(row), seg_id, round(secs, 2)))
    ranked = [s for s, _ in clean_secs.most_common() if s != UNKNOWN]
    return {"repo": REPO, "shards_scanned": len(shards), "rows": len(rows),
            "shard_urls": [e["url"] for e in shards], "shard_sizes": [e["size"] for e in shards],
            "rule": {"min_sec": MIN_SEC, "max_sec": MAX_SEC, "max_chars": MAX_CHARS,
                     "no_digits": True, "no_brackets": True, "unattributed_excluded": True},
            "unattributed_hours": round(total[UNKNOWN] / 3600, 2),
            "speakers": [{"name": s, "gender": gender.get(s), "hours_total": round(total[s] / 3600, 2),
                          "hours_clean": round(clean_secs[s] / 3600, 2), "clean_segments": n_clean[s]}
                         for s in ranked],
            "segments": {s: sorted(segments[s]) for s in ranked[:TOP]}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=0, help="first N shards only (0 = all)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    shards = list_shards(REPO, CONFIG, SPLIT)
    if args.shards:
        shards = shards[: args.shards]
    t0 = time.time()
    print(f"scanning {len(shards)} shards of {REPO} with DuckDB ...", flush=True)
    rows = scan([e["url"] for e in shards], args.threads)
    print(f"  {len(rows):,} rows in {time.time() - t0:.0f} s", flush=True)
    rec = build(rows, shards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{rec['rows']:,} rows in {len(shards)} shards; {len(rec['speakers'])} named speakers; "
          f"unattributed {rec['unattributed_hours']} h; "
          f"clean hours over named speakers: {sum(s['hours_clean'] for s in rec['speakers']):.1f}")
    print(f"{'speaker':<32} {'gender':<7} {'total h':>8} {'clean h':>8} {'segments':>9}")
    for s in rec["speakers"][:25]:
        print(f"{s['name']:<32} {str(s['gender']):<7} {s['hours_total']:>8.1f} {s['hours_clean']:>8.1f} {s['clean_segments']:>9}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
