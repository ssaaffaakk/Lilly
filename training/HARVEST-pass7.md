# OCR pass-7 — data collection (started 29 Aug 2026)

Owner signed off autonomous collection. No Flickr (Pro required). NARA deferred
(free API key signup at archives.gov — not yet in `.env`).

## ETA (8 GB Mac, serial, plugged in)

| Step | ~duration |
|---|---|
| Commons categories | 45–90 min |
| OSM sign text | 15–45 min |
| 50k synthetic | 2–3 h |
| Photo harvest (500 kept, OCR screen) | 6–12 h |
| 20k photo-style synthetic | 3–5 h |
| **Total** | **~12–20 h** from step 1 start |

## Watchdog

`scripts/harvest_watch.sh` — **double-fork + new session** (`scripts/harvest_detach.py`)
so Cursor/agent shells cannot kill the job. `nohup` alone is not enough: harvest
was staying in the agent's process group (`pgid`) and dying with no `FAIL` line.

Every **5 min** runs `harvest_report.py` and restarts only when status is `STALLED`
or `FAILED` (exit code 1). Does **not** restart if the lockdir is held.

Start: `./scripts/harvest_watch.sh` (returns immediately; job detaches).

Honest status (`harvest_report.py`):

| Status | Meaning |
|---|---|
| `RUNNING` | Counter moved or log/heartbeat fresh |
| `WAITING` | Process up, counters flat (Overpass slot wait — normal) |
| `STALLED` | No process, stale heartbeat, or duplicate orchestrators |
| `COMPLETE` | `=== pipeline complete ===` in log |

Metrics ledger: `logs/harvest/metrics.json` (counter snapshots for delta checks).

## Pipeline

Serial script: `scripts/harvest_pass7.sh` → log `logs/harvest/pass7.log`

| Step | Script | API key? |
|---|---|---|
| 1 | `fetch_commons_categories.py` | No |
| 2 | `harvest_sign_text.py` | No (Overpass) |
| 3 | `generate_ocr_data.py --count 50000` | No |
| 4 | `harvest_sign_photos.py` (500 target, 1 worker) | No |
| 5 | `generate_ocr_photos.py --count 20000` | No |

## Baseline (before pass-7)

| Asset | Count |
|---|---|
| harvested/ CREDITS | 287 |
| commons-cats/ CREDITS | 53 |
| sign-text.tsv rows | 9,505 |
| synthetic train | ~20k |

## After collection

- Package crops + CREDITS for Kaggle dataset `lilly-ocr-crops-v2`
- Launch OCR pass-7 notebook when speech v16 fetch/install done
- Update `docs/WHITE-PAPER.md` section 3 with sources + dates

## Photo storage (white paper / book)

- Binaries: `data/ocr/real-photos/{harvested,commons-cats,scored}/`
- Attribution: `CREDITS.tsv` per folder (in git)
- Scored test set: `scored-sources.tsv` + `truth.json` (in git)
