# Clean-evaluation audit — 2026-09-17

What current artifacts and clean datasets exist, what can run where, and the
plan for fresh measurement. `origin/main` at `5fef177`. This file is the audit;
fresh numbers land in the `RESULTS-fresh-*` files, kept separate from the
`RESULTS-*` files that hold older stored-output runs.

## 1. Current model artifacts (fingerprints = md5 of the weight files)

| ability | build | fingerprint | base of |
|---|---|---|---|
| Translate bs→en (shipped) | `models/lilly/translator` | `1aedcc11231cdf50817ff12f99ff0d1e` | fine-tuned int8 CT2 |
| Translate bs→en base | `models/lilly/translator-base` | `348a984c324510cee218dfce8a7228e8` | untuned int8 CT2 |
| Translate en→bs (shipped) | `models/lilly/translator-en-bs` | `6f240bb14aa56ea7ae1c8a19cb25faab` | adapter md5 `c28a02c6…` |
| Translate en→bs base | `models/lilly/translator-en-bs-base` | `6809c7a1665b9fd0246e6374cebcce52` | untuned int8 CT2 |
| Listen (shipped, owner override) | `models/lilly/listen` | whisper-large-v3 int8 | — |
| Listen (gated baseline) | `models/lilly/listen-previous` | whisper-small int8 | — |
| Read (shipped) | PaddleOCR PP-OCRv6 medium, `lang=bs`, floor 0.9 | via `app/ocr.py` | untrained |

The shipped translate fingerprints **match the published ones** — the live
weights are the ones the 43.25 / 30.73 BLEU numbers were taken on.

## 2. Clean datasets present

| ability | dataset | provenance | size |
|---|---|---|---|
| Translate | FLORES-200 dev+devtest | `data/scripts/download_flores.py` → NLLB `flores200_dataset.tar.gz` | 997 + 1,012 = 2,009 pairs |
| Listen | `data/speech/test.tsv` | FLEURS bs + Common Voice, held out | 925 clips |
| Read (product) | 40 Commons photographs | Wikimedia Commons, 2 human transcribers, 91.2% agreement | 28 with text |
| Read (robustness) | test-v2 | Wikimedia Commons pool | 132 with text (contains illegible / no-text photos — robustness only) |

## 3. What can run where

- **Translate**: int8 CTranslate2 weights present, ctranslate2 now installed. ~0.35 s/sentence on this CPU → a full fresh both-direction run is ~40–80 min, no GPU. Device drift Mac↔T4 is already proven negligible (`RESULTS-product.md`). **Runs LOCALLY, fresh, now** (in progress).
- **Listen**: whisper-large-v3 int8 needs a GPU to transcribe 925 clips in reasonable time; faster-whisper not installed locally. **→ Kaggle.**
- **Read**: PaddleOCR (paddle) not installed, install is heavy, reads 172 photos. **→ Kaggle**, through the real `app.ocr` path.

## 4. Fresh evaluation — protocol (no cherry-picking)

**Translate** (running): `evaluate_app.py --direction {bs-en,en-bs} --split all
--fresh` re-translates every FLORES pair through `app.translate.Engine` (the real
product path), scores with sacrebleu BLEU + chrF2, and stamps the build
fingerprint onto the output. FLORES is uniformly clean professional text, so the
clean/noisy/unreadable split does not apply here — every pair is legible.

**Listen** (Kaggle, pending approval): pre-tag every test clip by intelligibility
**before** decoding — clean / noisy-but-intelligible / unintelligible — using a
recorded rule (energy-band + human spot-check), then WER through `app.speech.
transcribe` per category, plus the two registered gates (Bosnian-term recall,
Croatian-substitution) from `bench/speech/terms.tsv` and `contrast-terms.tsv`.
Unintelligible clips reported separately, never folded into a "model is bad".

**Read** (Kaggle, pending approval): pre-label photos clean / noisy / unreadable
by human legibility **before** the reader runs, then recall + invented words per
category through `app.ocr`. The 40 Commons stays the product number; test-v2
stays a **separate robustness** result, not a replacement.

## 5. Training-data quality (point 8)

- **Translate**: SETIMES / TED2020 / Tatoeba / WikiMatrix, 337,790 pairs after
  filtering (empty, length-ratio, Cyrillic-on-Bosnian, symbols, untranslated,
  dupes). Reputable public corpora. No data-quality reason to retrain.
- **Listen**: FLEURS bs (3,091) + FLEURS hr + voxpopuli_hr, all clean recordings.
  Open issue is the Croatian-substitution leak, not dirty audio.
- **Read**: the only clean trainable data is `crops/labels-human-latin.tsv` =
  738 human-labelled Latin crops. Every `mapillary-train*` set is self-labelled
  and CLOSED (passes 14–19, do-not-relaunch). So a clean-data OCR retrain has
  ~738 crops and a closed queue — the real lever is Cyrillic recogniser coverage
  (config), not retraining the Latin model.

## 6. External leaderboards (point 7)

- **Translate**: no public *submission* leaderboard exists for FLORES-200; it is a
  dataset + sacrebleu, scored offline (which is exactly what runs here). Reported
  NLLB-200 baselines for Bosnian: chrF **bs→en 70.89 / en→bs 67.72** — but those
  are **chrF++ (word_order=2)**, while this repo reports **chrF2 (word_order=0)**;
  the two are different metrics and must not be compared head-to-head. BLEU is
  comparable if the sacrebleu signature matches (recorded in the fresh run).
- **Listen**: the **Open ASR Leaderboard** (HuggingFace / arXiv 2510.06961) is
  reproducible and documented, multilingual track includes bs/hr. Published
  Whisper FLEURS references: large-v2 **bs 29.7% / hr 27.0% WER**. Comparing our
  large-v3 number to these is fair *as a reference*; submitting our own model
  would mean running their harness and uploading a private fine-tune — **needs
  owner approval, no upload without it**.
- **Read**: no standard Bosnian scene-text OCR leaderboard exists.

## 7. Decisions needed from the owner

1. Approve the two Kaggle eval jobs (Listen, Read) — GPU, reproducible, fail-stop.
2. External submission: keep as offline comparison only (no upload), or authorise
   uploading predictions/weights to the Open ASR Leaderboard harness.
3. Nothing here changes a product default, publishes, installs, or retrains.
