# Clean-evaluation audit — 2026-09-17

What current artifacts and clean datasets exist, what can run where, and the
plan for fresh measurement. The original audit was at `5fef177`; this update
records the owner-authorized eval jobs. This file is the audit;
fresh numbers land in the `RESULTS-fresh-*` files, kept separate from the
`RESULTS-*` files that hold older stored-output runs.

## 1. Current model artifacts (fingerprints = md5 of the weight files)

| ability | build | fingerprint | base of |
|---|---|---|---|
| Translate bs→en (shipped) | `models/lilly/translator` | `1aedcc11231cdf50817ff12f99ff0d1e` | fine-tuned int8 CT2 |
| Translate bs→en base | `models/lilly/translator-base` | `348a984c324510cee218dfce8a7228e8` | untuned int8 CT2 |
| Translate en→bs (shipped) | `models/lilly/translator-en-bs` | `6f240bb14aa56ea7ae1c8a19cb25faab` | adapter md5 `c28a02c6…` |
| Translate en→bs base | `models/lilly/translator-en-bs-base` | `6809c7a1665b9fd0246e6374cebcce52` | untuned int8 CT2 |
| Listen (shipped, owner override) | `models/lilly/listen` | `e6bb58483586b06c` | whisper-large-v3 int8 |
| Listen (gated baseline) | `models/lilly/listen-previous` | `a76342f6ab59b382` | whisper-small int8 |
| Read (shipped) | PaddleOCR PP-OCRv6 medium, `lang=bs`, floor 0.9 | via `app/ocr.py` | untrained |

The shipped translate fingerprints **match the published ones** — the live
weights are the ones the 43.25 / 30.73 BLEU numbers were taken on.

## 2. Clean datasets present

| ability | dataset | provenance | size |
|---|---|---|---|
| Translate | FLORES-200 dev+devtest | `data/scripts/download_flores.py` → NLLB `flores200_dataset.tar.gz` | 997 + 1,012 = 2,009 pairs |
| Listen | `data/speech/test.tsv` | FLEURS `bs_ba` test, held out | 925 clips |
| Read (product) | 40 Commons photographs | Wikimedia Commons, 2 human transcribers, 91.2% agreement | 28 with text |
| Read (robustness) | test-v2 | Wikimedia Commons pool | 132 with text (contains illegible / no-text photos — robustness only) |

## 3. What can run where

- **Translate**: the local full run was stopped and its partial output removed.
  A 40–80 minute model evaluation is heavy inference and belongs on Kaggle; it
  is independent of, and does not delay, these two authorized jobs.
- **Listen**: whisper-large-v3 int8 needs a GPU to transcribe 925 clips in reasonable time; faster-whisper not installed locally. **→ Kaggle.**
- **Read**: PaddleOCR (paddle) is a heavy install and reads 40 original photographs. **→ Kaggle**, through the real `app.ocr` path.

## 4. Fresh evaluation — protocol (no cherry-picking)

**Translate** (not part of these launches): a future Kaggle job may run
`evaluate_app.py --direction {bs-en,en-bs} --split all --fresh` through
`app.translate.Engine`. The stopped local partial is not evidence. FLORES is
uniformly clean professional text, so the clean/noisy/unreadable split does not
apply there.

**Listen** (Kaggle, owner authorized): the committed manifest freezes each clip
by audio and transcript SHA-256 before decoding. The official held-out FLEURS
read-speech set is the clean cohort (925); noisy-but-understandable and
human-unintelligible remain explicit zero rows rather than manufacturing noisy
audio. `probe_speech_clips.py` records SNR, bandwidth, clipping and duration as
diagnostic metadata only; those measurements never assign human legibility.
WER runs through `app.speech.transcribe` per cohort, alongside the two registered
gates (Bosnian-term recall and Croatian substitution). The exact shipped build
and baseline are fingerprint checked. The Open ASR normalizer/scorer is also
run offline on the same predictions, as supplementary non-submission evidence.

**Read** (Kaggle, owner authorized): the 40 original-resolution Commons files
are pinned by page revision, original URL, byte count, dimensions and SHA-1.
Visual classification was frozen before fresh inference: 21 clean, 6
noisy-but-understandable, 1 human-unintelligible and 12 no-text controls. Recall
and invented words run through shipped `app.ocr.scan` by cohort and overall.
test-v2 stays a **separate robustness** result, not a replacement.

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
- **Listen**: the current Open ASR Leaderboard repository is pinned for an
  offline supplementary score. Its current public multilingual dataset matrix
  does **not** register Bosnian, so the result is explicitly not called an
  official leaderboard submission. No predictions or weights are uploaded.
- **Read**: no standard Bosnian scene-text OCR leaderboard exists.

## 7. Owner decisions

1. Both Kaggle eval jobs are authorized: Listen and Read, eval-only and fail-stop.
2. Open ASR is offline comparison only. No public submission or upload.
3. Nothing here changes a product default, publishes, installs, or retrains.
