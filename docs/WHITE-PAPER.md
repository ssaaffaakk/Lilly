# Lilly: A Multimodal Bosnian–English Translation System

**Safak Surmeli**
September 2026

---

## Abstract

Lilly is a multimodal translation system for the Bosnian–English language pair
that accepts typed text, spoken audio, and photographed signs as input and
produces translations in both text and speech. It runs entirely offline on
consumer hardware, with no external API calls at inference time. The system
bundles five fine-tuned or selected models under a single FastAPI server and a
browser-based interface.

On the FLORES-200 benchmark the translator scores 43.25 BLEU / 68.10 chrF2
(Bosnian → English) and 30.73 BLEU / 60.00 chrF2 (English → Bosnian). The
speech recogniser achieves 11.9% word error rate on 200 held-out FLEURS clips.
The photograph reader identifies 57.8% of words per photograph on a 132-image
test set with 450 invented words — stock PaddleOCR PP-OCRv6, chosen over a
fine-tuned EasyOCR reader by a rule written before the comparison.

Every measurement threshold was written before its training run began, every gain
is accompanied by a paired bootstrap significance test, and every result — including
the ones that show no improvement — is published in the repository.

---

## 1. Introduction

### 1.1 Motivation

The author, a third-year computer science student, needed a Bosnian translator for
daily use during an internship in Bosnia and Herzegovina. Existing translation
services treat Bosnian as interchangeable with Croatian and Serbian — the three
languages share grammar and most vocabulary but differ in specific lexical choices,
script conventions, and cultural terms. A system that writes *vjerovatno* (Croatian)
where a Bosnian speaker said *vjerovatno* in a Croatian pronunciation, or that cannot
read the diacritics č, ć, đ, š, ž that distinguish Bosnian Latin orthography, is not
a Bosnian tool.

Lilly was built to close that gap: a single application where you can type a sentence,
speak into a microphone, or photograph a sign, and receive a translation that respects
the Bosnian variety — measured, not asserted.

### 1.2 Design principles

Three principles shaped every decision in this project, from architecture to training
methodology:

1. **Offline-first.** Nothing the user says, types, or photographs leaves the machine.
   Every model runs locally, loaded once and kept in memory. This is a privacy
   constraint that also makes the system usable on unreliable connections.

2. **Measure before changing.** Every training run is pre-registered: the pass/fail
   thresholds, the test set, the scoring path, and the definition of failure are
   written down before any number exists. A later disagreement with the thresholds
   is recorded as a note beneath them, not as an edit. Two arms were run for the
   translator and the pre-written tie-break chose the arm with the *lower* headline
   BLEU, because the rule said chrF2 decides.

3. **Publish the failures.** The repository contains every results file, including
   null results (the BosnianBench term recall at p = 0.360), refused gates (the
   speech listener's Croatian drift), and closed lines (four OCR self-labelling
   attempts that never moved the held-out score). A README that only lists wins is
   not worth trusting.

---

## 2. System Architecture

### 2.1 Overview

Lilly is a Python application built on FastAPI. It exposes seven HTTP endpoints and
a browser-based single-page interface. Five distinct models serve five abilities:

| Ability | Input | Output | Model |
|---|---|---|---|
| **Translate** (bs → en) | Bosnian text | English text | OPUS-MT `opus-mt-tc-big-zls-en`, LoRA fine-tuned, CTranslate2 int8 |
| **Reply** (en → bs) | English text | Bosnian text | OPUS-MT `opus-mt-tc-base-en-sh`, LoRA fine-tuned, CTranslate2 int8 |
| **Listen** | Spoken audio | Transcribed text | Whisper large-v3, LoRA fine-tuned, CTranslate2 int8 |
| **Read** | Photograph | Extracted text | PaddleOCR PP-OCRv6 (detection + recognition), stock weights |
| **Speak** | Text | Spoken audio | Kokoro-82M (English); Piper sr_RS (Bosnian) |

All five sit behind one Python object (`app.lilly`) and one API surface. Speech
and photographs are first transcribed/read into text, then passed through the same
translator, so the translation quality is identical across all input modes.

### 2.2 Bidirectional operation

Since 8 September 2026 every ability runs in both directions. The listener
(Whisper) is multilingual and transcribes both Bosnian and English. The reader
(PP-OCRv6) reads Latin script regardless of language. The UI exposes a direction
toggle: swap the arrow and Lilly hears English, reads English photographs, answers
in Bosnian, and says the answer aloud in the Bosnian voice.

### 2.3 Request bounding

Every request is bounded before it reaches a model — uploads by size, text by token
count, images by pixel count — because the server is designed to face the open
internet. A correction endpoint accepts user feedback for verified retraining.

### 2.4 Model loading

Startup is instant because each model loads lazily on first use. Once
`scripts/fetch_models.py` has run, nothing reaches the network again. The
publisher (`scripts/publish_to_hf.py`) binds every weight file to a content
fingerprint and refuses to ship any other; the listener specifically requires a
fingerprint named on the command line, because it did not clear its gate.

---

## 3. Data

### 3.1 Translation

**Training corpus.** 313,612 Bosnian–English sentence pairs after cleaning:
WikiMatrix, SETIMES-v2, TED2020-v1, and wikimedia-v20260327, plus 1,924 rows from
NTREX-128 (published 15 months after the base model's training cutoff — genuinely
unseen by two independent grounds). 21,178 misaligned WikiMatrix pairs were removed
on three measured rules.

**A critical frame.** The base model's own release manifest
(`opusTCv20210807+bt`) lists 159,611,854 pairs across 44 corpora, containing
WikiMatrix-v1, SETIMES-v2, TED2020-v1, and wikimedia-v20210402 by name. The
project's `data/clean/train.tsv` — and `valid.tsv` and `test.tsv` — are inside
the base model's own training data. Fine-tuning was re-weighting material the model
already had, not showing it new material. That invalidates nothing and is not a
defect: it is the correct frame for reading +1.26 BLEU, and the best explanation
anyone has produced for why chrF2 refuses to move.

**Evaluation.** FLORES-200 Bosnian–English, 2,009 sentence pairs (dev + devtest).
NTREX-128 (1,924 rows) is genuinely unseen. BosnianBench, a 346-case benchmark of
terms that separate Bosnian from Croatian and Serbian, was built to test the
Bosnian-specific claim and found it does not hold for the forward direction
(91.7% → 92.2%, p = 0.360).

### 3.2 Speech

**FLEURS Bosnian (bs_ba).** 3,091 training clips, 925 test clips. 200 held-out
clips from the test set serve as the standard evaluation prefix for every speech
experiment. Common Voice has no Bosnian. VoxPopuli Croatian (CC0 + EP attribution)
and FLEURS Croatian provided supplementary data for the whisper-small fine-tune.

**YouTube voice corpus.** 22.5 hours of Bosnian lectures from seven Creative Commons
BY-licensed channels (20 videos, `training/speak-youtube/manifest.json` with sha256
per file). The shipped listener transcribed them on the Kaggle box; a rule (log-prob
≥ −0.6, 3–20 seconds, centroid filter 0.60) kept 8,329 clips totalling 1,105
minutes across all seven speakers — more audio than FLEURS or the Croatian parliament
data. This is the richest Bosnian speech corpus assembled for this project.

### 3.3 Photographs

**Wikimedia Commons harvest.** 286 photographs of Bosnian signs, 524 MB. 1,114
dropped / 286 kept / 219 skipped / 7 refused at a 17.6% keep rate. Full attribution
in `data/ocr/real-photos/harvested/CREDITS.tsv` — 286 rows, nothing missing a
licence, attribution, or source page.

**Test sets.** Two test sets of Bosnian signs from Wikimedia Commons, each
transcribed blind by two independent readers who saw neither each other's work nor
any model's output; only words both agreed on enter the answer key.

- **The 40:** the original set, 373 agreed words, 88.2% inter-annotator agreement.
  Two items are not photographs (a map render and a 1900s postcard); they inflate
  the headline by 1.7 points per-photo.
- **test-v2:** 280 photographs drawn from the same pool, 132 with text, 2,907
  agreed words, never trained on by anything. This is the evaluation set.

**Training crops.** 1,914 crops cut from harvested photographs, 1,702 usable, each
transcribed blind by twelve annotators. 39 marked as containing no text (detector
false positives), 173 as unreadable — both excluded.

### 3.4 Voice (text-to-speech)

**FLEURS line.** The FLEURS bs_ba training clips clustered into 8 speakers; 7
selected (54–103 minutes each). Trained a Piper voice from the `sr_RS` checkpoint.
Result: 53.9% word error — refused by bar 1.

**ParlaSpeech-HR line.** Five Croatian parliament speakers, three clean hours each,
mean embedding served. Result: 51.9% — refused.

**Control.** The `sr_RS` checkpoint's own 747 Sorbian recordings. Three arms (sr
phonemes, bs phonemes, bs + gentle optimizer). All three stayed within +3 to +4
points of the checkpoint (sound by the rule). This proved the pipeline is not broken
and the phonemiser is not the cause of the refusals.

**YouTube line.** The 8,329 clips from seven Bosnian lecturers. Training staged but
blocked on GPU quota as of 10 September 2026.

---

## 4. Models and Training

### 4.1 Translation — Bosnian → English

**Base model.** Helsinki-NLP `opus-mt-tc-big-zls-en`, a Marian NMT model (~230M
parameters) trained on 159.6M sentence pairs across 44 corpora covering South Slavic
languages.

**Fine-tuning.** LoRA (Low-Rank Adaptation) applied to the model's attention layers.
Two arms were pre-registered and run on Kaggle Tesla T4 GPUs:

- **Arm A — data only.** The cleaned corpus with misaligned pairs removed and unseen
  pairs added. Hyperparameters unchanged from the shipped model.
- **Arm B — data plus recipe.** The same corpus, plus hyperparameters chosen by a
  jury run (a grid search whose results were committed before either arm launched).

The tie-break was written before either arm ran: the arm with the higher **chrF2**
wins, because chrF2 counts characters and is the fairer measure for a language that
inflects as heavily as Bosnian. Not the higher BLEU. Arm B won.

**Post-training.** The LoRA adapter is merged into the base weights, which are then
converted to CTranslate2 int8 quantisation for inference. The served model is a
single directory of quantised weights that loads in under a second.

**The language tag defect.** The base model emits its own language tag (`>>eng<<`
and variants) into the translation text in 576 of 2,009 outputs (28.7%). The
fine-tuned model does it in 0 (0.0%). This is the fine-tuning's clearest single
win: it removes a visible defect from every third output. The app strips any leaked
tag since 8 September 2026, so the evaluation numbers are always given both ways.

### 4.2 Translation — English → Bosnian

**Base model.** Helsinki-NLP `opus-mt-tc-base-en-sh` (~77M parameters), a smaller
model than the forward direction's.

**Fine-tuning.** LoRA, same method. Pre-registered with four bars: chrF2 ≥ 58.96,
BLEU ≥ 29.57, Bosnian form rate ≥ 94.3%, and `>>bos_Latn<<` label steering gap
preserved. All four cleared on 8 September 2026: chrF2 58.96 → 60.00, BLEU
29.57 → 30.73, form rate 94.3% → 99.2% (244 of 246 decided targets write the
Bosnian form), label gap 21.8 → 22.5 points.

### 4.3 Speech recognition

**Base model.** OpenAI Whisper, in two sizes fine-tuned here:

- **whisper-small** (~244M parameters): the gated listener. Fine-tuned with LoRA on
  FLEURS Bosnian + VoxPopuli Croatian. Word error: 38.5% → 34.9% on 200 held-out
  clips. Bosnian term recall: 65.9% → 68.2%. Wrong-variety substitution: 5.1% → 3.3%.
  Cleared all three pre-registered gate rows. Stays in the bundle as the baseline.

- **whisper-large-v3** (~1.55B parameters): the shipped listener. Fine-tuned with LoRA
  on the same data. Word error: 11.9% on the 200-clip prefix, 14.1% on all 925 clips.
  **Refused at its gate**, twice:
  - Gate run (7 September): refused by one word on the Croatian substitution row.
  - Pre-registered last look (8 September, all 925 clips): refused again — Croatian
    substitution 1.1% → 6.1% (p = 0.018), the same two words (*Europom* for *evropom*,
    *vjerojatno* for *vjerovatno*).
  - By rule 3 of the pre-registration, whisper-large-v3 is closed to any further look.
  - **Shipped by the owner's decision on 8 September**, for what it gets right (14.1%
    vs 39.5% word error, 72% vs 50% Bosnian term recall) and accepting what it gets
    wrong. The gate's result stands written; the failed row is on the model card; the
    gated small stays the baseline.

**Inference.** Both listeners are converted to CTranslate2 int8. The publisher
refuses any listener not named by fingerprint on the command line.

### 4.4 Photograph reading (OCR)

**Shipped reader.** PaddleOCR PP-OCRv6 (`PP-OCRv6_medium_det` + `PP-OCRv6_medium_rec`),
stock weights, fetched at run time. A recogniser confidence floor of 0.9 is applied
because without it PP-OCRv6 reads 60.0% but invents 2,373 words.

**Selection rule.** The engine was chosen by a rule written before the comparison ran
(`training/PREREGISTRATION.md`), evaluated on test-v2 (the 132-photograph set), not
on the 40. The fine-tuned EasyOCR reader had claimed 54.7% on the 40 but reads only
34.6% on the 132 — the 40 were its optimistic end.

**The EasyOCR line.** The original reader was EasyOCR with a CRAFT detector and a
`latin_g2` recogniser fine-tuned on 1,294 hand-transcribed real crops plus 20,000
synthetic crops. Four subsequent OCR training attempts (self-labelling from Mapillary
photographs, passes 14–19) produced no held-out gain and the line is closed. The
EasyOCR reader remains as the `LILLY_READER=easyocr` fallback.

**Known limits.** Bosnian Cyrillic is unreadable — the Latin recogniser has no
Cyrillic output classes. 272 of the 1,702 hand-transcribed crops (16.0%) are
Cyrillic. Diacritics (č, ć, đ, š, ž) are the reader's weakest column: only 180 of
1,702 real labels carry any of them (10.6%), and đ appears 8 times total in the
labelled set.

### 4.5 Text-to-speech

**English.** Kokoro-82M, stock weights, Apache-2.0. No fine-tuning.

**Bosnian.** Piper `sr_RS-serbski_institut-medium`, pulled from `rhasspy/piper-voices`
at start time, not bundled. Filed under Serbian, trained on the Sorbian Institute's
Lower Sorbian recordings by its own model card. Through Lilly's own listener it is
heard with 22.3% word error on the test prefix, against 11.7% for human recordings.

Two trained voices were attempted and refused:
- FLEURS voice: 53.9% word error (+31.57 over the checkpoint, p = 0.0000)
- Parliament voice: 51.9% word error (+29.61, p = 0.0000)

The control proved the pipeline holds a voice it is handed (all three arms within
+3 to +4 points). The recordings, not the recipe, were the cause. A YouTube voice
(8,329 clips from seven native lecturers) is staged and awaiting GPU time.

---

## 5. Evaluation

### 5.1 Methodology

Every evaluation in this project follows a strict protocol:

1. **Pre-registration.** The deciding measurement, its threshold, and the definition
   of failure are fixed in `training/PREREGISTRATION.md` before any training begins.
   Amendments are allowed before any number from the run exists and are written as
   separate notes, not edits to the original text.

2. **Held-out data only.** Translation is scored on FLORES-200, which the base model
   was not trained on. Speech is scored on 200 held-out FLEURS clips. Photographs are
   scored on Wikimedia Commons images transcribed by independent human annotators who
   never saw any model's output.

3. **The app's own path.** Translation is scored through `app.translate.Engine`, not
   the raw model. The sentence splitter, the quantisation, and the tag stripping are
   the product's own. Scoring the raw adapter on whole rows gives a different answer
   (+0.54 BLEU / −0.79 chrF2) because feeding several sentences at once makes the
   model drop a clause, and the app never does that.

4. **Paired bootstrap significance.** Every claimed gain is accompanied by a paired
   bootstrap test (n = 1,000 or 2,000 draws, p < 0.05 required). A difference that
   does not clear the bootstrap is reported as a tie, not a gain.

5. **Counts beside percentages.** 57.8% on 132 photographs means something different
   from 57.8% on 40. The raw counts and the confidence interval are given beside every
   percentage.

### 5.2 Translation results

Measured 8 September 2026 on a Kaggle Tesla T4 after the ordinal splitter fix, through
the served code path, language tags stripped.

**Bosnian → English, 2,009 FLORES-200 pairs:**

| | BLEU | chrF2 |
|---|---|---|
| Base (untuned), tag stripped | 41.77 | 67.66 |
| Lilly (fine-tuned) | **43.03** | **67.81** |
| Gap | +1.26 | +0.15 |
| p (paired bootstrap) | 0.001 | 0.074 |

On the 1,012-pair devtest half: **43.25 BLEU / 68.10 chrF2** against the base's
42.08 / 67.85.

**The finding that matters most.** The fine-tuning does not move chrF2: −0.16 at
p = 0.128 on 2,009 pairs, a bootstrap tie. Its BosnianBench term recall moves
+0.5 at p = 0.36. What the fine-tuning did achieve: the base model prints its
language tag into 30.4% of translations and Lilly into none, and with tags stripped
from both, BLEU still moves +1.26 at p = 0.001. The base model was already excellent;
we measured that carefully enough to know it.

**The ordinal splitter fix.** Until 8 September the sentence splitter cut a Bosnian
date (*5. maja 1990. godine*) into three pieces. The fix was pre-registered and is
worth +0.89 BLEU / +0.37 chrF2 to the fine-tune (p = 0.001, 167 of 2,009 rows
changed). Device drift between the Mac's CPU and the T4 is within noise (−0.04 BLEU,
p = 0.25).

**English → Bosnian, 2,009 FLORES-200 pairs:**

| | BLEU | chrF2 |
|---|---|---|
| Base (untuned) | 29.57 | 58.96 |
| Lilly (fine-tuned) | **30.73** | **60.00** |
| Gap | +1.16 | +1.04 |
| p | 0.001 | 0.001 |

Bosnian form rate: 94.3% → 99.2% (244 of 246 decided targets). `>>bos_Latn<<`
label steering gap: 21.8 → 22.5 points.

**Against an outside system.** `facebook/nllb-200-distilled-600M` (600M parameters)
on the same 2,009 pairs: 36.49 BLEU / 63.80 chrF2 (bs→en) against Lilly's 43.03 /
67.81, and 26.07 / 56.22 (en→bs) against 30.73 / 60.00. But the untouched Helsinki
base already accounts for +5.11 of the +5.65 BLEU gap. The win belongs largely to
OPUS-MT. And NLLB-600M is the distilled small variant; Google, DeepL, the 3.3B NLLB,
and the large general models were not tested.

### 5.3 Speech results

200 held-out FLEURS Bosnian clips, one scorer, one process:

| Listener | Word error | Bosnian term recall | Croatian substitution |
|---|---|---|---|
| whisper-small, stock | 38.5% | — | — |
| whisper-small, fine-tuned (gated) | **34.9%** | 68.2% | 3.3% |
| whisper-large-v3, fine-tuned (shipped) | **11.9%** | 89.1% | 6.5% |
| Human recordings | 11.7% | — | — |

On all 925 test clips, the large-v3 listener reads 14.1% word error against the
small's 39.5%. On the project's own rubric scale (`training/RUBRIC.md`), anchored
to published systems: band 3 (39.5%) against band 8 (14.1%).

### 5.4 Photograph results

| Reader | the 40 (found / invented) | test-v2, 132 photographs (found / invented) |
|---|---|---|
| First recorded (EasyOCR, first reader) | 36.0% / 224 | — |
| EasyOCR, stock | 48.0% / 188 | 30.0% / — |
| EasyOCR, fine-tuned on real crops | 54.5% / 182 | 34.6% / 2,071 |
| **PP-OCRv6, stock, floor 0.9 (shipped)** | **67.0% / 65** | **57.8% / 450** |

### 5.5 Voice results

On the FLEURS test prefix (200 clips, 167 sentences), heard through the shipped
listener:

| Voice | Word error | vs. sr_RS checkpoint (22.3%) |
|---|---|---|
| Human recordings | 11.7% | — |
| sr_RS checkpoint (shipped) | **22.3%** | — |
| Control arm C (bs, gentle) | 25.5% | +3.22, p = 0.0000, **sound** |
| FLEURS voice | 53.9% | +31.57, p = 0.0000, **refused** |
| Parliament voice (mean of 5 speakers) | 51.9% | +29.61, p = 0.0000, **refused** |

The ~52% ceiling across two different corpora, the same mel plateau, and the control
staying within four points all point to the same conclusion: the recipe works, but the
recordings available do not produce intelligible Bosnian speech when fine-tuned through
Piper's framework.

---

## 6. The Rubric

All three abilities are graded on a 1–10 scale (`training/RUBRIC.md`) written on
27 August 2026, before the measurements it grades. The scale was not set by whoever
built the models: proposals came from agents given the task and the language pair and
deliberately not told how Lilly scores, then each proposal was handed to a second
agent asked to attack it — to name any invented citation and to find any band an
off-the-shelf model would clear by accident. What survived:

| Ability | Lilly's score | Anchored to |
|---|---|---|
| Translation (chrF2 68.10) | **10** | ≥ 67.2 = NLLB-200-3.3B, best of 30 published systems on this pair |
| Speech (WER 14.1% on 925 clips) | **8** | 10–15% = commercial cloud (Google, Azure) on FLEURS Bosnian |
| Photographs | **void** | RUBRIC.md refuses to grade below 200 real photographs with text; test-v2 has 132 |

---

## 7. Methodology: Pre-registration in Practice

The pre-registration file (`training/PREREGISTRATION.md`) is the spine of this
project's claim to honesty. It is structured as a sequence of versioned sections,
each written before the corresponding training run:

- **v1 — translate.** Two arms, tie-break by chrF2, written before either ran.
  Arm B won with the lower BLEU — the rule chose chrF2, so that is what happened.
- **v2 — speech half 2.** AFTER WER on the 200 held-out clips, never skipped.
- **v2a — the reader (amendment).** Corrected a phantom target (67.1 / 69.4) that
  had no source and would have cleared without training. The actual before number
  was 75.2 / 73.0.
- **v3 — outside baseline.** Pre-registered before any outside number existed.
- **v4 — ordinal splitter.** Pre-registered the re-measurement after the bug fix.
- **v5 — speak (FLEURS voice).** Pre-registered the bars and the judge.
- **v6 — speak (parliament voice).** Same bars, same judge.
- **v7 — speak (control).** Sound at +5, broken at +10 with p < 0.05.
- **v8 — speak (YouTube voice).** Same bars. Training staged, awaiting GPU quota.

Each section includes a "what failure looks like" paragraph written before the
run, so the definition of failure cannot be reinterpreted after the numbers are in.

---

## 8. Infrastructure

### 8.1 Training on Kaggle

All training runs on Kaggle GPU instances (Tesla T4 or P100), never on the author's
laptop (8 GB RAM, no discrete GPU). The laptop commits, pushes, launches via
`scripts/kaggle_train.py`, polls via `scripts/kaggle_poll.py`, and fetches results.
This boundary is enforced by project rules: heavy compute that takes more than a few
minutes of CPU does not run locally.

**Preflight.** `scripts/preflight_kaggle.py` checks that the tree is clean, that
GitHub has the exact HEAD by SHA (not the CDN-cached version), and that every
notebook's parameters are consistent with the pre-registration. The launcher refuses
to start if preflight fails.

**Fail-stop rules.** A known failure stops the kernel. A COMPLETE status, a zip from
an earlier run, or hitting the 12-hour wall does not override a gate. CANCEL and
ERROR are failures even if a zip was recovered. Recovery is not success.

### 8.2 Publishing

`scripts/publish_to_hf.py` pushes to the Hugging Face repository `Safak11/lilly`.
Every weight file is bound to a content fingerprint. The listener goes up only under
a fingerprint named on the command line (`--allow-listen <fingerprint>`), because it
did not clear its gate. `scripts/fetch_models.py` checks the listener it receives
against the gated one, saying so out loud if an older fetch left the wrong build.

### 8.3 Version control

The GitHub repository (`ssaaffaakk/Lilly`) is the single source of truth. Every
step is committed and pushed before any dependent action (including Kaggle launches,
which clone from GitHub). `scripts/state.py` reports uncommitted, unpushed, and
untracked work and queries GitHub for HEAD by SHA.

---

## 9. Known Limitations

This section exists because claiming to know one's limits is part of the claim.

1. **The Bosnian-specific claim is not proven for the forward direction.** BosnianBench
   (346 cases) moves 91.7% → 92.2% at p = 0.360. The base model is already trained
   across South Slavic and arrives there on its own. The reply direction is the first
   place this claim holds: 94.3% → 99.2%.

2. **The gain is concentrated in news prose.** Broken out by corpus, the fine-tuning
   is worth +3.05 BLEU on news text (SETIMES) and −0.82 BLEU on talks (TED2020).
   Nothing measured here separates "learned better Bosnian" from "adapted to news
   style."

3. **The training data is inside the base model's training data.** This is the correct
   frame for reading +1.26 BLEU, and it means the fine-tuning's contribution is
   narrower than a naive reading of the numbers suggests.

4. **The photograph score is not yet valid by the project's own scale.** The rubric
   requires 200 real photographs with text; test-v2 has 132. Another 160 photographs
   (test-v2b) are fetched and waiting for two blind transcriptions.

5. **English-in is unmeasured.** The listener hears English and the reader reads
   English photographs, but no held-out English set has been scored.

6. **Bosnian Cyrillic is unreadable.** 16% of hand-transcribed crops are Cyrillic.
   The Latin recogniser has no Cyrillic output classes.

7. **The voice is not Bosnian.** The Piper sr_RS voice is filed under Serbian, trained
   on Sorbian recordings. Two voices trained on Bosnian audio were refused (53.9% and
   51.9% word error). A YouTube voice is staged.

8. **One test set per ability.** FLORES is professionally translated and even in
   register. Real user input is not.

---

## 10. Closed Lines and Dead Ends

Measured and published so they are not re-tried:

| Line | What happened | Why it is closed |
|---|---|---|
| Mapillary street-level imagery | 10 frames, zero words above confidence 0.3, one "detection" was the dashcam watermark | Structural to vehicle-mounted capture |
| Panoramax | Independently reproduced; 7 of 8 frames under the 800px short-side bar | Same structural failure |
| KartaView | Metadata API answers, every image URL returns 502 | Service broken |
| Europeana / NUL BiH | ~30 items, all CC BY-NC-ND, text is publisher captions | Same distribution as synthetic data |
| podaci.gov.ba | Tabular statistics, no imagery, no machine API | No data to use |
| Turkisms in BosnianBench | 18.5M OpenSubtitles pairs measured; 0 of 23 terms clears the 98% share gate | The constraint is the rule |
| OCR self-labelling (passes 14–19) | Five training sets from 20,240 Mapillary photographs, no held-out gain | Labels from the model under training; the folded score never moved |
| FLEURS voice (v5) | 53.9% WER, +31.57 over checkpoint | Refused by bar 1 |
| Parliament voice (v6) | 51.9% WER, +29.61 over checkpoint | Refused; same ceiling as FLEURS |

---

## 11. Credits and Attribution

Lilly is a bundle, not a new architecture. The weights come from:

| Component | Source | Author | Licence |
|---|---|---|---|
| Translator (bs→en) | `opus-mt-tc-big-zls-en` | Helsinki-NLP / OPUS-MT project | CC-BY-4.0 |
| Translator (en→bs) | `opus-mt-tc-base-en-sh` | Helsinki-NLP / OPUS-MT project | CC-BY-4.0 |
| Listener | Whisper large-v3 | OpenAI | MIT |
| Reader (shipped) | PaddleOCR PP-OCRv6 | PaddlePaddle / Baidu | Apache-2.0 |
| Reader (fallback) | EasyOCR + CRAFT | JaidedAI; Clova AI Research, NAVER Corp. | Apache-2.0 (EasyOCR); MIT (CRAFT) |
| English voice | Kokoro-82M | hexgrad | Apache-2.0 |
| Bosnian voice | Piper sr_RS-serbski_institut-medium | rhasspy | MIT |

**Tools.** CTranslate2 (OpenNMT, MIT) for quantisation and inference. peft
(Hugging Face, Apache-2.0) for LoRA fine-tuning.

Full attribution and licence verification in `models/lilly/NOTICE.md`. The OPUS-MT
authors ask to be cited:

> Tiedemann, J. and Thottingal, S. (2020). OPUS-MT — Building open translation
> services for the World. Proceedings of EAMT 2020.

---

## 12. Reproducibility

Every training run can be reproduced from the repository:

1. Clone the repository and install dependencies (`requirements.txt`).
2. Run `scripts/preflight_kaggle.py` to verify the tree is clean.
3. Launch via `scripts/kaggle_train.py <job-name>`.
4. Poll via `scripts/kaggle_poll.py` until COMPLETE.
5. Fetch and install results via `scripts/kaggle_train.py <job-name> --fetch`.

The preflight refuses to launch unless GitHub has the exact HEAD by SHA. Every
notebook pins its dependencies and seeds. Results files record the commit hash, the
hardware, and the time.

All evaluation scripts are in `training/`. All pre-registration entries, including
amendments, are in `training/PREREGISTRATION.md`. All results files — including
nulls and failures — are committed to the repository.

---

## 13. Current Status and Next Steps

As of 10 September 2026:

**Shipped and published:**
- Translation, both directions, with the ordinal splitter fix
- Speech listener (whisper-large-v3, by decision; whisper-small as baseline)
- Photograph reader (PP-OCRv6, stock)
- English voice (Kokoro-82M, stock)
- Bosnian voice (Piper sr_RS, stock)
- Published weights at `Safak11/lilly` with fingerprint-bound content

**Staged, awaiting GPU quota:**
- speak-youtube (v8): batch-4 fix committed (6a04676), dataset uploaded, notebook
  ready. Relaunch command: `python3 scripts/kaggle_train.py speak-youtube`.
  The weekly 30-hour Kaggle GPU quota is exhausted; the run fires when it resets.

**Open work:**
- test-v2b: 160 photographs fetched, awaiting two blind transcriptions to make the
  photograph score valid (≥ 200 photographs with text)
- No labelled set of real phone photographs from Bosnia
- English-in quality unmeasured
- Bosnian Cyrillic out of scope

---

*Last updated: 10 September 2026.*
