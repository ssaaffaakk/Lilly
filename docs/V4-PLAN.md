# Lilly v4 — the map for what gets trained next, and what does not

Written 8 September 2026, at 02:10, while the last two v3 jobs were still
running. The owner asked for a plan to improve every part further. This file
is that plan, and it is written against the project's own evidence rather
than against the wish: two of the three lanes have already shown that more
training on the data at hand does not move a held-out number, and the third
has its win mostly on loan from its base. So the order below is by **what can
move a held-out number**, not by what can be launched.

Rules this plan lives under, unchanged: every run gets its bars written before
its numbers exist (`training/PREREGISTRATION.md`); only held-out sets decide;
counts and intervals beside every delta; heavy compute on Kaggle
(`.claude/CLAUDE.md`); nothing on the do-not-repeat lists
(`docs/kaggle-fail-stop.md`, `docs/OCR-ROADMAP.md`) is relaunched.

## Status board — 8 September 2026, 02:10

| lane | state | next |
|---|---|---|
| speech instrument (last look at large-v3) | **done, does not ship by the rule; closed to further looks (rule 3)** — Croatian 6.1% (8/131) against 1.1% (1/87), p = 0.018; word error and term recall pass; rubric WER 39.5% / 14.1% | **shipped 8 Sep evening by the owner's override**, refused row on the card; the gated small kept as the baseline |
| en-bs LoRA | **ships by the rule** — re-measured on the Mac: chrF2 60.00, BLEU 30.73, form rate 99.2%, label gap 22.5 points, all four bars hold; served build rebuilt with the adapter | published 8 Sep (`translator-en-bs/` in the bundle); next: runs A and B below |
| test-v2b | 160 photographs on the Mac, attribution committed, blind sheets written and empty | two blind passes, then `build_truth`, then one score on test-v2 ∪ test-v2b |
| published bundle | `listen/` is whisper-large-v3 again since 8 Sep evening, by the owner's named override (`--allow-listen e6bb58483586b06c`); reply build in since 11:31 UTC | `publish_to_hf.py` refuses an unnamed listener and binds the reply build to its numbers |

**Note added 11 September 2026, corrected 13 September — the voice lane.** The
board above predates v5–v8. The Bosnian voice's warm-start checkpoint was found
to be Lower Sorbian (West Slavic) rather than Serbian, and all three fine-tunes
from it lost points — including the control, on the checkpoint's own studio
data. The next voice step is a measurement, not a training run: hear
`sl_SI-artur` and `bg_BG-dimitar` as fetched against the 200-clip prefix, the
bar pre-registered first (ships only if strictly below 22.3% at p < 0.05, the
bar every other speech candidate has faced). That measurement has not been run.

**v8/v9 — YouTube audio to a voice — is closed, not held.** This note said the
opposite for thirty-three minutes. The audio was probed the same afternoon:
nought of twenty lectures carry speech energy to 11 kHz, which is the band a
22.05 kHz voice must fill, so the line closes on a number rather than on an
argument (`docs/speak-checkpoint-comparison.md`, "Close v8/v9 — YouTube audio to
a voice. Not held: closed, on the measurement above" — commit `0bbb1df`,
11 Sep 15:12, against this note's `645e0e4` at 14:39). Do not relaunch
`speak-youtube` against any checkpoint, at any batch size, for any number of
epochs. The same 18.4 hours are unharmed for *recognition*, because Whisper's
filterbank stops at 8 kHz — but their transcripts are the shipped listener's own
output, so their provenance has to be settled before they are even that.

## What the evidence says before anything is launched

- **Photographs.** Six self-labelled passes, three PaddleOCR fine-tunes and a
  detector-resolution sweep all failed to beat the shipped reader on test-v2.
  The reader's score is *void* below 200 photographs with text
  (`training/RUBRIC.md`); test-v2 has 132. The lever is the test set, not the
  model.
- **Speech.** Bosnian speech data is 3,091 FLEURS clips. Common Voice has no
  Bosnian. The candidate that reads 11.9% against 34.9% had its last look on
  8 September: the Croatian row failed (1.1% → 6.1%, p = 0.018), rule 3 closed
  it, and the owner shipped it by decision that evening with the failed row on
  record. The lever now is audio supply, not another epoch.
- **Translation.** Against NLLB-200 the base supplies +3.78 chrF2 and the
  fine-tune −0.79 (`training/RESULTS-outside-baseline.md`). The lever is data
  quality and real Bosnian targets, not parameters.

## Lane by lane, in priority order

### 1. Photographs — make the score valid, then decide whether to train

1. **Two blind passes over test-v2b** (`training/transcription_pass.py`,
   `training/transcribe/BRIEF.md`). By two people, or by two vision agents that
   never see each other's work or any reader's output — the method that
   produced test-v2's passes byte for byte. `check` refuses holes and copies.
2. `build_truth.py` → `truth-v2b.json`; score the **shipped** reader once on
   the union (pre-registered: "v3 — picture — test-v2b"). The number is
   box-relaxed and says so.
3. Only then, and each under its own pre-registration: the two detector levers
   the roadmap defers — the app's 2 MP working size, and a CRAFT-detect +
   PP-OCRv6-recognise hybrid. The recogniser lever is spent (steps 7, 7a, 7b).
4. `scripts/fetch_highres_and_score.py` on the Mac: no weights change; it
   measures the reader on the inputs people actually upload.
5. Training data for the reader exists only if labels come from people or
   blind passes (`OCR-ROADMAP.md` step 4) **and** the owner says shop signs
   are in scope (open decision 1). Neither is true today.

### 2. English → Bosnian — decide tonight, then two pre-registered runs

- ~~**Tonight:** `evaluate.py`, `bosnian_form_rate.py` ×2, `--diagnose`. Four
  bars, both directions, not either. Ship only if all hold; the owner
  publishes.~~ Done that evening: every bar held — chrF2 60.00, BLEU 30.73, form
  rate 99.2%, label gap 22.5 points — and the owner published
  (`docs/en-bs-launch.md:98-99`). Runs A and B below are what is left.
- **Run A — the full fine-tune arm.** The forward direction's Arm B won by a
  pre-registered tie-break; the reply direction has only run LoRA. Same corpus,
  same seed, same bars, one commit to flip `ARM`.
- **Run B — back-translation.** Real monolingual Bosnian translated to English
  by the shipped bs-en model: synthetic source, **real Bosnian target**. The
  source is settled (was decision 4, now resolved): **MaCoCu-bs 1.0 — 730M
  words, CC0** (CLARIN.SI 11356/1808), native Bosnian crawled from `.ba`. CC0
  rather than Wikipedia's CC BY-SA, so no share-alike propagates to the
  published weights. This is the strongest known lever for both chrF2 and the
  Bosnian form rate, because the targets are Bosnian written by Bosnians. Bars:
  FLORES chrF2 above the best shipped, BLEU floor, form rate floor, label gap
  not collapsed. It stays on the small `opus-mt-tc-base-en-sh` base — the big
  -sla base was measured and closed (`PREREGISTRATION.md`, "Outcome, 14
  September ... the line closes"), so back-translation does not get combined
  with it.
- **Decode-time Croatian suppression (§2) — measured 15 Sep and closed.** The
  static-list lever the reports raised: suppress the Croatian forms at decode
  with CTranslate2 `suppress_sequences`, no training. Measured on the served
  path (`training/RESULTS-suppress-en-bs.md`, `training/measure_suppress_en_bs.py`):
  **0 of 308 outputs change and form rate holds at 99.6%** — the LoRA already
  writes the Bosnian forms, so there is nothing left to suppress. Even the oracle
  upper bound on the untuned base recovers +0.4 points (1 of 14) at zero chrF2
  cost. Not shipped; the fine-tune subsumes it. No app flag added.

### 3. Bosnian → English — data quality first

- **Run C — alignment filter.** WikiMatrix is a sixth misaligned by the
  project's own note. Score pairs with a sentence-embedding alignment model,
  drop the tail, retrain the shipped recipe, paired bootstrap on FLORES.
- **Run D — back-translation the other way.** English monolingual text through
  the now fine-tuned en-bs model gives synthetic Bosnian sources with real
  English targets.
- **A bigger base (NLLB-200 1.3B)** is a separate, expensive pre-registration
  and only if C and D are flat. The rubric's band 9 anchor is that model.

### 4. Speech — the instrument decides, then the data does

- **Decided 8 September: it does not ship by the rule; rule 3 closes whisper-large-v3
  to further looks. The same evening the owner shipped it by explicit override**
  (`PREREGISTRATION.md`, "The owner's decision"). No new split, normaliser or instrument. (The run's app-decode rows carried a
  cache defect on 200 of 925 clips, recorded in the outcome note; the deciding
  row and the clean greedy decode agree, and the cause is fixed in
  `speech_bench.py`.)
- **Before new audio, the recipe (added 8 September, evening; run 12 September).**
  Every Croatian clip in the mix was trained under `<|bs|>`, which is the
  mechanism behind the Croatian row that closed large-v3. The mix now carries
  one language token per clip and the trainer honours it: both halves printed
  `<|bs|>` 6,182 against `<|hr|>` 6,050, and all 6,050 of those used to train as
  Bosnian. The base is `openai/whisper-large-v3-turbo`, chosen by the owner in an
  amendment written before the launch, so rule 3 — which closed large-v3 — is
  untouched. The run produced **12.77% AFTER WER on 200 clips** (498 / 3,901
  words) against the gated baseline's 34.9%
  (`training/RESULTS-speech-langtoken.md`, commit `052dce5`).
  **That is not a verdict, and must not be written up as one.** All three
  pre-registered bars are unmeasured: word error on the 925 clips with
  `--decode app`, Bosnian term recall, and Croatian substitution — which is the
  hypothesis, and which nothing in a 200-clip AFTER WER tests. The instrument
  notebook pins both builds by fingerprint and requires the candidate to read
  `openai/whisper-large-v3`, so it cannot score this candidate; that limit was
  recorded as an amendment *before* the launch, and it is the gate working.
  Nothing from the run is published, installed or in the bundle. The next speech
  step is therefore the instrument amendment that lets these weights be scored,
  not another epoch and not new audio.
- After that, only new audio moves anything — and **ParlaSpeech-HR is not that
  audio.** The owner closed it on 28 August: it is CC BY-SA, share-alike
  propagates to distributed derivatives, and this project publishes weights
  (`training/PREREGISTRATION.md:749`, commit `911b143`). The same lines name the
  substitute — **voxpopuli_hr, CC0**, ~11,000 clips, which "covers the
  requirement on its own", and 2,620 of which are in the mix above already. The
  v6 *voice* run did train on ParlaSpeech-HR, but it was refused at its gate and
  published nothing, which is the only reason the licence never bit; a listener
  trained on it would ship. What is left is a small human-transcribed Bosnian
  set. Self-transcribed audio is allowed only teacher ≠ student (large → small
  distillation) and only if the audio exists.

### 5. The product loop

`/api/feedback` already stores corrections. Approved corrections exported as
training pairs (roadmap phase 7) are the only source of improvement that comes
from use rather than from corpora.

### 6. Infrastructure debt seen on 7–8 September

- ~~`training/Lilly_Translation_Kaggle.ipynb` clones into `/kaggle/working` and
  its `run()` does not tee; both are on the fail-stop list.~~ Fixed 8 September
  2026: clone to `/kaggle/temp`, teed `run()`, `Offload` + `check_trainproof`,
  `python -u`; preflight applies `check_offload` to it, and
  `scripts/kaggle_train.py` checks the notebook's `ARM` against the arm each
  job's pre-registration names, as it already checked `DIRECTION`.
- Kernel Outputs expire; artefacts the next job needs live in Kaggle datasets
  (`lilly-listen-large-v3`, `lilly-listen-small-previous`) or on Hugging Face,
  never only in an Output.
- Every artefact the gate scores carries a fingerprint; a notebook checks it
  before it scores (`training/SPEECHBENCH-gate.txt`).
- `kaggle_train.py --watch` no longer pushes while a version is running.

## Decisions that belong to the owner

1. ~~Publish or not~~ — decided 8 Sep: large-v3 closed and taken out of the
   bundle; the en-bs build published.
2. test-v2b blind passes: vision agents, people, or both.
3. Shop signs in scope? Decides whether the 20,240 Mapillary photographs can
   ever become training data.
4. ~~ParlaSpeech-HR under CC BY-SA~~ — decided 28 Aug: it is not used, because
   share-alike would propagate to the weights this project publishes, and
   voxpopuli_hr (CC0) covers the requirement on its own
   (`training/PREREGISTRATION.md:749`, commit `911b143`). ~~What stays open is the
   monolingual Bosnian source and its licence for back-translation (run B).~~
   **Resolved: run B's source is MaCoCu-bs 1.0 — 730M words, CC0
   (CLARIN.SI 11356/1808), native Bosnian, no share-alike.**

## What this plan will not do

Relaunch OCR passes 8–19 or the three PaddleOCR fine-tunes; train on labels
written by the model under training; touch a bar after a number exists; score
test-v2b with any reader before its human key exists; start a multi-hour job
on the Mac.
