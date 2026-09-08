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
| speech instrument (last look at large-v3) | **done, does not ship; rule 3: large-v3 closed** — Croatian 6.1% (8/131) against 1.1% (1/87), p = 0.018; word error and term recall pass; rubric WER 39.5% / 14.1% | bundle reverted to whisper-small with the 8 Sep publish; README rewritten |
| en-bs LoRA | **ships by the rule** — re-measured on the Mac: chrF2 60.00, BLEU 30.73, form rate 99.2%, label gap 22.5 points, all four bars hold; served build rebuilt with the adapter | published 8 Sep (`translator-en-bs/` in the bundle); next: runs A and B below |
| test-v2b | 160 photographs on the Mac, attribution committed, blind sheets written and empty | two blind passes, then `build_truth`, then one score on test-v2 ∪ test-v2b |
| published bundle | `listen/` was whisper-large-v3 from the 4 September publish until 8 Sep 11:31 UTC | reverted; `publish_to_hf.py` now refuses an ungated listener and binds the reply build to its numbers |

## What the evidence says before anything is launched

- **Photographs.** Six self-labelled passes, three PaddleOCR fine-tunes and a
  detector-resolution sweep all failed to beat the shipped reader on test-v2.
  The reader's score is *void* below 200 photographs with text
  (`training/RUBRIC.md`); test-v2 has 132. The lever is the test set, not the
  model.
- **Speech.** Bosnian speech data is 3,091 FLEURS clips. Common Voice has no
  Bosnian. The candidate that reads 11.9% against 34.9% is at its last look,
  and rule 3 closes it if the Croatian row fails. After that the lever is
  audio supply, not another epoch.
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

- **Tonight:** `evaluate.py`, `bosnian_form_rate.py` ×2, `--diagnose`. Four
  bars, both directions, not either (`docs/en-bs-launch.md`). Ship only if all
  hold; the owner publishes.
- **Run A — the full fine-tune arm.** The forward direction's Arm B won by a
  pre-registered tie-break; the reply direction has only run LoRA. Same corpus,
  same seed, same bars, one commit to flip `ARM`.
- **Run B — back-translation.** Real monolingual Bosnian (licence chosen by the
  owner; bs Wikipedia CC BY-SA is the clean default) translated to English by
  the shipped bs-en model: synthetic source, **real Bosnian target**. This is
  the strongest known lever for both chrF2 and the Bosnian form rate, because
  the targets are Bosnian written by Bosnians. Bars: FLORES chrF2 above the
  best shipped, BLEU floor, form rate floor, label gap not collapsed.

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

- **Decided 8 September: it does not ship; rule 3 closes whisper-large-v3.**
  No new split, normaliser or instrument. (The run's app-decode rows carried a
  cache defect on 200 of 925 clips, recorded in the outcome note; the deciding
  row and the clean greedy decode agree, and the cause is fixed in
  `speech_bench.py`.)
- **Before new audio, the recipe (added 8 September, evening).** Every Croatian
  clip in the mix was trained under `<|bs|>`, which is the mechanism behind the
  Croatian row that closed large-v3. The mix now carries one language token per
  clip and the trainer honours it; the run is pre-registered in
  `training/PREREGISTRATION.md`, "v4 — listen — one language token per clip",
  and the base it runs on is the owner's decision under rule 3.
- After that, only new audio moves anything: ParlaSpeech-HR (1,800 h Croatian,
  CC BY-SA, owner approval pending in `docs/WHITE-PAPER.md`) behind the term
  recall and Croatian substitution gate; or a small human-transcribed Bosnian
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
4. ParlaSpeech-HR under CC BY-SA; the monolingual Bosnian source and its
   licence for back-translation.

## What this plan will not do

Relaunch OCR passes 8–19 or the three PaddleOCR fine-tunes; train on labels
written by the model under training; touch a bar after a number exists; score
test-v2b with any reader before its human key exists; start a multi-hour job
on the Mac.
