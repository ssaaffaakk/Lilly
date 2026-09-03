# OCR roadmap — what the reader does next, and what it will not repeat

Written 3 September 2026. Read this before touching anything under `training/`
or `data/ocr/`. It replaces the Mapillary guidance that used to sit in
`.claude/CLAUDE.md` and `.cursor/rules/ocr-mapillary.mdc`. When a step lands,
update the status board at the bottom and push the same day.

Owner: Safak. Decisions marked **owner** are theirs, not an agent's.

## When you are next at the Mac — in this order

```bash
git fetch origin && git checkout claude/new-session-455uxc      # or merge it to main first
.venv/bin/pip install paddleocr paddlepaddle                    # once; not in requirements.txt
.venv/bin/python3 scripts/bakeoff_ocr.py --arms lilly paddle-v5 --limit 3   # smoke, minutes
.venv/bin/python3 scripts/bakeoff_ocr.py                        # step 1, about an hour
git add training/bakeoff training/RESULTS-ocr-bakeoff.md && git commit && git push
.venv/bin/python3 data/scripts/fetch_test_v2.py                 # step 2 pixels, minutes
git add data/ocr/real-photos/test-v2 && git commit && git push  # CREDITS + log, not photos
git add -f data/ocr/real-photos/mapillary/CREDITS.tsv && git commit -m "Mapillary credits" && git push
.venv/bin/python3 training/build_test_mly.py                    # step 2b draw; commit test-mly/
.venv/bin/kaggle datasets status afaksrmeli/lilly-mapillary-photos   # just to know it is there
```

Then the two-person transcription of `test-v2` (step 2), and the box count
(step 3). Neither needs a GPU or Kaggle.

---

## Why this file exists — the post-mortem in six lines

1. Six passes (14–19) on five training sets, one answer: **no gain**. Cause: the
   labels were written by EasyOCR, the reader being trained. A model fine-tuned
   on its own confident output learns what it already knows. The pipeline
   reported that faithfully; the experiment could not have succeeded.
2. The training crops were shop fascias at 1.2 words a crop (`kuca`, `CITY`,
   `BAZAR`). The test set is memorial plaques and notices (`KRIVIČNO JE
   DJELO`). Commit `a12a2fb` diagnosed this — and then passes 18 and 19 ran on
   the same data anyway.
3. The gate's held-out set is **132 crops with 25 diacritic letters**. One
   letter is 4 points. Every accept/refuse was decided on 1–3 crops, which is
   noise. You cannot optimise what you cannot measure.
4. The "737 clean human crops" used to compare passes 18 and 19 against the
   shipped reader: **666 of the 737 are on the training side of the split**
   (`training/ocr_split.is_valid_text` over `labels-human-latin.tsv`). The
   shipped reader trained on them. Only 71 are held out. The 45.0% folded figure
   is mostly memorisation, and the comparison favoured the reader that saw those
   crops most recently.
5. The `bs_char.txt` bug (labels could never contain č ć đ,
   `docs/crop-labels-were-crippled.md`) is real and secondary: the folded score,
   which ignores those letters, did not move either.
6. Most of the hours went into Kaggle plumbing — literal newlines in string
   literals, stale staging copies, `git clone` exit 128 — and into gate
   refusals, not into the question of what would teach the reader something.

## The numbers that stand, and how far to trust each

| number | set | trust |
|---|---|---|
| **54.7%** words per photograph, 45.0% pooled, 180 invented words | 40 Commons photographs, 373 agreed words, excluded from every training set (`training/RESULTS-ocr-restored.md`) | the product number. Small: the pooled figure carries a ±5-point interval, and 28 photographs is all it is |
| 44.0% of diacritic words (11/25) | same set | ±19 points. Cannot move by design |
| 43.2% folded on the crop gate | 132 held-out human crops | honest but ±8 points |
| 45.0% folded on 737 crops (GPU, bare `readtext`) or 47.9% (Mac, app door with allowlist) | `labels-human-latin.tsv`, 666/737 training-side | **do not use to compare readers**. Two numbers exist for one reader on one set because two code paths were used; `training/score_crops.py` uses the app's door and names the split |

### Verified on 3 Sep 2026, so nobody re-checks it

- **The 54.7% is clean.** No crop in `crops/labels-human.tsv`, `crops2/` or
  `label-answers/` was cut from any of the 40 scored photographs — checked by
  crop filename and by Commons page URL through `harvested/CREDITS.tsv`. The
  harvester did *download* 15 of the 40 (they are in `CREDITS.tsv` and
  `screened.tsv`), but none of them reached a labelled file. The exclusion in
  `RESULTS-ocr-realcrops.md` holds at the crop level, which is the level that
  matters.
- **The shipped reader trained on 1,294 human crops and held out 132**
  (`RESULTS-ocr-realcrops.md`), 1,426 usable Latin crops in all — which is why
  666 of the 737 clean crops are training-side.
- **36.0% → 54.7% is per-photograph on both ends** (`RESULTS-ocr.md`,
  `RESULTS-ocr-restored.md`). The 36.0% also happens to be the old diacritic
  rate (9/25); the coincidence has misled at least one reader of the docs.
- **The shipped reader has two published numbers on the 737 crops: 47.9%
  folded (`docs/human-crop-set.md`, on the Mac, through `app.ocr` with the
  allowlist) and 45.0% (`RESULTS-ocr-pass19.md`, on the GPU, bare
  `easyocr.readtext` with no allowlist).** 21 crops apart on the same weights.
  Not the machine — the door. The allowlist keeps Cyrillic and Vietnamese
  lookalikes out of the guesses, and the app has it. Score through the app's
  door or say which door was used.
- **`data/ocr/crops-kaggle/labels.tsv` is not the 5,988-row file the manifest
  says it is.** Commit `ac2be14` overwrote it with the 915-row keep list; at
  HEAD the two files are byte-identical. The 5,988-row record is restored from
  `3e5fa6e` as `labels-latin-all.tsv`, and the manifest has a row saying so.
- **`data/ocr/real-photos/mapillary/CREDITS.tsv` is not in git.** It is the
  only map from `mly_<id>.jpg` to city and CC BY-SA attribution — the same
  shape of irreplaceable the `.gitignore` comment describes for the Commons
  credits. `.gitignore` now admits it; committing it is on the Mac checklist.
- **The label TSVs bite Python's csv module.** A label such as `"LUCIJA"`
  opens a quoted field and swallows every following line: the default reader
  returns 2,970 of the 5,988 rows. Read them with `quoting=csv.QUOTE_NONE` or
  split on tabs. The scripts in this repo that matter do; `screened.tsv` and
  the Commons `CREDITS.tsv` parse correctly either way, so the `test-v2` draw
  is sound.
- **The Mapillary harvest is pre-screened by the reader under test** — see
  step 2b.
- **`evaluate_ocr.py`'s reading cache is keyed on the weight files, not on
  which reader is loaded.** `LILLY_READER=stock` or `=paddle` with the default
  `--cache` would score the *trained* reader's cached readings under the other
  name — the exact failure the cache stamp was added to prevent. Fixed the same
  day: the stamp now carries `app.ocr.reader_identity()`, and the bake-off
  runner gives every arm its own cache file anyway.

## What the 28 already say about signs against boards

From the shipped reader's per-photograph table in `RESULTS-ocr-restored.md`,
split by agreed words in the key:

| | photographs | words per photograph | read in full |
|---|---|---|---|
| signs, 1–5 words | 13 | **61.9%** | 5 |
| short boards, 6–20 | 12 | 46.4% | 2 |
| long boards, 21+ | 3 | 56.9% | 0 |

Signs read better than boards, which is the owner's intuition — and 8 of the
13 signs are still not read in full, four of them at zero (`Gospodska_ulica_27`
0/2, `Sarajevo_Trebević_Sign` 0/6). Thirteen photographs is not a number to
build on; it is the reason `test-v2` and `test-mly` exist.

## The line that is closed

- Fine-tuning `latin_g2` on EasyOCR-labelled Mapillary crops — passes 14 through
  19, `training/RESULTS-ocr-pass14.md` … `pass19.md`. Do not relaunch any of
  them, in any configuration.
- Do not relabel the 20,240 photographs with `recog_network="lilly"` and try
  again. That fixes č ć đ in the labels and nothing else; it is still the model
  labelling its own training data. The relabel pass proposed in
  `docs/crop-labels-were-crippled.md` is **not scheduled**.
- The photographs are not deleted. 20,240 `mly_*.jpg` sit in the Kaggle dataset
  `afaksrmeli/lilly-mapillary-photos` and on the Mac under
  `data/ocr/real-photos/mapillary/`. They become training data only when their
  labels come from somewhere other than EasyOCR (step 4) — and only if the
  owner says shop signs are in scope (open decision 1).
- The 713-row list pass-19 trained on is now in git at
  `data/ocr/mapillary-train-clean/gt.txt`, so the record is complete. The
  915-row list is `data/ocr/mapillary-train/gt.txt`. Neither is a train set.

## Steps, in order

### 0. Freeze — done by the commit that adds this file

No OCR pass launches until step 1 has a number and step 2 has a set. The
`.claude/CLAUDE.md` OCR section, `.cursor/rules/ocr-roadmap.mdc`,
`docs/kaggle-fail-stop.md` items 16–18 and `scripts/preflight_kaggle.py` say the
same thing, so no agent can fall back into the old line by reading an old file.

### 1. Engine bake-off: PaddleOCR PP-OCRv5, untrained, against the shipped reader

**Why first.** It is the cheapest experiment left, it needs no GPU and no
Kaggle, and it decides whether the next months go into EasyOCR or not.
PP-OCRv5's `latin_PP-OCRv5_mobile_rec` names Bosnian, Croatian and Serbian
(Latin) in its language list and its dictionary holds all of `čćđšžČĆĐŠŽ`
(checked against the dict file, not assumed). `cyrillic_PP-OCRv5_mobile_rec`
covers Serbian Cyrillic. The detector is a generation newer than CRAFT, and the
detector has never been touched here.

**How — built 3 Sep 2026, waiting for the Mac.**
- `LILLY_READER=paddle` in `app/ocr.py`, behind the same door as everything
  else: `read_regions` returns the same `(box, text, confidence)` triples and the
  same paragraph grouping, so `training/evaluate_ocr.py` and
  `training/measure_detection.py` run unchanged. `LILLY_PADDLE_VERSION`
  chooses PP-OCRv6 (default) or PP-OCRv5.
- The bar is written: `training/PREREGISTRATION.md`, "v2 — read — bake-off".
  Not worse on words-per-photograph or invented words, better on at least one;
  the shipped arm is re-run and must reproduce 54.7 / 180.
- One command runs all four arms — shipped, stock, PP-OCRv6, PP-OCRv5 — on the
  40 photographs, on the 71 held-out crops (the 666 training-side in a row of
  their own), and times one 2 MP read, then applies the rule:
  `python3 scripts/bakeoff_ocr.py` (`--dry-run` shows the commands,
  `--arms lilly paddle-v5 --limit 3` smokes it). Report:
  `training/RESULTS-ocr-bakeoff.md`, raw counts in `training/bakeoff/`.
- `pip install paddleocr paddlepaddle` on the Mac first; it is deliberately
  not in `requirements.txt` — a measurement path, not the app.

**Where.** The Mac, or a Kaggle CPU notebook. Not a Claude Code cloud session:
its network policy refuses `kaggle.com`, `upload.wikimedia.org`, both
`bcebos.com` weight hosts and `huggingface.co`, so neither the photographs nor
the weights can arrive there.

**Cost.** Half a day. No training.

### 2. A test set that can see a change

**Why.** 373 words cannot distinguish a 3-point gain from noise; 25 diacritic
words cannot distinguish anything. Every decision downstream — engine, detector,
fine-tune — needs this set first.

**What — pool built and draw frozen 3 Sep 2026; pixels and people still to come.**
- `training/build_test_v2.py` builds the pool from files in git — every
  photograph the harvester screened (2,665), minus the 40, minus the 190
  photographs any labelled crop was cut from, minus drawings and refused
  licences — and draws **280** by the same filename hash `sample_photos.py`
  used for the 40. `data/ocr/real-photos/test-v2/pool.tsv` (2,350 eligible,
  every exclusion named) and `sample.txt` (the draw, frozen) are committed.
- Not excluded, on purpose: the photographs the harvester dropped for zero or
  one confident text region — 71% of the pool. A set that kept only what the
  detector liked would be selected by the detector under test. 25 of the 40
  were such drops and half of them carried text.
- On the Mac: `python3 data/scripts/fetch_test_v2.py` fetches the 280 at
  1280 px (the size the 40 were read at), re-checks licences, writes
  `CREDITS.tsv` and `fetch-log.tsv`; commit both, not the photographs.
- Transcription: same method as `truth.json` — two passes, blind to each
  other and to the machine, `legibility: clear` lines only. The harness is
  `training/transcription_pass.py`: `sheet --set test-v2 --pass a` writes the
  empty sheet, each pass fills its own copy (a person, or a vision agent that
  never sees the reader's output), `check` refuses holes, copies and anything
  that matches the reader's cached output, `pair` writes what
  `build_truth.py --result … --out data/ocr/real-photos/test-v2/truth-v2.json`
  reads. Agreement rate recorded. Same commands with `--set test-mly`.
  Expect roughly 1,500 agreed words over 150+ photographs with text; if it
  lands short, draw the next in rank with `--count`, never by choosing.
- Scoring, unchanged code:
  `python3 training/evaluate_ocr.py --truth data/ocr/real-photos/test-v2/truth-v2.json
  --photos data/ocr/real-photos/test-v2/photos --sample data/ocr/real-photos/test-v2/sample.txt
  --cache data/ocr/real-photos/test-v2/reader-output.json --out training/RESULTS-ocr-test-v2.md`
- Domain: Commons photographs of signs, plaques and boards, like the 40. If
  the owner's decision 1 adds shop signage, that is a second pool with its
  own draw, not a change to this one. Never Mapillary for this set.
- Split by **source photograph**, never by crop or by label text (the 272
  Cyrillic crops come from 37 photographs and five of them hold 61% —
  `HANDOFF.md`). The rest of the pool is what a future training set may be
  built from; `test-v2` is never trained on. Ever.

**Cost.** Mostly transcription time: two blind passes over 280 photographs,
many of them quick blanks.

### 2b. `test-mly` — the street-level half, the product's domain

- `training/build_test_mly.py` draws 240 of the 20,240 Mapillary photographs
  (Latin-signage cities only) by the same filename hash, from
  `data/ocr/real-photos/mapillary/CREDITS.tsv` — which exists only on the Mac
  and must be committed first (Mac checklist). Photos are already on the Mac;
  no fetch.
- Same transcription, same `build_truth.py`, same `evaluate_ocr.py` with
  `--photos data/ocr/real-photos/mapillary`. Output
  `training/RESULTS-ocr-test-mly.md`.
- **Stated bias:** the harvest kept a photograph only when Lilly's own reader
  found two or more words in it (`harvest_mapillary.py`, `has_readable_text`).
  This set is therefore "street photographs the shipped reader can find text
  in"; its recall is an upper bound and the selection favoured EasyOCR's
  detector. Both engines read the same photographs, so the comparison is
  still informative; a future harvest without that screen is the real fix.
- 1024px thumbnails, smaller than the 1280px Commons renderings. Say so
  beside the number.

### 3. Detection recall R_d — as pre-registered, never run

`training/PREREGISTRATION.md`, section "v2 — picture", defines it: draw every
box `training/measure_detection.py` returns, count by hand which of the 373
truth words are covered. `R_d` and `45.0% / R_d` say whether the ceiling is the
detector or the recogniser — the one question no pass has answered, and the
reason 180 invented words and 504 sub-16-px crops have stayed unexplained. Do it
on the 40 now (an hour of counting), for both detectors if step 1 puts
PaddleOCR in the running, and again on `test-v2` when it exists.

### 4. Only if fine-tuning continues: labels from anything but the model

- Allowed sources: people, or vision-model passes that never see the reader's
  output. Note what the record shows: the "hand-transcribed" crop labels in
  `data/ocr/label-answers/` are files named `agent-01.tsv` … `agent-27.tsv`,
  so the 1,701 "human" crops were, by the evidence in git, blind agent passes
  over the sheets `data/scripts/label_crops.py` builds. That is fine — it is
  the recipe — but the docs' word "human" should be read that way.
- The tool exists: `label_crops.py sheets` → blind transcription → `collect`.
  Run it on the 5,988 Latin Mapillary crops (`labels-latin-all.tsv` lists
  them; the PNGs are on the Mac) with two independent passes and only agreed
  labels kept, exactly as `build_truth.py` does for photographs. Report the
  agreement rate before any training.
- Forbidden: any EasyOCR reader, stock or fine-tuned, writing labels that
  EasyOCR then trains on.
- The training photographs must be the same kind as the test set. Shop signs do
  not teach plaques; that is the whole of passes 14–19.
- Scoring a trained reader: both readers in one process, same crops, same
  code, on the GPU that holds the weights (`training/cells/clean_crop_eval.py`
  is the pattern), on held-out crops only, with counts and the interval.

### 5. Cyrillic — after 1–3

Route by script rather than arbitrate by confidence (the confidence race
measured worse, `training/RESULTS-ocr-cyrillic.md`). Either PaddleOCR's
Cyrillic model, or `cyrillic_g2` fine-tuned on the 276 Cyrillic crops split by
source photograph. Not before the engine question is settled.

## How to report a run — so the next agent can trust it

Every RESULTS doc for the reader states, in this order: the set and its path;
which split it is (held-out / training-side / mixed, with counts); n; counts
beside every percentage (`329/737`, not `44.6%`); the 95% interval beside every
delta; and for two readers on the same crops, how many crops flipped each way,
not just the totals. A delta inside the interval is written as "no change".

## Do not repeat — the small mistakes that cost days

1. **Scoring on crops the model trained on.** `labels-human-latin.tsv` is
   666/737 training-side. Name the split before quoting the number.
2. **Deciding on a delta smaller than the noise.** 132 crops is ±8 points; 25
   letters is ±19. Write the interval; inside it is "no change".
3. **Labels from the model being trained.** Passes 14–19. See step 4.
4. **A bare `easyocr.Reader([...])` writing labels.** The stock `bs` character
   list cannot emit Č Ć Đ. Every reader goes through `app.ocr.read_regions`,
   which checks the weights can produce them.
5. **Training on a different domain than the test.** Shop signs versus plaques.
6. **Running another pass after the cause is diagnosed.** When a RESULTS doc
   says a line is dead, the next step is in this file, not in a notebook.
7. **Comparing readers through different paths.** Different machines (laptop
   CPU versus Kaggle GPU), different code, or a reading cache keyed on the
   photograph's name alone. Same process, same crops, same code; the cache
   carries the weights' fingerprint.
8. **Hand-editing notebook JSON, or pushing a stale staging copy.** Four runs
   died on one literal newline. Notebooks come from `training/build_*_notebook.py`,
   every cell `ast.parse`d, the builder writing the staging copy too.
9. **Treating transient infrastructure as a bug.** `git clone` exit 128, a
   full GPU queue: relaunch once, change no code.
10. **Counting impossible rows as failures.** Cyrillic through a Latin reader,
    the 12 blank photographs, crops under 16 px. Say what is in the denominator.
11. **Reading a percentage off 25 letters.** The gate's diacritic column moves
    4 points per letter. Quote the letters.
12. **Leaving the next step in a results doc.** The queue is this file. A
    "later" written anywhere else is lost.
13. **Leaving the real artefact on one machine.** The 713-row list lived only on
    the Mac; the dataset push had no script; the Mapillary credits still are.
    Labels, lists and manifests are committed the day they are made.
14. **Overwriting a record with a filtered copy under the same name.** The
    5,988-row `labels.tsv` became the 915-row keep list and the manifest kept
    pointing at it. A filtered file gets a new name; the record keeps its own.
15. **Reading label TSVs with `csv`'s default quoting.** One `"LUCIJA"`
    swallows three thousand rows in silence. `QUOTE_NONE`, or split on tabs.
16. **Screening a pool with the reader under test.** The Mapillary harvest
    kept only photographs Lilly's reader found two words in, so every number
    on it is an upper bound and every engine comparison on it leans toward
    EasyOCR's detector. Harvest without the reader; screen with people.

## Where things run

| task | where | why |
|---|---|---|
| PaddleOCR bake-off (step 1) | Mac, or a Kaggle CPU notebook | needs the Commons photographs and the weight downloads; cloud sessions cannot reach either |
| Kaggle pushes, polls, dataset uploads | Mac | the token lives in `~/.kaggle/kaggle.json` and nowhere else |
| Transcription (steps 2, 2b, 4) | a person, or a vision-agent pass that never sees the reader's output; two independent passes, agreed words only | the reader must not label its own test or training data |
| Box counting (step 3) | a person | by design — a machine counting its own boxes measures itself |
| Code, filters, docs, analysis over committed files | any agent, any machine | |

## Decisions — owner

1. **Answered 3 Sep 2026: Lilly reads small signs and street names. Long
   sentences are not the claim.** So the headline for the product is the
   sign class (1–5 agreed words), boards are reported beside it, and the
   street-level Mapillary photographs are the product's domain — as a *test*
   set first (step 2b), and as training material only with blind labels
   (step 4). Every RESULTS file for the reader now carries the split.
2. **The bar for switching engine** in step 1 stays the mean over every
   photograph, as pre-registered, with the sign row reported beside it. If the
   owner wants the bar on the sign row instead, that is written into
   `PREREGISTRATION.md` before the run, not after.

## Status board

| step | state | evidence |
|---|---|---|
| 0 freeze | done, 3 Sep 2026 | this commit |
| 1 bake-off | pre-registered, code ready, **needs the Mac** | `scripts/bakeoff_ocr.py`, PREREGISTRATION "bake-off" |
| 2 test-v2 | pool built, 280 drawn and frozen; **fetch + transcribe on the Mac** | `test-v2/pool.tsv`, `sample.txt`, `fetch_test_v2.py` |
| 2b test-mly | draw script ready; **needs the Mac** (credits file, then draw, then transcription) | `training/build_test_mly.py` |
| 3 R_d | overlays per detector ready; **needs the Mac and a person counting** | `training/measure_detection.py` (runs for `LILLY_READER=paddle` too), PREREGISTRATION "picture" |
| 4 labels | blocked on 1–3 | — |
| 5 Cyrillic | blocked on 1–3 | — |
