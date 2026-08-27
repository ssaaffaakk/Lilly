# Where the work stands — 27 August 2026, evening

Written so the next session does not have to rediscover any of this. Delete it
when it stops being true; a stale handoff is worse than none.

## The three numbers, all measured today

| | measured | on what | means |
|---|---|---|---|
| Translation | chrF2 **67.34** | FLORES-200 devtest, 1,012 pairs, through `app.translate.Engine` | level of NLLB-200-3.3B (67.2), best of 30 published systems on this pair |
| Photographs | **36.0%** of words per photo | 40 real Commons photographs, answer key from two blind transcribers at 91% agreement | was 75% on our own synthetic crops — that generator was too easy |
| Speech | **35.5%** word error | 200 held-out FLEURS Bosnian clips | untouched whisper-small reads 38.5% on the same clips, so the fine-tune is worth 3.0 points |

**The finding that matters more than any of them:** the translator's fine-tuning
does not move chrF2. −0.16 at p = 0.128, a bootstrap tie. Its BosnianBench term
recall moves +0.5 at p = 0.36. What the fine-tuning did achieve is real and
narrower: the base model prints its own language tag into 30.4% of its
translations and Lilly into none, and with that stripped from both sides BLEU
still moves +1.29 at p = 0.001.

So the honest claim is: the base model was already excellent, we measured that
carefully enough to know it, we fixed a visible defect, and our own training is
worth a small BLEU gain and nothing measurable on chrF2.

Scales for all three are pre-registered in `training/RUBRIC.md`, written before
the measurements and anchored to published systems by agents who were not told
how Lilly scores.

## Open, in order of urgency

1. **1,914 crops are being labelled blind** as 240 sheets under
   `data/ocr/label-sheets/`, twelve annotators, answers landing in
   `data/ocr/label-answers/`. Merge with
   `python3 data/scripts/label_crops.py collect`, then
   `prepare_ocr_data.py --labels data/ocr/crops/labels-human.tsv`, train, and
   score on the same 40 photographs. Note what the training data is today: all
   of it is synthetic — 18,060 `syn*` from one generator and 3,592 `photo*`
   from the other — so **the reader has never seen a real crop.**
2. **The harvest will not reach 1,500 and never could.** Measured at 1,499
   screened: keep rate 17.4% overall, 20.2% on live candidates, 263 kept, ~388
   still staged. That lands near ~340 photographs from this run. The ceiling is
   the pool: 3,207 candidates at ~20% is ~650 even if every one were screened,
   and `--max-screen` defaults to 1,400 per run on top of that. `--target 1500`
   is not a goal this pool can meet.
   (An earlier note here said ~215–250. That read the 1,400 staged as a cap on
   the run; it is a staging window the harvester refills, so the figure was low.)

## Settled today

**The Kaggle speech ERROR was version skew, not the TSV flush.** The run died
at `In [6]` on `download_extra_speech.py: error: unrecognized arguments:
--source fleurs_hr --hours 12`, thirty-five minutes in, after the 3 GB of
Bosnian audio had already come down. The usage line it printed lists
`--lang`/`--max-clips`, which is commit `1a20440`'s downloader. The run started
15:36 UTC; the rewrite `630823a` was committed 15:46 UTC. The notebook went up
from the working tree with the new interface and the `git clone` inside the run
brought back the old script. `scripts/kaggle_train.py` now refuses to launch
unless the tree is clean and GitHub has the exact HEAD by SHA, and there is
deliberately no `--force`. The flush fix in `c9ad34f` is still correct; it just
was not this.

## Dead ends, measured — do not re-try these

- **Mapillary**: 10 frames, zero words above confidence 0.3, one "detection" was
  the dashcam's own watermark.
- **Panoramax**: independently reproduced that result from a different provider,
  and 7 of 8 sampled frames are under the 800px short-side bar anyway. The
  failure is structural to vehicle-mounted capture, not specific to Mapillary.
- **KartaView**: metadata API answers, every image URL returns 502.
- **Europeana / National & University Library of BiH**: the entire BiH
  collection is ~30 items, all CC BY-NC-ND, and the only text in them is
  publisher captions printed flat across the sky. That is the same distribution
  as our synthetic data, which is exactly the thing that cannot close a 75-to-36
  gap.
- **podaci.gov.ba**: tabular statistics, no imagery, no machine API.
- **Turkisms in BosnianBench**: 18.5M OpenSubtitles pairs measured; not one of
  23 terms clears the 98% share gate, and the constraint is the rule rather than
  the corpus. Recorded rather than fixed by lowering the bar.

Wikimedia Commons is the only source that works.

## Traps this project has already fallen into

- The Kaggle notebooks **clone from GitHub**. Push and verify before launching,
  and verify by commit SHA — `raw.githubusercontent.com/<repo>/main/` is
  CDN-cached and served a two-commit-stale file today.
- `scripts/state.py` reports uncommitted, unpushed and untracked work and asks
  GitHub for HEAD. Run it before calling anything done.
- Score through `app.translate.Engine` and `app.ocr.scan`, never the layer
  underneath. Both have been measured on the wrong path before, and the
  translation one moved the answer by 3.36 chrF2.
- `pgrep -f` takes extended regex: `\|` is a literal pipe, not alternation.
- **The scored photographs were living inside the harvester's scratch area and
  it was deleting them.** `harvest_sign_photos.py` unlinks a staged rendering
  the moment the verdict is `drop` (:934) and empties the whole staging
  directory when a run ends (:1036) — both correct for scratch, neither
  survivable for a test set kept in one. Twenty-five of the forty were already
  gone when this was found and the run still going would have taken the rest.
  They are now in `data/ocr/real-photos/scored/`, which no harvester walks,
  restored from `screen_url` (the 1280px rendering that was read — the
  `keep_url` original is a different picture and would have re-scaled the whole
  measurement). Verified faithful: three re-fetched photographs read word for
  word identically to the cached readings from the 36% run.
- **`truth.json` was not on GitHub.** The answer key two people transcribed
  blind at 91.2% agreement, 6.8 KB, one disk failure from taking the entire
  photograph measurement with it. Tracked now, with `scored-sources.tsv` beside
  it so a fresh clone can rebuild all forty pictures. `data/ocr/` is ignored
  wholesale, so anything irreplaceable put under it has to be un-ignored
  explicitly — `labels-human.tsv` and `label-answers/` are, for the same reason.
- **Re-running `training/sample_photos.py` silently replaces the ruler.** Its
  default pool was that same staging directory, so post-cleanup it would draw a
  fresh sample from whatever survived and overwrite `scored-sample.txt`, leaving
  `truth.json` transcribed for photographs no longer in the sample. It now
  refuses when the pool is gone and says what re-sampling costs.
