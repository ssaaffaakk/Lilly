# Where the work stands — 28 August 2026, morning

Written so the next session does not have to rediscover any of this. Delete it
when it stops being true; a stale handoff is worse than none.

## The three numbers

| | measured | on what | means |
|---|---|---|---|
| Translation | chrF2 **67.34** | FLORES-200 devtest, 1,012 pairs, through `app.translate.Engine` | level of NLLB-200-3.3B (67.2), best of 30 published systems on this pair |
| Photographs | **54.7%** of words per photo | the same 40 Commons photographs, answer key from two blind transcribers at 91.2% agreement | was 36.0% before training on real crops |
| Speech | **34.9%** word error | 200 held-out FLEURS Bosnian clips, measured here | the listener it replaced reads 35.5% on the same clips through the same code. Kaggle measured the same model at 33.9% on a T4 — quote the local pair for the like-for-like gain, the Kaggle pair (38.4% → 33.9%) only against untrained whisper-small |

**The finding that still matters more than any of them:** the translator's
fine-tuning does not move chrF2. −0.16 at p = 0.128, a bootstrap tie. Its
BosnianBench term recall moves +0.5 at p = 0.36. What the fine-tuning did
achieve is real and narrower: the base model prints its own language tag into
30.4% of its translations and Lilly into none, and with that stripped from both
sides BLEU still moves +1.29 at p = 0.001. The base model was already excellent,
we measured that carefully enough to know it, we fixed a visible defect, and our
own training is worth a small BLEU gain and nothing measurable on chrF2.

Scales for all three are pre-registered in `training/RUBRIC.md`, written before
the measurements and anchored to published systems by agents who were not told
how Lilly scores.

## Open, in order of urgency

1. **Bosnian Cyrillic is unreadable and nothing in the current plan changes
   that.** `app/ocr.py` builds `easyocr.Reader(["bs","en"])`, which is Latin
   only; `latin_g2` has 351 output classes and none are Cyrillic. 16.2% of the
   hand-transcribed crops and 7.0% of the scored answer key are Cyrillic, so the
   photograph ruler is capped at 93%. That needs a second recogniser, not more
   fine-tuning of this one.

2. **Diacritics are the reader's weakest column and real signage barely teaches
   them.** 180 of 1,702 real labels carry any of č ć đ š ž; đ appears once in
   the entire set. The synthetic crops are the only thing teaching those letters,
   which is why they stayed in the mix. More real photographs will not fix this
   on their own.

3. **The speech gain over the previous fine-tune is 0.6 points and untested for
   significance.** The big number (38.4% → 33.9%) is against *untrained*
   whisper-small. Against the listener actually replaced it is 35.5% → 34.9%,
   and the two Bosnian-term measures are both ties (p = 0.15, p = 0.09). The
   second run the pre-registration asked for — a different `--share` — has not
   happened, and would need more than the 3,430 Croatian clips `--hours 12`
   returned, since Bosnian was already above the 0.35 floor at 47%.

4. **SpeechBench is weak against Croatian drift specifically.** 73 of its 85
   targets are yat pairs whose alternative is Serbian. The run it just judged is
   the one that added Croatian. It passed, and the instrument had little for
   that particular failure to land on.

## Settled today

**The speech listener passed both pre-registered gates and is installed.** Word
error 35.5% → 34.9% and term recall 65.9% → 68.2%, both measured here on the
same 200 clips through the same code; variety substitution fell 5.1% → 3.3%.
"Both, not either", both hold. The listener it replaced is at
`models/lilly/listen-previous/`. Neither term difference is significant, so the
claim is that the Croatian audio did not make the model less Bosnian — the
failure the gate was built to catch — not that it made it more so.

**The reader now reads 54.7% of a photograph's words, up from 36.0%**, after
training on 1,294 hand-transcribed real crops alongside the 20,000 synthetic
ones. Pooled words 16.9% → 45.0%, diacritic words 36.0% → 44.0%, and — the line
that stops this being recall bought by guessing — invented words 224 → 180. It
finds more and makes less up. Full write-up in `training/RESULTS-ocr-realcrops.md`.

**All 1,914 crops are transcribed**, blind, by twelve annotators who were never
shown the reader's own guess. 1,702 usable, 39 marked as containing no text at
all (the detector's false positives) and 173 as unreadable — both excluded
rather than guessed at. The reader was exactly right on 278 of 1,702, **16.3%**,
which is the crop-level number behind the 36%.

**The Kaggle speech ERROR was version skew, not the TSV flush.** The run died at
`In [6]` on `unrecognized arguments: --source fleurs_hr --hours 12`, thirty-five
minutes in, after 3 GB of audio had already come down. The usage line it printed
is commit `1a20440`'s downloader. The run started 15:36 UTC; the rewrite
`630823a` was committed 15:46 UTC. The notebook went up from the working tree
with the new interface and the `git clone` inside the run brought back the old
script. `scripts/kaggle_train.py` now refuses to launch unless the tree is clean
and GitHub has the exact HEAD by SHA, with deliberately no `--force`.

**The harvest is finished: 286 photographs**, 524 MB, 1,114 dropped / 286 kept /
219 skipped / 7 refused — a 17.6% keep rate. CREDITS.tsv is clean: 286 rows,
nothing missing a licence, attribution or source page, no orphans either way.
It stopped on `--max-screen` (1,400 screenings per run), not on `--target 1500`,
and that target was never reachable: 3,207 candidates at 17.6% is about 560
photographs even if every one were screened.

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
- **Training wrote the reader into a file the app does not open.**
  `train_ocr.py` wrote `latin_g2.pth` and printed "the app reads with this now".
  `app/ocr.py` builds its reader with `recog_network="lilly"`, so easyocr loads
  `lilly.pth`, and nothing wrote that. A run that took held-out crops from 62.2%
  to 85.4% changed nothing a user would see — and scoring it afterwards would
  have re-read all 40 photographs, measured the *old* reader, and printed the
  result under the new one's name. The trainer now installs to `lilly.pth`,
  keeps the previous one beside it, and leaves `latin_g2.pth` pristine, because
  that file is what `LILLY_READER=stock` loads as the "before" build and what
  `ensure_pristine()` restores from.
- **The reading cache was keyed on the photograph's name alone.** Correct until
  the reader is retrained, and then the first score after a retrain reads every
  answer back out of the old reader's cache. It now carries a fingerprint of the
  weight files and throws itself away when they change.
- **Re-running `training/sample_photos.py` silently replaces the ruler.** Its
  default pool was that same staging directory, so post-cleanup it would draw a
  fresh sample from whatever survived and overwrite `scored-sample.txt`, leaving
  `truth.json` transcribed for photographs no longer in the sample. It now
  refuses when the pool is gone and says what re-sampling costs.
