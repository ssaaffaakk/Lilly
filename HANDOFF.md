# Where the work stands — 28 August 2026, morning

Written so the next session does not have to rediscover any of this. Delete it
when it stops being true; a stale handoff is worse than none.

> **3 Sep 2026:** for the reader, `docs/OCR-ROADMAP.md` supersedes the OCR
> items below. The Mapillary line (passes 14–19) is closed; the queue, the
> numbers to trust and the do-not-repeat list are there.

## The three numbers

| | measured | on what | means |
|---|---|---|---|
| Translation | chrF2 **67.47** | FLORES-200 dev+devtest, 2,009 pairs, through `app.translate.Engine`, tag stripped | level of NLLB-200-3.3B (67.2), best of 30 published systems on this pair |
| Photographs | **54.7%** of words per photo | the same 40 Commons photographs, answer key from two blind transcribers at 91.2% agreement | was 36.0% before training on real crops |
| Speech | **34.9%** word error | 200 held-out FLEURS Bosnian clips, measured here | the listener it replaced reads 35.5% on the same clips through the same code. Kaggle measured the same model at 33.9% on a T4 — quote the local pair for the like-for-like gain, the Kaggle pair (38.4% → 33.9%) only against untrained whisper-small |

*Corrected 28 Aug: the translation row read "chrF2 67.34, FLORES-200 devtest,
1,012 pairs". That figure is the BASE model's score on the full 2,009-pair set —
it was the model Lilly is measured against, printed as if it were Lilly, with
the wrong split named beside it. The installed build is Arm B: 42.18 BLEU /
**67.47** chrF2 on 2,009 pairs, 67.69 on the devtest half alone. See
`training/PREREGISTRATION.md`, "Outcome — Arm B wins".*

**And the frame that changes how the whole row reads.** The base is
`Helsinki-NLP/opus-mt-tc-big-zls-en` — a Marian model, NOT NLLB; the NLLB
reference above is a published comparison, and it has already been misread once
as an identity. Its release `opusTCv20210807+bt` has a per-row corpus manifest:
159,611,854 pairs across 44 corpora, containing WikiMatrix-v1, SETIMES-v2,
TED2020-v1 and wikimedia-v20210402 by name. **`data/clean/train.tsv` is inside
the base model's own training data, and so are `valid.tsv` and `test.tsv`,
drawn from the same pool.** Fine-tuning was re-weighting material the model
already had, not showing it new material. That invalidates nothing and is not a
defect — it is the correct frame for reading +1.29 BLEU, and the best
explanation anyone has produced for why chrF2 will not move. NTREX (1,924 rows)
is genuinely unseen on two independent grounds: published 15 months after the
cutoff, and zero rows in the manifest.

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
   only; `latin_g2` has 351 output classes and none are Cyrillic. **272** of the
   1,702 hand-transcribed crops (16.0%) and 26 of the 373 words in the scored
   answer key (7.0%) are Cyrillic. That needs a second recogniser, not more
   fine-tuning of this one.
   *Corrected 28 Aug: this said 276 crops. Counted three ways it is 272, and no
   definition reproduces 276.* **The 93% cap is the POOLED ceiling, not the
   per-photo one.** Cyrillic is not spread evenly — it sits on 7 of the 28
   photographs that carry text, and heavily: `Putokaz2.jpg` is 8 of 14 words,
   capped at 42.9%. Per-photo ceiling is **90.6%**. A bar quoted against the
   wrong one of those is quoted against a ceiling 2.4 points away.
   The 272 crops come from only **37 photographs**, one of which
   (`Information_board_in_Jajce`) is 97 crops — 35.7% of the whole Cyrillic set,
   top five 61.4%. So any Cyrillic fine-tune must split by SOURCE PHOTOGRAPH,
   not by crop, and should not expect the gain the Latin move bought from 1,294
   crops across 186 photographs.
   **And the rare letters are not in the data at all:** Џ 0, Ђ 3, Љ 3, Ћ 6,
   Њ 14, Ј 39. `RESULTS-ocr-cyrillic.md`'s "all twelve of Ђ Ј Љ Њ Ћ Џ verified
   present" is about `cyrillic_g2`'s OUTPUT CLASSES, not about our training
   data. Both sentences are true, they are about different things, and reading
   one as the other plans a run that trains a class with no examples.

2. **Diacritics are the reader's weakest column and real signage barely teaches
   them.** 180 of 1,702 real labels carry any of č ć đ š ž (10.6%); đ appears
   once in lower case in the entire set. *Corrected 28 Aug: the per-letter counts
   here were lower case only, and signage is mostly upper case. Counting both:
   č 41, ć 63, š 62, ž 39, **đ 8** (1 lower + 7 Đ). A recogniser treats Đ and đ
   as different classes, so 8 is the number that matters — still by far the
   rarest, so the conclusion does not move.* The synthetic crops are the only
   thing teaching those letters, which is why they stayed in the mix. More real
   photographs will not fix this on their own.

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
- **The harvester's delete-on-drop path has now destroyed data twice, and the
  second time nobody noticed for a day.** `harvest_sign_photos.py` unlinks a
  staged rendering the moment the verdict is `drop` (:934) and empties the whole
  staging directory when a run ends (:1036). The first casualty was 25 of the 40
  scored photographs, above. The second is
  `data/ocr/real-photos/train-photos/`: all 285 entries are symlinks into that
  same staging directory and **not one of them resolves**. What makes it fixable
  rather than merely observed is *why* those particular photographs went:
  `.state/screened.tsv` gives 218 of the 219 that are not also in `harvested/`
  the verdict `drop`, reason "only 1 confident region(s)". The 1,914 real crops
  were cut from photographs the harvester then judged not worth keeping and
  deleted — so the crops outlived their own sources by design, and a directory
  of symlinks was the only thing recording what they had been cut from. 66 are
  still in `harvested/` by name, 218 are re-fetchable from `screened.tsv` by
  Commons title, and `Mostar_street-2.jpg` is unrecoverable. Nothing downstream
  is blocked, because the crops themselves are real files. The rule this leaves:
  **a symlink into a scratch directory is not a record of anything**, and any set
  that must survive a harvest belongs outside every directory a harvester walks —
  which is why `scored/` exists.
- **`harvested/` contains 15 of the 40 scored photographs, and the byte hash
  cannot see 11 of them.** Identity here is the canonical Commons `File:` page:
  `scored/` holds the 1280px `screen_url` rendering that was actually read and
  `harvested/` the larger `keep_url` original, so only 4 of the 15 are
  byte-identical. A content-hash de-duplication reports the other 11 as distinct
  images. Train `picture-egitim` from `harvested/` and 37.5% of the test ruler is
  in the training set. `train-photos/` was the thing enforcing that exclusion —
  zero overlap — and it is now dead links.
- **Two of the 40 scored photographs are not photographs.**
  `Banjaluka_streetmap.jpg` is an OpenStreetMap raster render and
  `Narrow-Gauge-Railway_Spalatobahn_Station-Travnik.jpg` is a 1900s postcard.
  They inflate the headline by 1.7 points per-photo and 2.3 pooled: without them
  the figures are 53.0% and 141/330 = 42.7%. **The ruler is deliberately not
  being changed** — comparability with 36.0% and 54.7% is worth more, and
  re-sampling is the trap `sample_photos.py` refuses — but the 26-item figure is
  reported beside it every time. Note what the postcard scores: **9/9, the only
  item in the whole set the reader reads perfectly**, and the one whose text is
  flat, high-contrast, axis-aligned type with no perspective or lighting. That is
  the synthetic distribution scoring 100% inside a set that otherwise averages
  53%, which is independent corroboration of this project's own thesis.
