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

1. **The Kaggle speech run is in ERROR and undiagnosed.** Fetch its log
   (`kaggle kernels output afaksrmeli/lilly-speech`). Most likely cause: the
   speech-data agent reported that the downloader flushes its TSV rarely, so
   36 WAVs were written after the last flush and their transcripts died in the
   buffer. The flush cadence is the defect; fix that, not the symptom.
2. **1,915 crops are cut and unlabelled**, in `data/ocr/crops/`, from 285
   photographs that exclude the 40 in the test set (verified zero overlap). Each
   crop carries our own reader's guess in column 2 of `labels.tsv` — that guess
   must NOT be shown to whoever labels them, or the reader gets trained toward
   its own errors. Blind transcription worked at 91% agreement on the test set;
   the same method scales here, single-annotator, because precision matters for
   the ruler and not for the clay.
3. **The harvest is still running** and has taken the collection from 325 to
   1,127 photographs. `--target 1500`.

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
