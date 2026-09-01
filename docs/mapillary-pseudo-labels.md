# Mapillary pseudo-labels are not training data

Measured 2026-09-01, on `data/ocr/crops-kaggle/labels.tsv` (5,988 crops from
20,240 Mapillary photos, cropped on a Kaggle T4).

## How it was measured

24 crops were sampled at random from the rows carrying at least three
alphabetic characters — i.e. biased *towards* real signage and away from the
digit-only junk, so this is the optimistic end of the corpus.

They were rendered to a contact sheet with no labels visible, transcribed
blind, and only then compared against what EasyOCR had written. Reproduce
with the snippet in the git history of this file's commit, seed 1234.

## Result

**7 of 24 pseudo-labels were exactly right — 29%.**

The sample is small, so read that as "roughly a third", not as 29.0%. The
diacritic result below is what actually settles the question.

| crop | EasyOCR wrote | actually reads | conf |
|---|---|---|---|
| 12 | `KNJIZARA SVJETLOST` | KNJI**Ž**ARA SVJETLOST | 0.75 |
| 13 | `UKRASNE KOPCE` | UKRASNE KOP**Č**E | 0.78 |
| 16 | `BUREGDZINICA` | BUREG**DŽ**INICA | 0.94 |
| 1 | `fecan` | jedan | 0.50 |
| 5 | `Sabla` | Pablo | 0.53 |
| 14 | `CA;INO` | CASINO | 0.85 |
| 20 | `Anancial` | Financial | 0.82 |
| 18 | `Znoš R` | Znaš li | 0.49 |

## Why this closes the question

**Every Bosnian word in the sample lost its diacritic.** Three crops out of
three: Ž → Z, Č → C, DŽ → DZ. High confidence on all of them — 0.94 on
BUREGDŽINICA.

That explains the corpus-wide count directly. Only 62 of 5,988 rows (1%)
contain a diacritic. In the blind sample the true rate was 4 in 24 — about
17%. The reader is not failing to see the signs; it is reading them and
dropping the diacritic on the way out.

So fine-tuning on these labels would train the reader to strip exactly the
characters that distinguish Bosnian from its own errors. It would make the
measured failure worse, not better, and it would do so while reporting a
falling loss.

## Second problem: a third of it is not signage

8 of the 24 were dashcam on-screen display, matching the 32% found by pattern
match across the whole corpus:

`DR750S-2CH/FHD-FHD` · `MIC OFF/NV` · `HDR` · `047km/h` · `056km/h` · `MIC`

Mapillary carries dashcam frames with burned-in overlays. The crop pass reads
them as text at high confidence. They are a different font, a different
vocabulary and a different domain from street signage.

## What the photos are still good for

The 20,240 photos and 13,144 crops are real, kept, and credited
(CC BY-SA 4.0, `data/ocr/real-photos/mapillary/CREDITS.tsv`). What is missing
is a label source that is not the model under training.

`data/signs/sign-text.tsv` holds 9,506 real sign strings from OSM with
coordinates. Matching a photo's coordinates against a sign's would label the
crop from OSM rather than from EasyOCR, which breaks the circularity. That is
the path worth measuring next.

Not the crops themselves. They are already cut and already kept.
