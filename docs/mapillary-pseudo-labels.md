# What the Mapillary pseudo-labels are worth

Measured 2026-09-01 on `data/ocr/crops-kaggle/labels.tsv` — 5,988 crops from
20,240 Mapillary photos, cropped on a Kaggle T4.

Two blind samples were read: crops were rendered to a contact sheet with no
labels visible, transcribed, and only then compared against what EasyOCR had
written.

## The answer depends entirely on confidence

| sample | n | exact | ignoring diacritics |
|---|---|---|---|
| any confidence, ≥3 letters | 24 | 29% | 54% |
| **conf ≥ 0.85, no dashcam OSD** | **30** | **77%** | **93%** |

At 0.85 and above the labels are good. Two errors in thirty, both on crops
whose edge was clipped: `Cen` read as `Cer`, `Cash` as `cast`.

Diacritics survive at that threshold. `ODJEĆE`, `BUREGDŽINICA`, `MUŠKA`,
`Pošta` all came back correct.

### Correcting an earlier reading of this

An earlier version of this file concluded that the reader "strips every
diacritic" and that the corpus was unusable. That was drawn from the
first sample only, which spanned all confidences, and it was wrong. Ž → Z and
Č → C do happen, but in the low-confidence tail, not at the threshold anyone
would actually train on. The 1%-of-rows-carry-a-diacritic figure counts the
whole corpus, most of which is junk, and does not describe the usable part.

## How much usable data there is

Rows with at least three letters, excluding dashcam on-screen display:

| threshold | crops |
|---|---|
| ≥ 0.95 | 618 |
| ≥ 0.90 | 795 |
| ≥ 0.85 | 967 |
| ≥ 0.80 | 1,109 |
| ≥ 0.75 | 1,253 |
| ≥ 0.70 | 1,438 |

Accuracy was measured at 0.85. The bands below it are unmeasured and should
not be assumed to hold 93%.

## The part that is genuinely junk

A third of the raw corpus is dashcam on-screen display, not signage.
Mapillary carries dashcam frames with burned-in overlays and the crop pass
reads them at high confidence: `DR750S-2CH/FHD-FHD`, `MIC OFF/NV`, `HDR`,
`047km/h`. Different font, different vocabulary, different domain. These are
excluded by pattern, not by confidence — confidence does not catch them.

Beograd and Banja Luka were dropped earlier for Cyrillic signage: 7,156 of
the original 13,144 crops, quarantined in `crops-kaggle/dropped-cyrillic/`
rather than deleted.

## Where that leaves training

The keep list is 915 rows, written 2026-09-01 by
`data/scripts/filter_mapillary_train.py` (conf ≥ 0.85, ≥3 letters, no dashcam
OSD). Quarantine folders sit beside the keep PNGs; nothing was deleted.
Train images are in `data/ocr/mapillary-train/`, loaded with
`train_ocr.py --train-dir` so they never enter `data/ocr/train` or
`data/ocr/valid`.

The product bar for this set is meaning through the translator, not surviving
hats: `kuca` → House is enough. Rows that EasyOCR folded (`kuća` labelled
`kuca`) stay in. The crop gate still counts čćđšž and can refuse a run; that
is the notebook, not a reason to drop the row.

Do not mix these 915 with sign-letter plates in one `train_ocr`. Do not
relaunch pass-8/9/10/11. Whether 915 shop-sign crops move the photograph
score is not known until a run returns.
