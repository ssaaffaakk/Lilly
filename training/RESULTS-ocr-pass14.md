# OCR pass-14 — refused

Kaggle `afaksrmeli/lilly-ocr-pass14` v2, 2026-09-02. Status ERROR, which here
is the crop gate working.

Trained once on the 915 Mapillary shop-sign crops through
`train_ocr.py --train-dir data/ocr/mapillary-train`, 3 epochs, batch 8,
lr 3e-6, from the shipped reader. 345 steps — the run happened.

## What the gate saw

```
before real:  42.4% words, 64.0% of 25 Bosnian letters (132 crops)
after  real:  41.7% words, 60.0%
before syn:   89.4% words, 92.5% of 2364 Bosnian letters (1955 crops)
after  syn:   88.3% words, 92.4%
```

`real-crop words did not rise — not replacing the shipped reader`

Nothing rose. Not a trade of letters for words, which is what the folded
labels had me expecting: all four numbers drifted down.

The real held-out set is 132 crops and carries 25 Bosnian letters across 23
of them, so 42.4 → 41.7 is under one crop and 64.0 → 60.0 is one letter.
These deltas are noise, and the honest reading is not "it got worse" but "it
did not get better, and the gate requires better".

## Why this was the likely outcome

Two reasons, and the second is probably the larger.

**The labels are the model's own output.** Training EasyOCR on crops EasyOCR
labelled reinforces what it already does, including what it already gets
wrong. A blind read put the 915 at 93% ignoring diacritics — good labels, but
good by the standard of the reader that wrote them.

**The domains do not match.** The held-out real crops are Wikimedia Commons
signage: information boards, memorial plaques, street-name signs, landmine
warnings. The 915 are shop fascias — KONZUM, TISAK, BBI Centar, PEKARA.
Brand lettering is a different typographic problem from an information board,
and improving on one need not move the other.

## What loosening the confidence threshold would do

Measured, not assumed. A second blind read of 30 crops from the 0.60–0.85
band scored 79% ignoring diacritics, against 93% at ≥0.85.

Worse, the dashcam filter leaks at low confidence, because it matches
correctly-spelled on-screen display. When the reader garbles the OSD itself
the row survives the filter: `053kih`, `NIC UFF /NU`, `auto8008b`. Six of the
30 sampled were dashcam frames.

Six such rows are in the 915 already, at high confidence: `053kih` at 0.91,
`ZCHFHD` at 0.99, `019kih`, `039kih`, `014kih`, `Auto-8088b`.

So the larger corpus at ≥0.60 — 1,880 crops — is both dirtier per row and
carries more out-of-domain frames. More of this data is not the lever.

## Do not

- Do not relaunch pass-14 unchanged. It ran, it was measured, it was refused.
- Do not weaken the crop gate or the photograph gate to get a COMPLETE. The
  refusal is the gate doing the one job it exists for.
- Do not read this as "Mapillary was wasted". 20,240 photographs and 13,144
  crops are kept and credited. What is exhausted is the idea of labelling
  them with the reader being trained.
