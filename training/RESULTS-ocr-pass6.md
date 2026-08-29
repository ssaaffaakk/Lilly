# OCR pass-6 — Kaggle result

Run: Kaggle `afaksrmeli/lilly-ocr` version 11, Tesla T4, 29 August 2026.
Code cloned from GitHub at `0bb341c`. Numbers below are from the Kaggle log
only. No Mac photograph eval.

## Verdict: not installed

The install gate refused the new weights. `lilly-read.zip` was not written.
The app still uses the pass-4 `models/lilly/read/lilly.pth`.

Fetched for inspection only:

- `models/kaggle-output/ocr-v11/lilly-read-trained.zip` (14.3 MB; contains
  `read-trained.pth`, 15,406,489 bytes)
- `models/kaggle-output/ocr-v11/lilly-ocr.log`

Do not unzip that zip over `models/lilly/read/`.

## Gate (pooled + split real vs synthetic)

| | words exact | Bosnian letters |
|---|---|---|
| before pooled (pass-4 `lilly.pth`) | **89.2%** | **95.1%** of 2010 |
| before real (132 crops) | **41.7%** | **64.0%** of 25 |
| before syn (3006 crops) | **91.3%** | **95.5%** of 1985 |
| after pooled | **88.4%** | **93.9%** of 2010 |
| after real | **42.4%** | **56.0%** of 25 |
| after syn | **90.4%** | **94.4%** of 1985 |

Real-crop words rose 41.7% → 42.4% but Bosnian letters fell 64.0% → 56.0%.
Synthetic words and letters both fell. Pooled metrics fell on both axes.
The trainer message: *"real-crop words rose but Bosnian letters fell — a trade.
Not replacing it."*

Training finished (~12 min on T4, not a crash). `train_ocr` exit 1 by design
when the gate refuses.

## What it trained on

| | |
|---|---|
| start weights | attached `lilly-read-pass1` / `lilly.pth` (pass-4) |
| real train crops | 1,294 × 2 (not ×6) |
| synthetic | 26,994 (30k generated, seed 43) |
| train rows after oversample | 29,582 |
| valid (not repeated) | 3,138 (132 real + 3006 syn) |
| epochs | 5 at 1e-5 |

The 75%/85.7% pair printed earlier in the log is the `--quick-test` smoke
(7 Bosnian letters), not the gate.

## vs pass-5

Pass-5 used real×6 and judged on the old pooled crop gate (mostly synthetic).
Pass-6 added explicit real/syn split reporting and tightened the install rule:
real words must rise **and** real letters must not fall; syn must not regress
on either axis. Real words ticked up slightly but letters dropped; syn also
regressed. Same underlying limitation: ~1,294 labelled real crops are not
enough to lift real-photo recall without hurting diacritics on the mixed valid
set.

## What this does not settle

Real-photograph recall on Wikimedia full frames was not measured here (train
first, measure last). Next lever is more labelled real crops (Flickr/NARA),
not another epoch pass on the same pool.
