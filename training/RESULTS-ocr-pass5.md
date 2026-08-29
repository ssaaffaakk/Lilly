# OCR pass-5 — Kaggle result

Run: Kaggle `afaksrmeli/lilly-ocr` version 10, Tesla T4, 29 August 2026.
Code cloned from GitHub at `785e459`. Numbers below are from the Kaggle log
only. No Mac photograph eval.

## Verdict: not installed

The install gate refused the new weights. `lilly-read.zip` was not written.
The app still uses the pass-4 `models/lilly/read/lilly.pth`.

Fetched for inspection only:

- `models/kaggle-output/ocr-v10/lilly-read-trained.zip` (14.3 MB; contains
  `read-trained.pth`, 15,406,489 bytes)
- `models/kaggle-output/ocr-v10/lilly-ocr.log`

Do not unzip that zip over `models/lilly/read/`.

## Crop gate (same valid set, same scorer, before and after)

| | words exact | Bosnian letters |
|---|---|---|
| before (pass-4 `lilly.pth`) | **88.8%** | **96.1%** of 332 |
| after 5 epochs, real×6 + 30k synthetic | **88.0%** | **96.1%** of 332 |

Words fell 0.8 points. Diacritics were unchanged. The trainer's own rule is
both must rise; neither did.

Training itself finished: 10,865/10,865 steps, last logged loss 0.1298.
The run did not collapse. It just did not beat the reader it started from
on the crop gate.

## What it trained on

| | |
|---|---|
| start weights | attached `lilly-read-pass1` / `lilly.pth` |
| real train crops | 1,294 × 6 |
| synthetic | 26,994 (30k generated, seed 43, split leftover) |
| train rows after oversample | 34,758 |
| valid (not repeated) | 3,138 |
| epochs | 5 at 1e-5 |

The 75%/85.7% pair printed earlier in the log is the `--quick-test` smoke
(7 Bosnian letters), not the gate.

## What this does not settle

The crop gate is mostly synthetic. Pass-4 already sat at 87.8% words / 94.8%
diacritics on that distribution; this pass started at 88.8/96.1 and ended
slightly worse on words. Real-photograph recall was not measured here
(train first, measure last). Repeating the same 1,294 labelled crops six
times did not lift the crop numbers, which is what the gate saw.

Kaggle's `kernels files` listed this zip as 853 bytes while the download
was 14.3 MB. Status COMPLETE plus a direct `kernels output` is the fetch
path; the files listing size is not.
