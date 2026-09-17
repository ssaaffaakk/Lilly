# Read clean-eval — Kaggle, 17 Sep 2026

Eval-only. No training, no default change, no publish.

Shipped reader: PP-OCRv6 medium, floor 0.9, through `app.ocr.scan`. Kernel
`safak3as/lilly-read-clean-eval` v2, git `94415fc`, Tesla T4, COMPLETE
18:56 UTC. Manifest SHA-256
`5614d4040b79801262301735351c7871a099cda8843d6ebb0aca1bb196d6b472`. Reader
fingerprint `20da45dcc3a4c9a5`. 40/40 Commons originals verified against the
committed SHA-1 list (attached dataset + SHA-1 copies; zip entry names are not
identity).

Cohorts were frozen **before** this inference
(`training/clean-eval/ocr-commons-40.tsv`). Unreadable photographs are not a
verdict on the reader.

## Side by side

The reader did not change. The mix did.

| | what is in the mix | score |
|---|---|---|
| **All 40 — why it is 67%** | 21 clean + **6 blurry** (a person can still read them) + **1** a person cannot read + **12 empty** (no text) | **67.0%** words/photograph published; this run **67.9%**, 72 invented |
| **Outdoor photos the person building this app actually takes** | the street shots that person gets outside: **normal frames and slightly blurry ones a human can still read** (**21** in this set) | **82.5%** (273 / 331 words), 53 invented |

67% is pulled down by empty frames and the one nobody can read. **82.5% is the
same reader on the outdoor photographs the person building this app actually
shoots** — normal and slightly blurry, still readable. Neither replaces
`test-v2` (**57.8% found, 450 invented** on 132 photographs with text).

## Frozen cohorts (detail)

Word-pooled recall. Labels frozen before inference, not by the model.

| cohort | photographs | found / words | recall | invented |
|---|---:|---:|---:|---:|
| **clean** | 21 | 273 / 331 | **82.5%** | 53 |
| noisy-but-understandable | 6 | 14 / 36 | 38.9% | 15 |
| human-unintelligible | 1 | 4 / 6 | (not a capability score) | 2 |
| no-text controls | 12 | 0 / 0 | — | 2 |
| **legible overall** (clean + noisy) | 27 | 287 / 367 | 78.2% | 68 |
| all 28 with text | 28 | 291 / 373 | 78.0% pooled | 70 |

Pooled 78.0% is not either headline: `Spanish_square_08034.JPG` holds 144 of
373 key words. Equal weight per photograph on all 40 is **67.9%**. Diacritic
words 18/25 (72%); folded 23/25 (92%).

The 14-photo Latin-only reslice at 89% (`training/RESULTS-ocr-clean-legible.md`)
is a different cut of stored output, not this Kaggle run.

v1 of this kernel ERROR'd at 9/40 because Kaggle's zip hid `Međugorje_Banner.jpg`
under a Unicode filename. That was a missing photograph, not a reader score.
v2 is the score.
