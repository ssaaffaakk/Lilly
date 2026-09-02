# OCR pass-15 — refused, and worse than pass-14

Kaggle `afaksrmeli/lilly-ocr-pass15`, 2026-09-02. ERROR: the crop gate
working.

All 5,988 non-Cyrillic Mapillary crops, batch 16, 2 epochs, 750 steps, lr
3e-6, from the shipped reader. Pass-14 was the same run on the 915 that
survived a confidence and dashcam filter.

## More data made it worse, in proportion

| | crops | real words | real letters | syn words |
|---|---|---|---|---|
| pass-14 | 915 | 42.4 → 41.7 | 64.0 → 60.0 | 89.4 → 88.3 |
| pass-15 | 5,988 | 41.7 → 38.6 | 64.0 → 56.0 | 89.4 → 87.2 |

`real-crop words did not rise — not replacing the shipped reader`

This is the result worth keeping. After pass-14 the honest reading was "these
deltas are under one crop, call it noise". Pass-15 removes that defence: 6.5x
the data moved every number the same direction, roughly 4x as far. A dose
response is not noise.

## What it settles

The filtering was not the problem, so loosening it was not the fix. Both the
strict 915 and the whole 5,988 degrade the reader, and the larger set
degrades it more. The measured 93%-at-0.85 vs 79%-at-0.60 label accuracy
turns out not to be the operative variable.

What the two runs have in common is the thing that matters: the labels are
EasyOCR's own output. Training a reader on its own transcriptions teaches it
its own errors, and the more of them there are the more thoroughly it learns
them. Label accuracy measured against that reader cannot detect this, which
is why the crop gate — which measures against human labels — is the only
instrument here that was ever going to answer the question.

Domain is likely a second factor: the held-out crops are Commons information
boards, plaques and street signs; the Mapillary crops are shop fascias.

## Do not

- Do not run a third pass on Mapillary crops labelled by EasyOCR at any
  filter setting. Two points, monotone, both refused.
- Do not weaken either gate. Both refusals were correct and both were cheap.
- The 20,240 photographs and 13,144 crops stay. What is exhausted is
  labelling them with the reader being trained.

The next lever, if there is one, is a label source that is not the model:
`data/signs/sign-text.tsv` holds 9,506 OSM sign strings with coordinates.

## The actual mechanism, from the log

The "circularity" reading above was a hypothesis. The log gives the specific
version of it.

**The loss never converges.** 750 steps, oscillating 0.5–2.4 with no trend:

```
 50: 1.47   300: 2.14   550: 0.90
100: 2.22   350: 2.01   600: 1.33
150: 2.39   400: 1.21   650: 0.49
200: 1.36   450: 0.60   700: 1.49
250: 0.81   500: 1.33   750: 0.83
```

**Two held-out diacritic crops broke, none were fixed:**

```
PUTNIČKI      before: PUTNIČKI       after: PUTNICKI
iznad česme   before: iznad časme    after: i/nad jasut
```

**And the reason is a distribution mismatch, not per-row label error:**

| label source | rows carrying a diacritic |
|---|---|
| human gold (`crops/labels-human.tsv`) | 180 / 1,701 = **10.6%** |
| pass-14 train (915) | 26 / 915 = **2.8%** |
| pass-15 train (5,988) | 62 / 5,988 = **1.0%** |

EasyOCR drops diacritics when it crops, so feeding its output back trains the
reader on a world where Bosnian letters appear at a tenth of their real rate.
It learns that they are rare. `PUTNIČKI` → `PUTNICKI` is that lesson applied.

This also explains the dose response exactly: pass-15's set is both sparser
in diacritics (1.0% vs 2.8%) and larger, so it taught the lesson harder.

Note what this corrects. An earlier note measured diacritics inside the
conf ≥ 0.85 subset, found them preserved, and concluded the stripping claim
was wrong. Per-row accuracy in that subset was indeed fine. The operative
variable was never per-row accuracy — it is the rate across the set, and no
confidence threshold fixes a distribution that is missing nine tenths of its
diacritics.
