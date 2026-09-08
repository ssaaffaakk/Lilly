# The full speech instrument — 925 clips, both listeners, one run

Kaggle, Tesla T4. Pre-registered: PREREGISTRATION.md, 'v3 — speech — the full instrument'.

## Gate rows (the product's path, --decode app)

| threshold | listen-previous (small) | listen (large-v3) | |
|---|---|---|---|
| word error, strictly below | 52.4% | 33.0% | pass |
| term recall, not below | 49.6% | 72.1% | pass |
| Croatian substitution, not above | 1.1% | 6.1% | **FAIL** |

Croatian: 8 of 131 decided (177 targets) against 1 of 87; paired bootstrap p = 0.0180.
Term recall: +22.5 points, p = 0.0000.

## Rubric WER (greedy, temperature 0, BasicTextNormalizer, 925 clips)

| listener | WER | words |
|---|---|---|
| listen-previous | 39.5% | 18468 |
| listen | 14.1% | 18468 |

**Verdict by 'Both, not either': DOES NOT SHIP — and by rule 3 of the pre-registration, large-v3 is closed.**
