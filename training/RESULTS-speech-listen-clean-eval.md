# Listen clean-eval — Kaggle, 17 Sep 2026

Eval-only. No training, no default change, no publish.

Shipped listener: whisper-large-v3 int8 `e6bb58483586b06c` through
`app.speech.transcribe`. Baseline: whisper-small `a76342f6ab59b382`. Kernel
`safak3as/lilly-listen-clean-eval`, git `48471303`, Tesla T4, COMPLETE 20:08 UTC
(started 19:24 UTC). Manifest SHA-256
`908cb93c92c5fe262f7e6b3cb7bc43ba45a02f8ce3f36a83cc5c189ca1d50f7c`. 925/925
pinned FLEURS Bosnian test clips. v1 ERROR'd because `kaldialign==0.9.1` could
not import `batch_error_rate`; v2 is the score (`kaldialign==0.12.0`).

The official FLEURS test split is the frozen **clean** cohort. Noisy and
unintelligible rows stay explicit zeros; this set was not padded with synthetic
noise.

## Side by side

Same two listeners, product decode path. Product headline on the 200-clip
holdout stays **11.9%**. This run is all 925.

| listener | clips | errors / words | WER |
|---|---:|---:|---:|
| listen-previous (small, gated) | 925 | 6663 / 18836 | **35.37%** |
| listen (large-v3, shipped) | 925 | 2170 / 18836 | **11.52%** |

Open ASR multilingual normalizer, offline, not a submission: **12.23%**
(2252 / 18409) on the same shipped predictions.

## Registered gates

| gate | baseline small | shipped large-v3 | held? |
|---|---:|---:|---|
| Bosnian term recall not below baseline | 60.4% | **88.5%** (p = 0.000) | yes |
| Croatian substitution not above baseline | 0.96% | **6.25%** (+5.3 points, p = 0.0175) | **no** |

Verdict **FAIL** — the Croatian column is the same refusal already on the
product row (1.1% → 6.1% on the earlier 925-clip instrument). This run does not
reopen large-v3. Serbian 4.17% → 2.33% (p = 0.105, not a claim).

Evidence: `training/clean-eval/results/speech-gate.json`,
`speech-clean-eval-decision.json`, `open-asr-offline.json`,
`speech-legibility.md`.
