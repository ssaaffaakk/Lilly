# RESULTS — truecase (ALL-CAPS input) — 14 September 2026

The ALL-CAPS defect documented in
`docs/REPORT-what-would-raise-the-numbers-2026-09-13.md` §1, measured through
`app.translate.Engine` (the served int8 build, the path the endpoints call),
and a sentence-case restorer staged behind `LILLY_TRUECASE` (**default off**).

## What it is

Signs are written in capitals. Uppercase word forms are effectively
out-of-vocabulary for this model, so shouting at it costs a great deal:
200 FLORES devtest pairs per direction, three variants of the same source
against one reference.

Command:
`scripts/measure_truecase.py --limit 200 --workers 4`
Scorer: sacrebleu 2.6.0, `corpus_bleu` and `corpus_chrf(word_order=0)` — the
same calls as `training/evaluate.py`'s `score()` (word_order=0 is chrF2).

| direction | variant | BLEU | chrF2 |
|---|---|---|---|
| en-bs | A — as written | 31.40 | 61.06 |
| en-bs | B — uppercased | 14.37 | 35.67 |
| en-bs | C — uppercased, restored | 27.99 | 58.75 |
| bs-en | A — as written | 43.64 | 68.71 |
| bs-en | B — uppercased | 24.39 | 44.53 |
| bs-en | C — uppercased, restored | 39.85 | 66.56 |

| direction | gap | BLEU | chrF2 |
|---|---|---|---|
| en-bs | B − A | −17.03 | −25.39 |
| en-bs | C − A | −3.41 | −2.31 |
| bs-en | B − A | −19.26 | −24.17 |
| bs-en | C − A | −3.80 | −2.15 |

The restorer recovers **23.08 of 25.39 chrF2 (90.9%)** in en-bs and **22.02 of
24.17 (91.1%)** in bs-en. This corroborates the report's en-bs figures
(A 30.81 / 60.60, B 14.24 / 35.05, C 27.22 / 58.11) to within ~0.5 chrF2, and
adds bs-en, which the report did not measure: caps hurt it about as much.

## Status

**Staged, off by default.** `app/translate.py` calls the restorer only when
`LILLY_TRUECASE` is set, following the repo's existing opt-in pattern
(`LILLY_READER`, `LILLY_PADDLE_CYRILLIC_RESCUE`). With the flag unset, behaviour
is byte-for-byte unchanged. Shipping it as the default is a product change and
needs its own pre-registration and a photograph-bar measurement first (the
report says the honest bar is a score on photographs, not uppercased FLORES). That photograph bar now exists: `training/RESULTS-truecase-photos.md` drives the flag through the served `bs-en` engine on the 40 Commons photographs' own OCR, off against on, and lays out the 13 it changes side by side — the A/B the default-on decision is gated behind. The decision itself stays open.

## Known limitation

An all-caps acronym inside a longer shout is not protected: `NASA STOP` (8
letters, 100% upper) restores to `Nasa stop`. Short tokens on their own
(`WC`, `ATM`, `STOP`) are under the trigger threshold and pass through, and
mixed-case text is untouched. This is the same class as the report's
proper-noun residual (C − A ≈ −2.2 chrF2) and is left as measured, not hidden.
