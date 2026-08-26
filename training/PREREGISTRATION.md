# Pre-registration — translation retraining, 27 August 2026

Written **before** the run starts and before any of its numbers exist. Nothing
below may be edited after the run begins; a later disagreement with it is
recorded underneath as a separate note, not as a correction.

## Why this file exists

The same fine-tuning has been scored three ways: the raw adapter on whole rows
(+0.54 BLEU / −0.79 chrF2), through the app's own path (+1.23 / −0.22), and as
a user sees it, tags and all (+4.44 / +0.12). Each change of method had a real
reason and none of them was chosen to flatter the result. But all three moved in
the flattering direction, and that pattern is what the garden of forking paths
looks like from the inside — every individual step defensible, the sequence not.

The defence is to fix the target before firing. So: the measurement below is the
one that decides, whatever it says.

## What is being changed

Data. The training corpus goes from 334,790 rows to 313,612 (21,178 misaligned
WikiMatrix pairs removed on three measured rules), plus 38,277 previously unused
pairs from wikimedia-v20260327 and NTREX-128. Hyperparameters follow the recipe
chosen by the jury run recorded alongside this file.

## The measurement that decides

`python3 training/evaluate_app.py` — both builds int8 CTranslate2, both through
`app.translate.Engine`, 2,009 FLORES-200 pairs, language tag stripped. That last
choice matters: with tags left in, the base model's `>>eng<<` leak is worth 3.2
BLEU of apparent gain that is not translation quality. Stripped is the harder
and honest number.

Against the **current fine-tune**, not against the base:

| | current | threshold to replace it |
|---|---|---|
| BLEU | 42.04 | ≥ 42.04, and the gain over base still at p < 0.05 |
| chrF2 | 67.11 | **≥ 67.34** — the base model's own score |

The chrF2 bar is the strict one on purpose. The present fine-tune costs 0.22
chrF2 against the base at p = 0.028, and chrF2 is the fairer measure for a
language that inflects as heavily as Bosnian — this project's own documents say
so. A retrain that raises BLEU while leaving chrF2 below the base has repeated
the trade, not fixed it.

## What failure looks like

Stated now so it cannot be reinterpreted later:

- **chrF2 below 67.34.** The retrain did not fix the trade. Keep the current
  model, publish the current numbers, and say the chrF2 cost is unresolved.
- **BLEU below 42.04 with chrF2 above 67.34.** A different trade, not a better
  model. Keep the current one; record that more data moved the balance rather
  than lifting both.
- **Both below.** The extra data hurt. Keep the current model and say so.
- **Both above.** Replace, and publish the new numbers with this file beside them.

`models/lilly/keep-2026-08-26/` holds the current adapter and translator. If the
thresholds are not met, that is what ships and this file is the record of why.

## What this run cannot settle

The gain is concentrated in clean news prose — SETIMES +3.05 BLEU against
TED2020 −0.82 — and the training corpus leans the same way. So "the model
learned better Bosnian" and "the model adapted to news style" are not
distinguished by anything measured here, and no result of this run distinguishes
them. That needs a measure aimed at the claim itself, which does not exist yet.
Whatever this run returns, that sentence stays true and belongs in the model
card.
