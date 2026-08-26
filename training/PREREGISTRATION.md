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

## Amendment, 02:15 — two arms, written before either runs

Running one experiment that changes both the data and the hyperparameters would
leave the result uninterpretable: a gain could come from either, and we would be
free to tell whichever story fit. So there are two arms, and this is written
before either has started.

**Arm A — data only.** The corpus changes (21,178 misaligned WikiMatrix pairs
removed, 38,277 unseen pairs added). Every hyperparameter stays exactly as the
current model's: the recipe in `training/train_translation.py` as committed.

**Arm B — data plus recipe.** The same corpus, plus the hyperparameters chosen
by the jury run recorded alongside this file.

Both are judged by the thresholds already stated above — BLEU ≥ 42.04 **and**
chrF2 ≥ 67.34, measured by `training/evaluate_app.py` with the language tag
stripped. Nothing about the thresholds changes because there are two arms.

**If both clear the bar**, the one with the higher chrF2 wins, because chrF2 is
the measure this project has repeatedly said is the fairer one for Bosnian and
it is the one the current model loses on. Not the higher BLEU. Stated now so
the choice is not made after seeing the numbers.

**If only one clears**, that one. **If neither clears**, the current model ships
and both arms are reported as failures — including Arm A, which would mean the
extra data did not help, a result worth publishing precisely because it is
disappointing.

Arm A also answers a question no single run could: how much of any gain is the
data. If A clears the bar and B does not, the recipe hurt.
