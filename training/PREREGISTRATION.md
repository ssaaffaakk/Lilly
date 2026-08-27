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

## Amendment, 01:52 — the reader, written before its "after" number exists

The reader run started at 01:48 and is training now. Its `before` number is on
disk; its `after` number does not exist yet. This is written in that gap on
purpose.

**The target recorded in `RESUME.md` and `NIGHT-LOG.md` — "leak-free valid, word
> 67.1%, diacritic > 69.4%" — is withdrawn, because it has no source.** Those two
numbers appear nowhere else: no script writes them, no log records them, no
commit introduces them. `training/evaluate_ocr.py`, the script the training
script's own docstring names as the thing that produced them, does not exist.
The only reader measurements that exist on disk are:

| when | valid set | words exact | diacritics |
|---|---|---|---|
| 18:36, `remaining.log` | old synthetic, 327 letters | 93.8% | 91.1% |
| 01:49, `ocr-train.log` | today's regenerated set, 289 letters | **75.2%** | **73.0%** |

Neither is 67.1/69.4. And 67.11 is this file's own chrF2 figure for the current
translator, three lines up — the likeliest explanation is that a translation
score was copied into the reader's row and then inherited as a target.

That mistake is not cosmetic. **The untouched reader already scores 75.2 / 73.0**,
so a bar at 67.1 / 69.4 is one the shipped model clears by eight points without
being trained at all. Any run judged against it would be certified as an
improvement while having made the reader worse. This is the exact failure this
file was written to prevent, and it was pointing the wrong way for four hours.

**The bar, from here:** the run's own `before`, measured minutes ago from the
published weights (md5 `469869130aad1a34e8f9086f4262bc59`, verified pristine at
load) on the same 500 held-out crops, with the same scorer, in the same process:

| | before | threshold to replace the shipped reader |
|---|---|---|
| words exact | 75.2% | > 75.2% |
| diacritics | 73.0% | **> 73.0%** |

Diacritics is the one that decides, because dropping them is the failure the
retrain exists to fix; a run that lifts whole-word accuracy while losing Bosnian
letters has traded away the point. `train_ocr.py` already enforces exactly this
comparison in code — it refuses to overwrite the reader unless `after`
diacritics beat `before` diacritics, and it keeps the published weights beside
the new ones. The code was honest; only the prose target was wrong.

The valid set was checked for leakage before the run: 2,348 valid rows against
21,652 train rows, **zero shared strings**, so "leak-free" is now a measured
claim rather than an inherited one.

## Note on the clock

Entries in `NIGHT-LOG.md` run about 74 minutes ahead of this machine: the log
says commit `c2d80db` landed at 02:25, `git` says 01:11. Times in this amendment
are the machine's. The morning report should not treat the log's timeline as
wall-clock.

## Correction to the 01:52 amendment — reader thresholds

The 01:52 amendment raised the reader's bar to 75.2% words / 73.0% diacritics,
on the grounds that the earlier 67.1 / 69.4 figure "came from nowhere". That
reasoning was wrong, and the direction of the error matters.

67.1 / 69.4 is the pristine reader on the **whole** held-out set — 2,348 crops,
1,940 synthetic and 408 photographs. It was measured, just not by a script in
the repo, which is why a later search for its source found nothing.

75.2 / 73.0 came from `train_ocr.py`'s own held-out subset, which took the first
500 rows of `data/ocr/valid`. After the split was rebuilt by label text the
synthetic crops landed first, so those 500 rows were 500 synthetic images and
not one of the 408 photographs. The pristine reader gets 73.2% of synthetic
words and 38.0% of photographed ones, and that gap is the entire discrepancy.

So the amendment raised the bar using the easy half of the job. The subset now
strides through the file instead (414 synthetic, 86 photographs — 17.2% against
the set's 17.4%), and both numbers are reported rather than one replacing the
other.

**It changes no verdict.** Measured on the whole set: words 67.1% → 86.8%,
diacritics 69.4% → 86.2%. On the subset the run itself reported 75.2% → 89.0%
and 73.0% → 87.5%. The trained reader clears every bar either version of this
file ever set, so nothing here was decided by which number was in force.

Photographs gained most: words 38.0% → 74.3%, diacritics 59.7% → 80.0%.

## Outcome — Arm A, decided against the thresholds above

Measured by `training/evaluate_app.py`, 2,009 FLORES-200 pairs, both builds int8
CTranslate2 through `app.translate.Engine`, language tag stripped:

| | base | previous fine-tune | **Arm A** | threshold |
|---|---|---|---|---|
| BLEU | 40.81 | 42.04 | **42.21** | ≥ 42.04 ✓ |
| chrF2 | 67.34 | 67.11 | **67.35** | ≥ 67.34 ✓ |
| length vs reference | 1.019 | 0.992 | 0.997 | — |

Both bars cleared, so Arm A is installed. BLEU gains 1.40 over the base at
p = 0.001.

The chrF2 number needs saying carefully, because it clears by 0.01. The right
reading is not that chrF2 improved: at p = 0.365 the difference from the base is
indistinguishable from nothing. The reading is that **the regression is gone.**
The previous fine-tune lost 0.22 chrF2 at p = 0.028 — a small loss, but a
measured one. Arm A does not lose it. That was the point of setting the bar at
the base's own score rather than at the previous model's.

Nothing but the data changed. Same LoRA rank, same learning rate, same epochs.
21,178 misaligned WikiMatrix pairs out, 38,280 unseen pairs in.

As the user sees it — tags left in, which is what arrives on screen — the gap is
+4.61 BLEU and +0.35 chrF2, both at p ≤ 0.002. Most of that is the base model
printing `>>bos_Latn<<` into 28.5% of its own translations, which Arm A never
does. That is a real improvement to what a reader gets and not a translation-
quality gain, and both numbers are published for that reason.

**What this did not change.** BosnianBench, built the same night: term recall
base 91.5%, Arm A's predecessor 92.0%, p = 0.354. The fine-tuning does not
measurably improve understanding of Bosnian-specific terms, because the base is
already at 91.5%. Arm A moves BLEU and leaves that untouched, and the model card
has to say so.

## Outcome — Arm B wins, by the tie-break written before the numbers

Both arms cleared both bars. Product path, tags stripped, 2,009 FLORES pairs:

| | BLEU (bar 42.04) | chrF2 (bar 67.34) |
|---|---|---|
| Arm A — data only | **42.21** | 67.35 |
| Arm B — data plus recipe | 42.18 | **67.47** |

The rule above says: *if both clear the bar, the one with the higher chrF2 wins
— not the higher BLEU.* Arm A's BLEU is 0.03 higher and that does not matter,
because the rule was written to stop exactly this choice being made after the
fact. **Arm B ships.**

One thing the rule could not settle, and it is worth stating rather than
burying: judging two arms on the set we also report is selection on the test
set. Split by FLORES's own halves, from the same saved translations:

| | devtest (1,012) | dev (997) |
|---|---|---|
| base | 41.10 / 67.51 | 40.50 / 67.15 |
| Arm A | 42.59 / 67.58 | 41.81 / 67.10 |
| Arm B | 42.49 / **67.69** | 41.86 / **67.25** |

Arm B holds the higher chrF2 on both halves, so the choice is not an artifact of
where it was made. Had the two disagreed across the halves, the honest answer
would have been that neither arm is distinguishable and the tie-break was noise.

**What Arm B cannot tell us.** It changed the corpus *and* the epoch count, so a
win cannot be attributed to either. Arm A isolates the data: 42.21 / 67.35 with
one epoch and the corpus alone. The difference between them — +0.12 chrF2 for
−0.03 BLEU — is what the length-band rebuild and the second epoch bought
together, and nothing here separates them.
