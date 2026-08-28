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

---

# Speech retraining — written before any of it runs

Lilly hears 35.5% of words wrong. It was trained on 3,091 Bosnian clips, and
there is far more Croatian and Serbian speech available. Adding it is the plan.

## The trap this is written to catch

Croatian and Serbian are close to Bosnian but not the same. A model fed mostly
Croatian will get better at Croatian, and the overall word error rate can fall
while **Bosnian gets worse**. That is not a hypothetical: it is the same shape
as the mistake the translator made, where BLEU rose on the metric we were
looking at and chrF2 fell on the one that mattered more.

An average that improves while the thing we sell degrades is the failure this
project keeps almost making, and a single WER figure cannot see it.

## The measurement that decides

`training/evaluate_speech.py` on the 200 held-out FLEURS Bosnian clips — the same
clips as before, so the numbers are comparable — **and** a Bosnian-specific term
measure, which is being built separately.

| | now | threshold to replace it |
|---|---|---|
| word error, 200 Bosnian clips | 35.5% | **below 35.5%** |
| Bosnian-specific term recall | to be measured | **not below its baseline** |

Both, not either. A model that reads Bosnian words less well than the current one
does not ship, whatever it does to the average.

## What failure looks like

- **Overall error falls, Bosnian terms fall too.** The neighbours drowned the
  Bosnian. Keep the current model and raise the Bosnian share.
- **Neither moves.** The extra data was not close enough to help. Keep the
  current model, and record that the neighbour-language route did not work —
  that is a result worth publishing because the obvious next thing to try is
  more of it, and this says not to.
- **Overall error rises.** Something is wrong with the data or the recipe rather
  than with the idea. Diagnose before retrying; do not simply add more.

`models/lilly/listen.before-training` holds the untrained weights and
`models/lilly/listen` the current ones. If neither bar is cleared, `listen` stays
as it is.

## The mixing ratio is measured, not assumed

`data/scripts/build_speech_mix.py --share` repeats the Bosnian clips until they
hold a stated fraction of the examples. The right fraction is not knowable in
advance, so it is a parameter, and the plan is to train at two settings and score
each on Bosnian specifically rather than to pick a number and defend it
afterwards. If only one run is affordable, 0.35 is the starting point — high
enough that 3,091 clips cannot be drowned by ten times as many neighbours, low
enough that the neighbours still teach something.

## v2 — picture — picture-olcum

Written before the run, per rule 8. Nothing below is reinterpreted afterwards.

**The question.** The reader finds 54.7% of a photograph's words. That number is
the product of two stages and cannot say which one is the ceiling: a word the
detector never boxed is lost before the recogniser is asked, and a word boxed
but misread is lost after. Fine-tuning fixes the second and does nothing for the
first. `picture-egitim` is about to pick a recipe, and picking one without this
split is guessing.

**The set.** The 40 scored photographs in `data/ocr/real-photos/scored/`, listed
in `scored-sample.txt`. The answer key is `truth.json`: 373 agreed words across
the 28 that carry text, transcribed by two blind transcribers at 91.2%
agreement. The same set and the same key as the 36.0% and 54.7% figures, so the
result is comparable to both. No re-sampling — `training/sample_photos.py`
refuses now, and it should keep refusing.

**The scorer.** `training/measure_detection.py`, which draws every region the
detector returned and computes nothing. Then a human count against `truth.json`.

**The two figures, named now.**

- **Detection recall R_d = (truth words covered by at least one detected box) /
  373.** A word counts as covered when a box overlaps the glyphs of that word on
  the drawn overlay. Partial overlap of a word counts as covered, because the
  detector's job is to point at the text, and a box clipping one letter is a
  recognition problem rather than a miss. A box that covers a *different*
  instance of the same string does not count for this one; the key is a multiset
  and so is this count.
- **Recognition given detection = 45.0% / R_d**, using the pooled figure because
  it is the one whose denominator is the same 373 words. The per-photo 54.7%
  weights photographs equally and cannot be divided by a word-level recall
  without changing what the ratio means. **The pooled figure is the one this
  decomposition uses, and it is named here so it cannot be swapped later for
  whichever number flatters the result.**

**Detection precision, which needs no run and is fixed now at:** of the 1,914
regions the detector produced across 285 photographs and twelve blind
annotators, 39 (2.0%) contain no text at all and 173 (9.0%) contain text no
human could read. 1,702 (89.0%) are usable. These are already on disk in
`data/ocr/label-answers/`; they are quoted, not recomputed.

**What counts as this task being done.** `SCOPE-V1.md` asks for the detector's
find-rate on real photographs to be measured. R_d is that number. There is no
threshold to clear, because this is a measurement and not a gate — inventing a
bar for it after seeing it would be the exact failure this file exists to
prevent. What is committed in advance is the definition, the set, and which of
the two live recall figures the ratio is taken against.

**A validity check that can fail, and stops the task if it does.** Before the
decomposition is reported, `app.ocr.scan` is run over all 40 photographs and
must reproduce 45.0% pooled and 54.7% per-photo. If it does not, the detector
being measured is not the one behind the shipped number, and the decomposition
is not reported at all — the discrepancy is reported instead.

**What each outcome would mean.**

- **R_d high (say above 85%).** The detector finds nearly everything and the
  ceiling is recognition. `picture-egitim` should spend its effort on the
  recogniser, and the Cyrillic fine-tune is the strongest remaining move.
- **R_d middling (say 60–85%).** Both stages are losing words, and the 75% bar
  in `picture-egitim` cannot be reached by recogniser training alone — the
  arithmetic ceiling is R_d itself, which would need saying out loud before that
  run rather than after it.
- **R_d low (below 60%).** The detector is the ceiling, stock CRAFT is the thing
  to fix, and fine-tuning the recogniser further is polishing behind a closed
  door. That would be a genuinely unwelcome result and is the reason for writing
  this paragraph before seeing the number.

### Note — provenance count corrected

The section above reads "1,914 regions the detector produced across **285**
photographs". The correct figure is **200**. 285 was the candidate pool;
200 of them actually yielded crops. Verified independently by the lead, three
times, and by me at crop level.

**It changes no figure in the section and no threshold anywhere.** The precision
numbers are 39/1,914 = 2.0% and 173/1,914 = 9.0%, whose denominator is
*regions*, not photographs. Nothing above moves, and no bar shifts.

Recorded as a note rather than an edit because this file's own rule (lines 3–5)
is that nothing may be rewritten after the fact, and that rule binds hardest
when the change would be an improvement — otherwise it is not a rule.

Provenance of the error, since it is the second time this number has come back
after being corrected: 285 is still printed uncorrected in
`training/RESULTS-ocr-realcrops.md`, which is where it keeps being re-read from.
A pointer has now been added there so the source stops reissuing it.

## v2 — picture — picture-egitim

Written before the run and before any of its numbers exist. The four decisions
below were ruled by the lead after the ceilings were measured and before any
training was attempted; they are recorded here so neither of us can choose
after the fact.

### The bar — both, not either

| | threshold | now |
|---|---|---|
| words found per photograph | **> 75.0%** | 54.7% |
| words invented that are on no sign | **≤ 180** | 180 |

**Both must hold.** The acceptance test as written on the board — *kelime >
%75* — constrains recall and nothing else, and this lane has a **measured** way
to buy recall with garbage: the union rule in `RESULTS-ocr-cyrillic.md` gains
2.3 points of per-photo recall and takes invented words from 180 to 546. A
single-sided bar certifies that as a pass. For an app that translates what it
reads, an invented word becomes an invented sentence and the user cannot tell
which words are real.

The invented-words bar is **no-regression, not improvement**, and that asymmetry
is deliberate: recall is what this run is for, hallucination is what it may not
pay with.

### Which figure, and why it is the per-photo one

**Per-photo.** Not chosen now — inherited. `RESULTS-ocr-realcrops.md` already
designates it "the number that describes pointing a camera at a sign", and
`evaluate_ocr.py:316` prints that sentence. Adopting a designation fixed before
this decision existed beats making one against a result.

Disclosed plainly: per-photo is also the **softer** of the two bars in relative
terms — 75% is 82.8% of its reachable range against 80.6% of pooled's, and
clearing it means taking 56.6% of the remaining headroom rather than 62.5%.

A second reason, from the lead: **pooled is 39% one photograph.**
`Spanish_square_08034.JPG` holds 144 of the 373 words, so "pooled" is
substantially "how well does it read Spanish_square". Per-photo weights every
photograph equally, which is what a user experiences.

**Pooled is reported beside it every time, never instead of it.**

### The set — all 28, unchanged

The 40 scored photographs, the 28 of them carrying text, `truth.json` as it
stands. Comparability with 36.0% and 54.7% is worth more than the 1.7 points of
purity available from dropping the two items in the set that are not
photographs (`Banjaluka_streetmap.jpg`, an OpenStreetMap render, and a 1900s
postcard). **The 26-item figure — 53.0% per-photo, 42.7% pooled — is reported
beside the 28-item figure every time**, so the caveat never stops being visible.

The ruler is not re-sampled. `training/sample_photos.py` refuses, and it keeps
refusing.

### Which build is judged

**The shipped configuration**, which today is Latin-only. Its ceilings:
per-photo **90.6%**, pooled **93.0%** — the two differ because Cyrillic is
concentrated on 7 of the 28 photographs rather than spread.

If the run proposes a **Cyrillic-enabled** build, that build is judged against
**the same two bars**, with its own ceiling stated alongside — because enabling
Cyrillic lifts the per-photo ceiling from 90.6% toward 100% and makes 75%
materially easier, while the invented-words bar is precisely where every
Cyrillic union rule has already failed. It is reported as a secondary line.
**Neither build gets a bar chosen after its number exists.**

### Constraints on the training pool, fixed now

- **The 15 scored photographs present in `harvested/` are excluded by canonical
  Commons `File:` page identity, not by byte hash.** The hash sees only 4 of the
  15, because `scored/` holds the 1280 px `screen_url` rendering and
  `harvested/` the larger `keep_url` original. Training from `harvested/`
  unfiltered puts 37.5% of the ruler into the training set.
- **The six flat-artwork items are excluded when crops are cut** — three
  postcards, a travel poster, a document scan and an OpenStreetMap render, named
  in `data/ocr/real-photos/EXCLUDE-flat-artwork.txt`. Their text is the
  synthetic distribution, which is the thing that cannot close this gap.
- `train-photos/` is 285 dangling symlinks and **must not be silently replaced
  by `harvested/`**; that directory was what enforced the exclusion above.

### What failure looks like

- **Recall clears 75% and invented words rise above 180.** Not a pass. The run
  bought recall with hallucination, which is the trade this bar exists to
  refuse. Report both numbers and keep the current reader.
- **Invented words hold and recall lands short of 75%.** An honest miss.
  Report it, and report R_d from `picture-olcum` beside it, because if detection
  is the ceiling then no amount of recogniser training reaches this bar and the
  bar was aimed at the wrong stage.
- **Neither holds.** The recipe hurt. Keep the current reader and say so.
- **Both hold.** Install, and publish these thresholds beside the result.

---

## v2 — read — read-egitim

Written **before** the full fine-tune starts and before any of its numbers
exist. The `--full-finetune` path in `training/train_translation.py` has never
been run on this project, so nothing below is informed by a result.

### The question

`training/train_translation.py`'s own docstring states the capacity arithmetic
that motivates this run:

    LoRA r=16 on this model  =  270,336 x r  =  4,325,376 parameters
    at ~2 bits/parameter     =  8.65 M bits of capacity
    corpus target side       =  9.87 M tokens, ~1 bit/token upper bound

The two numbers are the same size, which is the regime where LoRA is reported to
start losing to full fine-tuning. Everything Lilly has ever shipped is LoRA. So:
**does training all 237.7 M parameters beat the r=16 adapter on the product
path?** That is an empirical claim the project has argued from arithmetic and
never measured.

### What the two arms are

**LoRA arm — already run, already measured, and NOT re-run.** Arm B is exactly
the LoRA arm of this comparison: the same `train-mix.tsv`, the same 2 epochs,
the same `ntrex-holdout.tsv` validation, r=16, lr 2e-4. Its numbers are in
"Outcome — Arm B wins" above: **42.18 BLEU / 67.47 chrF2**. Re-running it would
spend three to four GPU hours to reproduce a number that already exists under
the identical recipe, and the seed is fixed. If a re-run is later judged
necessary, that is a new decision and it is not this one.

**Full fine-tune arm — the new run.** Identical corpus (`train-mix.tsv`,
361,621 examples), identical validation (`ntrex-holdout.tsv`, 462 professional
pairs), identical 2 epochs, identical seed. One thing changes: all 237.7 M
parameters train, at `FULL_LR = 2e-5` rather than LoRA's 2e-4. That learning
rate is not a free choice made here — it is the constant already resolved in
`train_translation.py` from the mode, on the measured ten-to-one LoRA/full ratio
the file cites. Changing the corpus *and* the training mode at once would repeat
the mistake the Arm A / Arm B split was created to avoid.

### The measurement that decides

    python3 training/evaluate_app.py --tuned <the candidate build>

Both builds int8 CTranslate2 through `app.translate.Engine`, 2,009 FLORES-200
pairs (dev + devtest), language tag stripped. Nothing else counts. This project
has already measured the same fine-tune three ways and watched the answer move
3.36 chrF2 with the layer.

### The threshold

Taken from the board, unchanged, and not reinterpretable:

| | installed (Arm B) | full fine-tune must reach |
|---|---|---|
| BLEU | 42.18 | **> 42.18** |
| chrF2 | 67.47 | **> 67.47** |

**Both, not either.** Strictly greater, not equal: replacing an installed model
requires exceeding it, so an exact tie leaves Arm B in place.

**On the bar itself.** I will re-measure the installed build with the command
above before judging the candidate, because I have not personally reproduced
42.18 / 67.47 and `HANDOFF.md` carried a wrong translation figure until today.
Stated now so it cannot be adjusted later: **if my re-measurement of the
installed build disagrees with 42.18 / 67.47, the bar stays at 42.18 / 67.47.**
The disagreement becomes a reported finding about the measurement path, not a
new and conveniently lower threshold. A bar that moves when you measure it is
not a bar.

### The tie-break, written before either number exists

- **Both above.** The full fine-tune ships. Report it, install it, and put this
  section beside the numbers.
- **BLEU above, chrF2 not.** Does not ship. This is the trade the whole project
  has been fighting — BLEU bought with terseness — and taking it here would undo
  the one thing Arm B was chosen for.
- **chrF2 above, BLEU not.** Does not ship. Same rule, applied honestly in the
  direction that is less tempting.
- **Neither.** Does not ship. Arm B stays installed and the result is published
  as what it is: at this corpus size a 4.3 M-parameter adapter matches or beats
  training all 237.7 M, and the capacity arithmetic in `train_translation.py`
  predicted the wrong winner. That is a genuinely useful negative and it is the
  outcome I would bet on.

### Significance is reported, and does not move the bar

Paired bootstrap against Arm B on both metrics, 1,000 resamples, reported
whatever it says. It does **not** change the ship decision — the bar above is
the bar. But a win that clears the bar while failing p < 0.05 must be described
in the model card as a difference not distinguishable from noise, in the same
words used for the chrF2 tie. Pre-committed so the sentence cannot be softened
after seeing which way it falls.

### Robustness checks that cannot change the decision

Recorded now so they cannot be promoted to evidence afterwards:

- **Split-half.** devtest (1,012) and dev (997) scored separately from the same
  saved translations. If the candidate wins on the pooled set but loses on a
  half, the write-up says so. It does not overturn the pooled decision.
- **Length ratio vs reference.** Arm B is 0.997-ish. Terseness is this project's
  known failure mode and it is watched every time.
- **BosnianBench.** Reported for continuity. It has never moved (91.5% -> 91.7%,
  p = 0.465) and `read-olcum` established that its Turkism category cannot be
  built from any available corpus, so it is not expected to say anything and is
  not evidence for or against this run.

### What this run cannot settle

The same sentence that was true of Arm A and Arm B stays true: the gain is
concentrated in clean news prose (SETIMES +3.05 BLEU against TED2020 -0.82) and
nothing here distinguishes "learned better Bosnian" from "adapted to news
style". Also new, from `read-veri`: `data/clean/train.tsv` is inside the base
model's own training data (opusTCv20210807 contains WikiMatrix-v1, SETIMES-v2,
TED2020-v1 and wikimedia-v20210402 by name), so both arms are re-weighting
material the base already has rather than teaching it new text. Only the 1,924
NTREX rows are genuinely unseen. That frames any result here and is not changed
by it.
