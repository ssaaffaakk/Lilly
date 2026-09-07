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

---

# v2 — listen — listen-egitim

Written before the run. Nothing below may be reinterpreted after it.

## Why this section exists at all

The v1 speech section set two thresholds and said "both, not either": word error
below 35.5%, and a Bosnian-specific term measure not below its baseline. Both
were cleared and the listener shipped. Two things have changed since, and each
one invalidates a number rather than an argument.

**The instrument changed.** `training/speech_bench.py` reported ONE variety
substitution column, and 73 of the 85 targets it computed it over were yat pairs
whose alternative form is Serbian. The run it judged is the run that added 3,430
clips of *Croatian*, and Croatian is ijekavian like Bosnian, so those 73 targets
could not see the failure the gate existed to catch. Croatian and Serbian drift
are now separate columns over 221 targets, 69 of which can show Croatian drift.

**The decode path changed.** `training/evaluate_speech.py` and
`training/speech_bench.py` each constructed their own `WhisperModel` beside the
app's. Both now call `app.speech.transcribe`, so the numbers are produced by the
path a user's audio takes. Every speech figure on record — 38.5, 35.5, 34.9,
term recall 65.9 → 68.2, substitution 5.1 → 3.3 — was produced by the OLD path.

## The selection rule for terms, stated before the targets are counted

A term is admitted by corpus evidence alone: rate per million tokens, exclusivity
in both varieties, alignment, Dice specificity, agreement between the two mining
directions, and a veto from a second independent Bosnian corpus. The thresholds
are in `data/scripts/build_speech_terms.py` and they are frozen at the committed
values for this run.

**No term is admitted, refused, or weighted by whether it lands in the graded
set, or by how often.** All 349 graded transcripts are removed from every
corpus before anything is counted. A term has to be scorable to produce a
target, which is a property of the sentences; that is not the same as choosing
terms because they score.

Disclosed, because it is the one place the rule was under strain: while choosing
MIN_RATE I compared 4, 6 and 8 and saw the Croatian target count (72, 71, 69)
alongside the control quality. I kept 8, the setting with the FEWEST targets,
because the looser ones raised the Bosnian marker floor from 0.56 to 0.98. That
decision was made before any listener was scored on the new instrument, and the
thresholds do not move again.

## The re-measured baseline, and the one degree of freedom this closes

The old thresholds cannot be compared against anything measured through the new
path. So before the training run:

1. `models/lilly/listen-previous` and `models/lilly/listen` are both scored
   through the new path, in one process, on the same clips.
2. Those figures become the baseline. They are recorded here before the run.

**Precommitment, so that re-measuring cannot become a way of moving a
threshold.** If the re-measured `listen-previous` word error differs from 35.5%
by more than 0.5 points in either direction, the run does not proceed: the path
difference is investigated and reported first. A baseline that drifts upward
would make the gate easier, and the only defence against that is deciding now
what counts as too much drift.

## The gate

**Both, not either.** A listener that fails any of these does not ship,
whatever it does to the others.

| | threshold |
|---|---|
| word error, the 200-clip prefix, new path both sides | **strictly below the re-measured `listen-previous`** |
| Bosnian term recall, 349 sentences, mined terms | **not below the re-measured baseline** |
| Croatian substitution | **not above the re-measured baseline** |

Word error is the hard gate because it has 3,901 reference words behind it. The
Serbian column (158 targets) and the marker rate (every word of every
transcript) carry the Bosnian-specific weight.

### What the Croatian column may and may not be used for

Measured by `training/speech_bench.py --power`, on this run's real targets,
before any listener was scored:

| planted difference | croatian, 69 targets | serbian, 158 targets |
|---|---|---|
| 5 points | 20% | 26% |
| 10 points | 42% | 74% |
| 15 points | 67% | 89% |
| 20 points | **76%** | 100% |
| 30 points | 98% | 100% |

So: **the Croatian column is a coarse alarm with its power stated, and a null
result is recorded as "the instrument could not see", NEVER as "no drift".**
Sixty-nine targets need about twenty points before the column speaks at all.

This is a limit of the TEST SET, not of the term list: about 70 of the 349
FLEURS Bosnian sentences contain a word that has a Croatian counterpart at all,
and nothing in this project's control raises that.

For calibration, from the cached runs: the whole +5.6-point Croatian move
between the untrained model and the installed one is ONE occurrence of ONE word
— `vjerojatno` for `vjerovatno`, 1 of 18 decided targets. That is what a
movement of this size looks like from the inside.

## The mixture: 0.47 against 0.25, superseding the 0.35 starting point

The v1 section says the plan is "to train at two settings and score each on
Bosnian specifically rather than to pick a number and defend it afterwards. If
only one run is affordable, 0.35 is the starting point." That is a starting
point under a stated constraint. The constraint has gone, and three reasons
replace it:

1. **The constraint was fleurs_hr, and it was mechanical.** FLEURS hr train IS
   1,474 FLORES sentences, about 3,430 clips, and that is the entire corpus — no
   `--hours` value reaches a 35% share because there is no more of it. Croatian
   audio in general is not scarce: voxpopuli_hr holds ~11,000 clips under CC0.
2. **No threshold moves.** Word error and term recall are untouched. The share
   is a design parameter; rule 8 governs thresholds.
3. **The wider gap is the design more likely to produce an unwelcome answer.**
   The Croatian column needs about twenty points to speak. 47% against 35% is a
   small perturbation whose most likely outcome is two ties, which this project
   would then be tempted to read as "the Croatian audio was harmless". 47%
   against 25% gives drift a real chance to appear. If it does not appear at
   25%, that is informative; a null at 35% would only have been quiet.

At 25%, 3,091 Bosnian clips means 9,273 Croatian and 12,364 rows, 1.90x the
first run's 6,521. The epoch wall-clock is estimated from the first run's
measured rate and checked against the Kaggle session limit BEFORE launch.

**ParlaSpeech-HR is not used.** It is CC BY-SA, share-alike propagates to
distributed derivatives, and this project publishes weights. That is the owner's
decision about the licence of a released artefact and it is not taken here.
voxpopuli_hr is CC0 and covers the requirement on its own.

## What failure looks like, unchanged from v1 and restated

- **Word error falls, Bosnian terms fall too.** The neighbours drowned the
  Bosnian. Keep the current model, raise the Bosnian share.
- **Neither moves.** Record that the neighbour-language route did not work.
  That is worth publishing, because the obvious next thing to try is more of it.
- **Word error rises.** Diagnose before retrying; do not simply add more.

One addition, which is what this whole section is for: **a null on the Croatian
column is not the second case.** It is the instrument being unable to say.

---

# v3 — reply — English to Bosnian, written before the run

Written **before** the fine-tune starts and before any of its numbers exist. The
reverse direction ships today as the untuned base and `built.json` says so
(`fine_tuned: false`). This section fixes what would have to be true for that to
change.

## What is being changed

A LoRA fine-tune of `opus-mt-tc-base-en-sh`, the same recipe as the forward
direction, run through `training/train_translation.py --direction en-bs`. The
corpus is the same file: `read_tsv` flips source and target, so the pairs and
the length band are the ones the forward corpus was already rebuilt to hold. No
second corpus is built, and none should be.

**The base is `tc-base`, not `tc-big`.** Helsinki publishes a big model into
English and nothing big back out, so this direction starts at 29.57 BLEU /
58.96 chrF2 where the forward one started at 40.81 / 67.34. Every threshold
below is written against **this** base. Comparing the two directions' scores to
each other is not a result and will not appear in the write-up as one.

## The trap this is written to catch

This decoder serves five South Slavic languages and picks between them from
`>>bos_Latn<<` on the front of the source sentence. A fine-tune can weaken the
label's grip on the decoder without weakening anything a translation metric
looks at, because Croatian scored against Bosnian references still scores well —
they are close, and chrF2 is a character measure.

So the failure mode is fluent, plausible, confidently wrong-language output that
*improves* on chrF2. `scripts/fetch_translate_base.py` already documents this
shape for a label the tokenizer does not know. The same shape is available to a
model that still tokenizes the label and has stopped listening to it, and no
number in the table below would catch it on its own.

## The measurement that decides

`training/evaluate.py --direction en-bs` on the 2,009 held-out FLORES-200 pairs,
the same set and the same label as the base numbers in
`training/RESULTS-en-bs.md`, so the columns differ only by the fine-tuning.

| | base | threshold to replace it |
|---|---|---|
| chrF2, FLORES-200 | 58.96 | **above 58.96** |
| BLEU, FLORES-200 | 29.57 | **not below 29.57** |
| Bosnian form rate, output side | measured before the fine-tune runs | **not below that baseline** |

Both directions of the bar, not either. A model that scores higher while writing
less Bosnian does not ship.

**chrF2 decides, BLEU is the floor.** In this direction Bosnian is the *output*,
so the morphology being scored is the morphology being generated, and a character
measure is the fairer instrument for a heavily inflecting target. If the two
disagree, chrF2 wins. This tie-break is written here, before either number
exists, because the forward direction's arm choice showed how easily the other
rule looks better afterwards.

## The Bosnian form rate, and why this direction can finally measure it

`training/bosnian_bench.py` says plainly what it cannot do: "The direction is
bs->en, so nothing Bosnian survives into the output — you cannot look at English
and ask whether it is ijekavica." That limitation is why the forward direction's
Bosnian claim came back +0.5 points at p = 0.360 and stayed unproven.

It does not apply here. In this direction Bosnian **is** the output, so the
question becomes directly checkable: given an English sentence whose
professional Bosnian translation uses a Bosnian-only term, does the model write
that term or its Croatian or Serbian counterpart? One target, one bit, nothing
averaged away.

The data for it is already in `bench/cases.tsv` and no new collection is needed:
`en` is the source to feed, `terms` is the Bosnian-only target term, and
`variant_swaps` names the Croatian or Serbian counterpart it is being weighed
against. The scorer reads the output and asks which of the two it wrote. Fixing
the columns here means the measure cannot be quietly redefined once a number
exists.

This is the first time the project can test its own central claim head-on rather
than around the edge of it. Because that makes the measure load-bearing, three
things are fixed now:

1. **The baseline is measured on the base before the fine-tune is launched** and
   written into this file. A baseline recovered afterwards is not a baseline.
2. **The cases come from `bench/`, reused, not rebuilt.** A new case set chosen
   while a reverse model is in hand is a set chosen to be passed.
3. **A case whose Bosnian and Croatian forms score the same is dropped before
   the run, not after.** The forward direction found only 3.6% of swapped
   targets scored differently; the same audit runs here first, and if the
   discriminating subset is too small to say anything, that is reported as the
   instrument being unable to speak — not as a pass.

## Controls that are reported and cannot change the decision

Run after the deciding numbers are in, stated here so they cannot be presented
as the headline if they happen to flatter:

- **The label still tokenizes.** `>>bos_Latn<<` survives tokenisation as one
  piece, the check `fetch_translate_base.py` already performs.
- **The label still steers.** Decode the same FLORES source twice, once with
  `>>bos_Latn<<` and once with `>>hrv<<`. If the outputs are identical, or
  nearly so, the fine-tune has deafened the decoder to its own selector and the
  model does not ship whatever the table says. The base's own gap on this pair
  is measured first, for the same reason as everything else here.
- **Significance.** Paired bootstrap on chrF2 and BLEU, reported with the
  numbers. It does not move the bar; the bar is the table.

## What failure looks like

- **chrF2 rises, Bosnian form rate falls.** The model got better at
  Serbo-Croatian. Keep the base, and record that the reverse LoRA bought
  fluency by spending the thing this project is for.
- **Both flat.** `tc-base` had nothing left to give on this corpus. Publish the
  null. It is the more useful result of the two, because the obvious next move
  is a bigger base and this says whether that is the reason.
- **The label stops steering.** Diagnose the recipe before retraining. Do not
  raise the learning rate and try again.
- **BLEU falls below 29.57 while chrF2 rises.** The floor holds and it does not
  ship. The forward direction accepted a chrF2 that merely stopped getting worse;
  this direction does not get the mirror of that concession, because here chrF2
  is the deciding measure and not the consolation.

## What this run cannot settle

The ceiling. `tc-base` is a smaller model than the forward direction's base and
this fine-tune does not change that. A good result here is a good result for
`tc-base` and says nothing about what the direction could do on a larger one.
Nothing in the write-up may imply that English → Bosnian and Bosnian → English
are of comparable quality; on the evidence available they are not, and the
README already says so.

## v2 — read — bake-off — the engine question, written before any number exists

Written 3 September 2026, before `LILLY_READER=paddle` has read a single
photograph. `docs/OCR-ROADMAP.md`, step 1. Nothing below is reinterpreted
afterwards.

### The question

Six fine-tuning passes on EasyOCR's `latin_g2` (14–19) did not move the
reader, and the reason was the labels, not the recipe. The question that is
still open is not "more training" but whether a newer engine, **untrained**,
already reads these photographs better than the reader those passes could not
improve — in which case the next months go there, and if not, the engine
question is closed with a number instead of an opinion. The detector has never
been measured or changed here; a different detector is half of what a new
engine brings.

### The arms

- **Shipped** — `app.ocr` as installed: EasyOCR's CRAFT detector and the
  fine-tuned `lilly` recogniser. Its numbers exist (54.7% words per
  photograph, 180 invented words) and are **re-run, not quoted**, so that both
  sides of every comparison come from the same code on the same day.
- **Stock** — `LILLY_READER=stock`: EasyOCR's own `latin_g2`. The control,
  so the fine-tune's gain stands beside the engine gap.
- **PaddleOCR PP-OCRv6** — `LILLY_READER=paddle`: `PP-OCRv6_medium_det` and
  `PP-OCRv6_medium_rec`, what paddleocr 3.7 selects for `lang="bs"`.
- **PaddleOCR PP-OCRv5** — `LILLY_PADDLE_VERSION=PP-OCRv5`:
  `PP-OCRv5_server_det` and `latin_PP-OCRv5_mobile_rec`, the Latin model whose
  dictionary was checked to hold all of čćđšžČĆĐŠŽ.

Both PaddleOCR arms are the published weights with document preprocessing off
and nothing tuned: no threshold sweep, no resolution sweep, no allowlist. The
app's own two-megapixel path, through `app.ocr.scan`. Tuning a threshold on
the scored photographs would make the score a fit, and the 40 photographs are
the only test set there is.

### The measurement that decides

    python3 scripts/bakeoff_ocr.py

`training/evaluate_ocr.py` for every arm, in its own process with its own
reading cache, on the same 40 photographs and the same `truth.json`. The
reading cache is stamped with the reader's identity now, so an arm cannot be
scored on another arm's cached readings.

### The threshold

The bar is the **shipped arm as re-measured in this run**. It must reproduce
54.7% / 180; if it does not, the run is invalid and the reason is found before
anything is compared.

A PaddleOCR arm replaces the shipped reader if, on the 40 photographs, it is
**not worse on either row and better on at least one**:

| | shipped | a PaddleOCR arm must be |
|---|---|---|
| words per photograph | 54.7% | ≥ 54.7 |
| words invented that are on no sign | 180 | ≤ 180 |
| and | | > on at least one of the two |

A tie on both leaves the installed reader in place. The second row is in the
bar because recall can always be bought by guessing more, and for an app that
translates what it reads an invented word becomes an invented sentence.

If both PaddleOCR arms clear it: the higher words-per-photograph, then the
fewer invented words, then PP-OCRv5, whose dictionary is the one verified.

### Reported, and unable to change the decision

- Paired per-photograph deltas against the shipped arm, with a paired
  bootstrap p. Significance is reported and does not move the bar, as
  everywhere else in this file.
- The 71 held-out human crops, exact and folded, with the interval. The 666
  training-side crops in a row of their own, which compares nothing.
- Seconds per two-megapixel photograph on the Mac's CPU. The app runs CPU-only
  in Docker, so this is a product number, but speed is not in the bar.
- Words per photograph over the sign class alone (1–5 agreed words), beside
  the short and long boards. The owner said on 3 September that the product
  reads small signs and street names; the bar above is still the mean over
  every photograph, because that is what 54.7% is, and 13 sign photographs
  cannot carry a bar. Moving the bar to the sign row is the owner's call and
  is made here, before the run, or not at all.
- Whether the winning recogniser's dictionary can write all ten letters. If the
  arm that clears the bar cannot, the owner decides; this file does not
  pre-decide it and says so rather than inventing a rule after the number.

### What failure looks like

- **No PaddleOCR arm clears the bar.** The shipped reader stays; the engine
  question is closed with numbers; the next step is the roadmap's test set.
- **Recall rises, invented words rise with it.** Not adopted. The invented
  count becomes the thing to fix — a detector threshold, a size floor — as a
  new pre-registration, not a re-run of this one.
- **The shipped arm does not reproduce 54.7% / 180.** No comparison is made.
  The code path changed, and finding where comes first.

### What this run cannot settle

Whether fine-tuning PaddleOCR would help; anything about Cyrillic; and the
detector question on its own — `R_d` in the "picture" section above is still
the measurement for that, and it has still not been run.

### Amendment, 4 September 2026 — the bar under the stack the other engine needs

Written **after** the run, with the numbers known, and it says so. Authorised
by the owner on 4 September 2026 after the alternative — leave the bar at
54.7% / 180 and record no verdict — was put to them in the same words.

**The conflict.** The threshold above requires two things of the shipped arm:
that it be *re-measured in this run*, and that it *reproduce 54.7% / 180*.
Do-not-repeat 17 in `docs/OCR-ROADMAP.md` records, measured on the Mac with
the weights fixed at md5 `2010a2d4`, that these cannot both hold in any
environment the Paddle arms run in: installing `paddleocr` replaces the `cv2`
EasyOCR reads through (5.0.0 → 4.10.0), and under 4.10.0 the shipped arm reads
**54.5% / 182**. Restoring cv2 5.0.0 gives 54.7% / 180 back exactly and
segfaults both Paddle arms. The reader is the same file; the imaging stack
moved it by 0.2 points and 2 words.

**The amendment.** The bar is the shipped arm as re-measured in the same run
and under the same imaging stack as the arms it is compared against. The
reproduction check accepts drift from 54.7% / 180 of at most **0.5 points per
photograph and 5 invented words**, and only when (a) `read/lilly.pth` is the
scored reader by md5 (`training/RESULTS-ocr-weights.md`) and (b) the cause of
the drift is identified and recorded as an environment difference, as item 17
does. Drift beyond that, or drift with no cause found, leaves the run invalid
exactly as before: no comparison is made.

Why ±0.5 / ±5: it covers the measured drift with headroom and is below what a
28-photograph set can resolve — one word on one photograph moves the
per-photograph mean by up to 3.6 points — so it cannot flip a decision this set
is able to see. The rule itself is unchanged: not worse on either row, better on
at least one, ties to the installed reader, tie-break as written. The
"reported, unable to change the decision" list is unchanged.

**What it does not do.** It does not decide the 4 September run. Under either
reading of the bar — 54.7% / 180 or 54.5% / 182 — paddle-v6 at 67.7% / 106
clears it by thirteen points of recall and seventy-odd invented words, paired
bootstrap p < 0.001, with the held-out crops (50/71 against 30/71) and the
timing pointing the same way. Had the margin been inside 0.5 points or 5
words, this amendment would not have been written and the answer would have
stayed "no verdict".

## test-v2 — invented words — the definition, registered 4 September 2026

Registered by the owner after `training/RESULTS-ocr-test-v2.md` showed the
column on this set is partly transcription coverage (1,120 of PP-OCRv6's 2,373
on one museum panel both transcribers declined to finish) and after the two
readings were put to them in plain words. **The definition is the strict one
and does not change:** `evaluate_ocr.py`'s count — on each photograph whose key
holds at least one agreed word, the reader's words beyond those matching the
key — against the agreed key, not the union of the passes, with no board
excluded. The reason is the product's: an invented word becomes an invented
sentence, and a reader that reads what people could not is not, for this app,
worth a reader that says what is not there.

Under it, on test-v2 the shipped reader (`2010a2d4`) returns 2,071 and
PP-OCRv6 2,373; the bake-off rule's second row does not clear, and the app is
not switched on the 40's verdict alone.

## v2 — read — PP-OCRv6 confidence floor — written before the run

Written 4 September 2026, before any threshold has been tried. The bake-off
section's "what failure looks like" named this case in advance: *recall rises,
invented words rise with it — not adopted; the invented count becomes the
thing to fix, as a new pre-registration.* This is that pre-registration.

### The question

PP-OCRv6 reads 60.0% of a photograph's words on test-v2 against the shipped
reader's 34.6%, and returns 302 more words that are on no agreed sign. It
returns every recognition regardless of its own confidence
(`text_rec_score_thresh` 0, the library default; the app sets nothing). Does a
floor on that confidence remove the invented words without giving the recall
back — enough to clear the rule on test-v2?

### The arm

`LILLY_READER=paddle`, PP-OCRv6 medium det + rec, untrained, oneDNN off, the
same door as the bake-off (`app.ocr.scan`), with one new knob:
`LILLY_PADDLE_REC_THRESH` = the recogniser confidence below which a region is
dropped. Nothing else moves: detector thresholds, unclip ratio, resolution and
paragraph grouping stay at the bake-off's values. One lever, because the
question is whether *confidence* separates invented text from read text; if it
does not, a second lever is a second pre-registration.

### Two sets, two jobs, never crossed

- **Choose on the 40.** Sweep the floor over {0.5, 0.6, 0.7, 0.8, 0.9} on the
  40 scored photographs (`truth.json`), each setting in its own cache. The
  floor chosen is the **highest** one whose words-per-photograph is still
  ≥ 54.5% — the shipped arm's figure under the stack the Paddle arms need,
  `RESULTS-ocr-bakeoff.md` — so the 40 pick the setting and test-v2 never
  does. If no setting keeps 54.5%, the floor is 0 and the experiment has
  answered: confidence does not separate them.
- **Decide on test-v2, once.** One run at the chosen floor on the 280
  photographs, scored against `truth-v2.json`, the strict invented count.
  The bar is the shipped reader as scored on test-v2 on the Mac,
  `training/bakeoff/test-v2-lilly.json`: **34.6% words per photograph, 2,071
  invented.** The rule is the bake-off's, unchanged: not worse on either row,
  better on at least one. PP-OCRv6 is identical to the digit across the Mac
  and the cloud machine on both sets, so the arm may run on either; the
  shipped reader's row is the Mac's and is not re-measured here.

No second look: the threshold is not moved after the test-v2 number, and
test-v2 is not read at more than one setting. A run that peeks is a fit.

### Reported, and unable to change the decision

The whole sweep table on the 40 (recall, invented, diacritic, per setting);
paired per-photograph deltas on test-v2 against the shipped row from the
committed json, with a bootstrap p; the sign row (n=76) beside the mean; where
the remaining invented words sit (`training/invented_words.py`); seconds per
photograph.

### What failure looks like

- **No floor keeps 54.5% on the 40.** Confidence is not the separator. The
  next lever (detector box threshold, a size floor) is a new pre-registration.
- **The chosen floor clears recall on test-v2 and not invented words.** Not
  adopted. Same as above.
- **Clears both.** Adopt PP-OCRv6 at that floor: the switch is then made as a
  product change (default reader, `requirements.txt`, the Docker image, the
  cv2 pinning of do-not-repeat 17), reviewed by the owner, not by this run.

## v2 — picture — picture-olcum — addendum for two detectors, written 5 September 2026, before the count

The `picture-olcum` section above was written when the shipped reader was
EasyOCR (stock CRAFT detector, `lilly.pth` recogniser) and names a validity
check against 45.0% pooled / 54.7% per-photo. On 5 September 2026 the app
switched to PP-OCRv6 (`docs/OCR-ROADMAP.md`, step 6), so the detector the
decomposition is for has changed. Nothing above is rewritten; this addendum
says how the same measurement is taken now, and it is written before a single
word has been counted.

**Same definition, same set, same key.** R_d = (truth words covered by at least
one detected box) / 373, on the 40 in `scored-sample.txt` against `truth.json`,
partial overlap counts, a different printing of the same word does not. The
overlays come from `training/measure_detection.py`, unchanged.

**Two arms, both counted.**
- **PP-OCRv6 detector**, `paddle:PP-OCRv6_medium_det`, with the recognition
  floor OFF (`LILLY_PADDLE_REC_THRESH=0`). The floor drops a box *after* the
  recogniser has read it, so a word it removes is a recognition-stage loss and
  must not be charged to detection. The shipped configuration (floor 0.9) is
  the same detector with a stricter recogniser.
- **CRAFT**, `easyocr` with `lilly.pth`: the detector behind the 54.7%, so the
  section above gets its number too and the two detectors are compared on the
  same key.

**Validity check, restated per arm.** The boxes file each overlay run writes
carries the recogniser's text for every box (the same `read_regions` call the
scorer makes). Before the count is reported, that text is compared photograph
by photograph with the arm's bakeoff cache on the 40 (`training/bakeoff/
paddle-v6-photos.json`, `lilly-photos.json`); a mismatch means the detector
drawn is not the one scored, and the decomposition for that arm is not
reported. The EasyOCR arm reproduces 54.5% / 44.5% / 182 under cv2 4.10.0
rather than the 54.7% / 45.0% / 180 the section above names; that drift is the
4 September amendment's (do-not-repeat 17) and is inside its tolerance.

**Recognition given detection**, pooled / R_d as the section above fixes it.
The pooled figures are already on disk and are named here so they cannot be
chosen later: PP-OCRv6 floor off **71.0%**; PP-OCRv6 at the shipped floor 0.9
**69.4%** (`training/RESULTS-ocr-paddle-floor.md`); EasyOCR **44.5%**
(`training/RESULTS-ocr-bakeoff.md`). The first divides by R_d(PP-OCRv6) and is
the recogniser's own figure; the second divides by the same R_d and is the
shipped configuration's; the third divides by R_d(CRAFT).

**Who counts — a deviation, recorded.** The section above says "a human count".
The count is made by **two vision agents**, not a person: the owner delegated
the forward plan on 5 September 2026 ("sen nasıl uygun gördüysen öyle") and
there is nobody to make 746 word judgements per detector. So the count is done
the way the key itself was built — two counters, blind to each other and to the
reader's text (the overlay shows box indices only; the boxes file is never
given to a counter), each judging every one of the 373 words on the 28
photographs with agreed text, per `training/transcribe/COUNT-BRIEF.md`, merged
by `training/count_detection.py`. **R_d is the agreed count: a word is covered
when both counters say so.** Each counter's own figure and the number of
disagreements are reported beside it.

**A reliability check that can fail, set now.** If the two counters disagree on
more than 10% of the words (38 of 373) for an arm, that arm's count is reported
as unreliable and no decomposition is drawn from it. Words a counter could not
locate at all are a separate bucket (`not_located`), reported, and count as
not covered.

**What each outcome means** is unchanged from the section above, read for the
PP-OCRv6 arm: above 85% the recogniser is the ceiling and step 7 (recogniser
fine-tune) is aimed at the right stage; 60–85% both stages lose words and the
fine-tune's ceiling is R_d itself, said before that run; below 60% the
detector is the ceiling and step 7 is polishing behind a closed door. The
CRAFT figure is reported for comparison and decides nothing now.

**Bias disclosed.** The arm's name is in the overlay path a counter is given;
the counters have no stake in either detector and are never told which is
shipped, and the check above catches a counter that stops looking. The 40 are
also the set the confidence floor was chosen on (decision, 5 September); R_d is
a measurement with no bar, so nothing is being cleared on a spent set.

## v2 — read — Cyrillic rescue under PP-OCRv6 — written before the run, 5 September 2026

**The hole.** PP-OCRv6's recogniser dictionary has 18,708 characters and not
one Serbian Cyrillic letter (checked 5 September 2026, `docs/OCR-ROADMAP.md`
step 5). test-v2's key carries **628 Cyrillic words on 21 of its 132**
photographs with text — 21.6% of the key's 2,907 words, 328 of them on
`Info_Atik_džamija.jpg` and 132 on `Life_scupture_Banja_Luka_01.jpg`; the 40
carry 26 on 7 (bilingual signposts). Every one of those words is a certain
miss today, and stays one however well the recogniser is fine-tuned on Latin.

**The prior is negative and is not being ignored.** With EasyOCR, choosing
between the Latin and Cyrillic recognisers by confidence was worse at every
margin, and every union rule paid in invented words — the best trade was +2.3
points per photograph for 180 → 246 (`training/RESULTS-ocr-cyrillic.md`). The
reason recorded there, that a Cyrillic model reads Latin text as confident
lookalikes, applies to any second recogniser. So the rule below does not
arbitrate between two readings of a box that already counts.

**The rule, fixed now, with no free parameter.** `LILLY_PADDLE_CYRILLIC_RESCUE=1`
in `app/ocr.py`: the PP-OCRv6 pipeline runs with its floor off. A box whose
PP-OCRv6 confidence is at or above **0.9** is emitted exactly as the shipped
configuration emits it. A box below 0.9 — one the shipped configuration drops
— is cropped and read once by `cyrillic_PP-OCRv5_mobile_rec` (Baidu's
published weights, untrained, the same Hugging Face mirror); if that reading's
confidence is at or above **0.9** it is emitted, otherwise the box is dropped
as before. Both floors are the shipped floor, chosen on 4–5 September before
this section existed; nothing is swept. By construction the rescue cannot
remove or change a word the shipped configuration emits: words found per
photograph and pooled can only rise or stay, and invented words can only rise
or stay. The whole question is whether the rescued boxes are Cyrillic words or
confident garbage, and only test-v2 can answer it.

**The bars, both, from picture-egitim's rule, against the shipped
configuration's own test-v2 figures** (`training/RESULTS-ocr-paddle-floor.md`:
57.8% per photograph, 450 invented):
- words found per photograph must **rise** — paired against the shipped arm,
  the 95% bootstrap interval of the per-photograph difference excludes zero;
- invented words (strict, decision 4) must be **≤ 450**. No-regression, not
  improvement: a rescue that adds words it made up does not ship, however many
  real ones it adds beside them.
Reported beside the bars, deciding nothing: how many of the 628 Cyrillic key
words the rescued path finds, and how many rescued words are on no sign.

**Order.** The 40 first, as a smoke test only — 26 Cyrillic words decide
nothing. There the untouched boxes must reproduce the shipped arm's 67.0% /
65 mechanically (rescue off equals shipped), and the rescue arm's own figures
are recorded. Then **one** run on test-v2 with `evaluate_ocr.py` unchanged
under the rescue arm's own cache stamp. No second look, no second floor, no
variant of the rule after the number.

**What each outcome means.**
- Both bars hold: the rescue is the shipped configuration (step 6's
  environment gains one line), and step 7 fine-tunes the Latin recogniser
  knowing Cyrillic is handled at read time.
- Recall rises and invented words exceed 450: does not ship. The count of
  rescued words that are key words against rescued words on no sign is
  reported, and the Cyrillic question moves to step 7 as a training and
  dictionary question, not a read-time one — the EasyOCR result again, on a
  second engine.
- Recall does not rise: the Cyrillic recogniser at 0.9 reads almost none of
  the dropped boxes. Reported, and the read-time line is closed for good.

## v2 — read — step 7, PP-OCRv6 recogniser fine-tune — written before the run, 5 September 2026

**Why now, and what R_d said.** Step 3 measured the shipped configuration's
two stages on the 40 (`training/RESULTS-ocr-detection.md`): the PP-OCRv6
detector boxes 84.7% of the key's words and the recogniser then reads 83.8%
of what it is handed; the two stages lose about 15 and 14 words in every 100.
A recogniser fine-tune is therefore aimed at a stage that still loses words,
not at a closed door — and it can at most recover the recogniser's share.
The roadmap's own expectation stands: "expect little"; the EasyOCR fine-tune
was worth +4.6 points on test-v2 and PP-OCRv6 untrained is 25 points ahead.

**The data, fixed now.** The blind-labelled Commons crops already on disk and
on Kaggle (`lilly-ocr-crops`): `data/ocr/crops/labels-human.tsv` (1,701) and
`data/ocr/crops2/labels-human.tsv` (98 rows with a label), cut by CRAFT from
186 Commons photographs, labelled by agent passes that never saw a reader's
output (roadmap step 4 says how to read the word "human"). From them:
- **Cyrillic labels dropped** (272): PP-OCRv6's dictionary has no Cyrillic and
  this run does not change the dictionary; Cyrillic is step 5's question.
- **Labels longer than 25 characters dropped** (85 of the Latin 1,527): the
  config's `max_text_length` is 25 and the NRTR head is built for it; raising
  it would change the head the pretrained weights fit. Disclosed: this drops
  long board lines, which are the hardest, so the training set is a little
  easier than the photographs.
- `__NOTEXT__` / `__UNSURE__` dropped (there are none in these files).
- **Split by source photograph**, never by crop: a crop is validation when
  `blake2b(source photograph name, 4 bytes) % 10 == 0`. On today's files that
  is 395 validation crops from 22 photographs and 1,132 training crops from
  163 (`training/paddle_rec_data.py` prints the counts and which boards fell
  where; the two densest sources, the Jajce information board with 218 and
  the Bistrik observatory board with 135, are wherever the hash put them).
  No crop's source photograph is in test-v2 or in the 40 (checked: 0 and 0).
- Not used: Mapillary crops (shop signs, closed line), synthetic renders,
  plates, anything a reader labelled.

**The recipe, fixed now.** PaddleOCR **v3.7.0** (the tag matching the
installed `paddleocr==3.7.0`), `configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml`
unchanged except these overrides:
`Global.pretrained_model` = Baidu's `PP-OCRv6_medium_rec_pretrained.pdparams`
(the training checkpoint; the inference model on the Hugging Face mirror
cannot be trained from), `Global.epoch_num=30`,
`Optimizer.lr.learning_rate=0.0001` (a fifth of the from-scratch 0.0005),
`Optimizer.lr.warmup_epoch=2`, `Global.eval_batch_step=[0, 18]` (one epoch is
about 18 iterations at the config's batch of 64), `Global.save_epoch_step=5`,
`Global.cal_metric_during_train=true`; dictionary, image shape (3×48×320),
augmentation (RecConAug, RecAug) and both heads as in the config. **Model
selection is PaddleOCR's own `best_accuracy` on the validation crops** — the
only thing that ever chooses a checkpoint; test-v2 chooses nothing. One run,
one seed (the config's). GPU: Kaggle T4, launched from the Mac with
`scripts/kaggle_train.py ocr-paddle`; notebook
`training/Lilly_OCR_Paddle_Kaggle.ipynb` written to `docs/kaggle-notebooks.md`.

**The crop gate, on the box, before any zip.** The best checkpoint is
exported to inference format and scored **in one process with the stock
PP-OCRv6_medium_rec** on the 395 validation crops through PaddleX's
`TextRecognition` module: exact-match and diacritic-folded accuracy, counts
beside percentages, a paired bootstrap interval on the exact difference. If
the fine-tuned model's exact accuracy is below the stock model's, the run
ends with `SystemExit` and nothing is zipped. A NaN loss, a run that never
reached epoch 30, or a missing pretrained file is `SystemExit`, not a warning.
Passing the crop gate ships nothing; it only lets the model be measured.

**The photograph bars, on test-v2, one look.** The exported model is loaded
through the app's door (`LILLY_PADDLE_REC_DIR`, added for this; identity
stamped with the weights' md5) at the **shipped floor 0.9 — not re-swept**,
which is disclosed as a possible disadvantage for a model whose confidences
are shaped differently. `evaluate_ocr.py` unchanged, `training/rescue_report.py`'s
two-arm layout for the report. Both bars, from picture-egitim's rule, against
the shipped configuration's test-v2 figures (57.8% / 450):
- words found per photograph **rise**, paired 95% interval excluding zero;
- invented words (strict, decision 4) **≤ 450**.
The 40 are reported beside (67.0% / 65) and decide nothing. No second run,
no second floor, no second checkpoint after the number.

**What each outcome means.** Both hold: the fine-tuned recogniser ships
(`models/` carries it, the model card says so, and the Hugging Face bundle
gets a `read-paddle/` directory). Recall rises and invented words exceed 450:
does not ship; reported with the count of new invented words. Recall does not
rise: the recogniser is not the stage to train on these 1,132 lines, and the
next move is labels from the test-v2 pool remainder (206 undrawn `keep`
photographs and 318 one-region ones, two blind passes) or the detector's
small-type misses — each its own pre-registration. Neither holds: reported,
and the shipped configuration stays.

### Note, same day — the split's counts after the over-length drop

The counts above (395 / 1,132) were computed before the 85 over-length labels
were removed. `training/paddle_rec_data.py` on today's files: **371 validation
crops from 22 photographs, 1,071 training crops from 163.** The Jajce
information board (218 lines after the drop) fell on the validation side and
the Bistrik observatory board (128) on the training side — the hash's doing,
recorded here so it cannot be re-rolled. Rules unchanged.

### Outcome, 6 September 2026 — neither bar holds, the shipped configuration stays

The fine-tune ran on Kaggle (T4, run `20260906-2143`), passed the crop gate
(exact 264 → 278 on the 371 validation crops, +3.8 points, 95% −0.5 to +8.1;
folded 290 → 287, so the gate let the model be measured and, as written above,
decided nothing), and was scored **once** on the cloud through
`LILLY_PADDLE_REC_DIR` at the shipped floor 0.9, exactly as fixed above. On
test-v2 (n=132), paired against the shipped configuration
(`training/paddle-floor/test-v2-floor0.9.json`):

- words per photograph 57.8% → 60.8%, paired mean Δ +3.0 points, **95% −1.3 to
  +7.2** (35 up, 14 down, bootstrap p 0.164) — the rise bar (interval excluding
  zero) **does not hold**;
- invented words 450 → **703** — the invented bar (≤ 450) **does not hold**.

Neither bar holds. By the rule above ("Neither holds: reported, and the
shipped configuration stays") the fine-tuned recogniser does not ship and the
shipped configuration is unchanged. The 40, reported beside and deciding
nothing: 67.0% → 68.2% words per photograph, invented 65 → 55, diacritic
68.0% → 80.0% (paired +1.2, 95% −1.4 to +4.5).

What moved: diacritic-word recall rose 62.1% → 77.1% (folded 81.8% → 89.3%) —
the recogniser learned the letters it was trained on — but at the un-swept
floor it keeps 253 more invented words, and the per-photograph rise is not
separable from zero. This look is spent: no second run, no second floor, no
second checkpoint. The next lever (more labelled test-v2-pool lines, or the
detector's small-type misses from step 3) is a new pre-registration.
See `training/RESULTS-ocr-paddle-finetune.md`; candidate weights stay on
`Safak11/lilly-ocr-paddle-runs/20260906-2143`, not in `models/`.

## v2 — read — the fine-tuned recogniser's confidence floor — written before the run, 6 September 2026

Written 6 September 2026, before this recogniser has been read at any floor but
0.9 and before any test-v2 number below or above 0.9 exists. Step 7's "what
each outcome means" named this case in advance: *recall rises and invented
words exceed 450: does not ship; reported with the count of new invented
words* — and its outcome disclosed the un-swept floor as a possible
disadvantage "for a model whose confidences are shaped differently." This is
the pre-registration that tests that disclosed caveat. It is a **new lever**:
step 7's look is spent (no second floor under it), and the shipped reader's
floor was swept for the shipped reader, never for these weights.

### The question

The step-7 fine-tune reads better than the shipped recogniser — on test-v2,
diacritic-word recall 62.1% → 77.1%, folded 81.8% → 89.3%, words per
photograph 57.8% → 60.8% — but at the shipped floor 0.9 it kept 703 words that
are on no sign against the shipped reader's 450 (`training/paddle-finetune/`,
`training/paddle-floor/test-v2-floor0.9.json`). The floor was the shipped
reader's, not this one's. Does a floor **calibrated to these weights** cut the
253 extra invented words back to the shipped count without giving the reading
gain back — enough to clear both bars on test-v2 against the shipped
configuration?

### What is fixed, and the one free knob

Fixed, not free: the step-7 fine-tuned recogniser exactly as exported —
`Safak11/lilly-ocr-paddle-runs/20260906-2143`, reader `e939d2a3c9fe2913`,
weights md5 `87ad04a5`, loaded through the app's door
(`LILLY_PADDLE_REC_DIR` + `app.ocr.scan`). **No retraining, no new checkpoint,
no new labels, no detector change** — the detector stays the same
`PP-OCRv6_medium_det`, resolution and paragraph grouping stay at the shipped
values. The **one** free knob is `LILLY_PADDLE_REC_THRESH`, the recogniser
confidence below which a region is dropped, applied in-code as
`text_rec_score_thresh`. One lever, because the question is whether *this
recogniser's* confidence separates its invented text from its read text; if it
does not, a second lever is a second pre-registration.

### Two sets, two jobs, never crossed

- **Choose on the 40.** Sweep the floor on the 40 scored photographs
  (`truth.json`), each setting in its own cache, over {0.5, 0.6, 0.7, 0.8,
  0.9} and — because cutting invented needs a floor at or above the shipped
  0.9 — continuing above 0.9 in steps of 0.02 (0.92, 0.94, 0.96, 0.98) until
  the 40's words-per-photograph first falls below **67.0%**, the shipped
  reader's the-40 figure at floor 0.9 (`training/paddle-floor/the40-floor0.9.json`).
  The floor chosen, **f\***, is the **highest** swept setting whose the-40
  words-per-photograph is still ≥ 67.0% — so the 40 pick the setting and
  test-v2 never does. Disclosed in advance, not hidden: on the 40 this
  recogniser already invents *fewer* words than the shipped reader (55 vs 65
  at floor 0.9), so the +253 blow-up lives in test-v2's dense,
  large-denominator photographs, and a floor chosen on the 40 may be chosen
  too low to cut them — that transfer risk is accepted, and test-v2 is the
  decider that exposes it.
- **Decide on test-v2, once.** One read at f\* on the 280 photographs
  (`truth-v2.json`), the strict invented count (owner decision 4), scored
  through the same door. The bars are the **shipped configuration's** test-v2
  figures at floor 0.9 — **57.8% words per photograph, 450 invented**
  (`training/paddle-floor/test-v2-floor0.9.json`, reader `aea5890abfdb7ba3`) —
  the same comparator step 7 used, because the product question is whether the
  fine-tune at its own floor beats what ships today, not whether it beats
  itself at 0.9.

No second look: f\* is not moved after the test-v2 number, and test-v2 is not
read at more than one setting. A run that peeks is a fit.

### The two bars, and the metric

Against the shipped configuration, both must hold to ship — the same two-sided
shape as step 7:

- **words found per photograph rise** — paired against the shipped arm, the
  95% bootstrap interval of the per-photograph difference
  (`training/rescue_report.py` `interval()`: 10,000 resamples, seed 0,
  percentile) **excludes zero**;
- **invented words (strict, decision 4) ≤ 450** — no-regression, not
  improvement.

The deciding metric is **exact** words per photograph, identical to step 7 and
to what `rescue_report.py` computes, so the two looks compare like with like.
**Folded** (diacritic-blind) recall — the product's meaning-not-hats view
(`.claude/CLAUDE.md`), on which this recogniser already leads 89.3% vs 81.8% —
is reported beside but does **not** decide here; moving the decider from exact
to folded after step 7 scored exact would be moving the goalposts, so a
folded-metric bar is its own future pre-registration, not a switch made now.

### Reported, and unable to change the decision

The whole floor-sweep table on the 40 (words per photograph, invented,
diacritic, folded, per setting) and the chosen f\*; the paired per-photograph
deltas on test-v2 against the shipped row with a bootstrap p, the up/down
photograph counts, counts beside every percentage and the interval beside the
delta; folded, diacritic and pooled recall at f\*; where the remaining invented
words sit (`training/invented_words.py`); seconds per photograph. The crop gate
(exact 264 → 278, folded 290 → 287) is already spent and decides nothing here.

### What each outcome means

- **Both bars hold.** The re-swept fine-tune ships, as a product change
  reviewed by the owner: the weights go into `models/`, the model card says so,
  and the Hugging Face bundle gets a `read-paddle/` directory. Not by this run.
- **Invented ≤ 450 holds but the recall interval still straddles zero.** The
  floor *was* the invented driver (step 7's cause confirmed), but the reading
  gain is not separable from zero at any recall-preserving floor. Does not
  ship. The next lever is the detector's small-type misses (step 3: PP-OCRv6's
  detection misses concentrate on few-pixel type) or the per-photograph
  variance itself — each its own pre-registration, not this recogniser again.
- **Invented still > 450 at f\*.** The 253 extra words were not floor-removable
  without losing recall; this recogniser's confidence does not separate its
  invented text from its read text. Does not ship; the shipped configuration
  stays.
- **No swept floor keeps the-40 words-per-photograph ≥ 67.0%.** Confidence
  cannot cut invented on this recogniser without dropping below the shipped
  reader's own recall. Does not ship; the shipped configuration stays.

### What this run cannot settle

Even both bars passing leaves the true recall ceiling at the folded 89.3%
(the detector's limit at this operating point) and the residual diacritic gap;
this run decides the shippability of the step-7 recogniser at a calibrated
floor, not the domain ceiling, and not the exact-versus-folded metric question.
No training runs here, so the crop gate does not apply and nothing is zipped;
the weights move into `models/` only if both bars hold, the candidate otherwise
stays on Hugging Face, and the per-floor reader-output caches are regenerable
scratch, not committed — but the test-v2 read at f\* keeps its per-region
output so the invented change is attributable, not inferred.

### Outcome, 7 September 2026 — invented fixed, recall not; does not ship

The re-sweep ran on the cloud (`scripts/paddle_finetune_floor.py`), no
retraining. On the 40 the floor grid picked **f\* = 0.94** — the highest floor
keeping the-40 words-per-photograph ≥ 67.0% (at 0.96 it falls to 65.3%);
invented on the 40 fell 55 → 48 across 0.9 → 0.94. test-v2 was read once at
0.94, against the shipped configuration (57.8% / 450):

- invented words 450 → **432** — the bar (≤ 450) **holds**. The floor was the
  invented driver, as diagnosed: a floor calibrated to these weights cuts the
  step-7 blow-out (703) back under the shipped count.
- words per photograph 57.8% → **54.5%**, paired mean Δ **−3.3 points**, 95%
  **−8.4 to +1.5** — the rise bar **does not hold**, and the point estimate is
  negative. The floor that cuts invented to ≤ 450 also drops correct words on
  the many sparse signs, so per-photograph recall lands below the shipped reader.

Both bars do not hold together. By the rule above (invented holds but the recall
interval straddles zero → does not ship), **the fine-tune does not ship at any
floor and the shipped configuration stays.** The 40, beside and deciding
nothing: 67.0% → 67.9% words/photo, invented 65 → 48 (paired +0.9, 95% −2.0 to
+4.3). Both step-7 looks — the recogniser and its floor — are now spent.

Noted, not acted on (switching the decider now would be moving the goalposts):
at f\* = 0.94 the fine-tune beats the shipped reader on pooled (64.8 → 66.1),
diacritic (62.1 → 76.2) and folded (81.8 → **86.9**) with **fewer** invented
words (432 < 450) — it loses only on the exact per-photograph metric this run
fixed as the decider. The product bar is meaning, not hats (folded), so deciding
this same candidate on the folded metric is a live next pre-registration,
alongside the detector lever. See `training/RESULTS-ocr-paddle-finetune.md`.

## v2 — read — the fine-tune on the meaning (folded) metric — written before the folded number, 7 September 2026

Written 7 September 2026, before the folded per-photograph interval has been
computed. Step 7 (at floor 0.9) and step 7a (at the re-swept floor 0.94) both
decided the fine-tune on **exact** per-photograph recall and both said it does
not ship. But `.claude/CLAUDE.md` fixes the product bar as **meaning, not
hats** — `kuca` → House is enough, a dropped hat on a word the translator still
gets is not a reason to throw the row out. On the diacritic-blind (folded) view
the fine-tune already leads the shipped reader (81.8% → 86.9% folded) with
**fewer** invented words (432 < 450) at the step-7a floor. Step 7a's outcome
named this as its own pre-registration; this is it.

### The question

Read on the product's own meaning-metric — folded, diacritic-blind — does the
fine-tune beat the shipped reader on test-v2 by the same two-sided rule the
exact looks used, or is even the meaning-level per-photograph rise
indistinguishable from zero?

### What is fixed, and what is new

Fixed, not free: **the same candidate weights** (`Safak11/lilly-ocr-paddle-runs/
20260906-2143`, reader `f104826cee294a28` at the floor below) **and the same
floor f\* = 0.94** that step 7a's pre-registered rule already chose and already
read on test-v2. **No new floor sweep** (that would be a second look on the
floor), **no retraining, no new read of any photograph.** The only new thing is
the **deciding metric**: folded (diacritic-blind) per-photograph recall in place
of exact. Reusing 0.94 is not a free choice dressed up: the invented bar (≤ 450,
a test-v2 quantity) is the binding constraint, 0.94 is where test-v2 invented
first crosses under 450 (432; at 0.9 it is 703 and fails), and a higher floor
only cuts folded recall — so 0.94 is the floor that clears invented while
keeping folded recall highest, and step 7a already landed and read it.

### The measurement that decides

The folded per-photograph paired 95% bootstrap interval, candidate @0.94 against
the shipped configuration @0.9, computed from the **committed reader-output
caches** (`data/ocr/real-photos/test-v2/reader-output-paddle-v6-floor0.9.json`
for shipped; the step-7a candidate cache for the fine-tune) and `truth-v2.json`
— no photograph is re-read. `training/folded_decision.py` mirrors
`evaluate_ocr.py`'s word tokenisation and fold, and `rescue_report.py`'s paired
percentile bootstrap (10,000 resamples, seed 0), so the folded figure is the
exact figure's method with the fold applied. This number does not exist yet.

### The two bars, against the shipped configuration @0.9

Both must hold to ship:

- **meaning found per photograph rises** — folded words-per-photograph, paired
  against the shipped arm, 95% bootstrap interval of the per-photograph
  difference **excludes zero**;
- **invented words ≤ 450** — the **strict, exact** invented count (owner
  decision 4), unchanged and **not** loosened to a folded count. Recall moves to
  meaning; hallucination stays measured strictly. At 0.94 this is already 432,
  so it holds; it is stated so the rule is complete, not to be re-opened.

### Reported, and unable to change the decision

The exact per-photograph delta from step 7a (−3.3, the losing metric) beside the
folded one; pooled folded (64.8/81.8 shipped vs 66.1/86.9 fine-tune, already
known); diacritic recall; the 40's folded numbers; counts beside percentages and
the interval beside the delta.

### What each outcome means

- **Both bars hold.** The fine-tune ships **on the meaning metric**, as an
  owner-reviewed product change (weights into `models/`, model card, the HF
  bundle's `read-paddle/` directory). The exact-metric loss on sparse signs
  (−3.3) is disclosed as the known cost. Two things still need the owner/Mac and
  are not this run's to decide: the HF publish (needs `HF_TOKEN`, Mac only) and
  how the app fetches the fine-tuned weights at runtime.
- **The folded interval still straddles zero.** Even on the product's own
  metric the per-photograph rise is not distinguishable from zero. The fine-tune
  does not ship; the recogniser lever is fully spent, and the next lever is the
  detector's small-type misses (step 3), its own pre-registration.

### What this run cannot settle

It decides the fine-tune's shippability on the meaning metric at the step-7a
floor; it does not re-open the floor, the exact metric, or the detector ceiling.
It re-scores committed caches only — no training, no reading — so nothing is
zipped and no crop gate applies; the weights move into `models/` only if both
bars hold, and the candidate otherwise stays on Hugging Face.

### Outcome, 7 September 2026 — worse on the meaning metric; does not ship

Computed from the committed caches (`training/folded_decision.py`), no re-read.
The exact per-photograph delta reproduces step 7a's −3.3 (sanity check that the
folded scorer is `evaluate_ocr.py`'s method). On the deciding metric — **all
words, diacritics folded, per photograph** — candidate @0.94 against shipped
@0.9 (n=132):

- folded (all words) words per photograph 59.9% → **54.8%**, paired mean Δ
  **−5.1 points**, 95% **−10.1 to −0.6** (p 0.026, 22 up / 30 down) — the rise
  bar does not hold, and the interval is **entirely below zero**: the fine-tune
  is significantly *worse* on the meaning metric per photograph.
- invented (strict, exact) 432 ≤ 450 — holds.

Both bars do not hold; the fine-tune does not ship on the meaning metric. The
diacritic-words-*only* folded row did rise (+7.8, 95% +0.1 to +17.0) — the
accented slice the fine-tune learned — but that is ~214 of ~2,900 words and does
not carry the photograph; pooled all-words folded is a hair higher for the
fine-tune (66.2 → 67.0) on the dense boards, yet per photograph it is 5 points
lower. The "81.8 → 86.9 folded" cited in step 7a was this diacritic-only slice,
corrected here. The recogniser lever is now spent across all three looks
(exact @0.9, exact @0.94, meaning @0.94); the next lever is the detector's
small-type misses (step 3), its own pre-registration. See
`training/RESULTS-ocr-paddle-finetune.md`.

## v2 — picture — the detector's small-type misses: higher-resolution detection — written before the run, 7 September 2026

Written 7 September 2026, before the detector has been run at any side length but
the default. The recogniser lever is spent (steps 7, 7a, 7b — all no). Step 3
measured the shipped PP-OCRv6 detector at **R_d = 84.7%** (it never boxes ~15 of
every 100 key words), its misses concentrated on **few-pixel type**: 28 of its
57 misses on the 40 sit on one 144-word memorial whose text is 9–21 px high on a
1280 px render, where CRAFT boxes 138/144 and PP-OCRv6 116/144
(`training/RESULTS-ocr-detection.md`). That file named the detector move — "a
larger `text_det_limit_side_len`, or PP-OCRv6's server detector" — as its own
pre-registration. This is the first, cheapest route: the **same** medium
detector run at a **higher resolution**, so it sees the full working image
instead of the ~960 px version it is down-scaled to before detection. Inference
only — no training, no new labels, no new model.

### The question

The app shrinks a photograph to two megapixels, then PaddleOCR shrinks it again
to the detector's `text_det_limit_side_len` (~960) before finding boxes — so
9–21 px type becomes a handful of pixels and is never boxed. Does raising that
detection side length recover the small-type words the default loses, enough to
raise words per photograph on test-v2 without inventing more from brickwork?

### What is fixed, and the one free knob

Fixed: the shipped configuration — PP-OCRv6 **medium** detector and **medium**
recogniser, the recogniser floor 0.9, the app's two-megapixel working size, and
all detector thresholds (unclip ratio, box score) at their defaults. **The one
free knob is `text_det_limit_side_len`** (with `text_det_limit_type='max'`),
exposed through a new env `LILLY_PADDLE_DET_SIDE_LEN` added to `app/ocr.py`'s
pipeline construction **before the run** and never tuned on test-v2. Only *where
text is found* changes; the recogniser and its floor do not move. Disclosed: the
two-megapixel working-size shrink still applies and caps the benefit — whether
to raise the working size itself is a separate product knob and a separate
pre-registration, not this run.

### Two sets, two jobs, never crossed

- **Choose on the 40.** Sweep `text_det_limit_side_len` over {960 (default),
  1280, 1600, 2048} on the 40 (`truth.json`), each setting its own cache. The
  value chosen, **s\***, is the one that **maximises the-40 words-per-photograph
  subject to the-40 invented ≤ 65** (the shipped detector's the-40 count) — so
  recall is recovered, not bought with hallucination. Seconds per photograph are
  recorded (higher resolution is slower; a product cost the owner weighs). If no
  value beats the default, s\* = 960 and the experiment has answered: resolution
  is not the detector lever at this working size.
- **Decide on test-v2, once.** One read at s\* on the 280 photographs
  (`truth-v2.json`), scored against the shipped configuration @0.9 — **57.8%
  words per photograph, 450 invented** (`training/paddle-floor/test-v2-floor0.9.json`).
  No second side length on test-v2; a run that peeks is a fit.

### The two bars

Against the shipped configuration, both must hold to ship (the same two-sided
rule every look has used):

- **words found per photograph rise** — paired against the shipped arm, the 95%
  bootstrap interval of the per-photograph difference (`rescue_report.py`,
  10,000 resamples, seed 0) **excludes zero**;
- **invented words ≤ 450** — strict, exact, unchanged.

The deciding metric is exact per photograph, folded reported beside — no metric
switch mid-stream (the step-7b lesson).

### Reported, and unable to change the decision

The whole side-length sweep on the 40 (recall, invented, seconds per photograph,
per setting) and the chosen s\*; the paired test-v2 delta with its interval and
up/down counts; folded and diacritic recall at s\*; recognition-given-detection
if a two-blind-counter detection count is later run on test-v2 (an optional
diagnostic, not required for the decision); the 40 beside; counts beside
percentages, the interval beside the delta.

### What each outcome means

- **Both bars hold.** Higher-resolution detection ships as an owner-reviewed
  product change: the side-length knob defaults to s\*, the read-time cost
  disclosed. Not by this run.
- **Recall does not rise (interval straddles zero).** Resolution is not the
  detector lever at the two-megapixel working size. The next moves — a different
  detector (CRAFT-detect + PP-OCRv6-recognise hybrid, or the server detector),
  or raising the app's working size — are each their own pre-registration.
- **Recall rises but invented > 450.** Higher resolution finds more real text
  and more brickwork; does not ship as-is; reported with the invented count.

### What this run cannot settle

It tests detection resolution *within* the two-megapixel working size; it does
not test the working size itself, a different detector model, or a detector
fine-tune (which would need box-level labels that do not exist and a GPU). Even
a pass leaves the recogniser's own ceiling (recognition-given-detection 81.9%)
where it is. Inference only — nothing is trained or zipped; the knob defaults to
s\* only if both bars hold.

### Outcome, 7 September 2026 — no gain; resolution is not the lever; does not ship

Ran on the cloud (`scripts/paddle_detect_sidelen.py`), inference only. The-40
sweep: 960 → 65.7%, **1280 → 67.1%**, 1600 → 67.1%, 2048 → 67.1% words per
photograph (invented 60/64/65/64) — recall plateaus at 1280 and higher side
lengths find nothing more, so the library default already detects at roughly
1280. **s\* = 1280.** On test-v2 against the shipped configuration @0.9:

- words per photograph 57.8% → **57.3%**, paired mean Δ **−0.4 points**, 95%
  **−3.1 to +1.9** (12 up, 11 down, p 0.758) — the rise bar does not hold; the
  change is flat, only 23 of 132 photographs move at all.
- invented 450 → **438** — holds.
- diacritic and folded unchanged (62.1%, 81.8%).

The 40 beside (decides nothing): Δ +0.0, one photograph changed. By the rule the
higher-resolution detector does not ship; the shipped configuration stays, and
the `LILLY_PADDLE_DET_SIDE_LEN` knob defaults off.

A clean negative with a clear meaning: the medium detector at the two-megapixel
working size already finds what raising the detection side length can find, so
the step-3 small-type misses are lost **upstream** — to the app's two-megapixel
shrink, before the detector — or to the detector model itself, not to the
detection resolution. The next detector-side pre-registrations are (1) raising
the app's working size (read at more than two megapixels; `evaluate_ocr.py
--full-res` measures the ceiling) and (2) a different detector (CRAFT-detect +
PP-OCRv6-recognise hybrid, or the server detector).
See `training/RESULTS-ocr-detection.md`.

## v2 — picture — the reader on full-resolution inputs — written before the run, 7 September 2026

Written 7 September 2026, before any full-resolution number exists. Found while
closing the detector line: the committed test-v2 photos are **downscales** —
`20130606_Mostar_034.jpg` is 1280 px on disk but 3968×2976 on Commons (Commons
API, 2/2 sampled). The app reads at up to 2 MP (~1633 px), so a real
high-resolution upload gives the reader ~1633 px of a sign, not the benchmark's
1280 px — a regime the step-8 side-length sweep could not test because it ran on
the 1280 px files. Real users upload high-resolution phone photos, so the
shipped reader's 57.8% words/photo on test-v2 may **understate** real use.

### The question

Re-read the same test-v2 photos at their **full Commons resolution** through the
shipped reader (PP-OCRv6, floor 0.9, `app.ocr.scan` — unchanged), scored against
the same `truth-v2.json` (truth is by word, resolution-independent). Does
words-per-photograph rise over the 1280 px benchmark?

### What is fixed, and what changes

Fixed: the shipped reader and the whole app path, including the 2 MP working-size
cap — **nothing about the model or the code changes.** The only thing that
changes is the **input photograph's resolution** (the full original instead of
the 1280 px downscale), which is the user's own photo, not something the product
controls. This is a **measurement of the shipped reader on realistic inputs**,
not a change to be shipped.

### Method

`scripts/fetch_highres_and_score.py`, **on the Mac** (bulk-fetching Commons
originals is rate-limited from the cloud — HTTP 429, 0/6). For each test-v2
photo with text: fetch the Commons original, read through `app.ocr.scan`, cache
the reading, delete the original (disk-safe, resumable). Then score the cache
against `truth-v2.json` with `evaluate_ocr.py`, and compare to the committed
downscaled score `training/paddle-floor/test-v2-floor0.9.json`. If more than 15%
of photos cannot be fetched, it refuses to score a holey set.

### What each outcome means

- **words per photograph rise, paired 95% interval (rescue_report.py) excluding
  zero.** The benchmark understated real use; the shipped reader is better on the
  high-resolution photos a user actually takes. Record the higher figure as the
  realistic one beside the benchmark; whether raising the 2 MP cap itself helps
  further is a separate pre-registration.
- **flat.** The 1280 px downscale was not costing recall; the benchmark is fair
  and 57.8% stands for real use too.

Invented is reported beside (more resolution can find more real text and more
brickwork). Some Commons titles will not resolve; the miss count is reported and
the caches (`training/highres/reader-output-fullres.json`) are regenerable
scratch, not committed — only the scored summary is.

---

# Amendment, 7 September 2026 — the Bosnian form rate, defined before the base number exists

The "v3 — reply — English to Bosnian" section above put a third row in its
deciding table — **Bosnian form rate, output side** — and then, correctly,
refused to define it loosely: it fixed three properties (baseline measured on
the base *before* the fine-tune launches, cases reused from `bench/` rather than
rebuilt, non-discriminating cases dropped *before* the run) and left the scorer
to be written. `training/bosnian_bench.py` cannot be that scorer; it says so in
its own docstring, because it reads English output.

This amendment writes the scorer's rule down before it has produced a number
about any model. `training/bosnian_form_rate.py` implements it. Nothing below is
reinterpreted once a model output exists.

## The rule

Every surviving target is read twice against the same output, with the same
matcher: does the Bosnian surface form appear (exact form, word boundary, case
folded), and does its Croatian or Serbian counterpart appear?

    decided     the output contains exactly one of the two
    form rate = decided-for-Bosnian / decided
    silent    = attempted − decided, reported beside it and never averaged in

**A silent target is the instrument unable to say, not a miss.** Neither form
present (the model wrote a third word, or a different inflection) and both forms
present (the output hedged) are the same category and are reported as their own
column, exactly as `bosnian_bench.py` reports its 27-of-85. Folding silence into
either side would be a choice made after the fact.

**The matcher stays literal.** Exact surface form, word boundary, case folded,
both sides the same way. The pairs in `bench/` are inflection-matched
(`pobjedu>pobedu`, `dvije>dve`), so a model writing a different inflection is
lost from *both* columns equally and the loss lands in `silent`, where it can be
seen. Loosening the matcher to stems is a degree of freedom, and one chosen with
a number in hand is one chosen to be passed.

## The audit, run before any model output existed

Point 3 of the section above, executed. `--audit` loads no model. A target
survives only if it has a named counterpart, the two forms differ folded,
neither contains the other, and the professional's own Bosnian reference
actually used the Bosnian form. Result, from `training/form-rate/audit.json`:

| | |
|---|---|
| bench cases | 346 |
| targets (a case may carry more than one) | 385 |
| **discriminating targets kept** | **338**, across 308 cases |
| dropped — no counterpart named | 47 |
| dropped — forms identical folded | 0 |
| dropped — one form nested in the other | 0 |
| dropped — Bosnian form absent from the reference | 0 |
| distinct term pairs behind them | 179 |

338 is not a small set. The forward direction's instrument had 85 targets of
which 58 decided, and its +5.9 points came back at p = 0.10; this one starts
with four times the targets. **The instrument can speak.** That is recorded here,
before it has said anything.

## The asterisk, written before the number and not after it

`bosnian_bench.py`'s honest limit is that 73 of its 85 targets are yat pairs
whose alternative is Serbian, so drift toward *Croatian* had almost nothing to
land on. The same split, measured on this set:

| counterpart | targets |
|---|---|
| yat (ijekavica/ekavica — the alternative is Serbian) | 237 |
| lexical choice | 100 |
| `h` (e.g. *historija*/*istorija*) | 1 |
| of those, a listed **Croatian** form (*obitelj*, *zrakoplov*, *tisuću*, *rujna*…) | **78 / 338** |

23% Croatian-facing against the forward instrument's 14%, so this measure is
better against Croatian drift — and still Serbian-heavy. A form rate that holds
up here is evidence about ekavica first and Croatian second, and the write-up
says so in those words or does not make the claim.

## What this row can and cannot do to the decision

It is a **floor, not a headline.** The table above already fixes chrF2 as the
deciding measure and BLEU as its floor; this row is the third bar and works the
same way — *not below the base's rate*. A fine-tune that raises chrF2 while
writing less Bosnian does not ship, and a fine-tune that raises the form rate
while chrF2 falls does not ship either. Both directions, not either.

The base's own number goes into this file as soon as it is measured, in a note
appended below this line and not by editing anything above it.

### Outcome, 7 September 2026 — the base's own numbers, before the fine-tune exists

Measured on the untouched `opus-mt-tc-base-en-sh` with
`training/bosnian_form_rate.py`, whose rule is the amendment above and was
committed before it had read a single output. Full write-up:
`training/RESULTS-en-bs-formrate.md`. Appended here, not edited into anything
above it.

| | base |
|---|---|
| attempted / decided / silent | 338 / 245 / 93 |
| wrote the Bosnian form | 231 |
| wrote the counterpart | 14 |
| **Bosnian form rate** | **94.3%**  (95% Wilson 90.6–96.6) |

**So the third row of the deciding table is now filled in: not below 94.3%.**

The number changes how this section should be read, and the honest thing is to
say so before a candidate exists rather than after one fails. The row was
written as a bar the fine-tune had to clear; the base clears it at 94.3%, so it
is a bar with **5.7 points of headroom and 94.3 points of downside**. It was
called "a floor, not a headline" above, and that was right for a reason that was
not yet visible. Nothing about the bar moves — a fine-tune that writes less
Bosnian does not ship — but nobody may present this row as the thing the reverse
fine-tune is for. **chrF2 above 58.96 is what it is for.**

### The two controls, measured on the base for the same reason

| control | base |
|---|---|
| `>>bos_Latn<<` survives tokenisation as one piece | **yes**, id 5941 |
| form rate under `>>bos_Latn<<` vs `>>hrv<<` | **94.3% → 72.5%**, a 21.8-point gap |
| outputs identical under the two labels | 38 / 338 (11.2%) |
| targets flipped Bosnian → counterpart by the label alone | 53 |

The base's decoder is listening to its selector, and now there is a number for
how hard. After the fine-tune, this pair is re-run: a gap that has collapsed
toward zero means the adapter has deafened the decoder to the only thing
separating Bosnian output from Croatian, and by the rule above it does not ship
whatever the table says.

The measurement deviates from the section above in one stated way. That section
said "decode the same FLORES source twice". The 338 sources used here are the
bench cases — 265 of them FLORES, 38 NTREX, 35 SETimes — chosen because they are
the **same** sentences the form rate is measured on, which makes the control
paired with it rather than merely adjacent. It is a strictly stronger test on a
superset of source corpora, and it is recorded here rather than left for a
reader to notice.

### One limit found while measuring, which cannot move the bar

Of the 93 silent targets, none hedged (no output contained both forms) and 52
carry the Bosnian word in a different case ending — silence is mostly
inflection, symmetric across both columns because the matcher is literal on both
sides. But in **6** the model wrote an unambiguously Croatian *third* form that
is neither listed alternative (*povijest* for `historiji`/`istoriji`, *tisuću*
for `hiljada`/`tisuća`). For yat pairs the listed counterpart is the **Serbian**
form, so Croatian drift has somewhere to escape to. Six is a **floor** on drift
this two-way matcher cannot see, from a short hand-written stem list that can
only under-count. Reported, reproducible (`--diagnose`), and unable to change
the decision: widening the matcher after a candidate exists is exactly the
degree of freedom this file is written to close.

### The bar re-measured, 7 September 2026 — it reproduces exactly

The two deciding numbers above were published months before this section was
written and have been carried in prose since. Re-measured on the untouched base
with `training/verify_base_flores.py`, which imports every scoring step from
`training/evaluate.py` rather than restating it:

| FLORES-200, 2,009 pairs | committed | re-measured |
|---|---|---|
| BLEU | 29.57 | **29.57** |
| chrF2 | 58.96 | **58.96** |

Identical to the second decimal, so the table above stands unchanged and the
candidate will be weighed against a bar that has been checked rather than
inherited. Also measured, and unlike the forward direction: the en-bs base
leaks its language tag into **0 of 2,009** outputs, where the bs-en base leaks
it in 433 of 1,012. A fine-tune that starts leaking it would be introducing a
defect the base does not have, and that belongs in the write-up if it happens.

### Outcome, 7 September 2026 — whisper-large-v3 at "The gate", and a tension in this file

Both listeners scored in one process, same 200 clips, same code
(`training/SPEECHBENCH-gate.txt`); full write-up in `training/RESULTS-speech.md`.

| threshold | re-measured `listen-previous` | `listen` (large-v3) | |
|---|---|---|---|
| word error, strictly below | 34.9% | **11.9%** | pass |
| term recall, not below | 60.0% | **89.1%** (+29.1, p = 0.0000) | pass |
| Croatian substitution, not above | 5.3% | **6.5%** (+1.2, p = 0.4805) | **fails** |

By "Both, not either" this candidate does not ship. That is written first
because it is what the rule says.

**And this file contradicts itself on that row.** Ten lines below the gate
table, "What the Croatian column may and may not be used for" states its
simulated power — 20% at a planted 5 points, 76% at 20 — and concludes that
"sixty-nine targets need about twenty points before the column speaks at all",
with the calibration that a 5.6-point Croatian move was *one occurrence of one
word*. The move here is +1.2 points, and the instrument names it: `europom` for
`evropom`, one word, on top of the `vjerojatno` error both listeners make.

So the gate table asks a column to fail a candidate on a movement the same
document says the column cannot see. Nothing in this file resolves that, and it
is not being resolved retroactively by whoever reads it with a number in hand.
The tension is recorded, the candidate is unpublished, and the decision to
publish or not — an outward-facing act — belongs to the owner and gets written
down here with its reason.

What must not happen, and is worth naming because the result is tempting: the
Croatian row does not get deleted, reworded, or re-thresholded now. A 23-point
fall in word error and a 29.1-point rise in term recall at p = 0.0000 are not a
licence to edit the bar they were measured against.

#### The decision, same day: it does not ship

The owner was shown both readings and chose the literal gate. large-v3 stays
unpublished; the bundle keeps whisper-small at 34.9% word error while a listener
measured at 11.9% sits unreleased.

The reason is recorded because the cost is real: a bar that bends the first time
a result is spectacular is not a bar. Three OCR fine-tunes were refused this
year on this same discipline, each better on some column, none shipped. Bending
it here would retroactively make those three refusals look like failures of
imagination rather than of evidence.

The contradiction above is therefore left standing rather than repaired in the
candidate's favour, and this is what may be done about it: build an instrument
that can actually resolve the Croatian question — clips chosen for it, held out,
its bars written before large-v3 touches them — and judge **once**. Not this
gate re-run, not this threshold rewritten, and not this candidate scored
repeatedly until a version of the column lets it through.
