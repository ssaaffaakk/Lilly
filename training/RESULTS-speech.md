# Speech retraining — the result

Run: Kaggle `afaksrmeli/lilly-speech` version 12, Tesla T4, 28 August 2026.
Code cloned from GitHub at `ea231ff7`, which the launcher verified by SHA
before pushing the notebook.

## The number the run was built to produce

| | word error, 200 held-out FLEURS Bosnian clips |
|---|---|
| untouched `whisper-small` | **38.4%**  (1,497 wrong of 3,901 words) |
| after training on the mix | **33.9%**  (1,322 wrong of 3,901 words) |

Both measured in the same process, on the same clips, by the same scorer —
before and after — so the 4.5-point gap is the run's own comparison and not a
figure carried in from anywhere else. The previous listener read 35.5% on these
same 200 clips against a 38.5% baseline, so adding neighbour-language audio is
worth a further **1.6 points** over the Bosnian-only fine-tune.

## What it trained on

| | clips |
|---|---|
| Bosnian (FLEURS bs_ba, our own train split) | 3,091 |
| Croatian (FLEURS hr, `--hours 12`) | 3,430 |
| mixed | 6,521 rows, **47% Bosnian** |

`build_speech_mix.py --share 0.35` reports "asked for 35%". That is not a miss.
`--share` is a floor the mixer reaches by repeating Bosnian clips, and with only
3,430 Croatian clips the 3,091 Bosnian were already 47% of the mix — above the
floor, so nothing needed repeating. Reaching a genuine 35% would have needed
5,740 Croatian clips, 2,310 more than `--hours 12` returned. The run therefore
tests a 47% mix, not a 35% one, and the pre-registered plan to compare two
settings has had one of them run.

Also checked by the mixer, and worth recording because it is the failure that
would invalidate everything above: **0 extra clips were dropped for appearing in
valid or test**, and the first 400 clip paths all resolved.

## The gate this does NOT clear yet

`training/PREREGISTRATION.md` sets two thresholds and says "Both, not either":

| | now | threshold | measured |
|---|---|---|---|
| word error, 200 Bosnian clips | 35.5% | below 35.5% | **33.9% — passes** |
| Bosnian-specific term recall | baseline | not below baseline | **not yet run** |

The second gate exists precisely for this situation. The listener was just fed
3,430 Croatian clips, and the predictable failure is a word error rate that
falls while the model starts writing *tjedan* for *sedmica* and *mesto* for
*mjesto* — half a point of WER each, invisible in the average, and the whole
product. `training/speech_bench.py` is the instrument for that and the notebook
does not run it.

So: the trained listener has cleared one of its two bars. It is not installed
and should not be until SpeechBench has been run on it and on the listener it
would replace.

### The baseline the new listener has to beat — measured

SpeechBench, run locally on the same 200 clips, on the two listeners that exist
here today:

| | WER | term recall | variety substitution |
|---|---|---|---|
| `listen.before-training` | 38.5% | 60.0% | 12.1% |
| `listen` (the current one) | 35.5% | **65.9%** | **5.1%** |

85 targets, 58 of them decided one way or the other. The WER column reproduces
`evaluate_speech.py` exactly — 38.5% and 35.5% — which is worth saying because
it means the two instruments agree about the models they share.

What the current fine-tune bought, read honestly: +5.9 points of term recall at
**p = 0.1045**, which does not clear 0.05 and is a tie; and −7.0 points of
variety substitution at **p = 0.0050**, which does. The defensible claim is that
the current listener drifts toward Croatian/Serbian less often, not that it
knows more Bosnian terms.

**So the bar for the new listener is 65.9% term recall and 5.1% substitution.**
Below either and, by the pre-registered rule, it does not ship however good its
word error rate looks.

### Both gates run. It ships.

Same 200 clips, both listeners scored locally through the same code so the
comparison is like for like:

| | `listen` (was) | `listen-candidate` (now) |
|---|---|---|
| word error | 35.5% | **34.9%** |
| Bosnian term recall | 65.9% | **68.2%** |
| variety substitution | 5.1% | **3.3%** |

| pre-registered gate | threshold | measured | |
|---|---|---|---|
| word error, 200 Bosnian clips | below 35.5% | 34.9% | pass |
| Bosnian term recall | not below 65.9% | 68.2% | pass |

"Both, not either" — both pass, so it is installed, with the listener it
replaces kept at `models/lilly/listen-previous/`.

**What must not be overclaimed.** Neither term difference is significant:
recall +2.4 points at **p = 0.1475**, substitution −1.8 points at **p = 0.0925**.
Both are ties. The defensible claim is *not* that the Croatian audio made the
model more Bosnian — it is that **it did not make it less Bosnian**, which is
precisely the failure this gate was built to catch, and it did not happen. The
substitution rate moved the favourable way after 3,430 Croatian clips, which is
worth noting and not worth calling a result at p = 0.09.

And the honest asterisk on the gate itself: 73 of its 85 targets are yat pairs
whose alternative is Serbian. Drift toward *Croatian* has little here to land
on, so this instrument is weaker against exactly the drift this run risked than
its numbers make it look.

### One discrepancy, recorded rather than reconciled away

The same candidate reads **33.9%** on Kaggle and **34.9%** here — a full point
apart on identical clips. Kaggle ran a T4 with whatever faster-whisper and
ctranslate2 `pip install` gave it that hour; this machine runs int8 on CPU with
the pinned versions. Neither number is wrong and neither is the other's
correction.

It matters for how the headline is quoted. The gate above compares 34.9% against
35.5% — **both measured here, on the same hardware, through the same code** — so
the 0.6-point improvement over the previous fine-tune is the like-for-like one.
The 4.5-point figure at the top of this file is the candidate against *untrained*
whisper-small, both measured on Kaggle. They are different comparisons and
mixing them would overstate the gain by a factor of seven.

Two limits of this instrument, from its own output and not to be forgotten when
reading the next number: 27 of 85 targets (31.8%) resolve to neither variety,
and 16 of those have a word within two characters of the spoken form — heard,
misspelled, scored as neither rather than as drift. And 73 of 85 targets are yat
pairs whose alternative is Serbian, so **drift toward Croatian has very little
here to land on.** The run being judged added 3,430 clips of *Croatian*.

## Worst clips, for the record

    00073.wav — 80%
      said:  Njegov renome epicentra luksuza započet je oko 400. godine nove ere...
      heard: Njegu vrenom je epicentralukcu za započit joko 400-te godini novi erija...
    00132.wav — 77%
      said:  Nivo pH je prikazan količinom vodonikovih (H u pH) jona u testiranoj hemikaliji.
      heard: Ivo pH je prikazana u količinom vodnikovih HOPH i ona u tesiranoj hemikali.

Long sentences with numerals and foreign proper nouns, which is where the
remaining third of the errors live.
