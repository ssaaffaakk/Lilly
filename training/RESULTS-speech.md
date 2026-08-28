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

## Worst clips, for the record

    00073.wav — 80%
      said:  Njegov renome epicentra luksuza započet je oko 400. godine nove ere...
      heard: Njegu vrenom je epicentralukcu za započit joko 400-te godini novi erija...
    00132.wav — 77%
      said:  Nivo pH je prikazan količinom vodonikovih (H u pH) jona u testiranoj hemikaliji.
      heard: Ivo pH je prikazana u količinom vodnikovih HOPH i ona u tesiranoj hemikali.

Long sentences with numerals and foreign proper nouns, which is where the
remaining third of the errors live.
