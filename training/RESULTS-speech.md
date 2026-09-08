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

---

# whisper-large-v3 at the gate — 7 September 2026

The larger listener was trained on Kaggle and installed at `models/lilly/listen`
on 1 September. **Its gate had never been run.** The 68.2% in the section above
belongs to the whisper-small candidate, and `RESULTS-speech-half2.md:53` still
says "running locally" for the term-recall row. So the model on the Mac was
better than the one that ships and nobody had established it by the rule.

Run now, both listeners scored in one process on the same 200 clips through the
same code, `training/SPEECHBENCH-gate.txt`:

| | `listen-previous` (whisper-small, 237 MB) | `listen` (**whisper-large-v3**, 1.5 GB) |
|---|---|---|
| word error | 34.9%  (1,360 wrong of 3,901) | **11.9%**  (465 wrong) |
| Bosnian term recall | 60.0%  [50.7, 68.7] | **89.1%**  [81.9, 93.6] |
| variety substitution | 2.9% | 4.9% |
| — Croatian | 5.3% | 6.5% |
| — Serbian | 2.0% | 4.1% |

Paired bootstrap over sentences:

    term recall           +29.1 points   p = 0.0000
    variety substitution   +1.9 points   p = 0.1485   (does not clear 0.05)
    CROATIAN substitution  +1.2 points   p = 0.4805   (does not clear 0.05)
    SERBIAN  substitution  +2.1 points   p = 0.1765   (does not clear 0.05)

## Against the pre-registered gate, row by row

| threshold (`PREREGISTRATION.md`, "The gate") | measured | |
|---|---|---|
| word error **strictly below** the re-measured `listen-previous` | 11.9% against 34.9% | **pass** |
| term recall **not below** the re-measured baseline | 89.1% against 60.0% | **pass** |
| Croatian substitution **not above** the re-measured baseline | 6.5% against 5.3% | **fails, literally** |

Two of three pass by margins this project has never seen. Word error falls to a
third — the hard gate, with 3,901 reference words behind it. Term recall rises
29.1 points at p = 0.0000, where the whisper-small fine-tune the section above
celebrates managed +2.4 at p = 0.1475.

**And the third row is above its baseline, so by "Both, not either" it does not
ship.** That is what the rule says and it is written down first, before any
argument about it.

## What the third row actually contains

The instrument prints the targets it lost, and they are countable:

    written in Croatian by listen-previous: vjerovatno (1)
    written in Croatian by listen:          vjerovatno (1), evropom (1)

**The entire +1.2-point Croatian difference is one occurrence of one word** —
`europom` written where the speaker said `evropom`. Both listeners make the
`vjerojatno`/`vjerovatno` error; the larger one additionally makes `europom`.

Total substitutions of any variety: `listen-previous` 2 words, `listen` 5.

This is exactly the size the pre-registration described in advance, in a section
headed *"What the Croatian column may and may not be used for"*, written before
any listener was scored:

> the whole +5.6-point Croatian move between the untrained model and the
> installed one is ONE occurrence of ONE word — `vjerojatno` for `vjerovatno`,
> 1 of 18 decided targets. That is what a movement of this size looks like from
> the inside.

and, from the simulated power table in the same section:

| planted difference | power, Croatian column |
|---|---|
| 5 points | 20% |
| 10 points | 42% |
| 20 points | 76% |

> **the Croatian column is a coarse alarm with its power stated, and a null
> result is recorded as "the instrument could not see", NEVER as "no drift".**
> Sixty-nine targets need about twenty points before the column speaks at all.

So the pre-registration contains a row that fails this candidate and a section,
written the same day, saying that row cannot resolve a movement of this size.
**The tension is in the document, not in the reading of it**, and it is recorded
here rather than settled quietly in whichever direction flatters the result.

## One confound worth stating, which does not rescue anything

A listener that mishears a word cannot drift on it. `listen-previous` resolved
**42 of 110** targets to neither variety; `listen` resolved **7 of 110**. Part
of the smaller model's lower substitution rate is that it did not transcribe the
word at all. The paired comparison above controls for this — it scores the 68
targets *both* models decided — so the +1.2 points is not an artefact of that.
It is recorded because the raw single-model columns would overstate the smaller
model's cleanliness, and someone will quote them.

The marker rate has the same shape and the same small counts: Croatian 1 word of
3,916 for `listen-previous`, 4 of 3,879 for `listen`; Serbian 2 against 4.

## Decision — 7 September 2026: it does not ship

**Not published, by the rule as written.** The owner was shown both readings —
the literal gate ("Both, not either", and the Croatian row is above its
baseline) and the same document's power analysis (that column needs about
twenty points to speak; this is 1.2, and it is one word) — and chose the
literal one.

The reason is not that the drift is believed to be real. It is that the
project's credibility rests on the bar meaning what it said before the number
existed. Three OCR fine-tunes were refused this year on exactly this discipline
(`RESULTS-ocr-paddle-finetune.md`, steps 7, 7a, 7b) — each of them looked better
on some column and none of them shipped. A gate that bends the first time the
result is spectacular is not a gate, and every later refusal would have to be
read as "we could not find an argument this time".

So the released bundle keeps **whisper-small at 34.9% word error**, and a
listener measured at **11.9%** stays unpublished at `models/lilly/listen`, with
its full result recorded above. That cost is real and it is the price of the
rule; it is written here in those words so nobody has to reconstruct it later.

**What would change this, and what would not.** Not a re-run, not a re-reading,
not a threshold edited now. The Croatian column failed to resolve 1.2 points
because it has 32 Croatian-capable targets in this test set, and the
pre-registration says plainly that this is a limit of the *test set* — "about 70
of the 349 FLEURS Bosnian sentences contain a word that has a Croatian
counterpart at all, and nothing in this project's control raises that". An
instrument that could actually decide this question would need clips built for
it, held out, and pre-registered before large-v3 is scored on them again. That
is a new measurement with its own written bars, judged **once** — not this one
re-run until it passes.

---

# The full instrument — 8 September 2026: whisper-large-v3 is closed

The last look, pre-registered in `training/PREREGISTRATION.md` ("v3 — speech —
the full instrument") and run on a Kaggle T4 as `lilly-speech-instrument`
version 4: all 925 FLEURS bs_ba test clips, both listeners in one process,
both checked against the gate's weight fingerprints before scoring. Raw files:
`training/speech-instrument/`.

| threshold | `listen-previous` (whisper-small) | `listen` (whisper-large-v3) | |
|---|---|---|---|
| word error, strictly below | 52.4%  (9,865 of 18,836) | 33.0%  (6,209) | pass |
| term recall, not below | 49.6%  [45.5, 53.6] | 72.1%  [68.3, 75.6]  (+22.5, p = 0.0000) | pass |
| Croatian substitution, not above | 1.1%  (1 of 87 decided, 177 targets) | **6.1%  (8 of 131)**  (+5.0, p = 0.018) | **fails** |

`listen` wrote *evropom* five times and *vjerovatno* three where the speaker
said *evropom* / *vjerovatno*; `listen-previous` wrote *vjerovatno* once.
Serbian substitution moved the other way (4.2% → 1.9%, p = 0.081) and is not
a bar.

**Rubric WER**, the project's own scale (`RUBRIC.md`: greedy, temperature 0,
`BasicTextNormalizer`, 925 utterances), computed here for the first time:
whisper-small **39.5%**, whisper-large-v3 **14.1%** (18,468 words). Band 3
against band 8.

**Verdict, by "Both, not either": does not ship. By rule 3 of the
pre-registration, whisper-large-v3 is closed** — no other split, normaliser or
instrument. The bundle keeps whisper-small.

One defect in the run, reported in the outcome note and not repeated here at
length: the app-decode rows reused 200 of the Mac's cached transcripts under
clip names that mean different audio on Kaggle (the two downloads number the
clips one apart), so the two *passing* rows are computed with 200 of 925 clips
scored against the wrong sentence for both listeners; on the 725 clips the box
transcribed itself the word error is 35.7% against 11.4%. The greedy rows are
clean, and the deciding Croatian row is decided on real transcripts and agrees
with the greedy decode (3.0% → 6.2%, p = 0.023). The cause — a cache keyed on
file name — is fixed in `training/speech_bench.py`; the Kaggle cache was not
installed over this Mac's.

The published bundle carried this listener, ungated, from 4 to 8 September;
`listen/` went back to whisper-small (fingerprint a76342f6ab59b382) with the
8 September publish, 11:31 UTC, and `scripts/publish_to_hf.py` now refuses any
listener but the gated one.
