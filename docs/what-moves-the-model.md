# What moves the model — the family, and the levers that follow from it

Written 12 September 2026, no GPU spent. Every number below is quoted from a
results file already in this repository; nothing here is a new measurement.

## The fourteen words

`training/RESULTS-en-bs-formrate.md` measured the untouched English → Bosnian
base, `Helsinki-NLP/opus-mt-tc-base-en-sh`, on 338 discriminating targets. It
committed to one of the two forms on **245** and wrote the Bosnian one on
**231** — a form rate of 94.3% (Wilson 90.6–96.6). The instrument's value is
not that headline. It is that the **14** it loses are listed in full, and they
sort cleanly:

| the base wrote instead | axis | the form Bosnian shares with |
|---|---|---|
| reči, dece, negde, drugde, promenila, posetilaca | ijekavica → **ekavica** | **Croatian** |
| istorijska | the retained **/h/** dropped | **Croatian** |
| Točno, travnja, lipnja, tisućama, vlak, Europe, sudjelovati | **lexicon** | **Serbian** |

Seven and seven. **Every systematic loss is a drift toward Serbian; every
lexical loss is a drift toward Croatian.** The file called this "half Croatian,
half Serbian" when it was written; the split is sharper than that, and it has a
shape.

## What the shape says

Bosnian is not a third thing standing apart from two neighbours. On the
evidence of its own error list it is:

> **Croatian's phonology and orthography** — ijekavica, the retained /h/ —
> **plus a lexicon leaning Serbian** — *aprila* not *travnja*, *voz* not
> *vlak*, *hiljada* not *tisuća*, *Evropa* not *Europa* — plus its own
> turkisms.

Neither neighbour is the wrong standard. **Each supplies a different half**, and
the two halves are separable because they live on different axes: one is a sound
change applied across the whole vocabulary, the other is a finite list of words.

That separability is the whole lever. It means the two halves can be sourced
from different places and fixed by different means.

## Three things that follow

**1. Croatian data is the resource, not the contaminant.** Croatian is
ijekavian. On the axis that touches the most sentences — the yat reflex, which
appears in ordinary words like *vrijeme*, *djeca*, *mjesto* — Croatian teaches
exactly the form Bosnian wants, for free and at volume. `RESULTS-speech.md`
already measured this from the other side: the Bosnian-only listener read 35.5%
and adding 3,430 Croatian clips took it to **33.9%**. Filtering the corpus down
to "pure Bosnian" would throw away the best signal available on the axis that
matters most. `training/RESULTS-bosnian-audit.md` found the corpus is already
77% Bosnian by lexical marker; there is nothing to purify.

**2. Half the remaining gap is a dictionary, not a GPU.** The lexical losses are
a closed, known, enumerable list — months, *voz*/*vlak*, *hiljada*/*tisuća*,
*Evropa*/*Europa*, *učestvovati*/*sudjelovati*. A decoder biased against the
Croatian member of each pair fixes them with **zero training**. Seven of the
fourteen losses are of this kind.

**3. The ruler must not punish Croatian-shared forms.** A Bosnian-ness metric
that penalises everything Croatian would reject *vrijeme*, *djece* and
*historija* — all correct Bosnian. It must penalise three things and no others:
Serbian ekavica, the dropped /h/, and **distinctively** Croatian lexicon.
`training/audit_bosnian.py` is built this way already; `docs/BOSNIAN_METRIC.md`
is where the argument for a real instrument lives.

## Three corrections to the plan as it was put to me

These are recorded because the plan as stated would have collided with decisions
already in the repository.

**ParlaSpeech-HR cannot be the acoustic backbone.**
`training/PREREGISTRATION.md:749` records the owner's decision in words: *"**ParlaSpeech-HR is not used.** It is CC BY-SA, share-alike propagates to
distributed derivatives, and this project publishes weights."* The substitute is
named in the same line — **voxpopuli_hr, CC0**, ~11,000 clips, and it "covers
the requirement on its own." The v6 *voice* run did use ParlaSpeech-HR, and that
run was refused at its gate and published nothing, which is why the licence
never bit. A listener trained on it would ship.

**The per-clip language token is not a new idea — it is written, coded, and
waiting.** `training/PREREGISTRATION.md` ("v4 — listen — one language token per
clip") already carries the diagnosis, and it is sharper than anything I would
have added: the half-1/half-2 mix was 6,050 Croatian rows against 6,182 Bosnian,
so **half the training labels put `<|bs|>` in front of Croatian text**. Whisper's
decoder writes the spelling its language token names, and its pretraining gave
that token little to hold on to — **11 hours of Bosnian against 91 hours of
Croatian** (Whisper paper, appendix E). The model was taught by six thousand
examples that Bosnian is spelled *Europom* and *vjerojatno*. That is the exact
row that closed whisper-large-v3: Croatian substitution **1.1% → 6.1%** on 925
clips, p = 0.018, the same two words each time.

The pre-registration's own summary: *"the leak is a recipe defect, not a data
limit, and it costs one column in the mix file."* Both halves are implemented —
`data/scripts/build_speech_mix.py` writes the column and refuses an unregistered
source; `training/train_speech.py` reads it and refuses a code Whisper lacks.
**Nothing is missing but the run.**

**Bulgarian is out, but not for the reason given.** The argument offered was
that the translation base is scoped to the family and never included Bulgarian.
That does not hold as stated: the reply base `opus-mt-tc-base-en-sh` is
Serbo-Croatian and excludes **Slovenian**, while the forward base
`opus-mt-tc-big-zls-en` is *zls*, all South Slavic, and **does** include
Bulgarian. More to the point, the Bulgarian in question was
`bg_BG-dimitar` — a **Piper speech-synthesis checkpoint**, a different system
from OPUS-MT whose language coverage says nothing about it. The conclusion
survives on its own linguistic grounds, which is how
`docs/speak-checkpoint-comparison.md` already put it: Bulgarian is eastern South
Slavic, has lost its case system and has a different vowel inventory, so it is
the weakest of the three South Slavic Piper voices as a warm start. Out — on the
right reason.

## The levers, ranked by what they can move

| | lever | cost | what it moves |
|---|---|---|---|
| 1 | **Re-run the listener with the language column** | one training run, code ready | the row that closed large-v3. Diagnosed mechanism, pre-registered, implemented |
| 2 | **Bias decoding against the Croatian lexical pairs** | none | 7 of the 14 form-rate losses |
| 3 | Hear `sl_SI-artur` as fetched, no training | ~30 min | a possibly better Bosnian voice, and the non-commercial licence off the app |
| 4 | Score the fine-tune on FLORES's Bosnian-marked vs Croatian-marked halves | none | tests whether the −0.79 chrF2 is the model becoming *more* Bosnian |
| — | ~~YouTube audio → a voice~~ | — | **closed**: 0 of 20 lectures reach 11 kHz (`docs/speak-checkpoint-comparison.md`) |
| — | ~~Filter Croatian out of the mix~~ | — | **rejected**: it is the ijekavian signal, and it was worth 1.6 points |
| — | ~~ParlaSpeech-HR as backbone~~ | — | **blocked**: CC BY-SA, and this project publishes weights |

## The one sentence

**Croatian and Serbian are not the enemy — they are the two halves Bosnian is
built from; feed the family, keep the thin Bosnian layer on top, and label every
clip with the language it is actually in.**
