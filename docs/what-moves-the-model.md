# What would actually move the model — the brainstorm, ranked

Written 12 September 2026 from the repository's own numbers. No model was
loaded, nothing was trained, nothing was launched. This is the case for where
the next GPU hours should go, and the case against the places they keep going.

**The one sentence:** *Croatian and Serbian are not the enemy — they are the
95% Bosnian is built from; feed it that family, then bias the thin Bosnian
layer on top, and never waste a step on Bulgarian.*

**The correction that reframes the whole plan (12 Sep, owner):** the earlier
version of this file, and much of the project before it, treated Croatian and
Serbian as *contamination* to filter out. That is wrong. Bosnian, Croatian and
Serbian are one pluricentric language (BCS) — mutually intelligible, ~95%
shared. The abundant HR/SR data is not a threat to be scrubbed; it is the
resource. **Bulgarian** (`bg_BG-dimitar`, eastern South Slavic, lost its case
system, different vowels) is a genuinely different language and has no place in
this project. The base model already knows this: it is `eng-hbs` +
Slovenian — the BCS-plus-Slovenian cluster — and Bulgarian was never in it.

## The pattern in the record

Twelve training runs across four abilities. Almost every one came back null,
and the nulls have two causes between them:

| the run | why it could not win |
|---|---|
| translation fine-tune (bs→en) | 313,083 of `train.tsv`'s 313,612 rows are **inside the base model's own training data** (`RESULTS-data-audit.md`). It was re-weighting, not teaching. chrF2 moved −0.16, p = 0.128 |
| OCR passes 8–19 | labels written by the reader being trained (`OCR-ROADMAP.md`). A model fine-tuned on its own confident output learns what it already knows |
| PaddleOCR fine-tunes 7 / 7a / 7b | the recogniser *did* learn (diacritics 62.1% → 77.1%) and still lost the decider on test-v2 — a set of 132 photographs where the rubric calls the score **void below 200** |
| speak-bs, speak-parla, speak-control | warm-started from a **Lower Sorbian** checkpoint (West Slavic) filed as Serbian; all three arms lost, the control included, on the checkpoint's own studio data |
| speak-youtube | 0 of 20 source files carry speech energy to 11,025 Hz — a 22.05 kHz voice must *generate* that band |

Two causes, stated plainly:

1. **The data was already in the model,** or came from the model.
2. **The ruler could not see the thing being trained.** FLORES's Bosnian side
   is **49% Bosnian** (95% 36–61) against our training data's **77%** (76–77),
   intervals not overlapping (`RESULTS-bosnian-audit.md`). Every published
   translation number is scored there. A fine-tune that moves toward the
   standard of its own data is *marked down* by a reference that writes
   *tisuća*, *tjedan* and *povijest* in a third of its decidable sentences.

So the question "what training helps" has a precondition: a run only helps if
its data is new to the model **and** its instrument can see the gain.

## The family reframe — Bosnian is Croatian-spelling + Serbian-lexicon + turkisms

This is the single most useful thing in the record and it was hiding in a
footnote. The base already writes fluent BCS. The only question the whole
project turns on is: **on the handful of axes where the three standards split,
which one does it pick?** The form-rate instrument already answered it. The
base commits to a form on 245 targets and gets the Bosnian one on 231; the 14
it "misses" are not random — sort them by axis and they fall into two clean
piles:

| the base wrote | the axis | where Bosnian's form comes from |
|---|---|---|
| reči, dece, drugde, promenila, posetilaca, negde (6) | **ijekavica → ekavica** | the **Croatian** side |
| istorijska (1) | **kept-h dropped** | the **Croatian** side |
| Točno, travnja, lipnja, tisućama, vlak, Europe, sudjelovati (7) | **lexicon** | the **Serbian** side |

Read it twice, because it is the strategy:

- **Every systematic loss goes to Serbian.** Ijekavica (*vrijeme*, *djece*,
  *negdje*) and the kept *h* (*historija*) are Bosnian's rule-based layer, and
  on every one of them Bosnian and **Croatian are identical** and both oppose
  Serbian *ekavica*. This axis touches the most sentences (`BOSNIAN_METRIC.md`
  calls it "the most systematic"). **Croatian data is an ally here, not
  contamination — it teaches the exact forms the Bosnian ruler rewards.**
- **Every lexical loss goes to Croatian.** Month names (*aprila* not *travnja*),
  *voz* not *vlak*, *hiljada* not *tisuća*, *Evropa* not *Europa* — here
  Bosnian sides with **Serbian** against distinctively Croatian vocabulary.
- **Bulgarian appears in neither pile**, because it is a different language.

So Bosnian is literally **Croatian's phonology and orthography, Serbian's
leaning lexicon, plus its own turkisms and kept-h.** Neither neighbour alone is
"the wrong standard to filter out"; each supplies a different half of what
correct Bosnian is. Three consequences that reorder everything below:

1. **Stop filtering Croatian/Serbian out of the training data.** The 77%-Bosnian
   audit was reassuring, but the lesson is not "purify to 100%." Croatian in the
   mix raises the ijekavica form rate for free. Filtering it *out* would remove
   the best signal for the most systematic axis.
2. **The cheapest win needs no GPU at all: a Bosnian-form lexicon at decode
   time.** The lexical losses are a *finite, known list* — months, *vlak/voz*,
   *tisuća/hiljada*, *Europa/Evropa*. A constrained-decoding bias or a
   deterministic post-edit against a Bosnian-form dictionary fixes most of them
   without touching a weight. Half the form-rate gap is a dictionary, not a
   retrain.
3. **The Bosnian ruler must not punish Croatian-shared forms.** This is a trap.
   If the metric flags anything "Croatian-looking" as wrong, it will mark down
   *vrijeme*, *djece*, *historija* — which are correct Bosnian. The ruler may
   only penalise the axes where Bosnian genuinely diverges: Serbian *ekavica*,
   dropped *h*, and distinctively-Croatian *lexicon*. A ruler that treats HR as
   the enemy rejects real Bosnian.

## The levers, ranked by what can move a held-out number

### 1. Back-translation into English → Bosnian — the strongest candidate

Real monolingual Bosnian (bs.wikipedia, CC BY-SA is the clean default) run
through the shipped bs→en model: **synthetic source, real Bosnian target**.
It is the only lever in this repository where the targets are Bosnian written
by Bosnians and the material post-dates the base's 2021-08-07 data cut.

- It is the one thing that moves chrF2 *and* the form rate, because the form
  rate is a property of the target side and the targets here are human.
- The instrument already exists and already has a pre-run baseline: the base
  writes the Bosnian form on **231 of 245 decided targets, 94.3%** (95% Wilson
  90.6–96.6), 93 silent of 338 (`RESULTS-en-bs-formrate.md`). The shipped LoRA
  reads 99.2%.
- **The headroom is in the silences, not in the 5.7 points.** 93 of 338 targets
  produced neither form. A target the model declines to commit on is a Bosnian
  word it does not have. That is what new Bosnian targets can supply, and it is
  the row to pre-register on.
- Blocked on owner decision 4 (the monolingual source and its licence),
  `V4-PLAN.md`. Nothing else blocks it.

### 2. The Bosnian ruler — not training, but it gates everything above

FLORES cannot score Bosnian-ness; the form rate can, but on one axis and 338
targets. Until a held-out set exists that is actually Bosnian and large enough
to resolve a two-point delta, a real gain from lever 1 is invisible or
penalised. `docs/BOSNIAN_METRIC.md` already specifies it (200 targeted + 50
control sentences); `bench/terms.tsv` and `bench/cases.tsv` exist. This costs
no GPU and decides whether the next run can be read at all.

### 3. Speech — Croatian is the acoustic backbone; the token fix unlocks it

This is where the family point pays off most, and where "not Bulgarian" is
most literal. Whisper's pretraining has **11 hours of Bosnian against 91 of
Croatian**, and the two are acoustically near-identical — the roadmap itself
calls Croatian "acoustically near-identical" to Bosnian. **ParlaSpeech-HR is
1,800 hours of Croatian, CC BY-SA.** That is the single biggest supply of
in-family speech in existence, and Bulgarian audio would be useless for it
(different phonology). So Croatian is not a fallback — it is the acoustic
teacher.

The one thing that stopped it working was a **recipe defect, not the data**:
6,050 Croatian rows were trained under `<|bs|>`, which taught the decoder that
Bosnian is *spelled* *Europom* — the exact row that closed large-v3 (Croatian
substitution 1.1% → 6.1%, p = 0.018). The fix — **one language token per clip**
— lets Croatian audio train the *ears* under `<|hr|>` while never corrupting
Bosnian *spelling*. It is written, pre-registered, and separates the two layers
exactly the way the family reframe says to: shared phonology from Croatian,
Bosnian orthography kept clean. Blocked only on the owner's rule-3 ruling on
which base carries it, and on the ParlaSpeech-HR licence approval (decision 4).

### 4. Speech distillation — the 18.4 hours are good data pointed at the wrong ability

The YouTube corpus fails as voice data on its spectrum, but Whisper resamples
to 16 kHz and its filterbank stops at 8 kHz: those defects are harmless for
*recognition*, and variety across recording conditions is an asset there.
large-v3 → whisper-small is **teacher ≠ student**, so it is distillation, not
the circular move this project already closed on the reading side. Needs its
own pre-registration and a ruling on the transcripts' provenance.

### 5. Photographs — the ruler first, then Cyrillic, then nothing else

- The recogniser lever is spent across three looks; the detector lever is a
  clean negative at the 2 MP working size.
- **test-v2 is 132 photographs and the rubric voids a score below 200.**
  test-v2b's 160 are drawn and unread. Two blind passes make the union 292 and
  makes every reader number real. That is the highest-value photograph work
  and it needs no GPU.
- **Cyrillic is a capability gap, not a percentage point.** 628 Cyrillic key
  words on test-v2 and the reader cannot emit the alphabet. That is worth more
  than any point of Latin recall — and its labels must come from people.
- The 20,240 Mapillary photographs are not training data until their labels
  come from a human or a non-EasyOCR vision model.

### 6. The product loop — the only data that grows from use

`/api/feedback` already stores corrections. Approved corrections exported as
pairs are real Bosnian, written by users, unseen by any base. Volume is tiny
today; it is the only lever that compounds.

## What this says about the voice lane

Rank the candidate voices by **linguistic distance from Bosnian**, not by
what happens to be sitting in Piper:

    Croatian ≈ Bosnian  >  Serbian  >  Slovenian  >>  Bulgarian  >>>  Lower Sorbian (shipped)

The shipped voice is **Lower Sorbian — West Slavic**, the *worst* branch on the
whole list, further from Bosnian than Bulgarian is. Any BCS-family voice beats
it. So:

- **`bg_BG-dimitar` is struck from the plan.** Bulgarian is eastern South
  Slavic with a different vowel system and no case marking; a Bulgarian voice
  reading Bosnian is the same category of error as the Sorbian one, just less
  extreme. Do not fetch it, do not measure it.
- **`sl_SI-artur` (Slovenian, CC BY) is the least-bad *fetched* voice** — same
  sub-branch, and dropping the NC-SA restriction is a real independent gain.
  Measure it as fetched against the 200-clip prefix, ships only if strictly
  below 22.3% at p < 0.05.
- **The right voice, if one is trained, is a single clean Croatian speaker.**
  Croatian is ijekavica, phonologically identical to Bosnian. `speak-parla`
  already had the right *data* (Croatian Sabor) and still failed at 51.9% —
  but it warm-started from the Sorbian checkpoint and mean-pooled five
  speakers. One speaker, warm-started from Slovenian (South Slavic, not West),
  is a different experiment and has not been run. That is the only voice
  training worth a pre-registration.

## The order, if only one thing happens

0. **Build the Bosnian-form lexicon and bias decoding with it** — no GPU, no
   launch, no owner decision. The 14 lexical losses are a known finite list;
   a decode-time dictionary fixes most of the lexical axis immediately, and it
   is the same lexicon the ruler and the back-translation targets will reuse.
   Ship it first because it is free and it moves the exact metric.
1. **Build the Bosnian held-out ruler** — and write its rule so Croatian-shared
   forms (ijekavica, kept-h) count as *correct*, only Serbian-ekavica and
   distinctively-Croatian lexicon count against. No GPU; it makes every run
   below readable.
2. **Owner decides** the monolingual Bosnian source + licence (decision 4) and
   the ParlaSpeech-HR licence — one decision unlocks both the strongest
   translation lever and the strongest speech lever.
3. **Launch back-translation en→bs** with four bars: FLORES chrF2 above the
   best shipped, BLEU floor, **form rate floor, and the silent count as its own
   row** (93 of 338 — that is where the headroom is). Keep Croatian in the mix;
   do not purify.
4. **Launch the speech token-per-clip run on ParlaSpeech-HR** — Croatian ears,
   Bosnian spelling, the two layers kept apart.

Everything else on this page is ready to wait for that. Bulgarian is not on the
page at all, and that is the point.
