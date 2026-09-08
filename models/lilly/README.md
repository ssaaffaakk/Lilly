---
license: other
license_name: mixed-see-components
license_link: https://huggingface.co/Safak11/lilly/blob/main/NOTICE.md
language:
  - bs
  - en
pipeline_tag: translation
tags:
  - bosnian
  - translation
  - speech-recognition
  - text-to-speech
  - ocr
base_model:
  - Helsinki-NLP/opus-mt-tc-big-zls-en
  - Helsinki-NLP/opus-mt-tc-base-en-sh
  - openai/whisper-large-v3
  - hexgrad/Kokoro-82M
  - PaddlePaddle/PP-OCRv6_medium_det
  - PaddlePaddle/PP-OCRv6_medium_rec
---

> This card is the project's README, mirrored here so the numbers live in one place (`scripts/sync_model_card.py`). The folder layout and the licenses that travel with these weights are at the end.

<h1 align="center">Lilly</h1>

<p align="center"><strong>A Bosnian translator you can type into, talk to, or point a camera at — running entirely on your own machine.</strong></p>

<p align="center">
  <a href="#run-it"><img src="https://img.shields.io/badge/python-3.12+-3776AB.svg" alt="Python 3.12+"></a>
  <a href="#whats-inside"><img src="https://img.shields.io/badge/FastAPI-0.115+-009688.svg" alt="FastAPI"></a>
  <a href="#how-well-it-works"><img src="https://img.shields.io/badge/offline-no%20network%20at%20inference-2ea043.svg" alt="Offline"></a>
  <a href="https://huggingface.co/Safak11/lilly"><img src="https://img.shields.io/badge/weights-Safak11%2Flilly-FFD21E.svg" alt="Hugging Face weights"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ssaaffaakk/Lilly/main/docs/images/demo-translate.jpg" alt="Lilly translating a Bosnian sentence into English" width="760">
</p>

<p align="center">
  <a href="#why-i-built-it">Why</a> ·
  <a href="#see-it-work">Examples</a> ·
  <a href="#run-it">Run it</a> ·
  <a href="#whats-inside">Inside</a> ·
  <a href="#the-journey-so-far">Journey</a> ·
  <a href="#how-well-it-works">Numbers</a> ·
  <a href="#what-it-cannot-do-yet">Limits</a> ·
  <a href="#training">Training</a>
</p>

---

## Why I built it

I applied to a lot of internships and kept getting turned down for the same
reason: I did not know Bosnian.

I could not learn a language in a week, so I built the thing I needed instead.
Lilly is a translator I made for myself. In class and at work I press the
microphone, say what I just heard, and get it back in English. I point the
camera at a whiteboard or a sign and read it. I type what I want to say in
English and Lilly gives me the Bosnian to say back.

I tried the tools that already existed. Google Translate and the rest kept
getting my sentences wrong — close enough to look right, wrong enough to leave
me lost in the room. They also treat Bosnian as one more entry in a South Slavic
bucket, and that is not the same thing as understanding it.

So I trained the models myself. The first versions were bad — under 30% of the
words on a sign, more than half the words of a sentence heard wrong — and I
kept the numbers as they climbed, measured every change against the untuned
model underneath it, and published all of them, including the ones that say a
change did nothing. Lilly is how I show that I can work in Bosnian and that
I can build something serious when I hit a wall.

The longer version is in [`docs/STORY.md`](https://github.com/ssaaffaakk/Lilly/blob/main/docs/STORY.md).

---

## See it work

![Type, speak, or snap a photo](https://raw.githubusercontent.com/ssaaffaakk/Lilly/main/docs/images/lilly-modes.jpg)

Real output from the running app, not hand-picked from a benchmark.

**Bosnian → English**

| You type | Lilly returns |
| --- | --- |
| Molim vas, možete li ponoviti? Nisam razumio zadaću. | Please, can you repeat that? I didn't understand the assignment. |
| Sastanak je sutra u devet u kancelariji na drugom spratu. | The meeting is tomorrow at nine in the office on the second floor. |
| Rok za predaju projekta je sljedeći petak. | The deadline for handing over the project is next Friday. |

**English → Bosnian**, for saying something back

| You type | Lilly returns |
| --- | --- |
| Could you send me the file before the meeting? | Možete li mi poslati dosje prije sastanka? |
| I am still learning Bosnian, please be patient with me. | Još učim bosanski, molim vas budite strpljivi sa mnom. |

Speech and photographs go through the same translator, so a spoken sentence and
a photographed sign come back the same way a typed one does.

---

## Run it

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/fetch_models.py        # the bundle, then the reader's own weights
.venv/bin/uvicorn app.server:app --port 8000
```

Open http://localhost:8000. Allow the microphone, or photograph something with a
**đ** in it.

That gives you all five abilities. The translator arrives already fine-tuned
and quantised, so there is nothing to build for the forward direction, and the
bundle has carried the reply direction (English in, Bosnian out, the swap
button in the UI) since 8 September 2026, so `fetch_models.py` pulls it as
`translator-en-bs/`. The fetch also pulls PP-OCRv6 into PaddleX's cache
through the app's own reader, so the first photograph does not wait on a
download, and it checks the listener it was handed against the gated one,
saying so out loud if an older fetch left the closed whisper-large-v3 behind.
To rebuild the reply direction yourself from the upstream base instead:

```bash
.venv/bin/python scripts/fetch_translate_base.py --direction en-bs
.venv/bin/python scripts/build_translator.py --direction en-bs
```

Without `translator-en-bs/` the app still runs and `/api/reply` answers 503.
`python3 app/lilly.py` prints which parts are installed.

Startup is instant because each model loads on first use. Once
`fetch_models.py` has finished, nothing reaches the network again.
`python3 -m pytest tests` checks the parts that need no model: the sentence
splitter, the batch grouping, the correction store and the server's answers
to bad input.

---

## What's inside

![Lilly offline architecture](https://raw.githubusercontent.com/ssaaffaakk/Lilly/main/docs/images/architecture.png)

Every ability sits behind one object and one API.

```python
from app.lilly import lilly
lilly.translate("Dobar dan")          # Bosnian text   -> English text
lilly.reply("Good morning")           # English text   -> Bosnian text
lilly.listen("clip.m4a")              # spoken Bosnian -> Bosnian text
lilly.read("sign.jpg")                # photo          -> Bosnian text
lilly.speak("Good day", "out.wav")    # English text   -> spoken English
```

| Endpoint | Body | Returns |
| --- | --- | --- |
| `POST /api/translate` | `{"text": "..."}` | Bosnian in, English out |
| `POST /api/reply` | `{"text": "..."}` | English in, Bosnian out |
| `POST /api/speech` | audio upload | transcribes Bosnian, then translates it |
| `POST /api/photo` | image upload | reads Bosnian off the image, then translates it |
| `POST /api/speak` | `{"text": "..."}` | English speech as WAV |
| `POST /api/feedback` | a correction | stored for review and retraining |
| `GET /health` | — | liveness |

Every request is bounded before it reaches a model — uploads by size, text by
how much work it asks for, images by pixel count — because the server is written
to face the open internet.

### The correction button

When a translation is wrong, you press **This translation is wrong**, say what
it should have been, and the correction is stored for review. Verified
corrections go back into the training pool, so the model improves on the
sentences people actually hit rather than the ones a benchmark happens to
contain.

### Weights it is built from

Lilly is a bundle, not a new architecture. The translator and the listener are
fine-tuned here. The reader and the voice are off the shelf: the reader is
PP-OCRv6, chosen over the fine-tuned EasyOCR reader by a rule written before the
comparison ran, and four later attempts to fine-tune it never beat it. Full
attribution and licenses are in [`models/lilly/NOTICE.md`](https://github.com/ssaaffaakk/Lilly/blob/main/models/lilly/NOTICE.md).

| Ability | Built from | Fine-tuned here |
| --- | --- | --- |
| Translate | OPUS-MT [`opus-mt-tc-big-zls-en`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-zls-en) (Helsinki-NLP), CTranslate2 int8 | yes — LoRA merged into the weights |
| Reply | OPUS-MT [`opus-mt-tc-base-en-sh`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-base-en-sh) (Helsinki-NLP), CTranslate2 int8 | yes — LoRA merged into the weights; cleared its gate and published 8 Sep 2026 |
| Listen | [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) (OpenAI), converted to CTranslate2 int8 here | yes — LoRA. **Shipped by the owner's decision, refused at its gate** (see [Speech](#speech)); the gated [`faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small) fine-tune stays the baseline |
| Read | [PaddleOCR PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) (`PP-OCRv6_medium_det` + `_medium_rec`, PaddlePaddle), fetched at run time; [EasyOCR](https://github.com/JaidedAI/EasyOCR) + CRAFT stays as the `LILLY_READER=easyocr` way back | no — stock, chosen by a pre-registered rule; the EasyOCR fallback is fine-tuned |
| Speak | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (hexgrad) | no — stock weights |

---

## How well it works

### The journey so far

The first builds were worse than anything below. No measurement of them was
kept in the repository — the habit of committing a number before changing
anything came later, and is now the rule — so the first column comes from the
owner's own notes from that time, written as a bound. The next column is the
first number that was recorded, with the file it lives in. The last is today.

| | the first builds (unrecorded) | first recorded | today |
| --- | --- | --- | --- |
| Photographs — words found per photograph, the 40 | **< 30%** | 36.0% (`training/RESULTS-ocr.md`) | **67.0%** |
| Photographs — words found, pooled | **< 10%** | 16.9% (63 of 373) | **69.4%** |
| Photographs — words invented that are on no sign | **> 280** | 224 | **65** |
| Speech — word error, 200 held-out clips | **> 55%** | 38.5% (`training/RESULTS-speech.md`) | **11.9%** (whisper-large-v3, shipped by decision, refused at its gate; the gated whisper-small reads 34.9%) |
| Translation — BLEU on FLORES devtest, as the user sees it | **< 30** | 37.72, with the language tag leaked into 308 of 1,012 outputs (`training/RESULTS-devtest.md`) | **43.25**, leaked into **0** (re-measured 8 Sep on a T4 after the ordinal splitter fix) |
| Reply, English → Bosnian — chrF2 on FLORES-200 | — | 58.96, the base as downloaded (`training/RESULTS-en-bs.md`) | **60.00**, fine-tuned and published 8 Sep |

One recorded moment says what the early period was like: the reader scored
about 75% on synthetic text and **36% the first time it was pointed at real
photographs** (`training/RESULTS-ocr-dataset.md`). The 75% was never a real
number. Everything after that was measured on real photographs, real audio and
held-out sentences.

### Where the credit goes

The fairest thing to measure a change against is the untuned model it is built
on. Every "Lilly today" figure below is measured on data the model never
trained on, through the app's own code path, against exactly that base. The
pass marks were written down before each run (see
[thresholds](#thresholds-are-written-before-the-run)).

| Ability | Measured on | Untuned base | Lilly today | Where it stands |
| --- | --- | --- | --- | --- |
| **Translate**, Bosnian → English | 1,012 FLORES devtest sentences, as the user sees them | 42.08 BLEU / 67.85 chrF2 with its tag stripped; the base emits its language tag into 576 of 2,009 outputs, which the app strips since 8 Sep | **43.25 BLEU / 68.10 chrF2**, **0** leaks | shipped; re-measured 8 Sep on a T4 after the ordinal splitter fix |
| **Reply**, English → Bosnian | 2,009 FLORES-200 pairs | 29.57 BLEU / 58.96 chrF2 | **30.73 / 60.00**; writes the Bosnian form of a contested word **99.2%** of the time (base 94.3%) | cleared all four bars 8 Sep; **in the bundle since 8 Sep** |
| **Listen**, whisper-small | 200 held-out FLEURS clips | 38.5% word error | **34.9%** word error; Bosnian term recall 65.9% → 68.2% | the listener that cleared its gate; kept as the baseline |
| **Listen**, whisper-large-v3 | the same 200, then all 925 | — | 11.9% word error on the 200; **14.1%** on 925 against small's 39.5% | **shipped since 8 Sep by the owner's decision, refused at its gate**: writes Croatian forms more often (1.1% → 6.1%); closed to further looks, see below |
| **Read** | 40 Commons photographs; `test-v2`, 132 with text | first reader: 36.0% of sign words found, 224 invented | **67.0% found, 65 invented** on the 40; **57.8% found, 450 invented** on `test-v2` | shipped (PP-OCRv6, untrained) |

### How to read the numbers

- **Word error rate** (speech): the share of words heard wrong. Lower is better.
- **BLEU and chrF2** (translation): how much the output overlaps a professional
  translation. Higher is better. chrF2 counts characters, so it is the fairer
  measure for a heavily inflected language, and it is the one that decides here.
  A difference of a point or so is noise unless a paired bootstrap says
  otherwise; every gain quoted above has one.
- **Words found and invented** (photographs): the share of the words on the
  signs that the reader read correctly, and how many words it produced that are
  on no sign at all. The second number matters as much as the first, because
  recall can always be bought by guessing more.
- **Published**: the public bundle `Safak11/lilly` carries exactly the builds
  these numbers were measured on. The publisher checks each one by content
  fingerprint and refuses any other; the listener goes up only under a
  fingerprint named on the command line, because it did not clear its gate.

### Translation

Held-out FLORES-200 Bosnian–English, through the app's own path.

| | the first builds (unrecorded) | first recorded | today |
| --- | --- | --- | --- |
| BLEU, 1,012 devtest pairs | **< 30** | 37.72 — the model's language tag leaked into 308 of 1,012 outputs (`training/RESULTS-devtest.md`) | **43.25**, leaked into **0** |
| chrF2, 1,012 devtest pairs | — | 67.15 | **68.10** |

The "first recorded" column is the untuned model as downloaded. Scored with its
leaked tags stripped, so the defect cannot take credit, the fine-tuning is worth
**+1.26 BLEU** at p = 0.001 and **+0.15 chrF2** at p = 0.074, which does not
clear 0.05, on all 2,009 pairs (`training/RESULTS-product.md`; on the devtest
half +1.17 / +0.25). In plain terms: it makes word-level accuracy better, it
removes a defect from every third output, and it does not move chrF2 by an
amount the bootstrap can see. The first builds, under 30 BLEU, did not manage
any of that.

**Re-measured on 8 September 2026 after the splitter fix.** Until that day
`app.translate.Engine` cut a Bosnian date into pieces (*5. maja 1990. godine*
became three sentences) and translated each alone. The fix was pre-registered
(`training/PREREGISTRATION.md`, "v4 — translate — ordinals") and then measured
on Kaggle with both splitters on the same T4: the new rule is worth
**+0.89 BLEU / +0.37 chrF2** to the fine-tune and **+1.05 / +0.38** to the base
(p = 0.001, 167 of 2,009 rows changed), and the difference between the Mac's
CPU and the T4 on the old rule is inside noise (−0.04 BLEU, p = 0.25). The
figures above are the new ones; the old ones (42.49 / 67.69) are in the
records, not erased.

**The reply direction** (English → Bosnian) was fine-tuned on 8 September and
cleared all four of its pre-registered bars on the same 2,009 pairs: chrF2
**58.96 → 60.00**, BLEU **29.57 → 30.73** (a paired bootstrap over sentences
puts the gains at +1.04 [+0.70, +1.35] chrF2 and +1.16 [+0.70, +1.62] BLEU, 0
of 1,000 resamples at or below zero); the Bosnian form rate on 338 audited bench
targets **94.3% → 99.2%** (244 of 246 decided,
`training/RESULTS-en-bs-formrate.md`); and the `>>bos_Latn<<` label still
steers, its gap against `>>hrv<<` going 21.8 → 22.5 points. The served build
was rebuilt with the adapter merged and published on 8 September;
`scripts/fetch_models.py` pulls it as `translator-en-bs/`. Its base, `tc-base`,
is a smaller model than the forward direction's, so the two directions are not
of comparable quality.

### Speech

200 held-out FLEURS Bosnian clips, the same clips in every column.

| | the first builds (unrecorded) | first recorded | today |
| --- | --- | --- | --- |
| Word error rate | **> 55%** | 38.5% (stock Whisper-small, `training/RESULTS-speech.md`) | **34.9%** |
| Bosnian term recall | — | 65.9% | **68.2%** |
| Wrong-variety substitutions | — | 5.1% | **3.3%** |

**The larger listener: refused at its gate, shipped by decision.** A whisper-large-v3
fine-tune reads 11.9% word error against the shipped listener's 34.9% on the
same 200 clips (`training/SPEECHBENCH-gate.txt`). It has to clear three rows,
not one: word error, Bosnian term recall, and Croatian substitution — how often
it writes the Croatian form of a word where the Bosnian one was said.

- Its gate, run 7 September, refused it by one word on the Croatian row.
- The pre-registered last look, all 925 clips and both listeners on
  8 September (`training/speech-instrument/`), refused it again, and not by one
  word: Croatian substitution **1.1% → 6.1%** (1 of 87 against 8 of 131 decided
  targets, p = 0.018), the same two words over and over (*Europom* for
  *evropom*, *vjerojatno* for *vjerovatno*). Word error and term recall passed
  by wide margins.
- On the project's own scale (`training/RUBRIC.md`, computed for the first time
  in that run) the two listeners read **39.5%** and **14.1%**: band 3 against
  band 8.
- By rule 3 of that pre-registration **whisper-large-v3 is closed to any
  further look**: no other split, normaliser or instrument. That closes the
  question of whether it passes. What ships is a separate decision.
- **The owner's decision, 8 September, evening: it ships anyway.** With both
  readings on the table, the owner chose the larger listener for what it gets
  right, 14.1% against 39.5% of words wrong and 72% against 50% of
  Bosnian-specific words recovered, and accepted what it gets wrong: two
  Croatian spellings, *Europom* and *vjerojatno*, in 8 of 131 decided targets.
  The gate's result stands as written, in `training/PREREGISTRATION.md` with
  the reason. The bundle carries this listener only under its fingerprint
  named on the publish command (`--allow-listen e6bb58483586b06c`), every
  fresh install is told it is not the gated listener, and the gated
  whisper-small (34.9%) stays beside it as the baseline every listener is
  measured against.
- **How it got there, not tidied.** The 4–5 September reader publish swept the
  Mac's `models/lilly/listen`, already large-v3, into `Safak11/lilly` before
  any gate had run. That is the failure the fail-stop rules exist to prevent.
  It was taken out at 11:31 UTC on 8 September and put back that evening by
  the decision above, with the numbers on the table this time.

### Photographs

Two sets of Bosnian signs from Wikimedia Commons, each transcribed by two
readers independently, seeing neither each other's work nor any model's guess;
only words both of them saw are in the answer key. The 40 are the original set
(373 agreed words). `test-v2` is 280 photographs drawn from the same pool, 132
with text, 2,907 agreed words, never trained on by anything.

| | the 40 | `test-v2` (132 photographs) |
| --- | --- | --- |
| **the first builds (unrecorded)** | **< 30% found, > 280 invented** | — |
| **first recorded** — the first reader (`training/RESULTS-ocr.md`) | **36.0% found, 224 invented** | — |
| EasyOCR, stock | 48.0% found, 188 invented | 30.0% found |
| EasyOCR fine-tuned on real crops (the reader until 5 Sep 2026) | 54.5% found, 182 invented | 34.6% found, 2,071 invented |
| **PaddleOCR PP-OCRv6, untrained, confidence floor 0.9 — the reader now** | **67.0% found, 65 invented** | **57.8% found, 450 invented** |

The confidence floor is there because without it PP-OCRv6 read 60.0% and
invented 2,373. The engine was chosen by a rule written before the run
(`training/PREREGISTRATION.md`), on the big set, not the small one: the 40
alone had said the fine-tuned reader read 54.7%, and the 132 say 34.6%.

### Thresholds are written before the run

Deciding measurements and their pass marks live in
[`training/PREREGISTRATION.md`](https://github.com/ssaaffaakk/Lilly/blob/main/training/PREREGISTRATION.md), fixed before any
number exists. Two retraining arms were run for the translator and the
pre-written tie-break chose the one with the *lower* headline BLEU, because the
rule said chrF2 decides. Published scores are bound to the weights by content
hash, so the numbers and the model cannot drift apart.

---

## What it cannot do yet

This section exists because a README that only lists wins is not worth trusting.

- **The Bosnian-specific claim is not proven for the forward direction.** A
  benchmark of 346 cases built from terms that separate Bosnian from Croatian
  and Serbian moves 91.7% → 92.2% at p = 0.360. The base model is already
  trained across South Slavic and arrives at 91.7% on its own, so there is very
  little room above it. (The reply direction is the first place this claim can
  be tested head-on, and there it holds: 94.3% → 99.2%.)
- **The gain is concentrated in news prose.** Broken out by corpus, the
  fine-tuning is worth +3.05 BLEU on news text and −0.82 BLEU on talks. Nothing
  measured here separates *learned better Bosnian* from *adapted to news style*.
- **Against an outside system Lilly wins, and mostly not on its own merit.**
  `facebook/nllb-200-distilled-600M` on the same 2,009 FLORES-200 pairs scores
  36.49 BLEU bs→en against Lilly's 42.14, and 26.07 en→bs against 29.57 — a
  model 2.6× and 8× larger, beaten in both directions
  (`training/RESULTS-outside-baseline.md`). But the untouched Helsinki base
  already accounts for +5.11 of that +5.65 BLEU. What the fine-tune itself adds
  is small and its sign depends on the path: −0.79 chrF2 on whole rows, **+0.15
  on the path the product serves** (43.03 / 67.81 against a tag-stripped base at
  41.77 / 67.66 on all 2,009 pairs, re-measured 8 September;
  `training/RESULTS-product.md`). Its clearest win is not in either column:
  **308 of 1,012 base outputs leaked the model's language tag into the text, and
  0 do after.** In the reply direction the published bundle has no fine-tune
  yet, so that margin is the base's. **The win belongs largely to OPUS-MT**,
  which this project builds on and did not train. And NLLB-600M is the distilled
  small variant: Google, DeepL, the 3.3B NLLB and the large general models were
  not tested, so none of this is a claim about the state of the art.
- **English → Bosnian is not published yet, and stays the weaker direction.**
  The fine-tune cleared its bars (above) but the bundle serves the untuned base
  until the owner publishes. Even fine-tuned, it starts from a smaller base than
  the forward direction and reads 60.00 chrF2 where the forward direction reads
  68.10.
- **The photograph scores are recognition, not phone reality.** The evaluated
  images come from Wikimedia Commons. Real photographs taken on a phone in
  Bosnia would be the honest test, and there is not a labelled set of them yet.
  The Commons images are also *downscaled*: a sign whose Commons original is
  3968 px wide is scored here at 1280 px, and the app reads at up to 2 MP, so
  these numbers, if anything, understate what the reader does on a
  full-resolution photo. Measuring that is pre-registered and not yet run.
- **The photograph score is not yet a valid score by the project's own scale.**
  `training/RUBRIC.md` refuses to grade the reader below 200 real photographs
  with text; `test-v2` has 132. Another 160 photographs (`test-v2b`) are
  fetched and waiting for two blind transcriptions.
- **One test set.** FLORES is professionally translated and even in register.
  Real user input is not.

The full write-up of every limit is on the
[model card](https://huggingface.co/Safak11/lilly) and in
[`training/`](https://github.com/ssaaffaakk/Lilly/blob/main/training/).

---

## Training

![Kaggle training flow](https://raw.githubusercontent.com/ssaaffaakk/Lilly/main/docs/images/kaggle-flow.png)

Training runs on Kaggle GPUs. This Mac commits, launches, polls, and installs
the result; it does not run multi-hour fine-tunes. The boundary is written down
in [`docs/V2-BOUNDARIES.md`](https://github.com/ssaaffaakk/Lilly/blob/main/docs/V2-BOUNDARIES.md).

| Lane | What it does | Gate before anything ships |
| --- | --- | --- |
| Translation | LoRA or full fine-tune, either direction → adapter zip | pre-registered bars on FLORES, form rate and label steering |
| Speech half 1 | one epoch → `lilly-listen-half1.zip` | training exits 0; no quality claim here |
| Speech half 2 | resumes for epoch 2, then scores WER → `lilly-listen.zip` | WER must beat the shipped listener |
| Speech instrument | scores two listeners on all 925 clips | the three gate rows, both not either |
| OCR | harvest, real crops, synthetic crops → `lilly-read.zip` | install gate must pass; the line is paused, see the roadmap |

A known failure stops the kernel. A `COMPLETE` status, a leftover zip from an
earlier run, or hitting the 12-hour wall does not override a gate.

```bash
python3 scripts/preflight_kaggle.py
python3 scripts/kaggle_train.py speech          # half 1
python3 scripts/kaggle_train.py ocr             # in parallel if a GPU slot is free
python3 scripts/kaggle_train.py speech-half2    # only after half 1 is COMPLETE
python3 scripts/kaggle_train.py speech-instrument   # 925 clips, both listeners: the last look at large-v3
python3 scripts/kaggle_train.py translation-en-bs   # the reply direction, LoRA, pre-registered bars
python3 scripts/kaggle_train.py outside-baseline    # NLLB-200 on the same FLORES pairs
python3 scripts/kaggle_poll.py                  # CANCEL or ERROR counts as failure
```

What gets trained next, and what does not, is in [`docs/V4-PLAN.md`](https://github.com/ssaaffaakk/Lilly/blob/main/docs/V4-PLAN.md).
The reader's queue and do-not-repeat list are in
[`docs/OCR-ROADMAP.md`](https://github.com/ssaaffaakk/Lilly/blob/main/docs/OCR-ROADMAP.md). How to write a notebook that
fails loudly: [`docs/kaggle-notebooks.md`](https://github.com/ssaaffaakk/Lilly/blob/main/docs/kaggle-notebooks.md). The list
of failures already paid for: [`docs/kaggle-fail-stop.md`](https://github.com/ssaaffaakk/Lilly/blob/main/docs/kaggle-fail-stop.md).

---

## Repo map

| Path | What is in it |
| --- | --- |
| `app/` | FastAPI server, the four abilities, the web UI |
| `models/lilly/` | Offline weights, model card, attribution notice |
| `training/` | Notebooks, training and evaluation scripts, every results file |
| `bench/` | The Bosnian-versus-neighbours benchmark and how its cases are built |
| `scripts/` | `kaggle_train.py`, preflight, poll, fetch, publish |
| `docs/` | Plans, boundaries, fail-stop rules, white paper — index in [`docs/README.md`](https://github.com/ssaaffaakk/Lilly/blob/main/docs/README.md) |
| `data/` | Corpora and OCR crops (large, usually local only) |
| `space/` | Hugging Face Space packaging |

---

## Status

- [x] Translate, listen, speak, read, web app, correction pipeline
- [x] Published weights and model card with every score and every limit
- [x] Pre-registered thresholds and hash-bound results
- [x] English → Bosnian fine-tune — all four pre-registered bars cleared 8 Sep (chrF2 +1.04, BLEU +1.16, form rate 99.2%, label gap 22.5); built and **published 8 Sep**
- [x] Larger speech model — trained (11.9% word error); refused at the gate 7 Sep and at the pre-registered last look 8 Sep (Croatian 1.1% → 6.1%, p = 0.018); closed to further looks by rule 3; **shipped 8 Sep evening by the owner's decision**, refused row and all, under a fingerprint named on the publish command
- [ ] A valid photograph score: `test-v2b`'s 160 photographs transcribed blind, then one score on the union
- [x] Re-measure the served translation path after the ordinal splitter fix — done 8 Sep on Kaggle, both splitters on one T4: devtest 42.49 → **43.25** BLEU, 67.69 → **68.10** chrF2 (p = 0.001); device drift within noise
- [ ] A labelled set of real phone photographs from Bosnia

---

## Credits

The weights come from Helsinki-NLP and the OPUS-MT project, OpenAI and SYSTRAN,
JaidedAI and Clova AI Research, PaddlePaddle, and hexgrad. CTranslate2 and peft
shape the build. Please credit them rather than this repository. Every license
was checked against the project's own page and is listed in
[`models/lilly/NOTICE.md`](https://github.com/ssaaffaakk/Lilly/blob/main/models/lilly/NOTICE.md). The OPUS-MT authors ask to be
cited; the citation is on the [model card](https://huggingface.co/Safak11/lilly).

Built by [@ssaaffaakk](https://github.com/ssaaffaakk).

---

## Layout

| Folder | Ability | Built from | Format | Fine-tuned here? |
|---|---|---|---|---|
| `translator/` | Bosnian text → English text | OPUS-MT `opus-mt-tc-big-zls-en` | CTranslate2, int8 | **yes** — LoRA merged into the weights |
| `translator-en-bs/` | English text → Bosnian text (the reply) | OPUS-MT `opus-mt-tc-base-en-sh` | CTranslate2, int8 | **yes** — LoRA merged into the weights, 8 Sep 2026 |
| `listen/` | spoken Bosnian → Bosnian text | `whisper-large-v3` (OpenAI), converted to CTranslate2 here | CTranslate2, int8 | **yes** — LoRA. **Shipped by the owner's decision on 8 September 2026, knowing its pre-registered gate refused it:** on all 925 test clips it reads **14.1%** of words wrong against the gated whisper-small's 39.5%, recovers 72% of Bosnian-specific words against 50%, and writes the Croatian form of a Bosnian-specific word in **6.1%** of decided targets against 1.1% (p = 0.018), the row it failed. Details under "Measured quality". |
| `read/` | photo of Bosnian text → text | EasyOCR (CRAFT detector + Latin recogniser) | PyTorch checkpoints | **yes** — recogniser only, words 69.5% → 88.3%. **Since 5 Sep 2026 the app does not read with these weights: it reads with PaddleOCR PP-OCRv6** (Baidu's published weights, untrained, at a recogniser confidence floor of 0.9), fetched at run time by PaddleX from its Hugging Face mirror and **not part of this bundle**. `read/` is the way back — `LILLY_READER=easyocr` — and the "before" build every comparison is measured against. |
| `speak/` | English text → spoken English | Kokoro-82M | PyTorch checkpoint + one voice | no — stock weights |

`translator/built.json` records what went into the build (`fine_tuned`, `quantization`),
so the served model can say which it is rather than leaving you to guess from the folder
name.

```python
from app.lilly import lilly

lilly.translate("Dobar dan")          # Bosnian text   -> English text
lilly.reply("Good morning")           # English text   -> Bosnian text
lilly.listen("clip.m4a")              # spoken Bosnian -> Bosnian text
lilly.speak("Good day", "out.wav")    # English text   -> spoken English
lilly.read("sign.jpg")                # photo          -> Bosnian text
```

---

# Credits and licenses

Every weight here comes from one of these projects, and the two tools below shaped the
build. The bundle exists because of their work — please credit them rather than this
repository. Licenses were checked against each project's own page, not assumed.

### Weights

| Folder | Source | Author | License |
|---|---|---|---|
| `translator/` | [opus-mt-tc-big-zls-en](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-zls-en) | Helsinki-NLP / the OPUS-MT project, University of Helsinki | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |
| `listen/` | [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3), converted to CTranslate2 int8 by this project | OpenAI | MIT |
| `read/` | [EasyOCR](https://github.com/JaidedAI/EasyOCR); detector from [CRAFT](https://github.com/clovaai/CRAFT-pytorch) | JaidedAI; CRAFT by Clova AI Research, NAVER Corp. | Apache-2.0 (EasyOCR); MIT (CRAFT) |
| *(engine, not bundled)* | [PaddleOCR PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) — what the app reads with since 5 Sep 2026, fetched at run time | PaddlePaddle Authors, Baidu | Apache-2.0 |
| `speak/` | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | hexgrad | Apache-2.0 |

### Tools

Neither contributes weights; both are why the served models are the shape they are.

| Tool | Author | License | Used for |
|---|---|---|---|
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | OpenNMT | MIT | quantising and serving `translator/` and `listen/` |
| [peft](https://github.com/huggingface/peft) | Hugging Face | Apache-2.0 | the LoRA fine-tuning merged into `translator/` |

The OPUS-MT authors ask that their papers be cited:

```bibtex
@inproceedings{tiedemann-thottingal-2020-opus,
  title = "{OPUS}-{MT} {--} Building open translation services for the World",
  author = {Tiedemann, J{\"o}rg and Thottingal, Santhosh},
  booktitle = "Proceedings of the 22nd Annual Conference of the European Association
               for Machine Translation",
  year = "2020", address = "Lisboa, Portugal"
}

@inproceedings{tiedemann-2020-tatoeba,
  title = "The Tatoeba Translation Challenge {--} Realistic Data Sets for Low Resource
           and Multilingual {MT}",
  author = {Tiedemann, J{\"o}rg},
  booktitle = "Proceedings of the Fifth Conference on Machine Translation",
  year = "2020"
}
```

[NOTICE.md](NOTICE.md) carries the full attribution notice, and takes it one level
further back — to the datasets these models were themselves trained on, where those
datasets ask for it (Koniwa and SIWIS behind `speak/`, the Tatoeba Challenge and OPUS
collections behind `translate/`, SynthText and the ICDAR sets behind `read/`). Note that
NOTICE.md was written when the translation folder was named `translate/`; its
`translate/` entry is the source of what is now served as `translator/`.

## What was changed

- `translator/` — the OPUS-MT weights with this project's LoRA fine-tuning merged in,
  then converted to CTranslate2 int8. Not byte-identical to upstream, by design.
- `translator-en-bs/` — the OPUS-MT `tc-base-en-sh` weights with this project's LoRA
  fine-tuning merged in, then converted to CTranslate2 int8. Not byte-identical to
  upstream, by design.
- `listen/` — the Whisper large-v3 weights fine-tuned on Bosnian audio (LoRA merged), then converted to
  CTranslate2 int8. Not byte-identical to upstream.
- `read/`, `speak/` — weight files byte-identical to the originals. Two files in
  `speak/` were renamed so the app can load them by a stable path:
  `kokoro-v1_0.pth` → `model.pth`, and `voices/af_heart.pt` → `voices/default.pt`.
  Only the `af_heart` voice is included; the rest are in the upstream repository.
