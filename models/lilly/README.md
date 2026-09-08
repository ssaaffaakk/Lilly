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

# Lilly — Bosnian and English, five models in one folder

The weights the [Lilly](https://github.com/ssaaffaakk/Lilly) app runs on. Lilly turns
Bosnian into English three ways — typed text, spoken Bosnian, and photographs of Bosnian
text — reads the English back aloud, and gives you the Bosnian to say in reply.

This is **a bundle, not a new architecture**. Three of the five folders are fine-tuned
by this project — both translators and the listener — and each carries its own scores
on this page, measured against the untuned model it is built on. The voice is stock.
The EasyOCR reader in `read/` is fine-tuned too, but the app reads photographs with
PaddleOCR PP-OCRv6, chosen by a rule written before the comparison ran; it is fetched
by the app at install time and is not in this bundle. Every upstream model is credited
below. Once installed, nothing reaches the network at runtime.

Two directions. Bosnian → English is the shipped direction and carries the scores
below. English → Bosnian (`translator-en-bs/`, the reply the app offers under the swap
button) is a smaller base fine-tuned on 8 September 2026; its own numbers are in
"The reply direction" below, and nothing here says the two directions are of
comparable quality — on the evidence they are not.

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

# Where it started

The first builds were worse than anything on this page. No measurement of them was
kept — the habit of committing a number before changing anything came later, and is
now the rule — so the first column is the owner's own notes from that time, written as
a bound. The next column is the first number that was recorded, with the file it lives
in. The last is today.

| | the first builds (unrecorded) | first recorded | today |
|---|---|---|---|
| Photographs — words found per photograph, the 40 | **< 30%** | 36.0% (`training/RESULTS-ocr.md`) | **67.0%** |
| Photographs — words found, pooled | **< 10%** | 16.9% (63 of 373) | **69.4%** |
| Photographs — words invented that are on no sign | **> 280** | 224 | **65** |
| Speech — word error, 200 held-out clips | **> 55%** | 38.5% (`training/RESULTS-speech.md`) | **11.9%** (whisper-large-v3, shipped by the owner's decision, refused at its gate; the gated whisper-small reads 34.9%) |
| Translation — BLEU on FLORES devtest, as the user sees it | **< 30** | 37.72, with the language tag leaked into 308 of 1,012 outputs (`training/RESULTS-devtest.md`) | **42.49**, leaked into **0** |
| Reply, English → Bosnian — chrF2 on FLORES-200 | — | 58.96, the base as downloaded (`training/RESULTS-en-bs.md`) | **60.00**, fine-tuned and published 8 Sep 2026 |

Every number after the first column was measured on real photographs, real audio and
held-out sentences, and the rest of this page is those measurements.

# The reply direction — English → Bosnian

`translator-en-bs/` is `opus-mt-tc-base-en-sh` with a LoRA adapter merged in, fine-tuned
on 8 September 2026 on the same corpus as the forward direction with the pairs flipped,
and quantised to int8 the same way. It cleared all four bars written before it ran
(`training/PREREGISTRATION.md`, "v3 — reply"), measured on the 2,009 FLORES-200 pairs
with `training/evaluate.py` and `training/bosnian_form_rate.py`:

| | base as downloaded | fine-tuned | bar |
|---|---|---|---|
| chrF2, FLORES-200 | 58.96 | **60.00**  (+1.04, paired bootstrap p < 0.001) | above 58.96 |
| BLEU, FLORES-200 | 29.57 | **30.73**  (+1.16, p < 0.001) | not below 29.57 |
| Bosnian form rate, 338 audited targets | 94.3% | **99.2%**  (244 of 246 decided) | not below 94.3% |
| `>>bos_Latn<<` against `>>hrv<<`, form-rate gap | 21.8 points | **22.5 points** | not collapsed |

The form rate asks, for words that separate Bosnian from Croatian and Serbian, which
form the model wrote; the label gap checks the decoder still hears its language
selector. Both moved the right way. Two things this does not say: the numbers are the
adapter's through `training/evaluate.py`, and this int8 build has not been scored through
the app's own path yet; and `tc-base` is a smaller model than the forward direction's
base, so 60.00 here and 67.47 there are not the same quality. Bound to these numbers
by the served build's digest, `training/RESULTS-en-bs-formrate.md`.

# Measured translation quality

**Test set:** 2,009 FLORES-200 Bosnian–English pairs, held out — not in the base model's
training data and not in the fine-tuning data.

**How it was measured:** both the base and the fine-tuned model built the same way (int8
CTranslate2) and run through the app's own `app.translate.Engine`, so the sentence
splitting and the quantisation are identical on both sides and the only difference
between the columns is the fine-tuning. Significance is a paired bootstrap.

The base model prints its own language tag — `>>eng<<` and friends — into the translation
in 572 of 2,009 outputs (28.5%). The fine-tuned model does it in 0 (0.0%). That is a
defect a reader sees, so both numbers are given: with the tag, because it is what arrives
on screen; without it, because otherwise the fine-tuning is credited with translation
quality it did not gain.

### With the language tag stripped — the number to quote

| Model | BLEU | chrF2 | Length vs reference |
|---|---|---|---|
| Base (untuned) | 40.81 | 67.34 | 1.019 |
| Lilly (fine-tuned) | **42.18** | 67.47 | 1.001 |
| Gap | **+1.37** | +0.14 | |

BLEU moves +1.37 at p = 0.001. chrF2 moves +0.14 at p = 0.104, which does **not** clear
the 0.05 bar — so the right reading is not that chrF2 improved. It is that the fine-tuning
no longer costs anything there. An earlier version of this model lost 0.22 chrF2 at
p = 0.028; rebuilding the training corpus so its length variance matches the benchmark's
(sd of log character ratio 0.213 → 0.130 against FLORES's 0.129) removed that loss.
Output length went from 0.992 of the reference to 1.001.

### As a user sees it, tag and all

| Model | BLEU | chrF2 | Length vs reference |
|---|---|---|---|
| Base (untuned) | 37.60 | 67.00 | 1.048 |
| Lilly (fine-tuned) | 42.18 | 67.47 | 1.001 |
| Gap | +4.58 | +0.48 | |

BLEU moves +4.58 at p = 0.001 and chrF2 +0.48 at p = 0.001. Both clear the 0.05 bar, so
both are measured rather than leaned towards. But most of the +4.58 is the tag leak being
fixed rather than translation improving, which is why the stripped table above is the one
to quote — this one says what a reader gets, not what the model learned.

Source: `training/RESULTS-product.md`, generated by `training/evaluate_app.py`. That file
names the build its scores were measured on by content hash, and the publisher refuses to
upload weights whose hash does not match — so the numbers on this page and the weights
beside them cannot come apart.

**Chosen against a threshold written first.** Two retraining arms were run and both cleared
the bars set in `training/PREREGISTRATION.md` before either number existed. The tie-break
written at the same time says the higher chrF2 wins and not the higher BLEU; the other arm
scored 42.21 BLEU / 67.35 chrF2, so it had the better headline and lost. Split across
FLORES's own halves the choice holds — the shipped arm leads on chrF2 in both devtest
(67.69 vs 67.58) and dev (67.25 vs 67.10) — so it is not an artifact of selecting on the
set being reported.

### The same fine-tune, scored three ways

Worth stating plainly, because the spread is larger than the effect:

| Method | BLEU | chrF2 |
|---|---|---|
| Raw adapter, whole rows (earlier model) | +0.54 | −0.79 |
| Through the app's path, tag stripped | +1.37 | +0.14 |
| Through the app's path, as the user sees it | +4.58 | +0.48 |

Each change of method had a reason and none was chosen to flatter the result — but all
three moved in the flattering direction, and that pattern is what a garden of forking
paths looks like from the inside. The project's response was to write
`training/PREREGISTRATION.md` fixing the deciding measurement and its thresholds
**before** the next training run produced any numbers. If you are weighing these
results, read that file alongside this table.

**Training data:** 361,621 examples from 351,889 pairs. 334,790 cleaned pairs to start
with, then 21,178 misaligned WikiMatrix pairs removed (their two sides are about the same
subject rather than being translations of each other) and 38,277 pairs the base model had
never seen added from wikimedia-v20260327 and NTREX-128 — 351,889. The corpus is then
rebuilt as one sentence per example inside a length band matched to the benchmark's, which
is why there are more examples than pairs: a pair carrying two sentences becomes two.

**Build:** the LoRA adapter is merged into the weights and the result converted to
CTranslate2 int8, because an adapter cannot be applied after quantisation
(`scripts/build_translator.py`).

---

# Limits

**The thing this project says it is for is not measurably better.** Lilly's pitch is real
Bosnian rather than generic Serbo-Croatian, so a benchmark was built to test exactly that:
240 terms that separate Bosnian from Croatian and Serbian, chosen by counting rather than
by intuition — a term is kept only if it appears at least 20 times in professionally
translated Bosnian, holds 98% of the share against its alternative, and that alternative
occurs at least once in mixed text. 346 cases.

| | term recall | 95% interval |
|---|---|---|
| Base | 91.7% | [88.5, 94.1] |
| Lilly | 92.2% | [89.1, 94.5] |

**+0.5 points, paired bootstrap p = 0.360.** Nine targets fixed, seven broken. On the same
cases BLEU moves +5.7 against the base and this does not move at all. The reason is not
mysterious: the base is `opus-mt-tc-big-zls-en`, already trained across South Slavic, and
it arrives at 91.7% on its own. There is very little room above that.

**Does the benchmark work?** Two controls say the scoring code is sound rather than
generous: feeding it the professional reference recovers 100.0% of targets, and feeding it
a copy of the Bosnian input recovers 0.0%. A third says something about the test's reach —
only 3.6% of swapped targets score differently from the Bosnian one, so replacing a
Bosnian term with its Croatian or Serbian form usually changes nothing the model does.
That is a narrow signal, and it bounds what this benchmark can detect.

**A second measure, reported and not promoted.** The gap between a model's score on the
Bosnian form and on the swapped one is arguably a more direct look at this project's claim
than term recall is: a model leaning on generic Serbo-Croatian should do relatively better
when the term is swapped, so a wider gap is what "it really is Bosnian" would look like.
Lilly's gap is **+0.9 points wider** than the base's — and at p = 0.4388 that does not
clear the bar either. It is stated here because it was looked at, and stated second
because it was not the measure named in `training/PREREGISTRATION.md` and was tested after
the headline was already known. A number noticed afterwards that happens to favour us is
exactly what pre-registration exists to keep in its place.

An earlier version of this benchmark had 398 cases and 61 of them — 15.3% — were
sentences the model had trained on. NTREX was added to the training corpus and is also one
of this benchmark's sources, and holding 462 NTREX rows out did not help because the
benchmark drew from all of NTREX. The check now lives in `bench/build.py`, where cases are
made: a sentence that appears in the training corpus cannot enter the set. The leak
favoured the fine-tuned model, so this could only have moved its score down — and it read
+0.2 at p = 0.465 before and +0.5 at p = 0.360 after, which is the same answer either
way.

The benchmark's own limits are worth stating too, because they cut both ways. Every
Turkism was rejected — *kahva*, *čaršija*, *avlija*, *sokak*, *mahala* occur zero times in
the news and wiki text available, so the categories that would most obviously show
Bosnian character are absent from the record rather than absent from the language. What
remains is a test of ijekavian forms and lexical doublets. See `bench/` and
`training/bosnian_bench.py`.

**The gain is concentrated in clean news prose.** Broken out by source corpus, the
fine-tuning is worth **+3.05 BLEU on SETIMES** (news) and **−0.82 BLEU on TED2020**
(talks). The training corpus leans the same way. So two explanations fit the numbers
equally well — *the model learned better Bosnian*, and *the model adapted to news
style* — and **nothing measured here distinguishes them.** Telling them apart needs a
measure aimed at that claim, which does not exist yet. Until it does, treat the headline
gain as demonstrated on news-like prose and unestablished elsewhere.

**chrF2 does not improve, it merely stops getting worse.** +0.14 against the base at
p = 0.104, which does not clear the 0.05 bar — so on the measure that is fairer for a
language inflecting as heavily as Bosnian, this model is indistinguishable from the one it
started from. That is an improvement only relative to an earlier version of this same
fine-tune, which lost 0.22 chrF2 at p = 0.028; rebuilding the corpus so its length
variance matched the benchmark's removed that loss. The honest summary is that the
fine-tuning buys word-level accuracy and, now, costs nothing at the character level —
not that it improves both.

**One direction, one language pair.** Bosnian → English. The base model is multilingual
across South Slavic languages; the fine-tuning, the evaluation and the app are Bosnian
only, so nothing here says how it behaves on Croatian, Serbian, Slovenian or Macedonian.

**`speak/` carries no score on this page.** It is stock Kokoro-82M; for how it performs,
see upstream.

**`listen/` — speech.** Whisper large-v3 with a LoRA adapter, trained on Kaggle in
two halves on FLEURS Bosnian plus Croatian audio, converted to CTranslate2 int8. It is
the listener the owner chose to ship on 8 September 2026 **knowing its pre-registered
gate refused it**, and both sides of that are here.

| 925 FLEURS bs_ba test clips | whisper-small, the gated listener | **whisper-large-v3, shipped** |
|---|---|---|
| word error, greedy, Whisper's normaliser (`training/RUBRIC.md`) | 39.5% | **14.1%** |
| Bosnian-specific words recovered (581 targets) | 49.6% | **72.1%** |
| Croatian form written where Bosnian was said | **1.1%** (1 of 87 decided) | 6.1% (8 of 131), p = 0.018 |

On the 200-clip prefix the two read 34.9% and 11.9% word error. The gate has three
rows and says all three must pass; the third failed, on two words written the
Croatian way (*Europom* for *evropom*, *vjerojatno* for *vjerovatno*), and by rule 3
of its pre-registration this model gets no further look on any other split, normaliser
or term list. The owner then shipped it for what it gets right and accepted what it
gets wrong; the reason is written in `training/PREREGISTRATION.md` and the record in
`training/RESULTS-speech.md` and `training/speech-instrument/`. The gated whisper-small
(38.5% → 35.5% on the 200 clips by `training/evaluate_speech.py`, 34.9% by the bench's
scorer) is not in this bundle; it is the baseline every listener is measured against.

**`translator-en-bs/` — the reply direction.** See "The reply direction" below.

**`read/` — photographs.** Only the recogniser is fine-tuned; the CRAFT detector is stock.
Measured through easyocr's own recognise path with the app's allowlist, on 600 held-out
crops sampled to keep the set's mix (17.3% photographs, the rest synthetic):

| | shipped recogniser | fine-tuned |
|---|---|---|
| whole words exact | 69.5% | **88.3%** |
| Bosnian letters kept | 69.5% | **86.0%** |
| words, photographs only | 47.1% | **75.0%** |

The held-out set shares no label text with the training set — 873 texts once sat on both
sides and over half of every reader score this project printed was recital until that was
fixed. Photographs gain most, which is the half that matters: a stock recogniser reads
under half the words in a photograph of a Bosnian sign.

Two things this does not say. The crops are word images, so this measures recognition and
not the detector's ability to find text on a real photograph. And the crops' photographs
are synthesised.

**On real photographs, and why the app no longer reads with these weights.** The figures
above are crops; on whole photographs the same fine-tuned EasyOCR reader is far weaker,
and PaddleOCR PP-OCRv6 at floor 0.9 beats it on both rows of every comparison run:

| set | | fine-tuned EasyOCR | PP-OCRv6 @ 0.9 |
|---|---|---|---|
| the 40 Commons photographs | words found per photograph | 54.5% | **67.0%** |
| | words invented | 182 | **65** |
| `test-v2`, 132 held-out photographs | words found per photograph | 34.6% | **57.8%** |
| (2,907 agreed words, two blind transcribers) | words invented | 2,071 | **450** |

`test-v2` is the larger and entirely held-out set — 132 photographs carrying text against
28 — and it is the honest figure for this ability: **the fine-tuned EasyOCR reader finds
about a third of the words on a photograph of Bosnian text, not the 88.3% the crop table
above shows.** The two are different measurements, not a contradiction: word images versus
whole photographs, detector included. Full method and counts in
`training/RESULTS-ocr-bakeoff.md`, `training/RESULTS-ocr-test-v2.md` and
`training/RESULTS-ocr-paddle-floor.md` in the app repository.

**Single test set.** All translation numbers above come from one held-out set of 2,009
FLORES-200 pairs. FLORES is professionally translated and fairly uniform in register;
real user input is not.

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

