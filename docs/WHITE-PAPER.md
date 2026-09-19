# Lilly: An Offline Bosnian–English Translator

**Safak Surmeli**  
September 2026

[Code](https://github.com/ssaaffaakk/Lilly) ·
[Weights](https://huggingface.co/Safak11/lilly) ·
[Data](https://huggingface.co/datasets/Safak11/lilly-data)

---

## Abstract

Lilly is an offline Bosnian–English translator. A user can type, speak, or
photograph a sign; the app returns text and speech. Nothing leaves the
machine at inference time.

Five models sit behind one FastAPI server. Speech and photographs become
text, then go through the same translator.

On the path a user actually meets: **43.25 BLEU / 68.10 chrF2** Bosnian →
English (1,012 FLORES-200 devtest sentences) and **32.22 BLEU / 61.55 chrF2**
English → Bosnian (2,009 FLORES-200 pairs). The listener records **11.9%**
word error on 200 held-out FLEURS clips. The reader finds **67.0%** of the
words on 40 Commons photographs (65 invented) and **57.8%** on 132 held-out
photographs (450 invented). The 67% mix includes empty and unreadable frames;
the same reader scores **82.5%** (273/331) on the 21 outdoor shots a person
can still read. 82.5 does not replace 67.

Thresholds were written before each run. Gains use a paired bootstrap.
Failures stay in the repository.

---

## 1. Introduction

Bosnian, Croatian, and Serbian share grammar and most of their vocabulary.
They do not share every word, every script habit, or every cultural term.
Public translators often treat them as one language. A tool that writes a
Croatian form, or that cannot read č ć đ š ž, is not a Bosnian tool.

Lilly was built as a daily translator for an internship in Bosnia and
Herzegovina: type, speak, or photograph, get an answer that respects the
Bosnian variety, and keep the audio and the pictures on the device.

This paper reports the system, the data, and the measurements — including
the runs that did not help.

**Contributions.**

1. One offline app for both directions: translate, listen, read, speak.
2. Product scores measured through the serving path (`app.translate.Engine`,
   `app.speech`, `app.ocr.scan`), not the raw checkpoint.
3. A public dataset of the photographs and cleaned parallel sentences the
   result files name ([`Safak11/lilly-data`](https://huggingface.co/datasets/Safak11/lilly-data)).
4. Negative results in the same place as the wins (BosnianBench, OCR
   self-labelling, two refused voices).

Section 2 is the system. Section 3 is the data. Section 4 is how models were
chosen and scored. Section 5 is the numbers. Section 6 is what they mean.
Section 7 is what is still missing.

---

## 2. System

Lilly is a Python FastAPI service and a browser UI. Five abilities:

| Ability | In | Out | Model |
|---|---|---|---|
| Translate (bs → en) | Bosnian text | English text | OPUS-MT `opus-mt-tc-big-zls-en`, LoRA, CTranslate2 int8 |
| Reply (en → bs) | English text | Bosnian text | OPUS-MT `opus-mt-tc-base-en-sh`, LoRA, CTranslate2 int8 |
| Listen | Audio | Text | Whisper large-v3, LoRA, CTranslate2 int8 |
| Read | Photograph | Text | PaddleOCR PP-OCRv6, stock, confidence floor 0.9 |
| Speak | Text | Audio | Kokoro-82M (English); Piper `sr_RS` (Bosnian) |

Speech and photographs become text first. Translation quality is therefore
the same for type, talk, and camera.

Since 8 September 2026 every ability runs both ways. Whisper is multilingual.
PP-OCRv6 reads Latin script in either language. The UI swaps the arrow.

Requests are bounded by upload size, token count, and pixel count. After
`scripts/fetch_models.py`, inference does not use the network. Weights on
Hugging Face are bound to content fingerprints.

---

## 3. Data

### 3.1 Translation

**Train.** 313,612 Bosnian–English pairs after cleaning: WikiMatrix,
SETIMES-v2, TED2020-v1, wikimedia-v20260327, and 1,924 rows from NTREX-128
(published after the base model's training cutoff). 21,178 misaligned
WikiMatrix pairs were dropped.

The base model's own release (`opusTCv20210807+bt`) already lists WikiMatrix,
SETIMES, TED2020, and Wikimedia. Fine-tuning re-weighted material the model
had already seen. That is the right frame for reading +1.26 BLEU.

**Test.** FLORES-200 Bosnian–English (2,009 pairs, dev + devtest). NTREX-128
is unseen on two independent grounds. BosnianBench (346 cases that separate
Bosnian from Croatian and Serbian) tests the variety claim.

### 3.2 Speech

FLEURS Bosnian (`bs_ba`): 3,091 train clips, 925 test clips. Every speech
gate uses a 200-clip held-out prefix of that test set. Common Voice has no
Bosnian. The whisper-small fine-tune also used VoxPopuli Croatian and FLEURS
Croatian.

A YouTube corpus of 22.5 hours from seven CC-BY Bosnian lecture channels
yielded 8,329 clips (1,105 minutes) after a log-prob and length filter. That
set is staged for a Bosnian voice; it is not the shipped listener's train set.

### 3.3 Photographs

Wikimedia Commons harvest: 286 sign photographs, full attribution in
`CREDITS.tsv`.

Two test sets, each transcribed blind by two independent readers. Only words
both agreed on enter the key.

- **The 40:** 373 agreed words, 88.2% inter-annotator agreement. Two items
  are not photographs (a map render and a 1900s postcard).
- **test-v2:** 280 photographs from the same pool, 132 with text, 2,907
  agreed words. Nothing in this project trained on this set.

Training crops: 1,914 cut from the harvest, 1,702 usable, labelled by humans.
39 empty and 173 unreadable crops were excluded.

### 3.4 Where the bytes live

GitHub holds lists, keys, credits, and every results file. The photographs
and cleaned parallel sentences are
[`Safak11/lilly-data`](https://huggingface.co/datasets/Safak11/lilly-data):

| Archive | Contents |
|---|---|
| `translation-clean.zip` | 313,612 training pairs plus valid / test / NTREX holdout |
| `ocr-the-40.zip` | the 40 Commons photographs the product OCR number was measured on |
| `ocr-test-v2.zip` | test-v2 photographs, credits, both blind passes, the answer key |
| `ocr-harvest.zip` | harvested Commons sign photographs and `CREDITS.tsv` |
| `ocr-crops.zip` | crops cut from the harvest, with the human labels |

FLORES-200 and FLEURS stay with their publishers. The 20,240 Mapillary
frames are on Kaggle (`afaksrmeli/lilly-mapillary-photos`) and are not a
train set.

---

## 4. Methods

### 4.1 How a number is allowed to count

1. **Write the bar first.** Pass/fail, the test set, and what failure looks
   like sit in `training/PREREGISTRATION.md` before the run. Later disagreement
   is a note under the bar, not an edit of it.
2. **Held-out only.** FLORES-200 for translation. 200 FLEURS clips for speech
   headlines. Commons photographs with independent human keys for reading.
3. **The app's path.** Translation is scored through `app.translate.Engine`
   (splitter, int8, tag strip). Scoring the raw adapter on whole rows is a
   different experiment and is labelled as such.
4. **Paired bootstrap.** A claimed gain needs p < 0.05 (1,000 or 2,000
   draws). Inside the interval is a tie.
5. **Counts next to percentages.** 57.8% on 132 photographs is not 57.8% on
   40. 132 photographs is ±8 points; 25 letters is ±19.

### 4.2 Translation

**Bosnian → English.** Base: Helsinki-NLP `opus-mt-tc-big-zls-en` (~230M),
trained on 159.6M South Slavic pairs. Two LoRA arms on a Kaggle T4, tie-break
written first: **higher chrF2 wins**, not higher BLEU. Arm B (cleaned data plus
a pre-committed recipe) won, including when its BLEU was lower. The adapter is
merged and served as CTranslate2 int8.

**English → Bosnian.** Base: `opus-mt-tc-base-en-sh` (~77M). Same LoRA method.
Four pre-registered bars: chrF2 ≥ 58.96, BLEU ≥ 29.57, Bosnian form rate
≥ 94.3%, `>>bos_Latn<<` steering gap preserved. All four cleared on 8
September 2026.

### 4.3 Speech

Two Whisper sizes, LoRA, CTranslate2 int8.

- **whisper-small** (~244M): FLEURS Bosnian + VoxPopuli Croatian. Cleared
  all three gate rows. Stays as the baseline.
- **whisper-large-v3** (~1.55B): same data. **Refused at its gate** (Croatian
  substitution). Closed to further looks by the pre-registration. **Shipped
  by the owner's decision** on 8 September 2026 for the word-error drop,
  with the failed row on the model card.

### 4.4 Photograph reading

The shipped reader is stock PaddleOCR PP-OCRv6 (`medium_det` + `medium_rec`)
with recogniser floor 0.9. The engine was chosen by a rule written before the
comparison, on **test-v2**, not on the 40.

The previous reader was EasyOCR (CRAFT + `latin_g2`) fine-tuned on 1,294 real
crops and 20,000 synthetic crops. Mapillary self-labelling (passes 14–19)
produced no held-out gain and is closed. EasyOCR remains as
`LILLY_READER=easyocr`.

Labels never come from the model under training. Any reader that writes
labels goes through `app.ocr.read_regions`.

### 4.5 Speech synthesis

English: Kokoro-82M, stock. Bosnian: Piper `sr_RS-serbski_institut-medium`,
pulled at start, not bundled. Two trained Bosnian voices were refused (see
§5.4). A YouTube voice is staged.

---

## 5. Results

### 5.1 Translation

Measured 8 September 2026 on a Kaggle T4 after the ordinal-splitter fix,
through the served path, language tags stripped.

**Bosnian → English, 2,009 FLORES-200 pairs**

| | BLEU | chrF2 |
|---|---|---|
| Base, tag stripped | 41.77 | 67.66 |
| Lilly | **43.03** | **67.81** |
| Gap | +1.26 | +0.15 |
| p (paired bootstrap) | 0.001 | 0.074 |

On the 1,012-sentence **devtest** half, the product path is **43.25 BLEU /
68.10 chrF2** (base 42.08 / 67.85).

The base model prints `>>eng<<` (and variants) into 576 of 2,009 outputs
(28.7%). Lilly prints it into 0. That is the fine-tune's clearest single win.
chrF2 does not clear p = 0.05. BosnianBench term recall 91.7% → 92.2%
(p = 0.360) does not prove a Bosnian-specific gain on this direction.

The ordinal splitter used to cut *5. maja 1990. godine* into three pieces.
The fix is worth +0.89 BLEU / +0.37 chrF2 (p = 0.001, 167 of 2,009 rows).

**English → Bosnian.** Two columns, because they are not the same experiment.
Left: PyTorch base + LoRA, whole rows (the bars). Right: int8 through
`app.translate.Engine` (the product), measured 12 September 2026.

| | Adapter, whole rows | **Served: int8, app splitter** |
|---|---|---|
| Base | 29.57 / 58.96 | 31.23 / 60.93 |
| Lilly | **30.73 / 60.00** | **32.22 / 61.55** |
| Gap | +1.16 / +1.04 | +0.99 / +0.62 |
| 95% interval | [+0.67, +1.65] / [+0.69, +1.35] | [+0.47, +1.46] / [+0.33, +0.89] |
| p | 0.001 / 0.001 | 0.001 / 0.001 |

On the 1,012-sentence devtest half the served build is 31.49 / 61.03 →
32.45 / 61.75. Bosnian form rate: 94.3% → 99.2% (244 of 246 decided
targets). FLORES's Bosnian side is 49% Bosnian by lexical marker against
this project's train set at 77%; part of any chrF2 move measures which
standard the reference was written in.

**Against published systems (same 2,009 pairs, same loader, same sacrebleu)**

| Direction | System | Params | BLEU | chrF2 |
|---|---|---|---|---|
| bs → en | NLLB-200-distilled-600M | 600M | 36.49 | 63.80 |
| bs → en | Lilly base, untouched | ~230M | 41.77 | 67.66 |
| bs → en | **Lilly, shipped** | **~230M** | **43.03** | **67.81** |
| en → bs | NLLB-200-distilled-600M | 600M | 26.07 | 56.22 |
| en → bs | **Lilly, shipped (bars / product)** | **~77M** | **29.57 / 32.22** | **58.96 / 61.55** |

NLLB-200-3.3B is the best published chrF2 on this pair (67.2). Lilly's
devtest chrF2 is 68.10. The untouched Helsinki base already sits at that
tier. Google, DeepL, and the 3.3B NLLB were not run here, so this is not a
state-of-the-art claim. A 230M South-Slavic specialist is spending its
capacity on this family; NLLB splits 200 languages.

### 5.2 Speech

200 held-out FLEURS Bosnian clips, one scorer, one process:

| Listener | Word error | Bosnian term recall | Croatian substitution |
|---|---|---|---|
| whisper-small, stock | 38.5% | — | — |
| whisper-small, fine-tuned (gated) | **34.9%** | 68.2% | 3.3% |
| whisper-large-v3, fine-tuned (refused, shipped) | **11.9%** | 89.1% | 6.5% |
| same large ear, 925 clean FLEURS, product path (17 Sep) | **11.52%** (**16666 / 18836** heard right) | 88.5% | 6.25% (p = 0.0175 vs small 0.96%) |
| Human recordings | 11.7% | — | — |

11.9% on 200 is the product headline. 16666/18836 is the same listener on
all 925 clean clips; it does not replace 11.9%. An earlier instrument on
the 925 read 14.1% against the small's 39.5%. Commercial cloud APIs on
FLEURS Bosnian sit around 12–16.5% WER; 11.9% is in that band. The
Croatian-substitution row failed. That row stays on the card.

### 5.3 Photographs

| Reader | the 40 (found / invented) | test-v2, 132 photographs (found / invented) |
|---|---|---|
| EasyOCR, stock | 48.0% / 188 | 30.0% / — |
| EasyOCR, fine-tuned | 54.5% / 182 | 34.6% / 2,071 |
| **PP-OCRv6, stock, floor 0.9 (shipped)** | **67.0% / 65** | **57.8% / 450** |

Same shipped `app.ocr.scan`, 40 photographs split **before** inference
(17 Sep 2026):

| Mix | Score |
|---|---|
| **All 40 — why the product number is 67%** | 21 clean + **6 blurry** + **1 unreadable** + **12 empty** → **67.0%** (re-run 67.9% / 72 invented) |
| **Outdoor shots the person building this app actually takes** (normal and slightly blurry, still readable; 21 in this set) | **82.5%** (273/331), 53 invented |

67% includes empty frames and the one nobody can read. **82.5% is the same
reader on outdoor photographs a human can still read.** Neither number
replaces test-v2 (57.8% / 450). The project's own rubric needs ≥ 200
photographs with text before it will grade reading; test-v2 has 132, so
that grade is void.

The floor is a trade. On the 40, no floor is 67.7% found / 106 invented;
floor 0.9 is 67.0% / 65. On test-v2, no floor is 60.0% / 2,373; floor 0.9
is 57.8% / 450.

Bosnian Cyrillic is unreadable (Latin recogniser, no Cyrillic classes).
272 of 1,702 labelled crops (16.0%) are Cyrillic. Diacritics are sparse in
the labelled set (180/1,702 carry any; đ appears 8 times).

### 5.4 Voice

Heard through the shipped listener on the FLEURS prefix (200 clips, 167
sentences):

| Voice | Word error |
|---|---|
| Human recordings | 11.7% |
| sr_RS checkpoint (**shipped**) | **22.3%** |
| Control (Bosnian phonemes, gentle optimizer) | 25.5% (sound: within +4 of the checkpoint) |
| FLEURS voice | 53.9% — **refused** |
| Parliament voice (mean of 5 speakers) | 51.9% — **refused** |

The control shows the recipe can hold a voice it is handed. The two
Bosnian corpora did not.

---

## 6. Discussion

The translator that a user meets is already strong, and most of that
strength is Helsinki-NLP's. Fine-tuning removed a tag that appeared in
every third output, moved BLEU by a significant +1.26, and did not move
chrF2 past the bootstrap bar. On the reverse direction the product path
clears all four pre-registered bars and writes the Bosnian form of a
contested word 99.2% of the time.

The listener that a user meets is whisper-large-v3 at 11.9% word error,
shipped after it failed a Croatian-substitution gate. The gated
whisper-small at 34.9% remains the baseline that cleared the rule.

The reader that a user meets is untrained PP-OCRv6. Fine-tuning EasyOCR on
its own labels from Mapillary never moved the held-out score. The 40 and
test-v2 disagree enough that the 40 cannot be the only number.

Negative results, so they are not re-tried:

| Line | What happened |
|---|---|
| Mapillary / Panoramax vehicle capture | Near-zero usable words; structural to dashcam geometry |
| OCR self-labelling, passes 14–19 | Five sets, no held-out gain; labels from the model under training |
| FLEURS voice | 53.9% WER, refused |
| Parliament voice | 51.9% WER, refused |
| BosnianBench (bs → en) | +0.5 points, p = 0.360 |

---

## 7. Limitations

1. **The Bosnian-specific claim is not proven bs → en.** BosnianBench
   p = 0.360. The reply direction is where the form-rate claim holds
   (94.3% → 99.2%).
2. **The translation gain is concentrated in news.** SETIMES +3.05 BLEU;
   TED2020 −0.82 BLEU. Nothing here separates “better Bosnian” from
   “adapted to news.”
3. **Train pairs sit inside the base model's train data.** Read +1.26 BLEU
   as re-weighting, not new evidence.
4. **The photograph rubric is void** until ≥ 200 photographs with text
   exist. test-v2b (160 photographs) is waiting for two blind transcriptions.
   There is no labelled set of phone photographs from Bosnia.
5. **English-in is unmeasured.** The listener and reader accept English;
   no held-out English set has been scored.
6. **Bosnian Cyrillic is out of scope.**
7. **The spoken Bosnian voice is not Bosnian.** It is a Serbian-labelled
   Piper checkpoint trained on Sorbian recordings.
8. **One test genre per ability.** FLORES is even, professional prose.
   Real messages are not.

---

## 8. Availability

| What | Where |
|---|---|
| Code, keys, results | [`ssaaffaakk/Lilly`](https://github.com/ssaaffaakk/Lilly) |
| Weights | [`Safak11/lilly`](https://huggingface.co/Safak11/lilly) |
| Photographs and parallel sentences | [`Safak11/lilly-data`](https://huggingface.co/datasets/Safak11/lilly-data) |

`scripts/publish_to_hf.py` binds every weight file to a fingerprint. The
listener uploads only under a fingerprint named on the command line.
`scripts/pack_datasets.py` builds the five archives in §3.4.

Training runs on Kaggle GPU, launched from a clean GitHub SHA. Evaluation
scripts live in `training/`. Licences and notices: `models/lilly/NOTICE.md`.

---

## 9. Conclusion

Lilly is a small offline translator for one language pair, measured on the
path a user meets. Translation quality is largely the OPUS-MT bases, with a
fine-tune that stops the language tag leaking and a splitter that no longer
breaks Bosnian dates. Listening is a large Whisper the owner shipped after a
failed variety gate. Reading is stock PP-OCRv6, not a fine-tune of EasyOCR.
The numbers, the refusals, and the files they were measured on are public.

**Next measurements, not yet done:** two blind transcriptions on test-v2b;
a labelled set of Bosnia phone photographs; English-in scores; a Bosnian
voice if the staged YouTube run clears its bars.

---

## References

Tiedemann, J. and Thottingal, S. (2020). OPUS-MT — Building open translation
services for the World. *Proceedings of EAMT 2020*.

NLLB Team (2022). No Language Left Behind: Scaling human-centered machine
translation. *arXiv:2207.04672*.

Goyal, N. et al. (2022). The FLORES-200 Evaluation Benchmark for
Low-Resource and Multilingual Machine Translation. *Transactions of the
Association for Computational Linguistics*.

Kocmi, T. et al. (2024). Navigating the metrics maze: Reconciling score
magnitudes and accuracies. *ACL 2024*.

Federmann, C., Kocmi, T., and Xin, Y. (2022). NTREX-128 – News Test
References for MT Evaluation of 128 Languages. *Proceedings of the First
Workshop on Scaling Up Multilingual Evaluation*.

Radford, A. et al. (2022). Robust Speech Recognition via Large-Scale Weak
Supervision. *arXiv:2212.04356*.

Conneau, A. et al. (2023). FLEURS: Few-shot Learning Evaluation of Universal
Representations of Speech. *IEEE SLT*.

PaddlePaddle (2025). PaddleOCR PP-OCRv6. Apache-2.0.
<https://github.com/PaddlePaddle/PaddleOCR>.

Jaided AI. EasyOCR. Apache-2.0. <https://github.com/JaidedAI/EasyOCR>.

hexgrad. Kokoro-82M. Apache-2.0.

rhasspy. Piper voices (`sr_RS-serbski_institut-medium`). MIT.

---

*Last updated: 19 September 2026.*
