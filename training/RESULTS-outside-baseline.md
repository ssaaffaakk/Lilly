# Lilly against an outside system — NLLB-200 on the same FLORES pairs

Run 7 September 2026 on a Kaggle Tesla T4, pre-registered in
`training/PREREGISTRATION.md` ("v3 — an outside comparison, written before any
outside number exists"). Raw machine output: `training/outside/report-raw.md`
and the two JSON files beside it.

**Why this run existed.** Every number this project had published compared Lilly
to *itself*. "42.14 BLEU on FLORES" had no scale attached: it could have been
strong for Bosnian or well behind something anyone can download, and the
repository did not know which.

## The numbers

`facebook/nllb-200-distilled-600M` and Lilly's bases, same 2,009 FLORES-200
pairs, same loader, same length-sorted batching, same sacrebleu call.

### Bosnian → English (the shipped direction)

| system | params | BLEU | chrF2 |
|---|---|---|---|
| NLLB-200-distilled-600M | 600M | 36.49 | 63.80 |
| Lilly's base, untouched | ~230M | **41.60** | **67.58** |
| Lilly, shipped (base + LoRA) | ~230M | **42.14** | 66.79 |

    Lilly shipped − NLLB   +5.65 BLEU   +2.99 chrF2
    Lilly's BASE  − NLLB   +5.11 BLEU   +3.78 chrF2

### English → Bosnian (the reply direction, not fine-tuned)

| system | params | BLEU | chrF2 |
|---|---|---|---|
| NLLB-200-distilled-600M | 600M | 26.07 | 56.22 |
| Lilly, shipped = untouched base | ~77M | **29.57** | **58.96** |

    Lilly − NLLB   +3.50 BLEU   +2.74 chrF2

## The anchor, which is what makes the above readable

NLLB ran on a T4 and Lilly's published numbers were measured on CPU. The speech
lane already recorded a full point of word error between those two kinds of
machine on identical clips, so the bases were re-scored on the same GPU in the
same run:

| | published (CPU) | this GPU | drift |
|---|---|---|---|
| bs→en base | 41.60 / 67.58 | 41.48 / 67.55 | **−0.12 BLEU**, −0.03 chrF2 |
| en→bs base | 29.57 / 58.96 | 29.57 / 58.96 | **−0.00**, +0.00 |

The hardware is worth at most a tenth of a BLEU point. The gaps above are 5.65
and 3.50 — **about forty times larger** — so they are not an artefact of where
each model ran.

## What this actually shows, which is not the headline

**Lilly beats NLLB-200-distilled-600M in both directions, against a model 2.6×
and 8× its size.** That is true and it is the first outside evidence this
project has ever had.

**And most of that margin is not Lilly's.** Read the two rows again:

| | over NLLB, bs→en |
|---|---|
| Lilly's untouched Helsinki base | +5.11 BLEU / **+3.78 chrF2** |
| Lilly's own fine-tune adds | **+0.54 BLEU / −0.79 chrF2** |

The win belongs mostly to **Helsinki-NLP's `opus-mt-tc-big-zls-en`**, which
Lilly builds on and did not train. On chrF2 the fine-tune is *behind* the base
it started from, so against NLLB the untouched base is the stronger model on
that metric and the shipped one is stronger on BLEU. In the reply direction
there is no fine-tune at all, so **100%** of the +3.50 is the base's.

Anyone quoting "Lilly beats NLLB" without that paragraph is quoting a number
this project did not earn.

**Which path these numbers are on, and why it matters.** Both Lilly and NLLB
were scored here on whole rows through `evaluate.py`, so the comparison is like
for like. But `training/RESULTS-devtest.md` says plainly that the whole-row path
"is a useful diagnostic and not what anyone runs" — the product splits sentences
and runs int8 through `app.translate.Engine`. On that served path, over the
1,012 FLORES devtest segments, the fine-tune reads **42.39 BLEU / 67.34 chrF2**
against a base at 41.10 / 67.51 tag-stripped: **+1.29 BLEU, −0.16 chrF2**. So the
fine-tune's chrF2 contribution is negative on both paths and the −0.79 quoted
above is the larger of the two. The direction of the finding does not change;
the size does, and the served figure is the one the project's own results file
says to quote.


**Why a small specialist beats a large generalist.** `opus-mt-tc-big-zls-en`
translates South Slavic into English and nothing else; NLLB-200 covers two
hundred languages in one set of weights. This is the ordinary shape of that
trade-off and it is the honest answer to "why is a 600M model losing to a 230M
one": it is not losing at translation, it is spending its capacity on 199 other
languages.

## Limits, stated because the result is flattering

- **This is the distilled 600M NLLB, not the 3.3B one.** `nllb-200-3.3B` is a
  different model and this run says nothing about it. The comparison chosen was
  the one that fits on the hardware the project uses.
- **Google Translate, DeepL and the large general models were not tested.**
  "Beats NLLB-600M" is not "state of the art", and nothing here licenses that
  sentence.
- **BLEU and chrF2 against Bosnian references reward a good Serbo-Croatian
  model**, which is the whole reason `docs/BOSNIAN_METRIC.md` exists. Neither
  number here says anything about whether Lilly writes better *Bosnian* than
  NLLB. `training/bosnian_form_rate.py` reads any en→bs system's output and
  could ask that; it has not been pointed at NLLB, and until it is, the question
  is open.
- **NLLB paraphrases more than FLORES's references do**, and BLEU punishes
  correct rewording. Its first bs→en output here was *"He added that we now have
  four-month-old non-diabetic mice who were previously diabetic"* against the
  reference *"We now have 4-month-old mice that are non-diabetic that used to be
  diabetic," he added* — the same sentence, most of the n-gram credit gone. This
  is why chrF2 is reported beside BLEU, and Lilly leads on both; but the BLEU
  gap is the more flattering of the two and should not be quoted alone.
- **One configuration each, fixed in advance.** No decoding knob was tuned for
  or against NLLB, and none will be tuned now that the ordering is known.

## What it changes

The README may now say where Lilly stands against something built elsewhere,
with the base's share of the credit named in the same breath. And the reply
direction's fine-tune has a clearer job: it starts from a base that is already
3.5 BLEU ahead of NLLB, so a fine-tune that merely holds that is not worth
shipping — the bar in the pre-registration (chrF2 above 58.96) is the right one
and this says why.

---

Reproduce (Kaggle, ~11 minutes of T4):

    .venv/bin/python3 scripts/kaggle_train.py outside-baseline --watch
    .venv/bin/python3 scripts/kaggle_train.py outside-baseline --fetch
