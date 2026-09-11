# Is the training data actually Bosnian? — the result

Run 11 September 2026 on the Mac, no GPU, `training/audit_bosnian.py` over
`data/clean/train-mix.tsv`, `data/flores/dev.bs`, `data/flores/devtest.bs` and
`data/speech/train.tsv`. Nothing was trained and nothing was changed.

## The question

`training/RESULTS-outside-baseline.md` records the translation fine-tune at
**−0.79 chrF2** against NLLB-200 where the base supplies +3.78. The standing
hypothesis for that regression was contamination: published work puts the
precision of web-mined "Bosnian" language labels near 37%, most of the rest
being Croatian or Serbian, and if our corpus were mostly mislabelled then
fine-tuning on it would teach the model the wrong standard.

**The hypothesis is rejected.** The corpus is predominantly Bosnian.

## The judgment

A sentence votes only if it carries a word one standard uses to the exclusion
of the others; the shares below are of the voting sentences, not of the corpus.
Intervals are Clopper–Pearson 95%.

| | sentences | carry a marker | **Bosnian** | Croatian | Serbian |
|---|---|---|---|---|---|
| **training data** (`train-mix.tsv`) | 361,621 | 8,298 (2.3%) | **77%** | 12% | 11% |
| **FLORES bs**, dev + devtest — the benchmark | 2,009 | 68 (3.4%) | **49%** | 37% | 15% |
| FLEURS bs train — the listener's text | 3,091 | 113 (3.7%) | 50% | 29% | 20% |

Bosnian share, training data: **77%** (76–77). FLORES: **49%** (36–61). The
intervals do not overlap.

Per corpus, the training side:

| corpus | pairs | marked | Bosnian | Croatian | Serbian |
|---|---|---|---|---|---|
| SETIMES | 130,313 | 3,075 | **96%** | 2% | 2% |
| WikiMatrix | 145,057 | 3,079 | **72%** | 16% | 12% |
| wikimedia | 71,698 | 1,756 | **57%** | 21% | 20% |
| TED2020 | 10,022 | 216 | 37% | 6% | **56%** |
| ntrex | 4,047 | 165 | **67%** | 16% | 15% |

The 37%-precision figure describes corpora built by running a language
classifier over crawled web text. Ours are not that: SETIMES was edited into
separate Bosnian, Croatian and Serbian editions by its publisher, and the
Wikipedia-derived halves come from `bs.wikipedia`. SETIMES reads 96% Bosnian,
which is what an editorially separated corpus should read. The finding does not
transfer, and it should not have been carried over without this check.

## The part that was not expected

**The benchmark is less Bosnian than the data.** Every published translation
number in this repository is scored on FLORES, and FLORES's Bosnian side splits
roughly half Bosnian, a third Croatian.

That inverts the story about the −0.79. A fine-tune that moved the model toward
the standard of its training data — 77% Bosnian — would be marked down by a
reference set that writes *tisuća*, *tjedan* and *povijest* in a third of its
decidable sentences. On this evidence the regression may be the model becoming
**more** Bosnian, not less correct.

**That is a hypothesis, not a result.** It is not established here. Testing it
means scoring the fine-tune and the base separately on the Bosnian-marked and
Croatian-marked halves of FLORES and asking whether the fine-tune wins one and
loses the other. Sixty-eight decidable sentences cannot settle it; the test
needs a reference set built for the question, which is the same instrument
`docs/BOSNIAN_METRIC.md` already argues for.

What can be said without that: **chrF2 on FLORES does not measure whether the
output is Bosnian**, and no number in this repository has ever claimed it did.

## The tool, and a correction to it

`training/audit_bosnian.py`. Markers are whole words or explicit stems, three
lists, chosen to sit outside the standard they are not assigned to. Bosnian's
list is the shortest of the three — *historija* (Croatian *povijest*, Serbian
*istorija*), *sedmica* (*tjedan*, *nedelja*), *kahva* (*kava*, *kafa*), and the
retained /h/ of *lahko* and *mehko*. That brevity is the language: Bosnian
accepts a wider range of forms than either neighbour, so fewer words are
exclusively its own.

**The first version of this audit was wrong and its numbers are void.** It
matched stems, and three of them misfired:

| stem | intended | actually matched |
|---|---|---|
| `beo` | Serbian *beo*, Bosnian *bijel* | **Beograd**, 3,400 times |
| `vremen` | Serbian *vreme* | **vremena** — ijekavian inflects the same way |
| `vlak` | Croatian *vlak*, train | **vlakna**, fibres, shared by all three |

Those three carried about half of the "Serbian" and "Croatian" counts, and the
first run read 45% Serbian. The corrected run reads 11%. The stems are now
whole words wherever the ekavian/ijekavian split does not survive stemming.

What remains is checkable by eye: the commonest Croatian hits are *unatoč*,
*tijekom*, *zrakoplovstvo* and *točka*; the commonest Serbian are *uopšte*,
*gde*, *istorija*, *vreme* and *posle*. The commonest Bosnian are *sedmica*
(2,570) and *historija* (~3,000). `otok` is the weakest entry kept — Croatian
for island, but also swelling in all three — and it accounts for **236 of the
1,033** Croatian marker hits. Dropping it would move the Croatian column down,
not the Bosnian one up past where it already is.

## Limits

- 2.3% of sentences are decidable. The other 97.7% are the same in all three
  standards, which is the ordinary case and not a defect in the data.
- The shares are of that 2.3%. They are representative only if markers appear
  independently of a sentence's standard, which is assumed, not shown.
- The FLORES figure rests on 68 sentences. Its interval is 25 points wide.
- The audit reads the Bosnian side only. It says nothing about translation
  quality, only about which standard the text is written in.

## Reproduce

```bash
python3 training/audit_bosnian.py data/clean/train-mix.tsv
python3 training/audit_bosnian.py data/flores/dev.bs data/flores/devtest.bs
```
