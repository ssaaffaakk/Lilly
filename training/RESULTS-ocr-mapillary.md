# Mapillary crops do not move the reader — four passes, one answer

Passes 14, 15, 16 and 17, 2026-09-02, all on Kaggle T4, all refused.

## The result is the same every time

Word accuracy on the 132 held-out real crops, diacritics folded — the number
that matches the product bar:

| pass | training data | crops | before | after |
|---|---|---|---|---|
| 14 | conf ≥ 0.85, hand-filtered | 915 | 43.2% | **43.2%** |
| 15 | every non-Cyrillic crop | 5,988 | 43.2% | **43.2%** |
| 16 | conf ≥ 0.30, diacritic-balanced | 2,968 | 43.2% | **43.2%** |
| 17 | lexicon-checked, diacritics restored | 898 | 43.9% | **43.9%** |

Four datasets spanning 898 to 5,988 crops, four different label-selection
policies, from my own judgement through to the literature's. Zero of the 132
held-out crops changed in any of them.

## The model learns; it just learns something else

Pass-17's loss fell from 2.14 to 0.02 over 339 steps. That is memorisation of
898 crops, and the weights genuinely changed — the strict score moved 42.4% to
40.9%. So the training worked and transferred nothing.

The one measurable effect on the held-out set was diacritics: Bosnian letters
64.0% → 60.0%, in every pass, while the folded word score stayed still. The
only decisions the training flipped were the marginal ones, and on this data
the marginal decision is whether to write ć or c.

## Why: the two sets are different kinds of text

| | content | words per crop |
|---|---|---|
| training (Mapillary) | `kuca`, `CITY`, `SHOP`, `BAZAR`, `KONZUM` — shop fascias | 1.2 |
| held out | `PROSVJEĆIVANJU MLADIH NARAŠTAJA`, `KRIVIČNO JE DJELO`, `Uslijed posljednjih ratnih dešavanja.` — memorial plaques, legal notices | 1.9 |

Display lettering on a shop front and engraved text on a plaque are different
recognition problems. Getting better at one need not touch the other, and
measurably did not.

## What was ruled out along the way

- **Label quality.** Pass-17's labels were checked word by word against a
  301k-form lexicon and had their diacritics restored from it. Same result as
  pass-15, which took every crop unfiltered.
- **Quantity.** 5,988 did no better than 898. Nothing scaled.
- **The diacritic rate.** Pass-16 balanced it to the human 10.6% by
  duplication, pass-17 fixed it at source by restoration. Neither moved the
  word score.
- **The gate.** It was folded to score diacritic-blind on 2026-09-02, before
  passes 16 and 17, so none of these refusals is the old hat-sensitive gate.
- **Learning rate.** Loss reaches 0.02. The optimiser is not the constraint.

## What is left

Not another pass on this data. The question it answers has been answered four
times.

The honest options are about domain, and the choice is the product's, not the
model's:

1. **If the product photographs shop signs**, the held-out set is measuring
   the wrong thing. It is 132 Commons crops of plaques and information boards.
   Building a held-out set of hand-labelled shop signs would tell us whether
   these 898 crops help at all — right now we cannot see it either way.
2. **If the product photographs plaques and boards**, then Mapillary is the
   wrong corpus and the training data should come from the same place the
   held-out crops do.

The 20,240 photographs and 13,144 crops are kept and credited either way.
