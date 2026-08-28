# The Cyrillic recogniser — built, measured, left off by default

The gap it addresses is real and structural: `latin_g2` has 351 output classes,
none Cyrillic, so 7.0% of the scored answer key's words (across 7 of the 40
photographs) and 16.2% of the hand-transcribed crops were unreachable by any
amount of Latin fine-tuning.

The fix built here: a second recogniser (`cyrillic_g2` via easyocr's
`rs_cyrillic`, all twelve of Ђ Ј Љ Њ Ћ Џ and lower case verified present),
sharing CRAFT detection — the expensive half — so only recognition runs twice
over the same boxes. It reads Cyrillic properly where the Latin model produced
gibberish: `Сарајево` where before `Capajebo`, `АОБРО АОШАИУ БОСНУ` where
before `AOEPO AOWAMY EOCHY`.

And it is **off by default**, because on the 40 scored photographs it made the
product worse at every setting tried.

## Choosing between the two readers by confidence — worse at every margin

One capture of both readings per region, margins scored offline from it:

| margin | pooled | per-photo | diacritic | folded | spurious |
|---|---|---|---|---|---|
| **latin only (shipped)** | **45.0%** | **54.7%** | 44.0% | 60.0% | **180** |
| 0.00 | 40.8% | 50.2% | 44.0% | 56.0% | 177 |
| 0.05 | 41.8% | 51.5% | 44.0% | 56.0% | 177 |
| 0.10 | 42.9% | 52.2% | 44.0% | 60.0% | 171 |
| 0.20 | 43.2% | 51.9% | 44.0% | 60.0% | 169 |
| 0.30 | 44.0% | 52.6% | 44.0% | 60.0% | 169 |
| 0.50 | 45.0% | 54.4% | 44.0% | 60.0% | 172 |

The best margin is the one that never picks Cyrillic. The Cyrillic model reads
Latin text as confident lookalikes (СТОР for STOP), so confidence cannot
arbitrate: it displaces more correct Latin readings than it rescues Cyrillic
ones.

## Emitting both readings instead of choosing

| rule | pooled | per-photo | spurious |
|---|---|---|---|
| latin only (shipped) | 45.0% | 54.7% | **180** |
| union: always emit both | 46.9% | 57.2% | 546 |
| union if Latin weak (<0.5) | 46.1% | 55.8% | 320 |
| union if Cyrillic confident (>0.5) | 46.6% | 57.0% | 246 |
| union if Cyrillic confident and mostly-Cyrillic glyphs | 46.6% | 57.0% | 253 |
| swap when Latin weak and Cyrillic stronger | 43.7% | 52.4% | 170 |

The best honest trade on offer is +2.3 points of per-photo recall for +37% more
invented words (180 → 246). For an app that translates what it reads, an
invented word becomes an invented sentence, and the user cannot tell which
words are real. That trade is the wrong way round, so it does not ship.

## What this negative result actually says

Not that the second recogniser is wrong — that **arbitrating between two
untrained models at read time is not a substitute for training**. The move that
took the Latin reader from 36.0% to 54.7% was fine-tuning on 1,294 real crops.
The corresponding move here is fine-tuning `cyrillic_g2` on the 276 Cyrillic
crops already transcribed — those exist, blind-labelled, ready. Until that run
happens, the two-reader plumbing stays in `app/ocr.py` behind
`LILLY_READER=cyrillic`, and the shipped reader is the Latin one whose numbers
above are the shipped numbers.

Sweep artefacts: `/tmp/margin-capture.json` (one capture of both readings for
all 40), `margin-sweep.log`, `/tmp/rule_sweep.py`.

---

## Correction, 28 August 2026 — the crop count

This page says the corresponding move is "fine-tuning `cyrillic_g2` on the
**276** Cyrillic crops already transcribed". The count is **272**, re-measured
from `data/ocr/crops/labels-human.tsv` three ways with no disagreement between
them: labels containing any Cyrillic letter (272), labels entirely Cyrillic
(271), and labels containing any non-Latin letter at all (272, with zero labels
non-Latin without being Cyrillic).

Nothing else on this page moves — every figure in the sweep tables has crops or
words as its denominator, not this count.

Two things about that set, measured since and relevant to anyone acting on the
recommendation above:

- The 272 come from **37 photographs**, and `Information_board_in_Jajce` alone
  supplies 97 of them — 35.7%. A train/valid split **by crop** would put the
  same board on both sides. Split by source photograph.
- **`Џ` has zero examples** in the transcribed set, and `Ђ` and `Љ` have three
  each. The sentence higher up this page — "all twelve of Ђ Ј Љ Њ Ћ Џ and lower
  case verified present" — is about `cyrillic_g2`'s **output classes**, not about
  our training data. Both statements are true and they are about different
  things; read together carelessly they would send a fine-tune to train a class
  with no examples in it.

Full working in `training/RESULTS-ocr-dataset.md`.
