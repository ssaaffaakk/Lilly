# 27% of the human crop set cannot be read by a Latin-only reader

Measured 2026-09-02 on `data/ocr/crops/labels-human.tsv`, 1,701 crops.

| | crops | share |
|---|---|---|
| Cyrillic labels | 272 | 16.0% |
| digits or punctuation only | 100 | 5.9% |
| fewer than three letters | 188 | 11.1% |
| **measurable** | **1,241** | **73.0%** |

The Cyrillic rows are the ones that matter. `СЛУЖБИ`, `МИРА`, `ПРИПАДНИКЕ`,
`ЗНАК`, `СЛОБОДЕ`, `СЈЕЋАЊА` — the shipped reader is EasyOCR `latin_g2`,
whose 351-character set contains no Cyrillic at all. It cannot produce these
strings. They are not hard crops; they are impossible ones, and every score
computed over the full set has carried 16% guaranteed zeros.

The rest of the junk is milder but real: single digits, `3`, `2019`, and
crops cut through a word so the label is a fragment (`FORG` where the sign
reads FORGE, `MUSEU`, `kiselo` with the next word sheared off).

## What this changed

Pass-18's weights were compared against the shipped reader twice.

On 60 crops sampled from the whole file, pass-18 looked worse — 23/60 to
20/60 diacritic-blind. On 120 crops sampled from the measurable 73%, the two
readers are identical: 37/120 exact and 40/120 folded, both. The apparent
regression was noise from rows neither reader can do anything with.

So the verdict on the Mapillary training is unchanged and now agrees across
every measurement: no effect. The four gate refusals said the folded score
did not move; this says the same thing on a disjoint sample with the junk
removed.

## Where the reader actually stands

33.3% of measurable human-labelled crops read correctly with diacritics
folded. That is the honest number for this set, before and after pass-18.

## What to do about it

Scoring should exclude the Cyrillic rows rather than count them as failures —
they belong to a Cyrillic reader (`cyrillic_g2` ships with EasyOCR), and
routing by script before recognition is standard practice. Keeping them in
the denominator understates the reader by roughly 16 points and makes every
comparison noisier than it needs to be.
