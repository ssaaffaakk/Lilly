# Pass-19 — cleanest training set yet, still worse

Kaggle `afaksrmeli/lilly-ocr-pass19` v2, 2026-09-02. 713 Mapillary crops,
4 epochs, batch 8, 360 steps, no install gate. Weights in
`lilly-read-pass19-ungated.zip`.

## Scored on the GPU, against the reader it would replace

737 clean human crops — no Cyrillic, nothing under 16px, no fragments — both
readers in one process, on the machine that trained it:

| | exact | ignoring diacritics |
|---|---|---|
| shipped | 315/737 — 42.7% | 332/737 — **45.0%** |
| pass-19 | 308/737 — 41.8% | 329/737 — **44.6%** |

`delta: exact -7, folded -3`

## Every defect found along the way was fixed, and the answer did not change

The 713 crops are the most carefully built training set of the five:

- Cyrillic cities dropped — the recogniser has no Cyrillic in its character set
- dashcam OSD dropped, including the garbled forms (`053kih`, `NIC UFF /NU`)
- confidence at 0.30, not 0.85, per Noisy Student and Kahn et al.
- every label checked word-by-word against a 301k-form Bosnian lexicon,
  requiring five corpus occurrences, per Rijhwani et al.
- diacritics restored from that lexicon rather than duplicated
- crops under 16px tall or 32px wide removed — the defect that made
  `Coca-Cola` unreadable, 20.6% of the set

Five passes, five configurations, one answer:

| pass | training set | result |
|---|---|---|
| 14 | 915, conf ≥ 0.85 | folded 43.2 → 43.2 |
| 15 | 5,988, everything Latin | 43.2 → 43.2 |
| 16 | 2,968, diacritic-balanced | 43.2 → 43.2 |
| 17 | 898, lexicon-checked | 43.9 → 43.9 |
| 18 | same, gate skipped, installed and measured | −10 on 737 |
| 19 | 713, lexicon + size floor | **−3 on 737** |

## What is left

Not the data cleaning. Six things were wrong with the training set and all six
are fixed; the number did not move.

What has never been tested is whether these crops help on the text they
actually are. Both evaluation sets are Commons plaques, information boards and
street signs. The training corpus is shop fascias — `KONZUM`, `TISAK`,
`BAZAR`, 1.2 words a crop. There is no held-out set of shop signage anywhere
in this project, so "does Mapillary help read shop signs" has no measurement
behind it either way, and cannot get one without hand-labelling some.

The kernel's ERROR is `evaluate_ocr.py` failing afterwards on the photograph
report, downstream of everything above.
