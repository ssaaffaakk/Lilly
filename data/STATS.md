# Data report — Phase 1

| Corpus | Raw pairs | Kept | Top reasons dropped |
|--------|-----------|------|---------------------|
| SETIMES | 138,387 | 137,529 | untranslated 523, duplicate 258, mostly_symbols 60 |
| TED2020 | 11,638 | 11,343 | duplicate 148, empty 119, length_ratio 26 |
| Tatoeba | 535 | 534 | duplicate 1 |
| WikiMatrix | 210,691 | 188,384 | untranslated 19,785, mostly_symbols 1,874, cyrillic 436 |

**Final dataset:** 337,790 clean pairs — train 334,790 / valid 1,500 / test 1,500

Filters: empty lines, >2.5x length-ratio mismatches, >250-word lines, Cyrillic on the Bosnian side (almost always Serbian), mostly-symbol lines, untranslated lines, exact duplicates.

Split seed 41, so the split is reproducible.
