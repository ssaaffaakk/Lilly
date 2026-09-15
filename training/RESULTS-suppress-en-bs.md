# RESULTS — decode-time Croatian suppression (§2) — 15 September 2026

A candidate lever raised in `docs/REPORT-what-would-raise-the-numbers-2026-09-13.md`
and again in the 14 Sep strategy note: the base model's Bosnian-form errors are a
short static list of Croatian words (*Točno, travnja, lipnja, tisućama, vlak,
Europe, sudjelovati*, `training/RESULTS-en-bs-formrate.md`), so pass their token
sequences into CTranslate2's `suppress_sequences` at decode time and the decoder
is forced to the Bosnian alternative — no training, zero GPU, in the served path.

Measured, not assumed. `training/measure_suppress_en_bs.py` reuses the committed,
pre-registered form-rate machinery (`training/bosnian_form_rate.py`: same bench
cases, same audit filter, same Bosnian/variant/silent matcher) and drives it
through `app.translate.Engine` (int8 CTranslate2 4.8.1, the path a user meets),
which is where `suppress_sequences` applies. 338 surviving targets, 308 unique
English sources, each translated OFF and ON.

The suppression list is the **oracle upper bound**: every Croatian counterpart
the surviving targets weigh against (30 words → 54 token-sequences, nominative
plus case folds). A deployable fixed list cannot beat knowing every counterpart
in advance, so this bounds §2 from above.

## Result

| | served (LoRA) OFF | ON | base (untuned) OFF | ON |
|---|---|---|---|---|
| form rate | 99.6% | **99.6%** | 94.6% | **95.0%** |
| bosnian | 260 | 260 | 246 | 247 |
| variant | 1 | 1 | 14 | 13 |
| silent | 77 | 77 | 78 | 78 |
| chrF2 (bench) | 63.25 | 63.25 | 62.73 | 62.73 |
| outputs changed | — | **0 / 308** | — | 1 / 308 |
| variant→bosnian | — | **0** | — | 1 |
| bosnian→variant | — | 0 | — | 0 |

## Reading it

**On the shipped build §2 does nothing.** Zero of 308 outputs change; form rate
holds at 99.6%. The reason is direct: the LoRA already writes the Bosnian forms —
*Voz* not *vlak*, *Evropu* not *Europu*, *Hiljadu* not *tisuću*, *Fudbal* not
*nogomet* — so there is no Croatian token left for the decoder to be steered off.
The fine-tune already did, generally, what §2 proposed to do by a rule.

**Even on the untuned base the ceiling is +0.4 form-rate points.** With every
counterpart suppressed, exactly one of fourteen variant errors flips to Bosnian
(94.6% → 95.0%), at **zero chrF2 change and zero bad flips**. The other thirteen
do not flip: forcing off one Croatian token most often yields another Croatian
inflection or a form the strict matcher cannot decide, not the Bosnian target.
Suppression removes a specific string; it does not install Bosnian lexical
identity, which is the same thing bar 4 of the v9 gate found about the label.

**So §2 is not worth shipping.** It is harmless (no chrF2 cost, no bad flips) but
pointless on the product: the served build is already at the ceiling suppression
could reach, and reaches it more generally than a static list can. The lever the
document credited to §2 is spent inside the fine-tune. No flag is added to the
app.

Numbers reproduce with `.venv/bin/python3 training/measure_suppress_en_bs.py`
(CPU, ~4 min, no GPU). The served form rate here (99.6%, 261 decided) is the
int8 CT2 path on the full 338-target bench; the pre-registered 99.2% (246
decided) is the PyTorch-adapter path on its audited subset — same story, both
paths at ~99%.
