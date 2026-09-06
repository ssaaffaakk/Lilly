# Fine-tuning the PP-OCRv6 recogniser on the Commons crops — one look, both bars fail

Pre-registered before the run in `training/PREREGISTRATION.md`, "step 7,
PP-OCRv6 recogniser fine-tune". The recogniser (`PP-OCRv6_medium_rec`) was
fine-tuned on 1,071 blind Commons crop lines (371 held out for model
selection, split by source photograph so no crop's photograph is in either
test set), trained on Kaggle's T4 (run `20260906-2143`), exported to inference
format, and scored **once** through the app's own door (`LILLY_PADDLE_REC_DIR`)
at the shipped floor 0.9 — not re-swept. Nothing here touches `models/lilly/`
or the app; the weights are a candidate, not a release.

## The decision, on test-v2

280 photographs, 132 with legible text, held out entire — no reader in this
project has trained on any of it. Paired photograph by photograph against the
shipped configuration (`training/paddle-floor/test-v2-floor0.9.json`): same
detector, same floor, only the recogniser differs.

| arm | words per photograph | pooled | invented | diacritic | folded |
|---|---|---|---|---|---|
| shipped `aea5890a` | **57.8%** | 64.8% | 450 | 62.1% | 81.8% |
| fine-tune `e939d2a3` | **60.8%** | 67.9% | 703 | 77.1% | 89.3% |

Paired per photograph (n=132): mean Δ **+3.0 points**, 95% **−1.3 to +7.2**,
35 up, 14 down, bootstrap p 0.164.

**The two pre-registered bars:**

- **words found per photograph rise, 95% interval excluding zero — does not
  hold.** The point estimate rose 3.0 points, but the interval spans zero
  (−1.3 to +7.2): the rise is not distinguishable from none on 132
  photographs.
- **invented words ≤ 450 — does not hold.** The fine-tune returned **703**
  words that are on no sign — 253 more than the shipped reader.

Neither bar holds. **By the pre-registered rule the fine-tuned recogniser does
not ship; the shipped configuration stays.**

## What the fine-tune actually changed

It learned the diacritics it was trained on: diacritic-word recall rose 62.1%
→ 77.1% (+15 points) and the diacritic-blind figure 81.8% → 89.3% (+7.5). It
found more real words on 35 photographs and fewer on 14. But at the shipped
floor 0.9 it keeps more low-confidence regions, and 253 of the extra words it
keeps are invented — the cost the per-photograph figure and the confidence
interval both absorb. The recogniser is reading the letters better; the words
per photograph do not clearly follow, and the invented count moves the wrong
way. The floor was not re-swept, disclosed in the pre-registration as a
possible disadvantage for a model whose confidences are shaped differently —
and it is: the extra invented words are exactly what a floor tuned to this
recogniser would be asked to cut.

## The 40, reported beside — decides nothing

The 40 exhaustively-transcribed Commons photographs (28 with legible text),
reported for continuity; the pre-registration says this set decides nothing.

| arm | words per photograph | pooled | invented | diacritic | folded |
|---|---|---|---|---|---|
| shipped `aea5890a` | **67.0%** | 69.4% | 65 | 68.0% | 92.0% |
| fine-tune `e939d2a3` | **68.2%** | 71.0% | 55 | 80.0% | 84.0% |

Paired per photograph (n=28): mean Δ +1.2 points, 95% −1.4 to +4.5, 5 up, 4
down, bootstrap p 0.432. The rise does not hold; invented ≤ 65 holds (55).
Here diacritic recall rose (68 → 80) while the diacritic-blind figure fell
(92 → 84) and it invented ten fewer words — a different balance on a small,
dense set, and not the decider.

## What this closes and what it does not

The pre-registration is spent: no second run, no second floor, no second
checkpoint after the number. On these 1,071 lines the recogniser is not the
stage to train. The clear diacritic gain against a per-photo rise the interval
cannot separate from zero, together with the higher invented count, points
where the pre-registration already said the next move lies: more labelled
lines from the test-v2 pool remainder (206 undrawn `keep` photographs and 318
one-region ones, two blind passes), or the detector's small-type misses (step
3 found PP-OCRv6's misses concentrate on few-pixel type) — each its own
pre-registration.

The candidate weights stay on the public dataset
`Safak11/lilly-ocr-paddle-runs/20260906-2143` (the inference zip and the crop
gate). They are not in `models/` and the app does not load them.

---

Reader identities: shipped `aea5890abfdb7ba3`
(`paddle:PP-OCRv6_medium_det+PP-OCRv6_medium_rec:3.7.0:rec>=0.9`), fine-tune
`e939d2a3c9fe2913`
(`paddle:PP-OCRv6_medium_det+PP-OCRv6_medium_rec@87ad04a5:3.7.0:rec>=0.9`).
The crop gate on the 371 validation crops, one process, stock against
fine-tuned (`Safak11/lilly-ocr-paddle-runs/20260906-2143/crop-gate.json`):
exact 264 → 278 (+3.8 points, 95% −0.5 to +8.1), folded 290 → 287 — the gate
passed (exact rose) and let the model be measured; it decided nothing, as
pre-registered. Raw scored summaries in `training/paddle-finetune/`; the
per-photograph reader output caches (`reader-output-paddle-v6-tuned.json`) are
regenerable scratch and not committed.
