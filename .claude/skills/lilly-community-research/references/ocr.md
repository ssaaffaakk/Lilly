# OCR — reader: invented words, Cyrillic blindness, diacritics, synthetic-to-real

Community-sourced (Reddit) techniques for the camera reader. Sources in `sources.md` by `[O#]`.

## Context (the numbers to fix)

- Shipped reader PP-OCRv6 at confidence floor 0.9: **57.8% word accuracy but 450 invented words** on
  `test-v2` (132 held-out Commons photos). EasyOCR fine-tune reads 34.6%.
- **Cyrillic unreadable**: Latin-only recogniser (`app/ocr.py`, `easyocr.Reader(["bs","en"])`); 272 of
  1,702 crops (16.0%) are Cyrillic; rare letters barely in data (Џ 0, Ђ 3, Љ 3, Ћ 6, Њ 14, Ј 39). Safe
  direction is Cyrillic→Latin (unambiguous; no digraphs). Per `docs/OCR-ROADMAP.md` and
  `docs/diacritic-gate-literature.md` §5 (script routing is standard; EasyOCR raises on mixing scripts
  in one Reader).
- **Diacritics weakest column**: only 180/1,702 crops carry č ć đ š ž; **đ = 8 examples** (1 lower + 7
  Đ). Synthetic crops are the only teacher for these letters.
- Synthetic→real gap: synthetic distribution scores ~100% on flat postcards; real averages 53%; crop-level
  16.3% exact.
- Detector produced 39 "no text" false-positive crops.
- Postcard anomaly: the only perfect 9/9 item is flat, high-contrast, axis-aligned type = synthetic
  distribution scoring 100% inside the real set.

## Findings

### O1. "OCR accuracy is a system problem, not a model" — the floor isn't the lever — `sources.md [O1]`
Confidence *floor* alone still emitted 450 inventions. System-level gates: format checks, plausibility
checks, reject whole-line/histogram anomalies. For Lily: require word to parse as plausible Bosnian
(letter-frequency/vowel sanity) so gibberish is flagged, not emitted.

### O2. Word-level (not line-level) confidence bucketing — `sources.md [O2]`
Per-word confidence gating with human-review escape hatch on low-confidence words. Hallucinated words
are typically uniformly-mediocre *across* the whole token (one bad char ≠ hallucination). Emit clean
words, flag/drop uniformly-low words.

### O3. Two independent engines = hallucination detector — `sources.md [O3]`
Two engines agreeing → accept; disagreeing → reject/reroute. We already have EasyOCR fine-tune sitting
beside PaddleOCR. If PaddleOCR claims a word and EasyOCR disagrees → strongest free invention signal.
No training.

### O4. LLM post-correction — correct in place, NEVER regenerate — `sources.md [O4]`
Structured output (correct vs keep): LLM only fixes obvious letter-substitution/OCR errors against a
wordlist, returns the rest verbatim. Unconstrained LLM becomes a *second* hallucinator — the golden rule.
Pairs with backlog idea in `docs/OCR-ROADMAP.md` post-OCR stage.

### O5. Two separate thresholds: recognition score AND per-character confusion — `sources.md [O5]`
"Turn up score, turn down confusion until you don't pass labels with errors." Argmax still forces a glyph
when per-char confidence is genuinely low. Lower the *confusion* threshold → reject confusable matches
(đ vs d, č vs c, Ћ vs Ђ) instead of silently guessing. Explains how 450 inventions slip through a 0.9 floor.

### O6. Confusable pairs need *pair-crop* data, not a threshold — `sources.md [O6]`
PaddleOCR on license plates misreads D/0, S/5 — same failure class as đ/d, č/c. No threshold fixes a model
that can't see the mark. Synthesize the *pair* deliberately: minimally-edited crops differing only in the
stroke. đ=8 examples → synthesize đ/d as a pair, not đ alone.

### O7. Character whitelist = inventions structurally impossible — `sources.md [O7]`
Constrain the output alphabet to exactly the Bosnian charset (`a–ž č ć đ š ž 0-9`). Whitelist also forces
diacritic letters *back* into the allowed set so they aren't silently mapped to plain letters (đ→d). Double
win against inventions AND diacritics.

### O8. Cyrillic: route, don't retrain — we already own the recognizers — `sources.md [O8]`
PaddleOCR ships `rs_cyrillic` and `rs_latin`; EasyOCR ships `cyrillic_g2` (`rs_cyrillic` in its language
list). Script-detect first, route Cyrillic crops to `rs_cyrillic`, then transliterate Cyrillic→Latin (the
safe direction per `diacritic-gate-literature.md` §5). Zero-training fix for 16% of crops. Requires the
second Reader `app/ocr.py` currently refuses to build — that's the documented architectural gap.

### O9. Single multi-script recognizer as an alternative to routing — `sources.md [O9]`
Turbo-OCR: now Latin + Cyrillic (+ CJK/Ar) in one recognizer. Keeps mixed-script signs (Latin+Cyrillic on
one line) from being mangled by a router.

### O10. Synthetic augmentation must match the *target font* — `sources.md [O10]`
Transform clean text into images in the **same font** as the real images, then apply image noise. For đ:
render đ-words in the actual street-sign font family + sign-like noise so the mark is in-distribution.

### O11. Font choice determines diacritic survival — `sources.md [O11]`
Some fonts collapse accent marks (đ→d) at render resolution; some render them cleanly. Audit the synthetic
font list: bias toward fonts where č ć đ š ž render distinctly (and Cyrillic hooks stay visible). Cheap
and direct.

### O12. Synthetic backbone → real-data fine-tune — `sources.md [O12]`
Pre-train the backbone on synthetic, then fine-tune (at least the heads) on real data. Explains the
100%-postcard / 53%-real cliff: synthetic teaches the char table, real teaches the distribution. Targets
the 16.3% crop-level collapse; real 1,702 crops are the fine-tune set.

### O13. Fake noise is fake — replicate the actual camera — `sources.md [O13]`
Measure and replicate the phone camera's blur/JPEG/light function into the renderer (or paste real
background crops) so diacritics and Cyrillic degrade the same way as in the real 53% block.

### O14. Detector: score-filter boxes + pre-process glare — `sources.md [O14]`
EasyOCR uses CRAFT (better than connected-components); glare over text hampers any detection → pre-process
glare/sheen before recognition, or discard the box. PaddleOCR exposes per-box score — filter by textness.
Companion preprocessing rule (r/computervision [O14b]): deskew, crop, denoise, then **stop if it reads
cleanly** — over-thresholding wrecks OCR. Attacks our 39 "no text" false-positive crops.

## Recommended order

1. O8 `rs_cyrillic` routing (kills 272-crop blindness, zero training) — pairs with `docs/OCR-ROADMAP.md`.
2. O2 word-level gating + O3 two-engine agreement + O7 charset whitelist (kill the 450 inventions).
3. O1/O5 rejection instead of forced argmax.
4. O12 synthetic-backbone/real-fine-tune + O13 real camera noise = the structural fix for the 53% gap.
5. O10/O11 đ-pair synthesis in sign fonts = the diacritic column.
6. O4 LLM post-correct as the final glue — only after O2–O3 are in.