---
name: lilly-community-research
description: >-
  Lilly community-research skill. Community-sourced (Reddit) fixes, tools and
  failure-mode warnings for Lilly's four components: Whisper ASR (Croatian
  drift, KenLM rescoring, low-resource fine-tuning), MarianMT translation
  (chrF2 plateau, BCS lexicon drift, glossary/constrained decoding), OCR
  (invented words, Cyrillic blindness, diacritics, synthetic-to-real gap) and
  Piper TTS (voice ceiling, phonemizer/espeak problems). Use when working on
  any of those four components, or when the user mentions Croatian drift,
  invented OCR words, Cyrillic OCR, diacritics, rs_cyrillic, KenLM rescoring,
  whisper speech improvement, language-token mislabelling, low-resource NMT,
  backtranslation, glossary injection, constrained decoding, chrF not moving,
  Piper voice quality, TTS espeak phonemes, or "community tips". Provides
  concrete, measured-mistake-mapped techniques with Reddit sources.
---

# Lilly Community Research — community-sourced fixes, mapped to our measured mistakes

A permanent index of Reddit-sourced techniques for Lilly's four components. Built
13 Sep 2026 from a deep Reddit research pass. Each reference maps every technique to
the measured failure it fixes, and each has its provenance in `references/sources.md`.

> **GATE RULE — read first.** This skill is *advice*, not a verdict. No technique in
> here ships, changes a pre-registered number, or becomes a new baseline unless it runs
> the same pre-registered gate as any other arm:
> `PREREGISTRATION.md`, score through `app.translate.Engine` / `app.ocr.scan`, on the
> held-out rulers, never past a metric you carved for it. The proper shape of every
> idea here is: *new arm in `training/PREREGISTRATION.md`, run it, write RESULTS, then
> judge.* If a technique below contradicts a result file already in the repo, **the
> result file wins until a new arm beats it at a gate.**

## When to use

- Working on any of: speech listener (Whisper), translation (Marian/opus-mt, en↔bs),
  camera reader (OCR), or TTS voice (Piper).
- User mentions: Croatian drift, invented OCR words, Cyrillic, diacritics, KenLM,
  language-token mislabelling, chrF plateau, glossary/constrained decoding, espeak,
  WER not improving, "community tips", "what does the community do".

## When NOT to use

- Nothing here replaces `docs/*` (what-moves-the-model, diacritic-gate-literature,
  OCR-ROADMAP, HANDOFF.md). The repo docs are the settled record; this skill is the
  un-assessed idea shelf.
- If a technique contradicts a repo result file, the result file wins (see gate rule).
- Do not cite a Reddit thread as proof for a claim about Lilly's own numbers. Threads
  are evidence that *someone* tried the technique; our numbers decide if it works here.

## Known mistake → candidate fix map (the whole shelf at a glance)

| Component | Measured mistake / number | Candidate fix (details in reference) |
|---|---|---|
| Speech | Croatian substitution 1.1%→6.1%, large-v3 | **KenLM/Ijekavian n-gram rescoring** — no retrain |
| Speech | Croatian spelling via decode (vreme/dete) | `initial_prompt` + `suppress_tokens` on Croatian forms |
| Speech | `<bs>` token in front of Croatian text (root cause) | Force `language=bs` + `task=transcribe` per segment |
| Speech | drift cascades window→window | `condition_on_previous_text=False` + VAD gate |
| Speech | small-data fine-tune regressed WER | 1000–3000 steps, checkpoint-per-bucket eval, bs holdout |
| Speech | decoder picked up hr prior (11h bs vs 91h hr) | freeze encoder / tune only decoder (Distil-Whisper lesson) |
| Speech | appended hr data hurt old perf | one consolidated pass, bs upweighted, bs holdout stops |
| Translation | fine-tune does not move chrF2 (−0.16 tie) | data must be genuinely out-of-corpus; re-check overlap |
| Translation | Croatian lexicon losses (vlak/voz, travnja/aprila…) | glossary injection + constrained decoding at inference |
| Translation | Croatian-lexicon drift is "dictionary not GPU" | zero-training decode bias against hr pairs (lever #2) |
| OCR | 450 invented words at floor 0.9 | word-level gating + two-engine agreement + charset whitelist |
| OCR | Cyrillic unreadable (272/1,702 crops) | route crops to `rs_cyrillic` / `cyrillic_g2`, then transliterate |
| OCR | đ = 8 real examples (weak diacritic column) | synth đ/d *pair* crops, sign-font family, diacritic-clear fonts |
| OCR | 100% on postcards, 53% on real photos | synthetic backbone → real-data fine-tune; measure camera noise |
| OCR | detector false positives (39 empty crops) | box textness/score filter + glare pre-processing |
| TTS | all voices ~52% vs sr_RS 22.3% | treat TTS source as eval confound; eval buckets per phoneme set |
| TTS | espeak sr phonemes on 72% of bs utterances | verify `espeak.voice` + base model; phoneme-cache diagnostic |
| TTS | ~52% ceiling / mel plateau | grapheme-VITS run skips espeak (bs is phonemic) |

Every one of those has a Reddit thread in `references/sources.md`.

## Files

- `references/speech-asr.md` — Whisper, Croatian drift, KenLM, low-resource recipe
- `references/translation.md` — NMT, chrF2, back-translation, lexicon/constrained decoding
- `references/ocr.md` — invented words, Cyrillic, diacritics, synthetic-to-real
- `references/tts.md` — Piper voice, espeak phonemes, eval confounds
- `references/sources.md` — every Reddit URL with the finding it supports