# TTS — Piper voices: the ~52% ceiling, espeak phonemes, eval confounds

Community-sourced (Reddit) techniques for the speak component. Sources in `sources.md` by `[P#]`.

## Context (the numbers to fix)

- All trained Bosnian voices land near **52%** word-error through the shipped listener (v5 BS 53.9%,
  v6 parla 51.9%), vs the sr_RS voice at **22.3%** and human recordings at **11.7%**. sr_RS stays; no
  voice zip ships; control run (v7) showed "the pipeline is not broken; the phonemizer is cleared".
- espeak reproduces the checkpoint voice's own phonemes on 72% of utterances with `sr` and 4% with `bs`;
  both refused-lines were trained with `bs` phonemes.
- Data: 3h/speaker FLEURS and parliament lines, ~8h YouTube (7 speakers); FLEURS is 16 kHz; block/refused
  at multiple gates. `docs/speak-checkpoint-comparison.md` concluded Bulgarian/Slovenian warm-start
  trade-offs; YouTube line closed (0/20 lectures reach 11 kHz).

## Findings

### P1. ⚠️ The 52% may be a *measurement artifact*, not a bad voice — `sources.md [P1]`
Same ASR, same reference text, two TTS generators → **20.9% WER (espeak-ng audio) vs 4.65% WER (gTTS
audio)**, consistent across recognizers. The confound lives in the *audio generator's prosody/spectral
domain distance from what the ASR trained on*. espeak-ng's real role is the front-end (phonemizer), not
a synthesizer.
- Fixes the conclusion "my voices are bad": before any retrain, gate on same-text/same-batch/
  swap-only-TTS-source comparing candidate vs sr_RS. If the 30-point gap shrinks/vanishes, the model is
  fine and the *evaluation* is biased. This is an eval-metadata fix that replaces an expensive retrain.

### P2. Eval buckets instead of aggregates — locate the espeak split — `sources.md [P2]`
For low-resource speech: make tiny eval buckets *before* more training — same phrase across accents,
noisy phone audio, older speakers, code-mixed. Otherwise you improve aggregate WER while making one
group worse unnoticed. Keep raw text forever; use light normalized form only for retrieval.
- Our espeak finding (72% Serbian-phoneme / 4% Bosnian) should become buckets, not an aggregate. If the
  52% is a "Serbian-phoneme utterance" bucket, that's proof the model learned sr→mel — the fix is
  phonemizer/base model, not more data.

### P3. Piper from-scratch on few hours = catastrophic trap — `sources.md [P3]`
From-scratch 6h voice at 12,000 epochs / 6 GPU-days **still couldn't speak a word**. Thread rule: "Use a
base model, kids." Base trained on 4h = "nowhere near enough".
- The sr_RS voice (22.3%) is the base model we already have. Verify every run's base weights match sr_RS /
  piper's bs voice and that the `.onnx` config `espeak.voice` is `bs`/`hr`, NOT `sr`. If it's `sr`, that
  is precisely the 72%-Serbian-phoneme bug. Confirms the v5/v6/v8 failure shape.

### P4. For a phonemic script, skip espeak entirely — `sources.md [P4]`
VITS2 trains fine on **graphemes** (normalized text), results equal or better than espeak phonemes; the
paper itself notes this. Bosnian orthography is ~phonemic (grapheme≈phoneme), so espeak adds almost
nothing except a failure mode — and our data shows it emits the *wrong* (Serbian) phoneme set.
- Cheapest high-value T4 experiment: `use_phonemes=False` grapheme-VITS run. Isolates whether the 52%
  ceiling is the phoneme mismatch or the mel. If the grapheme model lifts WER, the espeak account is
  confirmed.

### P5. Phoneme-cache diagnostic before burning GPU hours — `sources.md [P5]`
Arabic fine-tune crashed because `phoneme_cache/*.npy` contained a single integer (espeak produced
empty/invalid token sequences → broken alignment/duration → stalled mel). `use_phonemes=False` fixed it.
- Bosnian has ~30 phonemes incl. ć/č/đ/dž; Serbian espeak output substitutes š/ž/č variants. Dump the
  phoneme cache *before* the next 10h run and inspect sequences. Empty/single-token caches = the "mel
  plateau" explained with a CPU-side check.

### P6. The four corpus-killers — all four are in our data — `sources.md [P6]`
What kills TTS corpora: (1) highly differing voice quality, (2) low sampling rate, (3) lack of text
normalization, (4) disadvantageous audio-transcript alignment.
- FLEURS is 16 kHz (Piper/MB-iSTFT expects 22050 Hz mono — silent mel truncation); parliament has varying
  mics; YouTube needs number normalization ("trideset dva" not "32") and re-alignment. Data-prep glue
  that can replace an entire retrain cycle.

### P7. (Side channel — before TTS) diacritic restoration + Cyrillic↔Latin glue — `sources.md [P7]`
Production `serbian-language-tools` (turanjanin) does "ćelava latinica"→č/ć/đ/š/ž restoration and
Cyrillic↔Latin conversion, with tokenizer handling punctuation/URLs/emojis before dictionary lookup.
Companion /r/croatian thread documents the ambiguity reality (piće vs pice — resolved via lemma/context).
- Two cheap wins: (a) ASR output usually arrives diacritic-less; restoring before TTS gives espeak the
  exact graphemes so it doesn't guess from bare `c`; (b) deterministic Cyrillic↔Latin normalization
  unifies YouTube/FLEURS/parliament text. Offline Python, fits the offline constraint.

## Recommended order for the next speak work

1. P1 — rerun the same-text/same-batch/swap-only-TTS-source eval vs sr_RS before anything trains.
2. P3 — fingerprint every trained voice's base weights + `espeak.voice` config; refuse runs where it's `sr`.
3. P4 + P5 — grapheme-VITS control arm + phoneme-cache dump (both isolate the espeak hypothesis cheaply).
4. P6 data-prep: resample 22050, normalize, re-align before the next data line.
5. P2 eval buckets per phoneme set, pre-registered per bucket.