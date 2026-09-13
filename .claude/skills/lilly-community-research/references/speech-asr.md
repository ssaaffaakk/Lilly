# Speech / ASR — Whisper, Croatian drift, KenLM rescoring

Community-sourced (Reddit) techniques for the listener component. Every finding names the
measured mistake it attacks. Sources live in `sources.md` by `[S#]`.

## Context (the numbers to fix)

- Fine-tuned `whisper-large-v3` on Bosnian: 11.9% WER on 200 held-out clips; gated whisper-small
  baseline 34.9%. Both shipped/listed; large-v3 was refused at its gate: **Croatian substitution
  1.1% → 6.1%, p = 0.018** on 925 clips.
- Root cause established in `docs/what-moves-the-model.md`: 6,050 of 12,232 mix rows are Croatian,
  and half the `<|bs|>`-labelled rows contain Croatian text — the decoder learned "bs token ⇒
  Croatian spelling". Whisper's pretraining had only 11h Bosnian vs 91h Croatian.
- "The leak is a recipe defect, not a data limit" — the language-column fix is coded and waiting.

## Findings

### S1. Whisper + KenLM n-gram rescoring (the biggest lever) — `sources.md [S1]`
Build a KenLM from pure Bosnian text (spelling-preferring: *vrijeme*, *dijete*, *historija*) and
rescore the beam (Whisper-LM paper `arxiv.org/abs/2503.23542`; lib `hitz-zentroa/whisper-lm-transformers`
or `marvinIV/whisper-KenLM`). Basque WER 10.52 → 5.15.
- Fixes: Croatian drift at the orthography level, **no retrain**.
- Trap: the OP's Finnish run got ~0 until the alpha/beta interpolation weights were hand-tuned.
  Tune them on the 200-clip Bosnian holdout. **Alpha/beta are not portable.**

### S2. `initial_prompt` + `suppress_tokens` — decode-time orthography bias — `sources.md [S2]`
- Pass a Bosnian exemplar sentence as `initial_prompt` (Whisper mirrors spelling patterns from it).
- Suppress the token ids for Croatian yat forms (*vreme*, *dete*, *Europa*) → decoder is forced to
  alternatives. Author: "solved 70% of needs", no fine-tuning.
- Fixes the 6.1% Croatian substitution column without training.
- Tool availability: both knobs exist in faster-whisper/CTranslate2 (the shipped engine).

### S3. Force `language="bs"` + `task="transcribe"` every segment — `sources.md [S3]`
A fine-tuned Whisper intermittently started *translating* instead of transcribing; fix was pinning
`language="xx"` + `task="transcribe"` in `generate_kwargs` / `model.transcribe`.
- Fixes our diagnosed root cause: the decoder must start from a Bosnian-conditioned state, and never
  fall onto the hr-friendly decoding path per segment.

### S4. Kill the drift cascade: `condition_on_previous_text=False` + VAD pre-gate — `sources.md [S4]`
Production write-up of 135 hallucination phrases: `condition_on_previous_text=False` stops one bad
window seeding the next ("one 'thank you' becomes 28"); Silero VAD as a pre-gate so Whisper never
runs on non-speech; plus per-language exact-string blocklists.
- Fixes the *cascade* mechanism behind our 1.1% → 6.1%: a Croatian-spelled window primes the next.

### S5. Mechanism warning: fine-tuned Whisper amplifies training patterns — `sources.md [S5]`
Model fine-tuned on 15h kept emitting "Houston, " phrases never in its data — it latched onto a
pattern associated with the audio. Same mechanism as our Croatian drift: the decoder latches onto a
stored completion ("Bosnian audio ⇒ Croatian spelling") beyond the data's real ratio.
- Lesson: rebalancing data alone is not enough; combine with decode-side countermeasures (S1/S2).

### S6. Small-data fine-tune can RAISE WER — `sources.md [S6]`
4h of data → WER 23.1% → 30%. Prescription: cap epochs, evaluate checkpoints **during** training on
the Bosnian-only holdout, stop when the drift axis regresses, verify lr/batch.
- Directly addresses "the Croatian regression was only discovered after the run".

### S7. Concrete whisper-small recipe — `sources.md [S7]`
Whisper learns fast and will overfit on low hours: cap at **1000–3000 steps**, save a checkpoint
every 250–500, eval the *latest* on the eval set, resample 16 kHz, dataset quality > quantity.
- Guardrails for the next listener run.

### S8. Freeze encoder, fine-tune decoder only (Distil-Whisper lesson) — `sources.md [S8]`
Distil-Whisper kept the encoder frozen ("inherits robustness to noise and audio distributions");
all spelling/language-prior cost lives in the decoder. Our failure is orthographic (decoder behavior),
not a hearing failure.
- Highest-leverage variant: protects the robust acoustic encoder from 11h of Bosnian audio while
  nudging the spelling/language subsystem.

### S9. Alternative base: Parakeet / NeMo can outdo Whisper on less-English Slavic — `sources.md [S9]`
"Claimed Slovak support, can't get a word right" → int8 Parakeet gave a massive usable jump; NeMo/ESPnet
+ a small n-gram LM recommended in companion thread. If the orthography fight stalls, Whisper's 11h-bs /
91h-hr prior is inherited; a 1-day bakeoff against a base that treats bs/hr as one acoustic language
(and lets YOU bolt on the Bosnian n-gram LM) is cheap.

### S10. ⚠️ `initial_prompt` behaves differently per engine — `sources.md [S10]`
Prompt insertion leaks (whole prompt appended to transcript) differs between OpenAI, runpod, deepinfra.
We run CTranslate2/faster-whisper — a different engine with its own prompt handling. Verify S2 actually
changes logits on our engine before shipping.

### S11. KenLM cannot be extended — linearly interpolate instead — `sources.md [S11]`
Can't update a trained KenLM. Instead interpolate a large generic LM with a small domain LM (the thread:
~2 TB general vs ~10 MB domain). A Bosnian yat-pair LM only needs to be strong on a small lexicon, so
10 MB of Bosnian text is plenty; interpolate with a bigger BCS LM.

### S12. Re-consolidate, do NOT append new data — `sources.md [S12]`
Continued/fine-tuned chaining (clean → noisy appended data) collapsed old-performance. Correct pattern:
one consolidated pass on a balanced mix across all data, with the target holdout as the stopping
criterion.
- This is exactly the "added Croatian audio, drift doubled" mistake. Upweight Bosnian in the final mix,
  and label orthography must match the language token before mixing.

### S13. Code-switching / mixed-language annotation — `sources.md [S13]`
For production code-switching: fine-tune on annotated data containing the actual switching phenomena
so the model learns *when not* to switch. If we keep any Croatian audio, its transcripts must be
transliterated to Bosnian orthography so the model never sees hr text under the bs token.

## Recommended order for the next listener work

1. Verify S3 (force language/task) + S4 (`condition_on_previous_text=False`) — decode-only, no GPU.
2. S1 KenLM rescoring on the 200-clip holdout, hand-tune alpha/beta (this is the "bigger than retrain" play).
3. S2 suppress_tokens on the known Croatian forms.
4. If any retrain happens: S7 recipe + S12 consolidated mix + S8 freeze-encoder option + S6 checkpoint-eval.
5. If the fight stalls at Whisper's prior: S9 bakeoff.