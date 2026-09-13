# Translation — NMT, chrF2 plateau, BCS lexicon drift, constrained decoding

Community-sourced (Reddit) techniques for the translation component. Sources in `sources.md` by `[T#]`.

## Context (the numbers to fix)

- Base `Helsinki-NLP/opus-mt-tc-big-zls-en` (Marian). Fine-tune does NOT move chrF2 (−0.16,
  bootstrap tie); BLEU +1.29 (p = 0.001) is real but narrow. `data/clean/train.tsv` was **inside the
  base model's own training pool** (WikiMatrix, SETIMES, TED2020) — fine-tuning was re-weighting seen
  material. NTREX is genuinely unseen.
- The base printed its own language tag into 30.4% of outputs; Lilly prints none → +1.29 BLEU.
- Form-rate measurement (`training/RESULTS-en-bs-formrate.md`): 245 of 338 discriminating targets
  commit to one of two forms; the **14 losses split cleanly: every systematic loss drifts to Serbian
  (ekavica, dropped /h/), every lexical loss drifts to Croatian (vlak, travnja, tisućama, sudjelovati)**.
- `docs/what-moves-the-model.md` lever #2: 7 of 14 losses are a closed, enumerable lexicon —
  fixable with **zero training** by biasing decoding against the Croatian member of each pair.

## Findings

### T1. Glossary injection at inference (the "dictionary not GPU" play) — `sources.md [T1]`
Controlled study: 5 LLMs × 5 EU languages, 42k judgments — injecting a glossary + style + locale
instructions into every translation request cut MQM **terminology errors 17–45%** (Mistral −44.6%,
DeepSeek −42.1%), p < 0.001. Core claim: "terminology drift isn't a model problem — it's a context
pipeline problem."
- Fixes exactly our seven Croatian-lexicon losses. Runs on the local LLM; no GPU training; deterministic
  gates still apply (glossary is deterministic).
- This is lever #2 in `docs/what-moves-the-model.md` done at inference.

### T2. Constrained decoding — zero-out the Croatian member of each pair — `sources.md [T2]`
Constrained decoding / token-level constrained generation: renormalize the probability distribution over
a required set and sample from it. Tools: `guidance-ai/guidance`, HF constrained beam search blog,
`aidancooper.co.uk/constrained-decoding`. Also `kbnf` (Rust, `sources.md [T2b]`) and formatron for
grammar-constrained generation with vocab support.
- Fixes the 7 lexical losses + can enforce the Bosnian charset on output, with zero training.
- Note from `sources.md [T2c]`: constrained decoding that zeroes probs works best when the model has
  some exposure to the grammar — a small bias term is kinder than a hard zero for unseen-in-training pairs.

### T3. "Fine-tuned NMT got WORSE" — same failure as our chrF2 tie — `sources.md [T3]`
mBART fine-tune on unseen low-resource language: loss 3.3 → 0.8 but BLEU stuck at 0.1. Practitioner
answers = our own conclusion: (a) train distribution ≠ test, (b) with small data very low training loss
= overfitting, (c) verify lr/batch/epochs, (d) evaluate checkpoints during training.
- Fixes the expectation for the en→bs branch: data must be genuinely out-of-corpus, or the run silently
  does nothing measurable on chrF2.

### T4. Compare seq2seq bases at our data scale — `sources.md [T4]`
M2M vs mT5 fine-tune on 10k low-resource pairs (Yorùbá→EN): M2M beat mT5 by a large margin (ROUGE 23 vs
45). Free walkthrough with code (`maroxtn/mt5-M2M-comparison`).
- Lateral comparison for en→bs: keep Marian as the main arm, add a second base as a separate arm. Tells
  us whether the base (not the data) is the ceiling.

### T5. Back-translation mechanics — when it helps and why — `sources.md [T5]`
`[D] Why does backtranslation work?` — back-translation is only as good as the reverse model; the ceiling
is set by the target→source model trained on the small parallel set. Companion thread ties it to style
transfer / data augmentation for new rows.
- For en→bs: our only free data multiplier is monolingual Bosnian + a reverse model. These threads spell
  out the failure mode to watch.

### T6. Explicit LM rescoring vs back-translation — `sources.md [T6]`
Classic paper (Sennrich 1511.06709) vs explicit LM: back-translation is really comparable to other
target-side augmentation; the alternative is simple rescoring with an explicit LM over the source.
- Same "small Bosnian LM interpolated into a bigger one" trick as speech `S11` can rescore BS-side
  translations for Bosnian-ness without retraining.

### T7. (Instrumentation) BCS is pluricentric — don't let the metric punish Croatian — `sources.md [T7]`
/r/croatian: the yat split is a historical accident (Croatian standard is ijekavian because of Dubrovnik
literature); /r/AskBalkans: "same language, one-pluricentric, like English". The Croatian/Serbian/Bosnian
ruler must penalise exactly: Serbian ekavica, dropped /h/, distinctively Croatian lexicon — never
*vrijeme/djece/historija*.
- Confirms `training/audit_bosnian.py` design and `docs/BOSNIAN_METRIC.md`.

## Recommended order

1. T1 glossary injection + T2 constrained decoding — zero-training, fix the 14 known losses.
2. T3 → re-run the en→bs data overlap audit so the next fine-tune is genuinely out-of-corpus.
3. T5/T6 back-translation pipeline for en→bs data multiplier.
4. T4 only if we want a second translation arm.
5. T7 as the ruler-rule reminder (already in `audit_bosnian.py`).