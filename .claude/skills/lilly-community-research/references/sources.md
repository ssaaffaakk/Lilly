# Sources — every Reddit thread behind `lilly-community-research`

Provenance for every technique in `references/*.md`, referenced by `[S#]`, `[T#]`, `[O#]`, `[P#]`.
Collected 13 Sep 2026. Reddit `r/<sub>/comments/<id>` stays in `reddit.com/r/...` form so they cut/paste
cleanly in a browser. Archived copies exist for the two removed sites.

## Speech / ASR (used by `references/speech-asr.md`)

| ID | Link | Finding |
|------|------|---------|
| S1 | `reddit.com/r/LocalLLaMA/comments/1plvhwt/` | Whisper-LM n-gram rescoring recipe (paper `arxiv.org/abs/2503.23542`); Basque 10.52→5.15 WER, but alpha/beta need hand-tuning per language |
| S2 | `reddit.com/r/MachineLearning/comments/xnzjpu/` | `initial_prompt` + `suppress_tokens` solved ~70% of spelling fixes without retraining |
| S3 | `reddit.com/r/LocalLLaMA/comments/1cm4wtx/` | Whisper silently *translating*; fix = force `language` + `task="transcribe"` every call |
| S4 | `reddit.com/r/LocalLLaMA/comments/1rlqfd7/` | Production write-up: 135 hallucination phrases; `condition_on_previous_text=False` + Silero VAD + per-language blocklists end the cascade ("thank you" × 28) |
| S5 | `reddit.com/r/speechrecognition/comments/18hqtnj/` | Fine-tuned Whisper amplified a training pattern ("Houston, ") never in data → mechanism behind our Croatian drift |
| S6 | `reddit.com/r/MachineLearning/comments/17jyd2m/` | Small-data Whisper fine-tune raised WER (23.1→30%); fix = cap epochs, eval checkpoints during training |
| S7 | `reddit.com/r/LanguageTechnology/comments/1oktrlh/` | Concrete whisper-small recipe: 1000–3000 steps, checkpoint every 250–500, eval latest, 16 kHz resample |
| S8 | `reddit.com/r/MachineLearning/comments/17kqvl6/` | Freeze encoder / fine-tune decoder only; encoder inherits robustness (Distil-Whisper lesson) |
| S9 | `reddit.com/r/LocalLLaMA/comments/1rvzx4u/` | Parakeet int8 beat Whisper on claimed-Slavic support; NeMo/ESPnet as alternatives |
| S10 | `reddit.com/r/OpenAI/comments/1ftc9go/` | ⚠️ `initial_prompt` handling leaks/differs per engine (OpenAI/runpod/deepinfra) — verify on CTranslate2 |
| S11 | `reddit.com/r/LanguageTechnology/comments/zusab8/` | KenLM can't be extended; interpolate general ~2TB LM with a small ~10MB domain LM |
| S12 | `reddit.com/r/LanguageTechnology/comments/1kegziy/` | Appending new data to fine-tuned model collapsed old performance; consolidate one balanced pass instead |
| S13 | `reddit.com/r/speechtech/comments/1mboz21/` | Annotate code-switching/mixed-language data so model learns *when not* to switch |
| — | `reddit.com/r/LocalLLaMA/comments/1dsdnbi/` | Generative Fusion Decoding (Whisper+LLM decode alternative) — keep on shelf, no gate decision |

## Translation / NMT (used by `references/translation.md`)

| ID | Link | Finding |
|------|------|---------|
| T1 | `reddit.com/r/localization/comments/1szsbcn/` | Glossary+style+locale injected per request cut MQM terminology errors 17–45% (p<0.001); "terminology drift is a context-pipeline problem" |
| T2 | `reddit.com/r/MachineLearning/comments/1cm9r0y/` | Constrained decoding via renormalization over a required token set |
| T2b | `reddit.com/r/learnmachinelearning/comments/1bzd4gt/` | Tooling roundup: `guidance`, HF constrained beam search blog, `aidancooper.co.uk/constrained-decoding`, `kbnf` (Rust), formatron |
| T2c | `reddit.com/r/MachineLearning/comments/1cm9r0y/` (companion comment) | Hard-zero constrained decoding best when model has seen the grammar; small bias gentler for unseen-in-training vocab |
| T3 | `reddit.com/r/LanguageTechnology/comments/1gwv2pw/` | mBART fine-tune loss 3.3→0.8 but BLEU 0.1 (overfitting on low-resource) — same shape as our chrF2 tie |
| T4 | `reddit.com/r/MachineLearning/comments/o8vc8e/` | M2M vs mT5 on 10k Yorùbá→EN pairs: M2M much better; free code walkthrough (maroxtn/mt5-M2M-comparison) |
| T5 | `reddit.com/r/MachineLearning/comments/d9n9m5/` | Why back-translation works / its ceiling is the reverse model |
| T5b | `reddit.com/r/MachineLearning/comments/19cou9l/` | Back-translation as style transfer / data augmentation mechanics |
| T6 | `reddit.com/r/LanguageTechnology/comments/6c6wn2/` | Back-translation vs explicit LM rescoring (Sennrich 1511.06709 lineage) |
| T7 | `reddit.com/r/croatian/comments/15tpe3r/` | Ijekavian is standard-by-literary-accident; superscripts-matter note — ruler must not punish ijekavian bs |
| T7b | `reddit.com/r/AskBalkans/comments/1j9q8m0/` | "One pluricentric language" (like English) — confirms bs/hr/sr ruler design |

## OCR (used by `references/ocr.md`)

| ID | Link | Finding |
|------|------|---------|
| O1 | `reddit.com/r/ArtificialInteligence/comments/1qq7tw5/` | "OCR accuracy is a system/maintenance problem, not a model problem" — gates beat floors |
| O2 | `reddit.com/r/n8n/comments/1ve4bhu/` | Word-level confidence bucketing with human-review escape hatch |
| O3 | `reddit.com/r/MachineLearning/comments/18uvrwj/` | Two independent engines = hallucination detector (agree→accept, disagree→reject) |
| O4 | `reddit.com/r/Python/comments/1eo6dxz/` | LLM post-correction: correct in place against a wordlist, never regenerate |
| O5 | `reddit.com/r/PLC/comments/1bzr612/` | Separate recognition score from per-char confusion threshold; "every error needs a threshold" |
| O6 | `reddit.com/r/computervision/comments/1jglmd0/` | PaddleOCR D/0, S/5 misreads = confusable-pair failure class; needs pair-crop data |
| O7 | `reddit.com/r/learnpython/comments/18rthuo/` | Character whitelist makes inventions structurally impossible |
| O8 | `reddit.com/r/computervision/comments/wqqr94/` | PaddleOCR ships `rs_cyrillic` / EasyOCR `cyrillic_g2` — route, don't retrain |
| O9 | `reddit.com/r/LocalLLaMA/comments/1swwns0/` | Turbo-OCR single recognizer handles Latin+Cyrillic per-type (mixed-script lines) |
| O10 | `reddit.com/r/LanguageTechnology/comments/k9u0gy/` | Synthetic augmentation must match the real target font |
| O11 | `reddit.com/r/VietNam/comments/1d708y2/` | Font choice shapes diacritic survival at render resolution |
| O12 | `reddit.com/r/computervision/comments/1m8ip8v/` | Synthetic backbone → real-data fine-tune (closes the 100%→53% domain cliff) |
| O13 | `reddit.com/r/computervision/comments/1mqvt4j/` | Replicate the actual camera blur/JPEG/backdrop into the renderer |
| O14 | `reddit.com/r/MachineLearning/comments/i98wr6/` | CRAFT-based detector + glare/deskew/crop/denoise; stop when it reads cleanly |
| O14b | `reddit.com/r/MachineLearning/comments/i98wr6/` (companion comment) | Over-thresholding wrecks OCR; preprocess glare before recognition |

## TTS / Piper (used by `references/tts.md`)

| ID | Link | Finding |
|------|------|---------|
| P1 | `reddit.com/r/LanguageTechnology/comments/1txkflj/` | Same ASR/reference, TTS-only swap: espeak-ng audio 20.9% WER vs gTTS 4.65% — generator spectral distance is a *confound*, not a bad voice |
| P2 | `reddit.com/r/MachineLearning/comments/1uhlvjv/` | Buckets (accents, noise, speech type, code-mix) before more training; keep raw text forever |
| P3 | `reddit.com/r/LocalLLM/comments/1jvuv8h/` | Piper from-scratch on a few hours = catastrophic ("still can't speak a word"); "Use a base model, kids." |
| P4 | `reddit.com/r/speechtech/comments/1eb8cs4/` | VITS2 trains fine on graphemes for phonemic scripts; espeak unnecessary |
| P5 | `reddit.com/r/tts/comments/1oar4jc/` | Phoneme-cache diagnostic: empty/single-int `.npy` from espeak ⇒ stalled mel ("mel plateau" explained CPU-side) |
| P6 | `reddit.com/r/speechtech/comments/osn0ei/` | Four corpus-killers: voice-quality variance, low sample rate, no text normalization, bad alignment |
| P7 | `reddit.com/r/programiranje/comments/nmx7tt/` | `serbian-language-tools` (turanjanin): diacritic restoration + Cyrillic↔Latin, tokenizer-aware |
| P7b | `reddit.com/r/croatian/comments/1427nay/` | piće/pice ambiguity resolved by lemma/context — restoration is contextual, not 1:1 |

## Caveats (read once)

- Threads were captured via indexed web snapshots, not the Reddit JSON API (no authenticated `opencli`/
  `rdt` backend installed for `agent-reach`). A handful of older IDs (`1plvhwt`, `xnzjpu`, `wqqr94`,
  `i98wr6`, `osn0ei`, `nmx7tt`) may have archived before their OP-facing summary; the cited technique
  is from the indexed text.
- Findings are *directional evidence*, never Lilly truth. Only our gates make truth
  (`training/PREREGISTRATION.md`, `app.*` scores, result files).
- New threads get an appendix ID here, not a new file; the SKILL.md table and one reference row change
  together.