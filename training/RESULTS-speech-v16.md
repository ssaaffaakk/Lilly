# Speech v16 — Kaggle result

Run: `afaksrmeli/lilly-speech` version 16, Tesla T4, launched 29 Aug 2026 ~17:19 UTC.
Cancelled 30 Aug 2026 05:24 UTC during the AFTER WER cell. Numbers below are from
the Kaggle log only. No Mac eval. **Not installed.**

## What finished on the machine

- Mix: 12,232 rows from 3,091 Bosnian, 2 epochs (`openai/whisper-large-v3`, LoRA)
- Train: 1,528 steps in 8h 46m, `train_loss` 0.216, `eval_loss` 0.248 at epoch 2.0
- `lilly-listen-trained.zip` was written on the VM *before* convert
- Convert of `listen-trained` → `models/lilly/listen` printed success
- AFTER WER (`evaluate_speech.py --limit 200`) started; the session died there
- Cell 9 never ran, so `lilly-listen.zip` was never packaged

## What is actually downloadable

The Output listing still names `lilly-listen-trained.zip` and the two
`model-0000N-of-00002.safetensors` shards. Direct download of those paths returns
**404**. Cancel dropped the large new files.

What did persist under `Lilly/models/lilly/listen/` is the **whisper-small**
listener already on this Mac (v12): `built.json` and `tokenizer.json` are
byte-identical to `models/lilly/listen/`. Installing that would replace nothing.

## Verdict

The hours of large-v3 training happened. The weights did not survive the
cancel. App listener stays pass-1 (v12, whisper-small, 33.9% WER on the
notebook's 200 clips).
