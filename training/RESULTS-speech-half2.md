# Speech half 2 — AFTER WER

Run: Kaggle `safaksideacc2/lilly-speech-half2`, Tesla T4, git `34ca980`.
Started 31 Aug 2026 17:55 UTC, ended 1 Sep 2026 00:37 UTC, `status: passed`.
Weights installed locally on 1 Sep 2026 into `models/lilly/listen/`
(`openai/whisper-large-v3`, int8, 1.5 GB `model.bin`). The whisper-small
listener it replaced is at `models/lilly/listen-previous/`.

## Word error — 200 held-out FLEURS Bosnian clips

Same scorer (`training/evaluate_speech.py` through `app.speech.transcribe`),
same first 200 clips of `data/speech/test.tsv` (3,901 reference words).

| listener | word error | wrong / words | where |
|---|---|---|---|
| untrained `whisper-small` | 38.5% | 1,497 / 3,901 | local, earlier |
| shipped small (`listen-previous`) | 35.5% | 1,386 / 3,901 | local, earlier |
| pass-1 small (Kaggle v12 mix) | 33.9% | 1,322 / 3,901 | Kaggle T4 |
| **half 2 large-v3, 2 epochs** | **11.8%** | **459 / 3,901** | Kaggle T4 AFTER WER |

`speech-wer.json`: `{"wer": 11.7662, "edits": 459, "words": 3901, "clips": 200}`.

The pre-registered bar is **below 35.5%**. 11.8% clears it.

This is **not** an isolated training effect. Half 1 skipped BEFORE WER on
purpose (12h wall). There is no untrained large-v3 score from this run, so
11.8% versus 35.5% mixes a bigger base with two epochs of the wider mix.
What the gate asked for is “beat the listener that ships,” and on WER it did.

## What it trained on

Resumed the half-1 LoRA (`checkpoint-1528` after epoch 2). Mix from
`build_speech_mix.py --share 0.47`:

| | clips |
|---|---|
| Bosnian (FLEURS bs_ba train, repeated 2×) | 6,182 rows from 3,091 |
| Croatian FLEURS hr | 3,430 |
| Croatian voxpopuli_hr | 2,620 |
| mixed | 12,232 rows, **51% Bosnian** (asked 47%) |

0 extra clips dropped for appearing in valid or test. Leakage gate PASS on
both neighbour sources. Encoder LoRA: 384/384 tensors received a gradient.
`train_loss` 0.095, `eval_loss` 0.254, 1,528 steps, 4h 27m for the resume.

## The gate this does NOT yet clear

`training/PREREGISTRATION.md`: both, not either.

| | now (small, shipped) | threshold | half 2 |
|---|---|---|---|
| word error, 200 Bosnian clips | 35.5% | below 35.5% | **11.8% — passes** |
| Bosnian-specific term recall | 65.9% (legacy SpeechBench) | not below baseline | **running locally** |

SpeechBench was not in the Kaggle notebook. A WER drop this large from a
larger Whisper is the exact shape the term gate exists for: neighbours in
the mix, average error down, Bosnian forms possibly drifting to Croatian
or Serbian. `training/speech_bench.py --clips first200` is running on this
machine against the installed large-v3 and `listen-previous`.
