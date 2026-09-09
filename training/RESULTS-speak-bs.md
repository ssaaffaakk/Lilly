# A Bosnian voice from FLEURS — the result

Run: Kaggle `afaksrmeli/lilly-speak-bs` version 3, Tesla T4, 9 September 2026,
commit `9d6beca`, launched by `scripts/kaggle_train.py speak-bs`. Pre-registered
before any run as "v5 — speak — a Bosnian voice from FLEURS" in
`training/PREREGISTRATION.md`, with two amendments (memory; the checkpoint
callback) written before any number existed. Versions 1 and 2 produced no number.

## The judgment

The test prefix — the first 200 clips of FLEURS bs_ba test, 167 distinct
sentences, the prefix behind every published speech number — heard by the
shipped listener (`e6bb58483586b06c`, through `app.speech.transcribe`, on the
GPU by env), one process, `training/evaluate_speak.py`. Intervals are 95%
bootstraps over sentences (2,000 draws, seed 11); the paired p is
`training/speech_bench.paired_bootstrap`'s.

| voice | word error | wrong / words | 95% interval |
|---|---|---|---|
| before — Piper `sr_RS-serbski_institut-medium`, speaker 0, the bytes the app fetches | **22.3%** | 726 / 3256 | 20.6–24.0 |
| candidate — this run, served speaker 2 (cluster 6) | **53.9%** | 1754 / 3256 | 50.8–56.9 |
| human recordings — the 200 clips themselves | **11.7%** | 456 / 3901 | 10.1–13.3 |

Candidate against before: **+31.57 points**, interval +28.6 to +34.6, paired
bootstrap p = 0.0000 (no draw of 2,000 went the other way). Per sentence the
candidate is worse on **151** of 167, better on **6**, and on **19** it loses at least
80% of the words.

**Bar 1 — ships: strictly below the before voice at p < 0.05 — FAIL.**
**Bar 2 — the owner's ask: at or below the human recordings — not reached.**

By the section's own rule the FLEURS line is a null. The sr_RS voice stays in
`models/lilly/speak-bs/`; no voice zip was written (`lilly-speak-bs.zip` does
not exist in the run's Output, by design), and this recipe is not relaunched
with more epochs, another threshold, another speaker or another judge.

## What the run did

- Data on the box, as on the Mac: FLEURS bs_ba train 3,091 clips → 8 clusters at
  cosine distance 0.3, 7 selected (103, 98, 98, 87, 83, 69, 54 minutes),
  592.5 minutes together, 14.9% same-sentence collisions
  (291 of 1952); 72 clips over 20 s left out; **2985 clips trained**.
- Piper 1.8.0, warm-started from the sr_RS checkpoint (md5 `3dd3439e…`, 803 of 804
  tensors copied, the speaker table restarted), espeak-ng `bs`, batch 8, fp32,
  Piper's default learning rates. **32,962 steps, 47 epochs, 5.6 h** — the
  5 h 30 min wall stopped it, not the 60-epoch cap. Last checkpoint exported.
- Served speaker chosen on 120 distinct valid sentences (never test):

| speaker (index: cluster) | valid word error | wrong / words |
|---|---|---|
| spk2 | 51.8% | 1255 / 2424 |
| spk1 | 59.6% | 1445 / 2424 |
| spk0 | 62.8% | 1522 / 2424 |
| spk5 | 63.5% | 1540 / 2424 |
| spk3 | 64.7% | 1567 / 2424 |
| spk4 | 69.3% | 1681 / 2424 |
| spk6 | 70.4% | 1706 / 2424 |

  Every speaker sits between 52% and 70% on valid; the choice of speaker was not the lever.

## Reading the failure (a reading, not a new measurement)

- The candidate speaks **1488 s** of audio for the 167 sentences where the before
  voice speaks **2145 s** — thirty percent shorter for the same text. The transcripts
  show real Bosnian words in the right places at the start of a sentence and
  garbled runs after: the listener writes *"predstavili se njeni država za
  reakciju na namjer"* where the sentence says *"predsjednik Sjedinjenih Država
  Donald Trump najavio je da"*. That is a voice that hurries and slurs, not one
  that speaks another language.
- The loss was still moving when the wall came: train mel 0.553 (epoch 0) →
  0.462 (5) → 0.432 (20) → 0.427 (40); val mel 0.492 (epoch 4) → 0.431 (39)
  → 0.444 (44). Forty-seven epochs from a Serbian checkpoint did not reach a
  plateau on ten hours of crowd audio from seven people. Whether a longer run
  would arrive somewhere better is exactly the question this section does not
  get to ask again.
- The instrument was, as stated in advance, kind to the candidate: the listener
  was fine-tuned on these very FLEURS train voices. The candidate lost anyway,
  by 31 points.

## What this does not settle

Whether one clean studio hour from a native speaker would reach the human row
(declined for now, not closed). Whether the largest cluster alone, or a run
several times longer, would do better — either is a new section with its own
bars, not a relaunch. Naturalness, which nobody measured.

## Files

`training/speak-bs/`: `speak-bs-test.json` (the judgment, every hypothesis),
`speak-bs-valid.json` (the selection), `speakers.json` (the clusters on the
box), `metrics.csv` (every logged loss), `voice.onnx.json` (the voice's config
as exported), `built.json` (what the zip would have carried), `speak-bs.md`
(the box's own report), `stdout.txt` (the tee). The voice itself was not
packaged: a refused voice leaves no installable file.
