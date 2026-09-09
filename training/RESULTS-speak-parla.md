# A voice from the Croatian parliament — the result

Run: Kaggle `afaksrmeli/lilly-speak-parla` version 1, Tesla T4, 9–10 September
2026, commit `59beecd`, launched by `scripts/kaggle_train.py speak-parla`.
Pre-registered before the run as "v6 — speak — a voice from ParlaSpeech-HR" in
`training/PREREGISTRATION.md`. First version, no amendment: every guard the
FLEURS line taught held.

## The judgment

The FLEURS test prefix — 200 clips, 167 sentences — heard by the shipped
listener (`e6bb58483586b06c`) through `app.speech.transcribe`, one process,
`training/evaluate_speak.py`; 95% bootstraps over sentences (2,000 draws).

| voice | word error | wrong / words | 95% interval |
|---|---|---|---|
| before — Piper `sr_RS-serbski_institut-medium`, speaker 0, the bytes the app fetches | **22.3%** | 726 / 3256 | 20.7–24.1 |
| candidate — the mean of the five speakers' embeddings, this run | **51.9%** | 1690 / 3256 | 48.3–55.5 |
| human recordings — the 200 clips themselves | **11.7%** | 456 / 3901 | 10.2–13.3 |
| for the record — spk3 (Grmoja, Nikola) | 51.6% | 1679 / 3256 |
| for the record — spk1 (Maras, Gordan) | 57.8% | 1883 / 3256 |
| for the record — spk2 (Pernar, Ivan) | 58.0% | 1888 / 3256 |
| for the record — spk4 (Bunjac, Branimir) | 58.1% | 1891 / 3256 |
| for the record — spk0 (Bulj, Miro) | 65.7% | 2140 / 3256 |

Candidate against before: **+29.61 points**, interval +26.2 to +32.9, paired
bootstrap p = 0.0000. Per sentence the candidate is worse on **153** of 167, better
on **7**, and on **26** it loses at least 80% of the words.

**Bar 1 — ships: strictly below the before voice at p < 0.05 — FAIL.**
**Bar 2 — the owner's ask: at or below the human recordings — not reached.**

By the section's rule the parliament line is a null. The sr_RS voice stays in
`models/lilly/speak-bs/`; no voice zip was written; this recipe is not
relaunched with more speakers, more hours, or another judge. The five members
of parliament were heard for the record and none is served.

## What the run did

- 5697 segments, 900 minutes, five speakers at three hours each
  (`training/speak-parla/selection.json`), fetched row group by row group:
  7.1 GB in 22 minutes, every shard size and row id as the
  selection recorded them.
- Piper 1.8.0 through `training/train_piper.py`, warm-started from sr_RS,
  espeak-ng `bs`, batch 8, fp32: **46,818 steps, 35 epochs** — the 9-hour
  wall stopped it. Last checkpoint exported with the mean speaker appended.
- Losses: train mel ep0:0.509  ep5:0.446  ep10:0.433  ep20:0.426  ep30:0.422; val mel ep4:0.435  ep9:0.426  ep19:0.429  ep29:0.419.
  The same plateau as the FLEURS run (train mel 0.43 at the wall there too).
- The candidate speaks **1430 s** of audio for the 167 sentences where the before
  voice speaks **2145 s**.

## Reading two refusals together (a reading, not a measurement)

Two lines, two very different corpora — ten hours of crowd-read FLEURS from
eight people, fifteen hours of podium speech from five — and the same ceiling:
53.9% and 51.9%, every individual speaker between 51% and 66%, the mel loss
parked at 0.42–0.43 in both, the synthetic audio shorter than the before
voice's for the same text in both. A floor that does not move with the data
points at the recipe rather than the recordings: the warm start from a
Serbian voice with `bs` phonemes, the duration model learning a hurried
delivery, or the 16 kHz sources feeding a 22.05 kHz vocoder. None of that is
established here. The one experiment that would settle whether the pipeline
itself degrades voices is a control: the same trainer, warm-started from the
sr_RS checkpoint on that voice's own recordings (the Sorbian Institute data is
public), which should stay near 22.3%. That is a new section, and it costs a
short session, not a long one. Until it runs, no more data goes through this
recipe.

## Files

`training/speak-parla/`: `speak-parla-test.json`, `metadata.csv.manifest.json`,
`metrics.csv`, `voice.onnx.json`, `built.json`, `speak-parla.md`, `stdout.txt`,
`selection.json`. The voice itself was not packaged: a refused voice leaves no
installable file.
