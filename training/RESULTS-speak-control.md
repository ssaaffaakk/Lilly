# The voice pipeline's control — the result

Run: Kaggle `afaksrmeli/lilly-speak-control` version 1, Tesla T4, 10 September
2026 (00:55–02:38 CEST), commit `c61293f`. Pre-registered before the run as
"v7 — speak — the control" in `training/PREREGISTRATION.md`. First version, no
amendment.

## The reading

Piper's `sr_RS-serbski_institut-medium` checkpoint fine-tuned for 3,000 steps
on **its own 747 training utterances** (164 minutes; Sorbian Institute
releases pinned by sha256; every utterance found by prompt code with identical
text), three arms, speaker 0 of each heard on the FLEURS test prefix (200
clips, 167 sentences) by the shipped listener `e6bb58483586b06c`, beside the
checkpoint as the app fetches it and the human recordings. 95% bootstraps
over sentences, 2,000 draws.

| voice | word error | wrong / words | 95% interval | vs before (interval) | p | sentences | reading |
|---|---|---|---|---|---|---|---|
| before — the checkpoint, speaker 0, as fetched | **22.3%** | 726 / 3256 | 20.6–24.1 | | | | |
| arm A — `sr` phonemes, Piper's rates | **26.6%** | 865 / 3256 | 24.7–28.5 | +4.27 (+2.9 to +5.7) | 0.0000 | worse 93, better 35 | **sound** |
| arm B — `bs` phonemes, Piper's rates | **26.0%** | 846 / 3256 | 24.2–27.8 | +3.69 (+2.2 to +5.1) | 0.0000 | worse 81, better 37 | **sound** |
| arm C — `bs` phonemes, gentle optimizer | **25.5%** | 831 / 3256 | 23.7–27.7 | +3.22 (+1.8 to +4.7) | 0.0000 | worse 76, better 40 | **sound** |
| human recordings | **11.7%** | 456 / 3901 | | | | | |

**All three arms read "sound" by the rule fixed in advance (under +5 points).**
None is broken. Each is a little worse than the checkpoint it started from —
three to four points, p = 0.0000 in each arm: a real, small drift on the
voice's own data in 3,000 steps, not a collapse. The three arms sit within
about a point of each other; `bs` phonemes (B) are not worse than `sr` (A),
and the gentle optimizer (C) is not measurably better than Piper's rates.

## What the run did

- Phoneme agreement with the voice's stored lists, on the box as on the Mac:
  `sr` 538 of 747 (72.0%), `bs` 32 of 747 (4.3%).
- 3,000 steps per arm at batch 2, fp32, two speakers, warm start 804 of 804.
  Train mel: A ep0:0.413  ep3:0.412; B ep0:0.424  ep3:0.424; C ep0:0.407  ep3:0.393.
- Audio for the 167 sentences: before 2145 s; A 1983 s, B 2054 s, C 2034 s.

## What the reading means, as written before the run

"A, B, C all sound: a short fine-tune preserves the voice and the ceiling is
not explained here; the suspects left are the long run and the data's own
quality, and the FLEURS and parliament refusals stand as data verdicts."

So: the pipeline is not broken. The phonemizer, the leading suspect going in
(4.3% agreement with `bs`), is cleared — arm B is not worse than arm A. The
fresh optimizer is cleared at 3,000 steps. What the two refusals measured
was the recordings: ten hours of crowd-read FLEURS and fifteen hours of
parliament hall speech did not make a voice this recipe can carry, where 164
minutes of the checkpoint's own studio recordings hold it within four points.

What this does not settle: whether the three-to-four-point drift seen here
compounds over the 33,000–47,000 steps the refused lines ran (the mel
plateau at 0.42–0.43 in both, against a fresh start, is consistent with
either reading); and whether a clean studio hour from a native Bosnian
speaker would be carried the way the Sorbian hour is. That second question is
the only voice line left open, and it needs a person, not a corpus.

## Files

`training/speak-control/`: `speak-control-test.json`, `metadata.csv.manifest.json`,
`metrics-A.csv`, `metrics-B.csv`, `metrics-C.csv`, `voice-A/B/C.onnx.json`,
`speak-control.md`, `stdout.txt`, `sources.json`. Nothing was installed.
