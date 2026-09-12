# One language token per clip — the AFTER WER, and what it is not

Run: Kaggle `afaksrmeli/lilly-speech` version 17 (half 1) and
`afaksrmeli/lilly-speech-half2` version 1 (half 2), Tesla T4, 12 September 2026,
commits `2c70333` and `1ad83b9`. Pre-registered as "v4 — listen — one language
token per clip" in `training/PREREGISTRATION.md`, with two amendments written
before the launch: the base, and the limit recorded below.

## The recipe change landed

Both halves printed the same split, which is the whole experiment:

| | rows |
|---|---|
| FLEURS bs → `<\|bs\|>` | 3,091 |
| fleurs_hr → `<\|hr\|>` | 3,430 |
| voxpopuli_hr → `<\|hr\|>` | 2,620 |
| **total** | **`<\|bs\|>` 6,182 · `<\|hr\|>` 6,050** |

Before this run every one of those 6,050 Croatian clips trained under `<|bs|>`.
That is the leak the section named: Whisper writes the spelling its language
token names, and six thousand examples taught it that Bosnian is spelled
*Europom*.

## The number this run produced

`training/evaluate_speech.py`, 200 clips of `data/speech/test.tsv`, written by
the notebook that trained the weights.

| | word error | wrong / words |
|---|---|---|
| gated baseline — `listen-previous`, whisper-small | 34.9% | — |
| **this run — whisper-large-v3-turbo, language column** | **12.77%** | **498 / 3,901** |
| for reference — whisper-large-v3, shipped by owner override, refused at its gate | 11.9% | — |

No interval is quoted because `evaluate_speech.py` does not compute one. A
delta without an interval is not a result under this project's own rule, which
is one more reason the section below stands.

Both halves ran clean: loss fell 0.644 → 0.239 in half 1 and 0.272 → 0.218 in
half 2, no NaN, no Inf, no zero-gradient encoder, `status: passed`, 3h23m and
4h18m on a T4.

## This is not the verdict, and must not be read as one

The pre-registration fixes **three** bars and says "Both, not either":

| row | threshold | measured |
|---|---|---|
| word error, `--decode app`, 925 clips | strictly below `listen-previous` | **not run** |
| Bosnian term recall | not below its baseline | **not run** |
| **Croatian substitution** | **not above its baseline** | **not run** |

The third row is the hypothesis. Nothing above tests it. 12.77% on 200 clips is
the training notebook's own AFTER WER, not the gate, and a word error that falls
while the model writes *tjedan* for *sedmica* is the exact failure this section
predicted in advance.

`Lilly_Speech_Instrument_Kaggle.ipynb` cannot score this candidate as written:
cell 5 pins both builds by fingerprint (`a76342f6ab59b382`,
`e6bb58483586b06c`) and requires the candidate's `built.json` to read
`openai/whisper-large-v3` exactly. It was built to re-measure the two builds the
gate had already scored. Any new candidate, on any base, fails both checks —
the gate working, not a defect. This was recorded as an amendment **before** the
launch, not discovered after.

**Nothing from this run is published, installed, or added to the bundle** until
the 925-clip instrument has its own amendment: the candidate's fingerprint
computed after training and pinned before scoring, the candidate base set to the
one the first amendment names, and no movement in the bars, the split, the
normaliser or the term list.

## Artefacts

`lilly-listen.zip` (684 MB, the converted build) and `lilly-listen-half2.zip`
(155 MB, the adapter) are in the half-2 kernel Output. They are not in
`models/lilly/`, by design.
