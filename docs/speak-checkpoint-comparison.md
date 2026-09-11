# The Bosnian voice starts from a Sorbian checkpoint — and what to do about it

Written 2026-09-11, before any v9 run, while the weekly Kaggle GPU quota is out.
Nothing here is a measurement of a new voice. It is the case for and against
spending the next quota on one, assembled from numbers already in this
repository and from checks run on this Mac in minutes.

## What the shipped voice actually is

`models/lilly/speak-bs/` holds Piper's `sr_RS-serbski_institut-medium`. Piper
files it under **Serbian** and `app/tts.py` phonemizes it as Serbian. Its own
card names the recordings it was trained on:

```
Language: sr_RS (Serbian, Serbia)
Dataset:  https://github.com/marytts/serbski-institut-dsb-data
License:  CC BY-NC-SA 4.0
Training: Finetuned from U.S. English lessac voice (medium quality).
```

`dsb` is the ISO 639-3 code for **Lower Sorbian**. "Serbski institut" is Sorbian
for *Sorbian Institute* (Bautzen, Germany), not Serbian anything. So the voice
chain is:

> American English (`lessac`) → **Lower Sorbian** recordings → read through
> **Serbian** phonemes → asked to speak **Bosnian**

Lower Sorbian is **West Slavic**, the same branch as Polish and Czech. Bosnian
is **South Slavic**. The two are about as close as Polish is to Bulgarian. The
language written on the tin (Serbian) is the one thing in that chain that never
supplied a single second of audio.

`app/tts.py:10-21` and `models/lilly/NOTICE.md:105-111` have both described the
Sorbian data correctly since they were written. What was never drawn is the
inference below.

## What that explains — and what it does not

`training/RESULTS-speak-control.md` fine-tuned this checkpoint on **its own 747
training utterances** — same language, same speakers, studio quality, 3,000
steps, three arms. The best arm came out **3.2 points worse** than the
checkpoint it started from, p = 0.0000.

At the time that read as a warning about the recipe. The Sorbian discovery
offers a competing story: training harder on Lower Sorbian audio deepens Lower
Sorbian phonetics, which can only hurt Bosnian intelligibility. Both stories fit
the same three numbers.

**We cannot tell them apart from anything we have measured.** That matters, and
the rest of this document is written as if either could be true.

## The three South Slavic voices Piper has

`rhasspy/piper-voices` contains exactly three, one voice each:

| | `sr_RS-serbski_institut` (shipped) | `sl_SI-artur` | `bg_BG-dimitar` |
|---|---|---|---|
| filed as | Serbian | Slovenian | Bulgarian |
| audio actually from | **Lower Sorbian** (West Slavic) | Slovenian (South Slavic, western) | Bulgarian (South Slavic, eastern) |
| branch vs Bosnian | **different branch** | same branch, same sub-branch | same branch, far sub-branch |
| speakers | 2 | **1** | 1 |
| source corpus | MaryTTS release | `ppisljar/artur_studio_tts` (studio) | — |
| licence on recordings | **CC BY-NC-SA 4.0** | **CC BY 4.0** | — |
| sample rate | 22,050 Hz | 22,050 Hz | 22,050 Hz |
| espeak voice in config | `sr` | `sl` | `bg` |
| measured on our test prefix | **22.3%** WER | **never measured** | **never measured** |

Bulgarian lost case marking and has a different vowel system; it is the weakest
of the three on linguistic grounds. Slovenian is the only one in Bosnian's own
sub-branch.

`NOTICE.md` already declares the shipped Bosnian voice non-commercial on the
strength of that NC-SA line. Moving to `artur` would drop that restriction from
the application. That is a real gain and it is independent of any word-error
number.

## Two checks run on this Mac, in minutes

**1. Phoneme inventories are identical.** Both configs carry the same 157
symbols; the set difference `sr_RS − artur` is empty. Anything the shipped voice
can pronounce, `artur` can represent.

**2. `artur` must be driven with `bs` phonemes, never `sl`.** Phonemizing
*"Ćevapi i đevrek, hiljada kuća, čaša žute šljive."*:

| espeak voice | output |
|---|---|
| `bs` | `tɕˈɛvæpɪʲ ɪ dʑˈɛvɾɛk, xˈiʎædæ kˈutɕæ, tʃˈaʃæ ʒˈutɛ ʃʎˈivɛ.` |
| `sl` | `tɕɛʋˈaːpi ˈiː dʒˈeːʋrɛk, xiljˈaːda kˈuːtɕa, tʃˈaːʃa ʒˈuːtɛ ʃʎˈiːʋɛ.` |

`sl` gets **đ wrong** (`dʒ`, which is dž, not `dʑ`), splits **lj** into `l`+`j`
instead of `ʎ`, and stamps Slovenian phonemic vowel length (`ː`) onto a language
that has none. Every symbol the `bs` line needs is in `artur`'s table, so the
fix is free — but it has to be done deliberately.

The residual risk this leaves: `artur`'s embeddings for `ʎ`, `tɕ` and `dʑ` were
trained on whatever share of Slovenian text produced them, which may be small.
The slots exist; how well they are trained is unknown. The shipped voice is in
the same position — the control measured `bs` phoneme agreement with its own
stored lists at **32 of 747 (4.3%)** against `sr`'s 538 of 747 (72.0%).

## The case against training at all

Every fine-tune this project has run on the speech-synthesis side, against the
same 200-clip FLEURS prefix and the same judge:

| run | data | result | vs 22.3% |
|---|---|---|---|
| v5 FLEURS (`RESULTS-speak-bs.md`) | 3,091 found clips, 7 speakers | 53.9% | **+31.57** |
| v6 parliament (`RESULTS-speak-parla.md`) | 5,697 found segments, 5 speakers | 51.9% | **+29.61** |
| v7 control arm C (`RESULTS-speak-control.md`) | the checkpoint's **own studio data** | 25.5% | **+3.22** |
| — human recordings, for scale | — | 11.7% | −10.6 |

Read that table as a ceiling. The control is the most favourable fine-tune this
pipeline can ever be handed — the exact corpus the weights came from, studio
clean, single language — and it **still lost 3.2 points**. The two runs on found
audio both landed near +30.

The staged v9 data is 18.4 hours of **YouTube lecture audio**, seven speakers.
That is found audio. It is the same class of input as FLEURS and parliament,
which produced +31.57 and +29.61. It is richer and cleaner than either, and
"richer and cleaner" has never been worth thirty points anywhere in this
repository.

Swapping the warm start from Sorbian to Slovenian changes where training
*begins*. It does not change the step that lost points in all three runs.

**So: no, a `sl_SI-artur` warm start is not likely to beat 22.3%.** Nothing
measured here predicts it. The earlier claim in conversation that it probably
would was reasoning from language family alone, and the control table outweighs
that.

## The cheaper thing to do first

Neither `sl_SI-artur` nor `bg_BG-dimitar` has ever been heard on our test
prefix. Measuring a checkpoint as fetched needs **no training**: download the
voice, speak the 167 sentences with `bs` phonemes, hand the audio to the shipped
listener, score it with `training/evaluate_speak.py` — the identical path behind
every number in the table above.

That is one short job, minutes of synthesis and roughly 36 minutes of audio to
transcribe, against six hours for a v9 training run. It has two outcomes and
both are worth having:

- **`artur` reads below 22.3%** — a better Bosnian voice ships with zero
  training, and the non-commercial restriction leaves the application with it.
- **`artur` reads at or above 22.3%** — the Sorbian checkpoint is vindicated on
  its own terms, and we learned it for half an hour instead of six.

A run that cannot lose is a better use of the next quota than a run our own
results table predicts will fail its gate.

## Recommendation

1. **Measure `sl_SI-artur` and `bg_BG-dimitar` as fetched**, `bs` phonemes,
   against the 200-clip prefix and the shipped listener. Pre-register the bar
   first: ships only if strictly below 22.3% at p < 0.05, the same bar every
   other speech candidate has faced.
2. **Hold v8/v9 YouTube training.** Do not close the line — the 18.4 hours are
   good data and they stay on Kaggle — but do not spend six hours on a recipe
   that has lost points on every corpus it has been given, including the one it
   was built from.
3. **Revisit training only if step 1 changes the picture** — for example if a
   fetched voice lands well below 22.3%, which would give a fine-tune real room
   to fall and still clear the bar.

## Sources

- `models/lilly/speak-bs/MODEL_CARD`, `built.json`, `voice.onnx.json` — the
  shipped voice, its dataset URL and its licence.
- `models/lilly/NOTICE.md:105-111` — the Sorbian Institute credit and the
  non-commercial declaration.
- `app/tts.py:10-21` — the accent limitation, recorded before this document.
- `training/RESULTS-speak-bs.md`, `RESULTS-speak-parla.md`,
  `RESULTS-speak-control.md` — every number in the fine-tune table.
- `huggingface.co/rhasspy/piper-voices` — `sl/sl_SI/artur/medium/MODEL_CARD`
  and the `sl`, `sr`, `bg` trees, fetched 2026-09-11.
- Phoneme checks: `piper.phonemize_espeak.EspeakPhonemizer` in the project venv,
  against both `phoneme_id_map`s.
