# The reverse direction, tested and audited — 12 September 2026

English in, Bosnian out, by chat, by voice and by photograph, plus an audit of
whether the figures this project publishes are the figures its artifacts
support. Written for the white paper.

Everything below was run against the repository at commit `c486123`, on the
owner's laptop, through the app's own HTTP endpoints and the app's own engine
classes — never around them. Two other sessions were working in the same tree;
nothing here edited their files, and no branch, working tree or published
document was changed to make a number come out better.

## 1. Does it work

All four reverse-direction paths answer, end to end. These are real responses
from a locally running server, not descriptions of intent.

| Path | Sent | Came back |
|---|---|---|
| Chat, `/api/reply` | "Where is the bus station?" | "Gdje je autobuska stanica?" |
| Chat | "I would like two coffees, please." | "Ja bih **dvije** kafe, molim." |
| Chat | "My daughter is seven years old and she speaks three languages." | "Moja **kćerka** ima sedam godina i govori tri jezika." |
| Chat | "The weather tomorrow will be rainy, so bring an umbrella." | "**Vrijeme** će sutra biti kišovito, zato ponesite kišobran." |
| Chat | "Can you help me? I have lost my passport and I need to find the police station." | "Možete li mi pomoći? Izgubio sam pasoš i moram pronaći policijsku stanicu." |
| Voice in, `/api/speech` `direction=en-bs` | spoken English, "Where is the bus station?" | heard "Where is the bus station?", answered "Gdje je autobuska stanica?" — 19 s including the model load |
| Voice out, `/api/speak` `language=bs` | "Gdje je autobuska stanica?" | 129 KB WAV, 22.05 kHz mono |
| Voice out, `language=en` | "Where is the bus station?" | 92 KB WAV, 24 kHz mono |
| Photograph, `/api/photo` `direction=en-bs` | a sign reading EXIT ONLY / NO PARKING | read "EXIT ONLY NO PARKING", answered "IZLAZ SAMO BEZ PARKIRANJA" |
| Photograph | DANGER / HIGH VOLTAGE / KEEP OUT | read correctly, answered "OPASNA VISOKA VOLTAŽA OSTAJE NAPOLJU" |
| Photograph | PHARMACY / OPEN 24 HOURS | read correctly, answered "OTVORENA **FARMACIJA** 24 SATA" |
| Photograph | PLEASE DO NOT FEED THE ANIMALS, on a canvas too narrow for it — **my error, see the correction below** | read "PLEASE DO NOT FEE", answered "Molim vas, ne jedite" |
| Photograph | the same sentence on a canvas wide enough | read **"PLEASE DO NOT FEED THE ANIMALS"** exactly, answered "Molim vas, nemojte hraniti životinje" — correct |

The Bosnian is Bosnian and not Croatian or Serbian on every chat row that could
have gone either way: *dvije* not *dve*, *vrijeme* not *vreme*, *kćerka*,
*kišobran*. That is the form-rate instrument's claim holding up in ordinary use.

**A correction, and it is mine.** The first version of this report claimed the
reader had truncated "PLEASE DO NOT FEED THE ANIMALS" to "PLEASE DO NOT FEE" and
called that the most user-visible defect found. **That was wrong, and the fault
was in my test, not in the reader.** I had drawn 30 characters at 92 px — 1,673
pixels of text — onto a 1,000-pixel canvas, so the sentence ran off the right
edge. The reader read exactly what was in the image; the visible text ends at
1,014 px, which is "PLEASE DO NOT FEE" to the pixel. Re-drawn on a canvas wide
enough, the reader returns the full sentence, and so does a 59-character line
2,049 pixels wide:

| image | read as |
|---|---|
| PLEASE DO NOT FEED THE ANIMALS, 1,672 px wide | "PLEASE DO NOT FEED THE ANIMALS" — exact |
| the same text at a smaller size, 1,016 px | "PLEASE DO NOT FEED THE ANIMALS" — exact |
| NO SMOKING ANYWHERE ON THESE PREMISES INCLUDING THE TERRACE, 2,049 px | exact |

There is no line-length defect in the reader. The retraction is recorded here
rather than quietly edited out, because a false defect sent to the owner is the
same kind of error as a flattering number.

**What the corrected test found instead is worth more than the false finding
was: the translator is much worse on ALL-CAPS input, and signs are written in
capitals.** Same sentences, once as a sign writes them and once as a person
would:

| sign text, uppercase | what Lilly answers | the same sentence in ordinary case |
|---|---|---|
| DANGER HIGH VOLTAGE KEEP OUT | "OPASNA VISOKA VOLTAŽA OSTAJE NAPOLJU" — *"dangerous high voltage stays outside"* | "Opasnost, visok napon. Drži se podalje." — **correct** |
| PHARMACY OPEN 24 HOURS | "OTVORENA FARMACIJA 24 SATA" — *farmacija* is the academic subject | "Ljekarna, otvorena 24 sata." — the right sense, though *ljekarna* is the Croatian word where Bosnian says *apoteka* |
| NO SMOKING ANYWHERE ON THESE PREMISES INCLUDING THE TERRACE | "NEMA PUŠENJA NIGDJE NA OVIM **PREMIJERAMA** UKLJUČUJUĆI TERASU" — *premijerama* is premieres, or prime ministers | — |
| PLEASE DO NOT FEED THE ANIMALS | "Molim vas, nemojte hraniti životinje" — correct even in capitals | identical |

The photograph path hands the reader's output straight to the translator, so on
that path the translator meets capitals and on the chat path it meets ordinary
prose. Nobody had measured what that costs. It is a serving-path question of
exactly the kind this project has already been paid for once: splitting input
into sentences before translating was worth +0.89 BLEU. Section 5 carries the
measurement.

Two faults survive the correction, both in the translator rather than the reader,
and neither would show up in a BLEU score on FLORES sentences: *farmacija* for a
pharmacy you walk into, and *ostaje napolju* for "keep out".

One more observation, from sending the English clip with the wrong direction
flag on purpose: told to expect Bosnian, the listener invents Bosnian-shaped
words out of English audio ("Gdje je bus stacija?") rather than failing. The
direction switch is therefore load-bearing, not cosmetic — and a user who
forgets to flip the arrow gets plausible nonsense rather than an error.

## 2. What the published numbers actually say

Every reply-direction figure on GitHub was recomputed from the artifact behind
it. `training/verify_published_en_bs.py` does the whole thing in one command and
exits non-zero if a published score stops reproducing.

| Published claim | Where | Artifact | Verdict |
|---|---|---|---|
| base 29.57 BLEU / 58.96 chrF2, FLORES-200 2,009 | `README.md:230`, `RESULTS-en-bs.md:10` | `training/hypotheses-en-bs.json` | **reproduces exactly** |
| Lilly 30.73 / 60.00, same pairs | `README.md:230`, `RESULTS-en-bs.md:12` | same | **reproduces exactly** |
| in-house 1,500: 31.94 / 58.74 → 34.06 / 60.11 | `RESULTS-en-bs.md:9,11` | same | **reproduces exactly** |
| Bosnian form rate 94.3% → 99.2%, 244 of 246 decided | `README.md:286` | `training/form-rate/{base,tuned}.json` | **reproduces** — recounted from the 338 per-target marks, not from the stored summary: 231/245 = 94.29%, 244/246 = 99.19% |
| 338 audited bench targets | `README.md:286` | `training/form-rate/audit.json` | **reproduces** — 385 targets − 47 with no counterpart = 338 |
| `>>bos_Latn<<` versus `>>hrv<<` gap, 21.8 → 22.5 points | `README.md:287` | the four form-rate files | **reproduces** — 94.29 − 72.53 = 21.75, 99.19 − 76.68 = 22.51 |
| language tag leaked into 0 of 2,009 outputs | `README.md:230` | this session's served-path run | **reproduces** — 0 of 2,009 on both the base and the fine-tune |
| the served build is the published one | `models/lilly/translator-en-bs/built.json` | md5 of `adapter-en-bs/adapter_model.safetensors` | **reproduces** — `c28a02c6…`, and the build fingerprints as `6f240bb14aa56ea7ae1c8a19cb25faab`, the digest the 8 September publish recorded |

So the reply direction's published numbers are sound. Nothing was inflated.

## 3. The number that is not what a reader will think it is

`training/RESULTS-en-bs.md`'s 30.73 BLEU / 60.00 chrF2 was produced by
`training/evaluate.py`: the PyTorch base plus its LoRA adapter, each test row
fed in whole. **That is not the path a user meets.** The app serves an int8
CTranslate2 build and splits input at sentence boundaries first. Until today
nobody had scored that build — `training/evaluate_app.py` had no `--direction`
option at all, so the reply side could not be put through it.

It can now. Same 2,009 FLORES pairs, both columns int8 through
`app.translate.Engine`:

| | Published path (PyTorch + LoRA, whole rows) | **Served path (int8, app splitter)** |
|---|---|---|
| Base | 29.57 / 58.96 | 31.23 / 60.93 |
| Lilly | 30.73 / 60.00 | **32.22 / 61.55** |
| Gap | +1.16 / +1.04 | **+0.99 / +0.62** |
| Interval on the gap | BLEU [+0.67, +1.65] | BLEU [+0.47, +1.46], chrF2 [+0.33, +0.89] |

Both intervals are 1,000 paired resamples over sentences, 0 at or below zero.
On the devtest half alone — 1,012 segments, the set other people's systems are
quoted on — it is 31.49 / 61.03 → **32.45 / 61.75**, +0.96 BLEU at p = 0.003.

Two things follow, and they point in opposite directions:

1. **The product is better than the published figure says.** What a user's
   sentence actually gets is 32.22 BLEU / 61.55 chrF2, which is +1.49 BLEU and
   +1.55 chrF2 above the number on the page.
2. **The fine-tuning deserves less of the credit than the published figure
   implies.** Its own contribution through the served path is +0.99 BLEU and
   +0.62 chrF2, not +1.16 and +1.04. The sentence splitter lifts both columns and
   lifts the untuned base more, because part of what the fine-tune learned was
   how to survive multi-sentence rows the app never hands it.

This is the same shape the forward direction already found and documented
(+0.54 BLEU raw against +1.23 through the app, `training/RESULTS-product.md`).
The forward direction says "as the user sees it" in the README and has an
app-path results file; the reply direction said neither until now. The new file
is `training/RESULTS-product-en-bs.md`.

**A caveat that belongs beside this number and not under it.**
`training/RESULTS-bosnian-audit.md` measures FLORES's Bosnian side at 49%
Bosnian by lexical marker against this project's training data at 77%. In this
direction the Bosnian text is the *reference*, so part of any chrF2 movement
here measures which standard the reference was written in rather than
translation quality. The form-rate instrument is the one that tests the
Bosnian-versus-Croatian question head-on, and it is unaffected, because it asks
which form the model wrote rather than how closely it matched a reference of
uncertain standard.

## 4. One published interval does not reproduce

The same +1.16 BLEU gap is published with two different 95% intervals:

- `README.md:284` — BLEU **[+0.70, +1.62]**, chrF2 [+0.70, +1.35]
- `training/RESULTS-en-bs.md:31` — BLEU **[+0.67, +1.65]**

A 1,000-resample paired bootstrap over the stored outputs
(`training/hypotheses-en-bs.json`, seed 11) returns BLEU **[+0.67, +1.65]** and
chrF2 [+0.69, +1.35]. The results file reproduces exactly. The README's BLEU
interval does not, and it is narrower at *both* ends, which is the signature of
a second bootstrap run under another seed rather than a typo — a typo usually
moves one bound. The chrF2 low end differs by 0.01, which is resampling noise
and not a disagreement.

Neither number is dishonest and the point estimate and p-value agree in both
places. But the README is the public face, and it carried the one interval a
reader cannot reproduce from the outputs that are on disk.

**Ruled by the owner on 12 September, after this was reported: corrected.**
`README.md` now publishes [+0.67, +1.65] for BLEU and [+0.69, +1.35] for chrF2 —
the values that come out of the stored outputs — and names
`training/verify_published_en_bs.py` beside them, which recomputes all of it in
one command and exits non-zero if any published score stops reproducing. The
Hugging Face model card was regenerated from the README by
`scripts/sync_model_card.py`, per the standing rule that the card says what the
README says. Uploading the card to the Hub remains the owner's act and has not
been done from here.

## 5. Smaller findings

- **An undocumented drift in the forward direction's base.**
  `training/form-rate/base-flores-bs-en.json` — written by
  `training/verify_base_flores.py`, whose own docstring says "a bar that has
  moved is a finding for the write-up" — holds 41.48 BLEU / 67.55 chrF2 against
  the committed 41.60 / 67.58 in `training/RESULTS.md:8`. That is −0.12 BLEU and
  −0.03 chrF2 on the *untuned base* of the *forward* direction, so it touches no
  claim about Lilly, and it is small. It is also sitting in an artifact with
  nothing written about it anywhere in `docs/` or `training/*.md`. The en-bs half
  of the same file reproduces exactly.
- **English hearing and English reading are still unmeasured, and the README
  says so** (`README.md:122`) — correctly. This session tested those paths but
  did not score them: the English clip fed to the listener was synthesized by
  the app's own English voice, not spoken by a person, and the four photographs
  were made locally rather than drawn from a photograph set with a truth file.
  Both are honest functional tests and neither is a number. The three faults in
  section 1 are observations on four synthetic signs — far too few to put a
  percentage on, and `training/RUBRIC.md` would void such a score anyway.
- **The reply direction's base is a smaller model than the forward
  direction's**, so 61.55 chrF2 here and 68.10 chrF2 the other way are not
  comparable quality. The README already states this and it stays true.

## 6. What should happen next

1. ~~The owner rules on the interval in `README.md:284`.~~ **Done 12 Sep** —
   ruled and corrected to the reproducible interval, see section 4.
2. ~~`README.md` gains, for the reply direction, the same "as the user sees it"
   treatment the forward direction has.~~ **Done 12 Sep, on the owner's
   instruction.** `README.md` and `docs/WHITE-PAPER.md` now quote
   32.22 / 61.55 (and 32.45 / 61.75 on devtest) as the served score, with
   30.73 / 60.00 kept and labelled as the adapter path the pre-registered bars
   were cleared on. Both are true; only one is what a user gets. The white
   paper's en-bs table now carries both columns side by side rather than one.
   The pre-registration was not touched: bars are judged on the path they were
   registered on, and they were.
3. The photograph faults in section 1 deserve a real measurement rather than
   four synthetic signs: an English-text photograph set with a truth file, and a
   first English word-error rate on human English speech. Neither is expensive.
4. The reader truncating a long line and the pipeline then answering fluently
   about the fragment is the most user-visible defect found today, and it is not
   about the reverse direction at all — the same reader does it in both.

---

Reproduce section 2 and section 4 with:

```bash
.venv/bin/python training/verify_published_en_bs.py --resamples 1000
```

Reproduce section 3 with (about four hours of laptop CPU for both columns):

```bash
.venv/bin/python scripts/build_translator.py --direction en-bs --no-adapter \
    --dest models/lilly/translator-en-bs-base
.venv/bin/python training/evaluate_app.py --direction en-bs
```
