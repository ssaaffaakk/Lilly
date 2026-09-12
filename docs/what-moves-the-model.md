# What would actually move the model — the brainstorm, ranked

Written 12 September 2026 from the repository's own numbers. No model was
loaded, nothing was trained, nothing was launched. This is the case for where
the next GPU hours should go, and the case against the places they keep going.

**The one sentence:** *feed it real Bosnian it has never seen, and judge it
with a Bosnian ruler — everything else has already come back null.*

## The pattern in the record

Twelve training runs across four abilities. Almost every one came back null,
and the nulls have two causes between them:

| the run | why it could not win |
|---|---|
| translation fine-tune (bs→en) | 313,083 of `train.tsv`'s 313,612 rows are **inside the base model's own training data** (`RESULTS-data-audit.md`). It was re-weighting, not teaching. chrF2 moved −0.16, p = 0.128 |
| OCR passes 8–19 | labels written by the reader being trained (`OCR-ROADMAP.md`). A model fine-tuned on its own confident output learns what it already knows |
| PaddleOCR fine-tunes 7 / 7a / 7b | the recogniser *did* learn (diacritics 62.1% → 77.1%) and still lost the decider on test-v2 — a set of 132 photographs where the rubric calls the score **void below 200** |
| speak-bs, speak-parla, speak-control | warm-started from a **Lower Sorbian** checkpoint (West Slavic) filed as Serbian; all three arms lost, the control included, on the checkpoint's own studio data |
| speak-youtube | 0 of 20 source files carry speech energy to 11,025 Hz — a 22.05 kHz voice must *generate* that band |

Two causes, stated plainly:

1. **The data was already in the model,** or came from the model.
2. **The ruler could not see the thing being trained.** FLORES's Bosnian side
   is **49% Bosnian** (95% 36–61) against our training data's **77%** (76–77),
   intervals not overlapping (`RESULTS-bosnian-audit.md`). Every published
   translation number is scored there. A fine-tune that moves toward the
   standard of its own data is *marked down* by a reference that writes
   *tisuća*, *tjedan* and *povijest* in a third of its decidable sentences.

So the question "what training helps" has a precondition: a run only helps if
its data is new to the model **and** its instrument can see the gain.

## The levers, ranked by what can move a held-out number

### 1. Back-translation into English → Bosnian — the strongest candidate

Real monolingual Bosnian (bs.wikipedia, CC BY-SA is the clean default) run
through the shipped bs→en model: **synthetic source, real Bosnian target**.
It is the only lever in this repository where the targets are Bosnian written
by Bosnians and the material post-dates the base's 2021-08-07 data cut.

- It is the one thing that moves chrF2 *and* the form rate, because the form
  rate is a property of the target side and the targets here are human.
- The instrument already exists and already has a pre-run baseline: the base
  writes the Bosnian form on **231 of 245 decided targets, 94.3%** (95% Wilson
  90.6–96.6), 93 silent of 338 (`RESULTS-en-bs-formrate.md`). The shipped LoRA
  reads 99.2%.
- **The headroom is in the silences, not in the 5.7 points.** 93 of 338 targets
  produced neither form. A target the model declines to commit on is a Bosnian
  word it does not have. That is what new Bosnian targets can supply, and it is
  the row to pre-register on.
- Blocked on owner decision 4 (the monolingual source and its licence),
  `V4-PLAN.md`. Nothing else blocks it.

### 2. The Bosnian ruler — not training, but it gates everything above

FLORES cannot score Bosnian-ness; the form rate can, but on one axis and 338
targets. Until a held-out set exists that is actually Bosnian and large enough
to resolve a two-point delta, a real gain from lever 1 is invisible or
penalised. `docs/BOSNIAN_METRIC.md` already specifies it (200 targeted + 50
control sentences); `bench/terms.tsv` and `bench/cases.tsv` exist. This costs
no GPU and decides whether the next run can be read at all.

### 3. Speech — one language token per clip

The only lane where a **diagnosed recipe defect** is the cause, not a data
limit. 6,050 Croatian rows trained under `<|bs|>` against 6,182 Bosnian, while
Whisper's pretraining gave that token 11 hours of Bosnian against 91 of
Croatian. Six thousand examples taught it that Bosnian is spelled *Europom*.
That is the exact row that closed large-v3 (Croatian substitution 1.1% → 6.1%,
p = 0.018). The fix is one column in the mix file, it is written, and it is
pre-registered. It is blocked only on the owner's rule-3 ruling on which base
carries it.

### 4. Speech distillation — the 18.4 hours are good data pointed at the wrong ability

The YouTube corpus fails as voice data on its spectrum, but Whisper resamples
to 16 kHz and its filterbank stops at 8 kHz: those defects are harmless for
*recognition*, and variety across recording conditions is an asset there.
large-v3 → whisper-small is **teacher ≠ student**, so it is distillation, not
the circular move this project already closed on the reading side. Needs its
own pre-registration and a ruling on the transcripts' provenance.

### 5. Photographs — the ruler first, then Cyrillic, then nothing else

- The recogniser lever is spent across three looks; the detector lever is a
  clean negative at the 2 MP working size.
- **test-v2 is 132 photographs and the rubric voids a score below 200.**
  test-v2b's 160 are drawn and unread. Two blind passes make the union 292 and
  makes every reader number real. That is the highest-value photograph work
  and it needs no GPU.
- **Cyrillic is a capability gap, not a percentage point.** 628 Cyrillic key
  words on test-v2 and the reader cannot emit the alphabet. That is worth more
  than any point of Latin recall — and its labels must come from people.
- The 20,240 Mapillary photographs are not training data until their labels
  come from a human or a non-EasyOCR vision model.

### 6. The product loop — the only data that grows from use

`/api/feedback` already stores corrections. Approved corrections exported as
pairs are real Bosnian, written by users, unseen by any base. Volume is tiny
today; it is the only lever that compounds.

## What this says about the voice lane

Nothing here recommends training a voice. The next voice step is a
measurement: hear `sl_SI-artur` (Slovenian, same sub-branch as Bosnian, CC BY
rather than NC-SA) and `bg_BG-dimitar` as fetched, against the same 200-clip
prefix and the same bar every other candidate faced. A fetched voice landing
below 22.3% is what would give a fine-tune room to fall and still clear.

## The order, if only one thing happens

1. Owner decides the monolingual source and licence (decision 4).
2. Build the Bosnian held-out set — no GPU, and it makes run 1 readable.
3. Launch back-translation en→bs with four bars: FLORES chrF2 above the best
   shipped, BLEU floor, **form rate floor, and the silent count as its own
   row**, since that is where the headroom is.

Everything else on this page is ready to wait for that.
