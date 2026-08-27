# What a score out of ten means

Written on 27 August 2026, before the measurements it grades. That order is the
only thing that makes it worth anything: a scale drawn after the numbers are in
is a description of the numbers, not a judgement of them.

The thresholds were not set by whoever built the models. They came from agents
given the task and the language pair and deliberately **not** told how Lilly
currently scores, then each proposal was handed to a second agent asked to
attack it — to name any invented citation, and to find any band an unmodified
off-the-shelf model would clear by accident. What follows is what survived.

Every band is anchored to a real published system. Where no published anchor
exists, that is said rather than papered over.

---

## Translation — chrF2 on FLORES-200 devtest, bos_Latn to eng

1,012 segments, single reference, sacreBLEU 2.x, signature
`chrF2|nrefs:1|case:mixed|eff:no|nc:6|nw:0|space:no`, generated through the
served pipeline — int8 and the sentence splitter both — with the splitter's
output re-joined to one line per segment.

**Not BLEU.** Kocmi et al. (ACL 2024, Table 2) measured how large a difference
each metric needs before it agrees with human judgement: chrF needs 1.00 points
for 70% agreement and 3.05 for 90%. BLEU needs 1.39 and *never reaches 90% at
any difference*. The entire published gap on this language pair between a free
zero-effort download and Meta's flagship is 2.7 chrF2 — narrower than BLEU can
resolve. So BLEU is reported, and it does not set the score.

| score | chrF2 | anchored to |
|---|---|---|
| 10 | ≥ 67.2 | NLLB-200-3.3B, the best of 30 systems on the OPUS-MT dashboard for this pair |
| 9 | 66.5 – 67.1 | between NLLB-1.3B (66.6) and the 3.3B |
| 8 | 65.5 – 66.4 | above every system below the NLLB 1.3B tier |
| 7 | 64.6 – 65.4 | clear of NLLB-200-distilled-600M (64.5) — the free download |
| 5–6 | 63.5 – 64.5 | at the free download's level |
| 3–4 | 60.0 – 63.4 | below it |
| 2 | 57.4 – 59.9 | near mozilla/tiny_bsen (57.4), a 17M student model |
| 1 | < 57.4 | below the smallest system on the board |

**A gain only counts if a paired bootstrap rejects a tie at p < 0.05**
(`sacrebleu --paired-bs`, n=1000). This is not ceremony. Kocmi et al. (WMT 2021)
found chrF's agreement with humans rises from 75.6% to 85.4% purely by
discarding differences the bootstrap calls tied, and among 203 pairs where BLEU
flipped the human ranking the median difference was 1.3 points. Marie et al.
(ACL 2021) moved BLEU 6.8 points on a WMT20 submission by corrupting a single
sentence, and the bootstrap still found nothing significant.

**Gap, recorded rather than hidden:** our own base model,
`opus-mt-tc-big-zls-en`, has *no published bos→eng FLORES score* — FLORES-101
had no Bosnian, so the model card has no row for it. Its published neighbours
from the same release are hrv→eng 63.9 and srp_Cyrl→eng 67.8. So the base's
position on this table has to be measured here, not looked up.

---

## Speech — word error rate on FLEURS bs_ba test

925 utterances, forced `<|bs|><|transcribe|>`, greedy at temperature 0, no
external language model, scored after Whisper's `BasicTextNormalizer`.

| score | WER | meaning |
|---|---|---|
| 10 | ≤ 8% | dictation people would actually use |
| 9 | 8 – 12% | |
| 8 | 12 – 16.5% | |
| 7 | 16.5 – 21% | |
| 6 | 21 – 25% | |
| 5 | 25 – 30% | |
| 4 | 30 – 34% | and the gain over the zero-shot baseline must clear a bootstrap interval |
| 3 | 34% – (baseline − 2.5) | |
| 2 | within ±2.5 of our own zero-shot whisper-small | indistinguishable from doing nothing |
| 1 | worse than baseline + 2.5 | |

The band-2 rule is the important one. A fine-tune that lands within noise of the
checkpoint it started from has not earned a score for the checkpoint's ability,
and the only way to know is to re-measure that zero-shot baseline ourselves
rather than citing someone else's.

**Lilly reads 35.5% today.** On this scale that is a 2 or a 3, depending on where
the zero-shot baseline actually sits — and we have not measured it. That is the
next thing to measure, not the next thing to assume.

---

## Photographs — strict end-to-end word F1

ICDAR2015 Task 4.4 matching: a word counts only when a detected box overlaps a
true box by more than half AND the transcription matches exactly, diacritics
included. Detector and recogniser scored together, because that is what a person
points a camera at.

**A gate before any band is assigned.** The test set must be at least 200 real,
incidentally-captured, held-out Bosnian photographs, transcribed by eye. Below
that the score is **void, not low** — the interval around it is too wide to mean
anything.

| score | F1 | recall floor |
|---|---|---|
| 10 | ≥ 68 | 60 |
| 9 | 62 – 68 | 55 |
| 8 | 57 – 62 | 50 |
| 7 | 53 – 57 | 46 |
| 6 | 48 – 53 | 42 |
| 5 | 42 – 48 | 36 |
| 4 | 35 – 42 | 30 |
| 3 | 27 – 35 | |
| 2 | 18 – 27 | |
| 1 | < 18 | |

The score is the **lower** of what F1 earns and what recall permits. A reader
that is precise about the few words it finds and silent about the rest is not a
good reader, and an F1 average lets it look like one.

**Where we stand against that gate:** 40 photographs are being transcribed now.
That is a fifth of what a valid score needs. What it can produce is an estimate
with an honest interval — enough to know whether the 75% measured on our own
synthetic text was ever a real number — and it will be reported as an estimate.

---

## What this scale cannot see

None of these metrics know whether a translation is *wrong* in a way a reader
would act on. chrF2 cannot tell a dropped negation from a dropped comma. WER
counts every word the same, so a mangled street name and a mangled "the" cost
the same. F1 says nothing about whether the words it missed were the ones on the
sign that mattered.

They are the right scales because they are comparable to published work. They
are not a substitute for reading the output.
