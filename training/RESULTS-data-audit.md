# read-veri — is the extra corpus clean, aligned, and actually unseen?

v2, `read` lane. Every number below is reproducible by the command printed
beside it. Nothing here is copied from an earlier document; where an earlier
document is confirmed or contradicted, it is named.

The task has three parts and they are not equally hard. Leakage and alignment
are bookkeeping. "Data the base model has not seen" is the real one, and it is
the one that had never been checked against evidence.

---

## 1. What the base model actually is

`models/lilly/translate/config.json` is a `MarianMTModel`, and
`tokenizer_config.json` records:

    "name_or_path": "marian-models/opusTCv20210807+bt_transformer-big_2022-03-17/zls-en"

So the base is **Helsinki-NLP/opus-mt-tc-big-zls-en** — South Slavic to English,
trained on **OPUS Tatoeba Challenge release v2021-08-07** plus back-translation,
built 2022-03-17. `models/lilly/NOTICE.md` says the same. This matters because
"unseen" is now a dated claim about a specific, published corpus rather than an
intuition, and it can be checked.

The `eng-hbs` training set of that release is **159,611,854 pairs over 44 OPUS
sub-corpora**. Its per-row corpus manifest (`train.id.gz`, 170 MB, fetched by
HTTP Range out of the 8.85 GB release tar) breaks down as:

| corpus | in the base's training data | rows |
|---|---|---|
| WikiMatrix-v1 | **yes** | 1,004,991 (164,104 tagged `bos_Latn`) |
| SETIMES-v2 | **yes** | 556,459 (134,450 `bos_Latn`) |
| TED2020-v1 | **yes** | 459,916 (11,167 `bos_Latn`) |
| wikimedia-**v20210402** | **yes** | 115,410 (10,269 `bos_Latn`) |
| Tatoeba | no — dev/test only | 0 in train |
| NTREX | no | 0 |

**The consequence nobody had written down: our entire `data/clean/train.tsv` is
inside the base model's training data.** WikiMatrix, SETIMES and TED2020 are
313,083 of its 313,612 rows. So is `valid.tsv`, and so is `test.tsv` — both are
drawn from the same WikiMatrix/SETIMES/TED2020/Tatoeba pool. Fine-tuning on
`train.tsv` was never showing the model anything new; it was re-weighting
material it already had. That is not a defect, and it does not invalidate a
number, but it is the correct frame for reading a +1.29 BLEU gain.

## 2. Is the extra corpus unseen?

`data/extra/extra-train.tsv`, 38,277 rows, md5 `98c1ab02e3ccbbda8b8450a1a2e29bee`:
36,353 `wikimedia` + 1,924 `ntrex`.

**NTREX — unseen, established by date.** NTREX-128 was published by Microsoft on
2022-11-24 (`github.com/MicrosoftTranslator/NTREX`, earliest commit 2022-10-31),
which is 15 months after the 2021-08-07 data cut. It also appears zero times in
the base's own 44-corpus manifest. Both independently.

**wikimedia — a different release, and the overlap was already removed.** OPUS
publishes the bs-en `wikimedia` corpus as v20210402 (12,090 pairs), v20230407
and v20260327 (57,865 pairs, current). Ours is v20260327. The base trained on
v20210402. Downloading v20210402 (1.5 MB) and comparing NFC-normalised,
whitespace-collapsed, case-folded against our 36,353 wikimedia rows:

    0 full pairs in common
    86 rows (0.24%) where only the Bosnian side matches

so the corpus-level claim holds.

**But a corpus-level claim is not a sentence-level claim, and this is where the
previous reasoning stopped.** 43 of those 44 corpora are web-mined — CCMatrix,
CCAligned, ParaCrawl, OpenSubtitles — and web mining scrapes Wikipedia, which is
where both our extra corpus and FLORES come from. Checking only the corpus named
`wikimedia` checks 115,410 rows out of 159.6M and declares the other 159.5M
clean by omission.

So the Bosnian side of the base's full training set was streamed and matched
sentence by sentence — 4.59 GB decompressed in flight, never written to disk.
Result at the time of writing, with the stream still running:

    (see "Membership result" at the bottom — this section is updated when it lands)

## 3. Leakage

`python3 data/scripts/audit_translation_leakage.py`

Whole-pair overlap, three training corpora against three held-out sets, on
NFC-normalised, citation-stripped, whitespace-collapsed, case-folded text:

| | FLORES (2,009) | test.tsv | valid.tsv |
|---|---|---|---|
| train.tsv (313,612) | **0** | 0 | 0 |
| extra-train.tsv (38,277) | **0** | 0 | 0 |
| train-mix.tsv (361,621) | **0** | 0 | 0 |

**FLORES leakage is 0 — and it is 0 on the file that is actually trained on.**
That last column is the one no previous check looked at. `build_training_mix.py`
splits every row into one sentence per example, so a held-out sentence sitting
inside a longer paragraph is invisible to a whole-row comparison against the two
*inputs* and would land as a standalone sentence-level match in the *output*.
Checking the inputs and not the output is checking the wrong artifact. It is
clean, but it was clean unverified until now.

One-side matches, which are not leaks but are reported rather than waved through:
train.tsv has 1 bs / 1 en against test and 1 bs / 4 en against valid;
extra-train.tsv has 1 en against test (`"Aftenposten (in Norwegian)."`).
Two of the valid ones are real sentences rather than boilerplate — SETIMES rows
whose English is *identical* to a valid row's English while the Bosnian differs
slightly. The exact reference is in the training data. valid.tsv drives early
stopping only, and it is 2 rows in 1,500.

### Paraphrase leakage

`python3 data/scripts/audit_translation_leakage.py --near`

Exact matching answers "was this copied", not "was this copied and edited".
FLORES is built from Wikinews/Wikijunior/Wikivoyage and our largest new corpus is
Wikimedia, so a paraphrase leak is the plausible one. Jaccard over word types,
English side, nearest training row for each held-out row:

    FLORES vs train.tsv    max 0.895, p99 0.429, median 0.219    >=0.9: 0  >=0.7: 1
    FLORES vs extra-train  max 0.562, p99 0.429, median 0.217    >=0.9: 0  >=0.7: 0

**The single hit at 0.895** is FLORES `dev`, and its twin is `train.tsv` line
58090, WikiMatrix:

    FLORES dev  en: Turkey is encircled by seas on three sides: the Aegean Sea to the
                    west, the Black Sea to the north and the Mediterranean Sea to the south.
    train 58090 en: The country is encircled by seas on three sides: the Aegean Sea to the
                    west, the Black Sea to the north and the Mediterranean to the south.

One sentence in 2,009, in the *old* corpus, so every previously published number
carries it too. It is recorded, not fixed: removing it would change `train.tsv`,
whose row count `build_training_mix.py` asserts, and 1/2009 = 0.05% of the
evaluation set cannot move BLEU or chrF2 at the reported precision.

The high scores against test.tsv (29 rows at >= 0.9) are an artifact of the
metric and not leakage. The token pattern is letters-only, so digits are
invisible: SETIMES bylines differing only in date and citation lines differing
only in page number score a perfect 1.000. This is written into the script's
docstring so the next reader does not re-derive it as alarm.

## 4. Alignment, by hand

**30 pairs, drawn at random (seed 20260828) from `extra-train.tsv`.** 29 aligned.
One misaligned: row 13600, where the Bosnian is missing the entire first clause
of the English ("It is likely that Saint had a working model but there is no
evidence of one; he was a skilled cabinet maker and ..."). Its log character
ratio is 0.445 against a band of [-0.28, +0.40], so **step 3 of the mix removes
it** and it does not reach training — verified: 0 fragments of it in
`train-mix.tsv`.

**A second 28 pairs, drawn from the rows the splitter actually splits**, because
that is where alignment can break and a random sample mostly misses it. 26
aligned. Two defects, and the second is a class of failure nothing in the
pipeline can catch:

- One partial: the English carries a clause the Bosnian does not.
- One **crossed**: equal sentence counts on both sides, different fact order.

      BS[0]  ... objavljen je 7. februara 2020. godine, a producirao ga je Nick Patrick.
      EN[0]  ... was released on 7 February 2020, in which he plays alongside the London Symphony Orchestra.
      BS[1]  Na albumu Hauser svira u pratnji Londonskog simfonijskog orkestra.
      EN[1]  The album was produced by Nick Patrick.

  Step 2 pairs sentences with `zip()` after checking only that the two sides have
  the same *count*. When a translator reorders two facts across a sentence
  boundary — normal, correct translation — the count check passes and `zip()`
  produces two confidently wrong pairs.

### How common is the crossed case

Measured, not guessed. For every equal-count multi-sentence row in
train.tsv + extra-train.tsv (10,803 of them), score the `zip()` alignment against
every other permutation using anchors that survive translation — digits and
capitalised tokens — and flag rows where a reorder scores strictly better:

    flagged: 169 of 10,803 (1.56%)
      WikiMatrix 22/325 (6.77%)   wikimedia 125/4,120 (3.03%)
      SETIMES    20/5,975 (0.33%) TED2020    2/366     Tatoeba 0/14   ntrex 0/3

The flag is not the answer. 20 flagged rows were read by hand and **6 are
genuinely misaligned** — the Hauser row above, a citation offset by one, a Val
Kilmer biography offset by one, a sentence about a diary paired with one about a
burial, a TED2020 row offset by an untranslated "U redu.", and a WikiMatrix row
pairing the Qur'an with Isaiah. The other 14 are correctly aligned and merely
score higher under a permutation by coincidence (symmetric anchors: a photo
caption and its `[Courtesy of X]` credit line share every name).

So precision is roughly 30%, giving **on the order of 50 genuinely crossed rows**
detected. Recall is unmeasured and certainly below 1 — a crossed pair with no
names or numbers in it cannot be flagged this way — and the 12-row hand sample of
split wikimedia rows contained one, which would imply a higher rate than 50. The
honest statement is: **the true count is at least ~50 rows and plausibly a few
hundred, out of 351,889 — under 0.1% of the corpus.** Too small to move a
training result, large enough that it should be recorded rather than discovered
again later.

## 5. What this does and does not license

Established:

- FLORES leakage is 0, exact and near-duplicate, against all three corpora
  including the mixed file that is actually trained on.
- valid/test whole-pair leakage is 0; the one-side hits are boilerplate except
  two SETIMES rows whose English reference is in training.
- Alignment on the extra corpus is good: 29/30 random, and the one bad pair is
  filtered before training.
- NTREX (1,924 rows) is unseen by the base model, by publication date and by the
  base's own manifest.
- The wikimedia corpus is a later OPUS release than the one the base trained on,
  with 0 full-pair overlap against it.

Not established, and neither should be asserted:

- That the base has never seen our wikimedia *sentences* through some other
  web-mined corpus. That is what the streaming check is for, and until it lands
  the claim is corpus-level only.
- That "the model learned better Bosnian" rather than "the model adapted to news
  style". Nothing in this audit touches that, and `PREREGISTRATION.md` already
  says no result of that run distinguishes them.

## Membership result

*(pending — the 159.6M-line stream is running; this section is filled in with the
count, the command, and the FLORES row, whatever it says.)*
