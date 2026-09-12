# What would actually raise the numbers — 13 September 2026

The owner asked a plain question: can the measured percentages be raised, with
Kaggle or Colab Pro, and they do not mind what it takes. This is the answer,
written against this project's own evidence and against primary sources on the
web. Nothing here was launched; two things here were **measured on the Mac**,
and those two are marked as measurements rather than proposals.

Every licence below was read from the model card or the repository itself, not
from a blog post. That distinction cost one of the most attractive candidates
its place — see section 6.

Rules this report lives under, unchanged: nothing on the do-not-repeat lists is
proposed again (`docs/OCR-ROADMAP.md`, `docs/kaggle-fail-stop.md`); a product
change still needs its own pre-registration before it ships; and the listener's
language-token re-run already in flight is not duplicated here.

---

## The short version, ranked by gain per GPU-hour

| | what | GPU cost | evidence |
|---|---|---|---|
| 1 | **Restore case before translating a photograph** | **none** | measured here: ALL-CAPS costs the translator **25.5 chrF2**, and a crude fix gives **23.1** of it back |
| 2 | **Move the reply direction onto `opus-mt-tc-bible-big-…-sla`** | **none to test**, one session to fine-tune | measured here: **untuned it already draws level with the fine-tuned shipped model** — 32.11 BLEU / 60.47 chrF2 against 30.81 / 60.60 on 300 pairs. Apache-2.0, transformer-**big**, speaks `>>bos_Latn<<` |
| 3 | **Suppress the seven Croatian words at decode time** | **none** | CTranslate2 4.8.1 has `suppress_sequences`; the seven words are already enumerated |
| 4 | Back-translate CC0 Bosnian | one Kaggle run | MaCoCu-bs is **730M words, CC0**; back-translation at this data shape is worth ~+2.7 BLEU in the literature |
| 5 | Filter WikiMatrix | none to filter | it is 56% of their mix and a sixth misaligned by their own audit |
| 6 | Hear `sl_SI-artur` as fetched (agent 1's standing item) | none, ~30 min | CC BY 4.0, the only South Slavic Piper voice in Bosnian's own sub-branch; listening cannot lose |
| — | a Bosnian voice from a 2026 zero-shot TTS | — | **nothing found that is both permissive and speaks Bosnian**; their own conclusion stands |

Three of the top five need **no GPU at all**. That is the headline: the cheapest
wins available to this project right now are in the serving path and in the
choice of base model, not in more training.

---

## 1. The all-caps defect — measured, not proposed

**This is a measurement made today, and it is the largest single number in this
report.**

While testing the photograph path I fed the reader four English signs. The
reader read them correctly. The *translator* did not, and the reason is that
signs are written in capitals:

| the sign | what Lilly answered | the same sentence in ordinary case |
|---|---|---|
| DANGER HIGH VOLTAGE KEEP OUT | "OPASNA VISOKA VOLTAŽA OSTAJE NAPOLJU" — *dangerous high voltage stays outside* | "Opasnost, visok napon. Drži se podalje." — correct |
| NO SMOKING … ON THESE PREMISES … | "…NA OVIM **PREMIJERAMA**…" — *premieres*, or *prime ministers* | — |
| PHARMACY OPEN 24 HOURS | "OTVORENA FARMACIJA 24 SATA" — the academic subject | "Ljekarna, otvorena 24 sata." — right sense |

So I measured it. 300 FLORES devtest pairs, the served int8 build, three
variants of the same source, one reference:

| variant | BLEU | chrF2 |
|---|---|---|
| A — as written (the chat path) | 30.81 | 60.60 |
| B — uppercased (what a sign looks like to the translator) | **14.24** | **35.05** |
| C — uppercased, then restored to sentence case | 27.22 | 58.11 |

| gap | BLEU | chrF2 | resamples at or above zero |
|---|---|---|---|
| B against A | **−16.57** [−18.48, −14.83] | **−25.54** [−28.22, −23.20] | 0 of 1,000 |
| C against A | −3.58 [−4.61, −2.57] | −2.49 [−3.18, −1.82] | 0 of 1,000 |

**Reading it.** Shouting at this translator costs it 25.5 chrF2 — the difference
between the shipped quality and something far below the untuned base. A four-line
`str.lower()` plus sentence capitalisation, with no knowledge of proper nouns at
all, gives back 23.1 of those 25.5 points. The remaining 2.49 is mostly that
crude restorer lowercasing *Sarajevo*; a restorer that kept known proper nouns
would close most of what is left.

**Why it happens** is well documented and not specific to this model: uppercase
word forms are effectively out-of-vocabulary, so the subword tokeniser splits
them aggressively into rare pieces whose embeddings are poorly trained
([On the Impact of Various Types of Noise on NMT](https://arxiv.org/pdf/1805.12282),
[Faithful Target Attribute Prediction in NMT](https://arxiv.org/pdf/2109.12105)).
The standard remedies are truecasing, factored models and inline casing; only
the first is available without retraining, and the measurement above says it is
enough.

**What this does and does not claim.** It is measured on uppercased FLORES
sentences, not on real sign text, and real signs are shorter. It says nothing
about the chat path, which already runs on variant A. It is a diagnostic, not a
product change: shipping a recaser needs its own pre-registration, and the
honest bar for it is a score on photographs, not on uppercased FLORES.

**One caution for whoever implements it.** Bosnian capitalisation is not English
capitalisation, and `str.capitalize()` on a word already containing a diacritic
must not mangle it. The measurement above used the crudest possible restorer
precisely so the number is a floor rather than a best case.

---

## 2. The reply direction's base is the only one Helsinki ever made

`README.md` calls the reply base "a smaller model than the forward direction's"
and leaves it there. Checked against the actual model list, it is worse than
that: **`Helsinki-NLP/opus-mt-tc-base-en-sh` is the only English→Serbo-Croatian
model in the tc series.** There is a `tc-big-sh-en` for the direction they are
not short on, and nothing big going the other way. The reply direction is not
using a small base by choice; it is using the only one that existed.

But there is a candidate that was not obvious, because its name does not mention
Serbo-Croatian at all:

**`Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-sla`**

| | |
|---|---|
| licence | **Apache-2.0** — clear for publishing a fine-tuned derivative |
| type | **transformer-big** — 907 MB in fp32, the same class as the forward direction's base |
| released | 2024-05-30, newer than the base in use |
| source languages | German, **English**, French, Portuguese, Spanish |
| target labels | includes **`>>bos_Latn<<`** — the exact label Lilly already sends |
| training data | `opusTCv20230926max50+bt+jhubc` — OPUS Tatoeba Challenge v2023-09-26, **plus back-translations**, plus the Johns Hopkins Bible Corpus |

The `bible` in the name is the release series, not the domain: the training data
is the full OPUS Tatoeba Challenge set with the bible corpus added for coverage.
It is a Marian model like the two already in the bundle, so
`scripts/build_translator.py` and `app.translate.Engine` would take it unchanged,
and the `>>bos_Latn<<` label the app already sends is one it already knows.

**The card publishes no per-pair score for English→Bosnian** — only an aggregate
43.8 BLEU over a multi-language Tatoeba set — so this is a lead, not a number.
Testing it needs **no GPU and no training**: download, convert to int8, and score
it through the app's own path against the same 2,009 pairs. That is a couple of
hours of laptop CPU, and it answers a question worth several points either way.
If an untuned big model beats the fine-tuned tc-base, the reply direction's whole
plan changes.

That test was started while writing this report; section 7 records where it got
to.

---

## 3. Seven words, no training

`docs/what-moves-the-model.md` established that 7 of the base's 14 form-rate
losses are a finite list of Croatian lexical choices — *Točno, travnja, lipnja,
tisućama, vlak, Europe, sudjelovati* — and concluded that "a decoder biased
against the Croatian member of each pair fixes them with zero training". Nobody
had checked whether the serving stack can actually do that.

**It can.** CTranslate2 4.8.1 — the version in `requirements.txt` — exposes on
`Translator.translate_batch`:

- **`suppress_sequences`** — "Disable the generation of some sequences of tokens"
- `target_prefix`, and `prefix_bias_beta` to bias towards a prefix rather than force it
- `return_alternatives`, which with a target prefix returns alternatives at the
  first unconstrained position
- `no_repeat_ngram_size`, `disable_unk`, `repetition_penalty`

Sources: [CTranslate2 Translator API](https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html),
[Decoding features](https://opennmt.net/CTranslate2/decoding.html).

`suppress_sequences` is the one that matters: the seven Croatian forms can be
tokenised once at load and suppressed for the life of the engine. The cost is a
list literal and one keyword argument.

**Two cautions.** Suppression is absolute — a suppressed sequence can never be
produced, so a sentence that genuinely needs *Europe* as a proper noun in another
sense would be harmed; the list must be checked word by word against real
sentences before it ships. And the form-rate instrument is the ruler here, not
BLEU: the seven words are 7 of 246 decided targets, so the effect on a corpus
BLEU score will be near zero even if the effect on Bosnian-ness is real. That is
an argument for measuring it with `training/bosnian_form_rate.py`, not with
sacrebleu.

---

## 4. Data: the CC0 answer to an open question

`training/RESULTS-*.md` and the v4 plan leave open question 4 as "licence of
monolingual Bosnian for back-translation". **It is answered, and the answer is
the most permissive licence there is.**

**MaCoCu-bs 1.0** — [CLARIN.SI, handle 11356/1808](https://www.clarin.si/repository/xmlui/handle/11356/1808):
2,749,435 texts, **730,342,880 words** of Bosnian, crawled from the `.ba` domain
in 2021–2022, cleaned, with per-text quality metadata for filtering. Licence:
**CC0 — no rights reserved.** Commercial use, redistribution and models trained
on it are all unrestricted, with no attribution requirement.

That is roughly two thousand times the target-side text in their current
training mix, free of every licence problem that has bitten this project.

Alongside it: the **MaCoCu parallel corpora are also CC0**
([MaCoCu/parallel_data](https://huggingface.co/datasets/MaCoCu/parallel_data)),
and **HPLT's parallel collection is English-centric and includes Bosnian** among
its 18 pairs ([HPLT deliverable](https://hplt-project.org/HPLT_D2_1___Initial_release_of_monolingual_and_parallel_data_sets-1.pdf),
[HPLT paper](https://arxiv.org/html/2403.14009v1)). There is also a
`CLASSLA-web.bs` Bosnian web corpus from the same community.

**What the literature says it is worth.** For English→Turkish with **300k real
pairs plus 3.2M back-translated** — almost exactly this project's shape, since
they have 361k pairs — the reported gain is **+2.7 BLEU on average**; across
studies the range for low-resource pairs is about **+1.0 to +4.7 BLEU**
([Analysis of Back-Translation Methods for Low-Resource NMT](https://www.researchgate.net/publication/336132251_Analysis_of_Back-Translation_Methods_for_Low-Resource_Neural_Machine_Translation),
[Improving NMT Models with Monolingual Data](https://arxiv.org/abs/1511.06709),
[Rethinking the Exploitation of Monolingual Data](https://direct.mit.edu/coli/article/50/1/25/118132/Rethinking-the-Exploitation-of-Monolingual-Data)).

Back-translation helps the **en→bs** direction most, which is the weak one:
synthetic data puts *real human Bosnian* on the target side, which is exactly
what target-side fluency needs. Note the shape of the job — the expensive part
is **inference over millions of sentences**, not training, and their bs→en model
is the one that would do the translating.

**A caution that is theirs already.** `training/RESULTS-bosnian-audit.md` found
FLORES's Bosnian side is 49% Bosnian by lexical marker against their training
data's 77%. Web-crawled `.ba` text will be genuinely Bosnian; FLORES may not
fully reward it. The form-rate instrument should be read beside any BLEU gain.

**Why this matters more than it looks.** `docs/what-moves-the-model.md` ranks the
zero-cost decoding lever (section 3) above data work *because the data side looked
blocked* — every large Bosnian source it could name was licence-encumbered, which
is exactly what happened with NLLB and ParlaSpeech-HR. Those two facts belong side
by side: **the data side is not blocked.** 730 million words of Bosnian under CC0
is a licence this project cannot be caught out by, and it reopens a lane that was
ranked low for a reason that no longer holds. The decoding lever is still cheaper
and should still go first; it is no longer the *only* thing left.

---

## 5. Kaggle or Colab Pro — the buying decision

Current as of September 2026, from the platforms' own documentation where
available and marked where it is community-reported.

**Kaggle, free.** ~**30 GPU-hours per week**, reset weekly; **T4 ×2** (16 GB
each, 32 GB total) or **P100** (16 GB); **12 hours** maximum per session (9 for
TPU); about 20 TPU-hours weekly. Sources:
[Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage),
[quota guide](https://aicreditmart.com/ai-credits-providers/kaggle-free-gpu-tpu-30-hours-week-access-guide-2026/).
This is what the project already uses, and it already works around the
Output-expiry trap with private datasets.

**Colab.** Pro **$9.99/month = 100 compute units**; Pro+ **$49.99/month = 500
units** plus **background execution up to 24 hours**; pay-as-you-go **$9.99 per
100 units**. Burn rates: **T4 ≈ 1.96 units/hour** (~51 h from 100 units),
**V100 ≈ 5/hour**, **A100 ≈ 15/hour** (~6 h on Pro, ~33 h on Pro+). Sources:
[Colab pricing summary](https://aicoolies.com/pricing/google-colab),
[GPU features and pricing](http://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/).

**The honest comparison for this project.** Compute units buy a *budget*, not a
reserved GPU — a Pro+ subscriber can still be handed a T4. Against Kaggle's 30
free T4-hours a week (about 130 a month), Colab Pro's 100 units buy roughly 51
T4-hours a month for $9.99. **So for T4-shaped work, Kaggle is already the better
deal and paying for Colab Pro would buy less compute, not more.**

Colab is worth money for exactly two things this project might want:

1. **An A100 (40 GB)**, which Kaggle does not offer at all. That is what a 7B
   QLoRA fine-tune or a comfortable 3B full fine-tune would want. Pro+ gives
   about 33 A100-hours a month for $49.99.
2. **24-hour background execution** on Pro+, against Kaggle's 12-hour ceiling.
   The voice runs already hit that ceiling — v5 stopped at a 5h30 wall and v6 at
   a 9-hour cap.

| job | fits free Kaggle? | note |
|---|---|---|
| test the big `-sla` base (section 2) | **no GPU at all** | CPU convert + score |
| recaser measurement (section 1) | **no GPU at all** | done, on the Mac |
| `suppress_sequences` (section 3) | **no GPU at all** | serving-path change |
| back-translate a few million CC0 sentences | yes, but it is the awkward one | inference at volume; several 12-hour sessions, or one A100 day |
| LoRA on a 1–3B MT model, 360k pairs | yes, T4 with 4-bit | tight but standard |
| 7B QLoRA | **no** | wants the A100 → Colab Pro+ |
| whisper-large-v3-turbo re-train | yes | already measured at 3h23m for half |
| TTS from ~1 hour of speech | yes | but see section 6 — the blocker is data and licence, not compute |

**Recommendation: do not buy Colab **Pro**.** For T4-shaped work it buys less
compute than Kaggle already gives away, and every one of the top three levers in
this report needs no GPU at all.

**Pro+ is a different question, and the argument for it is ergonomic rather than
volumetric.** Corrected after this section was first written: I claimed nothing
on today's list needs the 24-hour window, and that was wrong. The listener runs
as **two separate Kaggle kernels** solely because of the 12-hour wall — half 1
took 3h23m and half 2 4h18m, **7h41m in total, which is comfortably one job
inside a 24-hour background window**. Splitting a training run in two is not free:
it is two launches, two fetches, two chances to lose an artefact to Output expiry,
and a mix that has to be halved in the first place. The same wall already
truncated two voice runs — v5 stopped at a 5h30 cap and v6 at 9 hours with its
losses still falling.

So: Pro+ at $49.99/month buys one thing this project actually feels every week —
runs that finish in one piece — plus about 33 A100-hours for the jobs a T4 cannot
hold. Neither is urgent. Both are real. It is the owner's call, and it should be
made on the wall, not on the hours.

---

## 6. The voice: the attractive answer is licence-blocked

The obvious 2026 answer to "no Bosnian voice exists" is a zero-shot cloning
model, and there is one that looks perfect: **k2-fsa/OmniVoice**, 646 languages
listed **including `bs`**, cloning from 3–10 seconds of reference audio, 1.2
million downloads. Several write-ups describe it as Apache-2.0.

**Those write-ups are wrong, and this project cannot use it.** From the model
card itself: *"Our code is released under the Apache 2.0 License. The pre-trained
model is licensed under the CC-BY-NC due to constraints from its training data
(e.g., Emilia)."* Apache-2.0 is the **code**; the **weights are CC-BY-NC** — the
same licence that banned NLLB here.

This is the third time a licence has decided this project's direction, and the
second time a secondary source disagreed with the primary one. It is worth
recording as a pattern: **read the card, not the blog.**

The permissively licensed alternatives do not speak the language. Orpheus-3B
(Apache-2.0, 229k downloads) lists one language; Kokoro-82M (Apache-2.0, 11.6M
downloads) lists one, and this project already uses it for English. Qwen3-TTS is
not publicly resolvable on the Hub under that name.

**So the project's own conclusion survives contact with 2026: the remaining path
to a Bosnian voice is a clean hour from a native speaker.** That is a negative
result, and it is worth as much as a lead — it closes a line of enquiry that
looked open.

**One distinction, and one cheap thing that is still open.** OmniVoice is voice
*cloning* from seconds of reference audio; `docs/speak-checkpoint-comparison.md`
is about a Piper *warm start*. Different questions that happen to hit the same
licence wall, so the CC-BY-NC finding closes the first without touching the
second — and the second still has a recommendation that costs about **thirty
minutes and no training**: fetch **`sl_SI-artur`** (CC BY 4.0, Slovenian — the
only South Slavic Piper voice in Bosnian's own sub-branch, where the shipped
`sr_RS` voice turned out to be trained on Lower Sorbian) and simply *hear* it
against the 200-clip prefix. A fetched checkpoint that is only listened to cannot
lose: it either beats 22.3% or it does not, and either way nothing was spent.
That remains the cheapest open item in the voice lane and this report does not
displace it.

---

## 7. The base swap, measured — and it is a lead worth following

The section 2 candidate was fetched, converted to int8 with the repository's own
`scripts/build_translator.py`, and scored through `app.translate.Engine` on the
same 300 FLORES devtest pairs, the same scorer, the same serving path:

| | BLEU | chrF2 |
|---|---|---|
| candidate — `tc-bible-big-…-sla`, **untuned**, Apache-2.0 | **32.11** | 60.47 |
| shipped — Lilly, **fine-tuned** tc-base | 30.81 | 60.60 |
| gap | **+1.30** | −0.13 |

**An untuned base draws level with a fine-tuned one.** It is ahead on BLEU and
level on chrF2, having had no Bosnian fine-tuning at all. The fine-tuning that
was worth +1.16 BLEU / +1.04 chrF2 on the small base would be starting from
here instead, and the two effects are not obviously in competition.

**But it writes worse Bosnian, and that is the catch.** Four sentences by hand
show it immediately: *Gde je autobuska stanica* (Serbian ekavica, where Bosnian
is *Gdje*), *ćerka* for *kćerka*, *dvije kave* where Bosnian says *kafe*. Counted
over the 300 outputs with this project's own marker patterns: **10 ekavica hits,
0 Croatian-lexicon hits.** The shipped model exists partly to avoid exactly those
— its form rate is 99.2% against the small base's 94.3%.

So the reading is: **more raw translation ability, less Bosnian discipline.**
Which is the good version of this problem, because Bosnian discipline is the
thing this project already knows how to fix, and it fixed it with a LoRA that
cost one Kaggle session.

**What this is not.** 300 pairs is a small sample and no paired interval was
computed, so +1.30 BLEU is a lead, not a result. The honest next step is the
full 2,009 pairs with a paired bootstrap, both columns through the app's path,
plus `training/bosnian_form_rate.py` on the candidate — because if the form rate
collapses, a BLEU gain is the wrong trade for this product. That is a CPU
afternoon and no GPU quota.

```bash
.venv/bin/python scripts/build_translator.py --direction en-bs --no-adapter \
    --source <the downloaded model dir> --dest models/lilly/translator-en-bs-sla
.venv/bin/python training/evaluate_app.py --direction en-bs \
    --tuned models/lilly/translator-en-bs-sla
.venv/bin/python training/bosnian_form_rate.py   # the ruler that matters here
```

One practical note for whoever runs it: `ct2-transformers-converter` fails on
this model with `MarianMTModel.__init__() got an unexpected keyword argument
'dtype'` — the ctranslate2 4.8.1 against transformers 4.49.0 clash this
repository already documents. `scripts/build_translator.py` carries the shim and
converts it without complaint. Use the repository's script, not the raw
converter.

---

## What this report does not do

It launches nothing, changes no served weights, and edits no published claim.
Every item above that would alter what a user meets — the recaser, the suppressed
word list, a different base — needs its own pre-registered bars before it ships,
because that is how everything else in this project was decided.

It also does not touch the listener. The highest-ranked speech lever, the
per-clip language token, was already running in another session while this was
written, and re-proposing it would have been noise.
