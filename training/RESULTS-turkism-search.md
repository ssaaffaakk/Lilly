# read-olcum — can BosnianBench score Turkisms? A measured no, and why

v2, `read` lane. This is a negative result. It is written out in full because
the project's standing explanation for the Turkism problem is *incomplete*, and
acting on the incomplete version would send the next person to look in the one
place I can now show is closed.

## What the project believed, and what is actually true

`training/bosnian_bench.py` and `bench/build.py` both conclude:

> The reason is the admission rule, not the data. [...] Measuring Turkisms needs
> a different instrument, and this is not it.

That is correct as far as it goes, and the reasoning behind it is sound: the
0.98 share gate assumes the Bosnian form and its counterpart are in
**complementary distribution**, which holds for yat pairs (a Bosnian text writes
*svijet*, never *svet*) and fails for Turkisms (*kafa*, *ulica*, *susjed* are
ordinary Bosnian too, so both words live in the same sentence and no share test
can separate them). I reproduced that and I agree with it.

**But it is not the binding constraint.** Fixing the rule would not produce a
Turkism category, because there is no corpus to build one from. I looked
everywhere it could be, and every route is closed by measurement:

| where | what is there | why it is closed |
|---|---|---|
| FLORES, ntrex-holdout, valid.tsv | 0 Turkisms | register: news and encyclopedic prose do not use them |
| test.tsv | **1** (`sevdah`) | and that row is itself misaligned — its English is about karaoke birthday parties |
| SETIMES + WikiMatrix + TED2020 | ~250 occurrences | it is `train.tsv`. Benchmark cases cannot come from training data |
| OpenSubtitles v2024, 18.5M lines | `komšija` 2,838, `dućan` 896 | but `avlija` **0**, `pendžer` **0**, `sevdah` **0**, `čaršija` **0**, `insan` **0**, `musafir` **0**, `muhabet` **0**. The distinctive ones are simply absent |
| all 25 OPUS bs-en corpora | surveyed, counted | see below |
| wikimedia v20260327, the part we never trained on | 37 Turkism-bearing rows | **36 of them are in wikimedia v20210402, which the base model trained on** |

### The OPUS survey

Every bs-en corpus OPUS publishes was listed from its API and, where under
200 MB and not already held, downloaded and counted on a 34-group Turkism list.
Density is occurrences per million Bosnian words:

| corpus | pairs | Turkisms | density | English side |
|---|---|---|---|---|
| **Tanzil v1** | 246,913 | 1,770 | **457** | human — a published Quran translation |
| XLEnt v1.2 | 266,696 | 130 | 180 | **not sentences** — a mined entity dictionary; several "translations" are the source string copied |
| CCAligned / MultiCCAligned | 192,099 | 360 | 64 | machine-mined; confirmed misalignment; most `ćuprija` hits are the Serbian town **Ćuprija** |
| HPLT / MultiHPLT v1.1 | 240,015 | 113 | 43 | machine-mined; confirmed misalignment |
| QED v2.0a | 12,541 | 6 | 38 | human, but 6 hits |
| NeuLab-TedTalks | 6,136 | 4 | 47 | human, but 4 hits |
| GNOME, tldr-pages, 3x ELRC | — | 0 | 0 | — |

Not available for bs-en at all, contrary to what I assumed before checking:
**bible-uedin, GlobalVoices, infopankki, ParaCrawl and CCMatrix have no bs-en
bitext in OPUS.** Only HPLT v2/v3, MultiHPLT v2/v3 and NLLB v1 were skipped for
size (527 MB – 4.87 GB).

**Tanzil is the only clean, human-translated, high-density option, and it is
disqualified**: `Tanzil-v1` appears in the base model's own training manifest
with **240,594 rows**. The base has read it. A benchmark built from it would
measure memorisation, not comprehension, and would do so in the base's favour.
Its density is also an illusion — 1,710 of the 1,770 hits are `bašče` inside one
repeated Quranic formula, *džennetske bašče*, "gardens of paradise".

### The last clean pool, and how it closed

The one place left was wikimedia v20260327 (57,865 pairs, published after the
base's 2021-08-07 cutoff) minus the 36,353 rows we trained on. It holds 37
Turkism-bearing sentences. Then:

    avlija   4 candidates  -> all 4 are in wikimedia v20210402
    kahva   32 candidates  -> 32 in v20210402 (5 also in our training)
    džemat   1 candidate   -> clean

and wikimedia **v20210402 is inside the base model's training data** (115,410
rows in its manifest). Verified by direct grep rather than by set arithmetic,
because the claim is strong enough to deserve it:

    $ grep -n "Oko prostrane avlije je zid" wm21/wikimedia.bs-en.bs
    530:Oko prostrane avlije je zid.

12,218 of v20260327's 57,865 rows are in v20210402 — 21% — yet **36 of the 37
Turkism-bearing rows fall inside that 21%.** That is not bad luck. The
vocabulary lives in old, stable Wikipedia articles about Bosnian culture and
architecture, which is exactly the text that was already there in 2021. The
material that makes a corpus useful for this is the material most likely to
have been captured already.

**One case survives**, and it is worth recording precisely because one is not
a category:

    bs  Donja Bijenja u prvoj polovini XVIII vijeka predstavljala je centar džemata.
    en  ... Donja Bijenja was the center of the congregation.
    swap  džemata > zajednice

## The instrument defect I did find

Turkism counts everywhere in this project are inflated by **proper nouns**, and
nothing guards against it. Measured:

    data/extra/extra-train.tsv   mahala   27 matches, 21 are Tadž Mahala   -> 6 real
                                 ćuprija   1 match,    1 is Na Drini ćuprija -> 0 real
    data/clean/train.tsv         ćuprija   7 matches,  5 are the novel title -> 2 real
                                 čaršija  15 matches,  2 are Baščaršija      -> 13 real

`bench/build.py` counts with `FREQ[c][form.lower()]`, a case-insensitive
frequency index, so *Mahala* in *Tadž Mahala* counts as the common noun. It has
a `capitalised()` guard already, but it is wired only into `yat_candidates()`,
and the Turkism entries are hand-listed and bypass it.

**This changed no verdict** and I am not claiming it did: `mahala` was rejected
at a 0.219 share and removing one proper noun in thirty does not move it near
0.98. It matters because it corrupts exactly the kind of corpus search this task
required — my own first-pass counts were wrong until I guarded for it — and any
future search will hit it again.

## What would actually unblock this

Not a better rule. A corpus that is simultaneously:

1. **human-translated** (rules out CCAligned, HPLT, XLEnt, NLLB),
2. **outside `opusTCv20210807`** (rules out Tanzil, SETIMES, WikiMatrix,
   TED2020, wikimedia v20210402 — i.e. rules out most of OPUS, since the base
   was trained on 44 of its corpora),
3. in a **register that carries Turkisms** — literary, colloquial, or
   ethnographic (rules out FLORES, NTREX, news, and encyclopedic prose), and
4. **not already in our training mix**.

Nothing in OPUS satisfies all four. Realistic sources are newly translated
Bosnian literature, or commissioning translation of Bosnian source text — a
data-acquisition problem with a cost, not a measurement problem. That is the
honest handover, and it is a firmer statement than the one this task started
from: *the rule is wrong AND the corpus does not exist, and of the two the
corpus binds harder.*

## What this does not say

It does not say Turkisms are unmeasurable in principle. The design that would
work is specified and I could not populate it:

- Admit a non-complementary term on **cross-variety rate ratio** — how much more
  often Bosnian text uses it than Croatian or Serbian text does — rather than on
  within-Bosnian share, which is the statistic that structurally cannot work.
- Let the **variant swap carry the discrimination**, which is what it is for.
  For yat pairs the swap barely lands: only 3.6% of swapped targets scored
  differently, because the base reads ijekavica and ekavica equally well. A
  Turkism swap (`avlija` -> `dvorište`) should land much harder, since a model
  trained on generic South Slavic plausibly knows *dvorište* and not *avlija*.
  That is a testable prediction and it is the reason the category is worth
  having.

The cross-variety ratio needs Croatian and Serbian corpora in a register that
carries the vocabulary, which is the same corpus problem one step sideways.

**It also must not be built by selecting terms the base model fails on.** That
would guarantee the result and is the reason the admission rule has to stay
corpus-based and model-independent, however inconvenient that is.
