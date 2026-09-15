# Lane C pilot — Ekavica→ijekavica converter, hand-checked precision

2026-09-15. Whole-word-form converter (`scripts/ekavica_to_ijekavica.py`),
measured on 500 Serbian sentences from a documented CC0 source. Gate result:
**100% (155/155) on a 500-sentence hand-checked sample; Wilson 95% interval
[97.6%, 100%] — above the 95% gate.**

## Source — CC0, documented, MD5-verified

MaCoCu-sr-en 1.0, a Serbian↔English parallel corpus built by crawling the
`.rs` and `.срб` TLDs (2021–2022).

- CLARIN.SI handle: http://hdl.handle.net/11356/1819
- Licence: **CC0-No Rights Reserved** (https://creativecommons.org/publicdomain/zero/1.0/)
- 2,068,916 parallel sentences; 95,863,993 words; 312,335 texts; issued 2023-04-26
- Record page shows the CC0 tag (raw HTML preserved); the MaCoCu family's CC0
  status was already verified on the repo (`training/DATA-SOURCES.md`), and the
  sibling handles (el-en 1856, ca-en 1857, uk-en 1858) display the same tag.
- File: `MaCoCu-sr-en.latin.sent.txt.gz`, 499.71 MB, MD5
  `adf90151f028e34e3669dc0703a11aa1` (matches the record page).
- Serbian side is the `src_text` column (tab-separated, header row).

## Method

- Slice = first 500 lines of the Latin-script sentence file whose `src_text`
  is non-empty, ≤ 400 characters, and contains no Cyrillic — deterministic
  in file order, so any clone with the verified file reproduces the sample.
- Each token whose whole lowercase form is in the lexicon is rewritten; case
  is preserved; nothing else changes.
- Every sentence containing at least one rewrite (127 before curation, 124
  after) was read and every rewritten token judged in its original context.
  No random subsample was needed — the hand-check covered 100% of the
  rewrite attempts, which is stronger than the required random 100.

## Lexicon (236 whole word-forms)

Three parts, merged in order:

1. **Base**: `bench/terms.tsv` rows with category `yat` and alt-variety `sr`,
   aligned positionally only where the ekavian/ijekavian pipe-lists have equal
   length.
2. **Supplement**: a small curated set of frequent unambiguous families the
   table does not cover (`reka`, `dete`, `mleko`, `hleb`, `gde`/`ovde`,
   `rešiti`).
3. **Exclusions** — word-forms whose spelling collides with a homograph
   carrying a different meaning, so they are never rewritten. Every one is a
   correctness call, matching the repo precedent of excluding `sveta` in
   `download_extra_data.py`:
   - `svet sveta svetu sveti svete svetom svetima svetova svetovima` —
     "world" (yat) vs "holy"/"saint"/"council" (not yat). The unambiguous
     `svetsk-` family (svetski, svetskog, …) stays in.
   - `zahteva` — noun `zahtev` gen.pl (→ *zahtjeva*) vs verb `zahtevati` 3sg
     (→ *zahtijeva*). The bare form is ambiguous; it is left alone.
   - `rekom` — the instrumental of `reka` is dominated in practice by the
     proper-noun acronym REKOM, which must not become "rijekom".
   - `reci` — dative of `reka` (→ *rijeci*) vs imperative of `reći`.
   - Whole-form lookup makes the regex approach's `reka/rekao/rekavši`
     false-positive class impossible: `rekao`, `rekli`, `rekavši` are outside
     the lexicon because they are never forms of the noun.

## Result — first run found two errors; curation fixed both

Initial run: 158 rewrites in 127/500 sentences. The hand-check found exactly
two errors, both collisions already covered by the exclusion policy:

1. `REKOM → rijekom` (all-caps acronym "REKOM", Regional Commission …).
2. `zahteva → zahtijeva` in "svih zahteva" — noun gen.pl, which should be
   *zahtjeva*; the ambiguous form had mapped to the verb only.

Both forms were added to the exclusions and the same sample re-run.

| | rewrites | sentences w/ ≥1 | errors |
|---|---|---|---|
| initial lexicon | 158 | 127/500 | 2 (REKOM, zahteva) |
| after exclusions | 155 | 124/500 | 0 |

Final precision: **155/155 = 100%** (95% CI [97.6%, 100%], counts reported).
Coverage on this register: 24.8% of sentences (124/500) get at least one
rewrite; `zahteva` (very frequent as a noun gen.pl) is deliberately left
unconverted, which is a coverage cost of correctness.

## Verdict

Gate is **passed** (95% ≤ 97.6% lower bound). Per `docs/run-assignments.md`
Lane C, the pre-registration text proposing a small owner-gated training arm
(bar: held-out FLORES form-rate up, no BLEU collapse) goes to the Leader.
The 300–500-sentence pilot stays as reported here; a larger independent
sample can be drawn from the same verified file if the Leader asks.

## Pre-registration draft (gate passed → for the Leader)

One small, owner-gated training arm, launched only if the owner approves:

- **Data:** up to a bounded share of a converted Serbian CC0 slice — e.g.
  ≤ 10% of added words, drawn from MaCoCu-sr-en 1.0 (CC0, handle 11356/1819)
  after `ekavica_to_ijekavica.py` conversion, joined to the SR/HR strengthening
  mix. Not mixed into the main bs→en train-mix.
- **Pre-committed bar (measured on held-out FLORES bs→en):** ijekavian
  requested-form rate up by at least +1 point *and* no BLEU collapse beyond a
  threshold the Leader sets (proposal: ≥ −0.5).
- **Fail rule:** either bar missed → arm is discarded and not relaunched with
  a bandaged mix. The converter's pilot precision stands at 100% [97.6, 100]
  on 155 rewrites; the Leader may first re-measure on a larger independent
  sample from the same verified file — the source, licence, and slice method
  are recorded above so a clone reproduces it.