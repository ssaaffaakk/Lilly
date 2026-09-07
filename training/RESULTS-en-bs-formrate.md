# English → Bosnian — the Bosnian form rate on the base, measured before the fine-tune

Run 7 September 2026 on the untouched base, `Helsinki-NLP/opus-mt-tc-base-en-sh`,
**before any English → Bosnian fine-tune exists**. That order is the whole point:
`training/PREREGISTRATION.md`, "v3 — reply — English to Bosnian", fixes it in
words — *"The baseline is measured on the base before the fine-tune is launched
and written into this file. A baseline recovered afterwards is not a baseline."*

Instrument: `training/bosnian_form_rate.py`, whose rule was committed in the
amendment of the same date, before it had produced a number about any model.

## Why this measure exists at all

`training/bosnian_bench.py` says in its own docstring what it cannot do: *"The
direction is bs->en, so nothing Bosnian survives into the output — you cannot
look at English and ask whether it is ijekavica."* That is why the forward
direction's Bosnian claim came back +0.5 points at p = 0.360 and stayed unproven
for a year.

In this direction Bosnian **is** the output. The question becomes one bit per
target: given an English sentence whose professional Bosnian translation used a
Bosnian-only term, does the model write that term or its Croatian/Serbian
counterpart? This is the first time the project can test its own central claim
head-on rather than around the edge of it.

## The set — audited before the run, `--audit`, no model loaded

| | |
|---|---|
| bench cases (`bench/cases.tsv`, reused, not rebuilt) | 346 |
| targets | 385 |
| **discriminating targets kept** | **338**, across 308 cases, 179 distinct pairs |
| dropped — no counterpart named | 47 |
| dropped — forms identical folded / nested / absent from the reference | 0 / 0 / 0 |

Four times the forward instrument's 85 targets. The instrument can speak.

## The number

| | base, `>>bos_Latn<<` |
|---|---|
| attempted | 338 |
| decided (exactly one of the two forms present) | 245 |
| wrote the **Bosnian** form | **231** |
| wrote the Croatian/Serbian counterpart | 14 |
| **form rate** | **94.3%**  (95% Wilson 90.6–96.6) |
| silent (neither form; never averaged in) | 93 |

**The label already works.** When the base commits to one of the two forms it
writes the Bosnian one nineteen times in twenty. Read against the pre-registered
table this is the useful shape of the result and it is uncomfortable: the third
bar has **5.7 points of headroom and 94.3 points of downside**. The
pre-registration called this row "a floor, not a headline" before the number
existed; the number says it was right.

The fourteen it loses, in full: *tačno→Točno, riječi→reči, aprila→travnja,
djece→dece, juna→lipnja, drugdje→drugde, hiljadama→tisućama,
historijska→istorijska, voz→vlak, promijenila→promenila, evrope→Europe,
posjetilaca→posetilaca, negdje→negde, učestvovati→sudjelovati* — half Croatian,
half Serbian.

## The control: does the label still steer?

The same 338 English sentences decoded twice, changing nothing but the label on
the front. This is the base's own gap, measured now so a fine-tuned model has
something to be compared against.

| | `>>bos_Latn<<` | `>>hrv<<` |
|---|---|---|
| form rate | **94.3%** | **72.5%** |
| decided | 245 | 233 |
| wrote the Bosnian form | 231 | 169 |
| wrote the counterpart | 14 | 64 |

Paired across the same 338 sentences: **38 of 338 outputs (11.2%) came out
identical** under the two labels, and **53 targets flipped from the Bosnian form
to its counterpart** when the label changed.

**21.8 points, and 88.8% of the sentences come out different.** The decoder is
listening to its selector. The month names are the clearest tell — under `>>hrv<<`
the base writes *travnja, lipnja, srpnja, rujna* where under `>>bos_Latn<<` it
writes *aprila, juna, jula, septembra* — and so are *evropa→Europa,
porodica→obitelj, voz→vlak, hiljada→tisuća, učestvovati→sudjelovati*.

A fine-tune that shrinks this gap toward zero has deafened the decoder to the
only thing separating Bosnian output from Croatian, and by the pre-registered
rule it does not ship whatever its chrF2 says.

## Two limits of the instrument, found on the base and reported before a candidate exists

Both are reproducible: `bosnian_form_rate.py --diagnose training/form-rate/base.json`.

**1. Most of the silence is a case ending, not a miss.** Of the 93 silent
targets, **0 hedged** (neither form appeared in all 93; not one output contained
both), and **52 carry a word sharing the Bosnian form's first five letters** —
the model wrote the Bosnian word in another inflection: *posjet* for *posjetu*,
*rijeku* for *rijeke*, *pobjedu* for *pobjede*, *izvještaju* for *izvještaja*.
The matcher is literal on **both** sides, so this loss is symmetric and it lands
where it can be seen instead of being scored as a failure. Loosening it to stems
would be a degree of freedom, and the pre-registration closed that door before
the number existed.

**2. A Croatian third form can escape the two-way matcher.** In **6** of the 93,
the output carries an unambiguously Croatian word that is *neither* listed form:
*povijest* where the target pair was `historiji`/`istoriji`, *tisuću* where it
was `hiljada`/`tisuća`, *tjedan* and *lipnja* in the sentence targeting
`sjevernoj`. For yat pairs the listed counterpart is the **Serbian** form, so
Croatian drift has a third word to escape into. Six is a **floor** on that
drift, not a measurement of it: the stem list is short, hand-written and
deliberately excludes anything ambiguous, so it can only under-count.

## The asterisk, restated because it governs what may be claimed

| counterpart | targets |
|---|---|
| yat (ijekavica/ekavica — the alternative is Serbian) | 237 |
| lexical choice | 100 |
| `h` (*historija*/*istorija*) | 1 |
| of those, a listed **Croatian** form (*obitelj*, *zrakoplov*, *tisuću*, *rujna*…) | **78 / 338** |

23% Croatian-facing, against the forward instrument's 14%. Better against
Croatian drift, and still Serbian-heavy. **A form rate that holds up here is
evidence about ekavica first and Croatian second**, and no sentence in any
write-up may say otherwise.

## What this does and does not settle

It settles the baseline, and it settles it in the only order that counts —
before the thing being judged exists. It does not say the fine-tune will pass or
fail; it says what passing would have to mean. And it puts one uncomfortable
fact on the table early: on the measure this project exists for, the untouched
base is already at 94.3%, so the reverse fine-tune has to earn its place on
**chrF2**, with this row and BLEU as floors it must not fall through.

---

Reproduce:

    .venv/bin/python3 scripts/fetch_translate_base.py --direction en-bs
    .venv/bin/python3 training/bosnian_form_rate.py --audit
    .venv/bin/python3 training/bosnian_form_rate.py --label base
    .venv/bin/python3 training/bosnian_form_rate.py --label base-hrv --tag ">>hrv<<"
    .venv/bin/python3 training/bosnian_form_rate.py --diagnose \
        training/form-rate/base.json training/form-rate/base-hrv.json
