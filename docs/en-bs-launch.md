# Launching the English → Bosnian fine-tune

The reverse direction is the last unkept half of the project's own promise. It
ships today as the **untouched base** and `built.json` says so
(`fine_tuned: false`). `training/RESULTS-en-bs.md` carries only base rows.

Everything that had to happen **before** the run is done and committed. What
remains needs Kaggle credentials, so it happens on the machine that has them.

## What is already measured, and why the order matters

`training/PREREGISTRATION.md`, "v3 — reply — English to Bosnian", requires four
things to exist before a candidate does. All four are in git:

| | | measured |
|---|---|---|
| the deciding bar | chrF2 on 2,009 FLORES-200 pairs | **above 58.96** |
| the floor | BLEU on the same pairs | **not below 29.57** |
| the third bar | Bosnian form rate, output side | **not below 94.3%** |
| control | `>>bos_Latn<<` survives tokenisation | yes, id 5941 |
| control | the label still steers (`>>bos_Latn<<` vs `>>hrv<<`) | **94.3% → 72.5%**, 21.8 points |

The form-rate numbers and both controls: `training/RESULTS-en-bs-formrate.md`.
The scorer's rule was committed before it had read a single output, and the
case audit ran with no model loaded. That order is the point — a baseline
recovered afterwards is not a baseline.

**Read the form rate before deciding this run is worth GPU time.** The base is
already at 94.3%, so that bar has 5.7 points of headroom and 94.3 of downside.
This run has to earn its place on **chrF2**, not on Bosnian-ness.

## Preconditions

    cd ~/Desktop/Lilly && git pull
    .venv/bin/python3 scripts/fetch_translate_base.py --direction en-bs   # 286 MB, prints the tag check

`kaggle.json` in `~/.kaggle/`, never in the repo. The extra corpus
(`data/extra/extra-train.tsv`) has to be the same file the counts were measured
on — `push_corpus` re-uploads only if its md5 changed, so the usual case is a
status call.

## The trap

`training/Lilly_Translation_Kaggle.ipynb` cell 0 carries the direction it
trains, **committed**:

    DIRECTION = "bs-en"      ->      DIRECTION = "en-bs"

Change it and commit before launching. The launcher compares the notebook
against the job and refuses the run rather than letting it finish and hand back
a model trained the wrong way round. Flipping it back afterwards is part of the
job: while it says `en-bs`, the forward `translation` job will refuse.

`ARM` stays as committed. The two arms were decided by a pre-registration of
their own and this run is not the place to reopen that.

## The run

    .venv/bin/python3 scripts/kaggle_train.py translation-en-bs --watch
    .venv/bin/python3 scripts/kaggle_train.py translation-en-bs --status   # if the watch drops
    .venv/bin/python3 scripts/kaggle_train.py translation-en-bs --fetch

CANCEL and ERROR are failures even if a zip came back. Recovery is not success —
`.claude/CLAUDE.md`, and `docs/kaggle-fail-stop.md` for why that rule exists.

`--fetch` prints the per-direction install line. It is
`models/lilly/adapter-en-bs/`, never `models/lilly/adapter/`: the second one is
the shipped forward adapter and unzipping a reverse model over it is a failure
that does not announce itself.

## Deciding, afterwards

Three numbers and two controls, in this order, and the bars are the table above:

    .venv/bin/python3 training/evaluate.py --direction en-bs --adapter models/lilly/adapter-en-bs
    .venv/bin/python3 training/bosnian_form_rate.py --adapter models/lilly/adapter-en-bs --label tuned
    .venv/bin/python3 training/bosnian_form_rate.py --adapter models/lilly/adapter-en-bs --label tuned-hrv --tag ">>hrv<<"
    .venv/bin/python3 training/bosnian_form_rate.py --diagnose training/form-rate/tuned.json training/form-rate/tuned-hrv.json

**chrF2 decides, BLEU is the floor, the form rate is a floor.** Both directions,
not either: a model that scores higher while writing less Bosnian does not ship,
and neither does one that writes more Bosnian while chrF2 falls.

The label-steering control is not decoration. If the tuned model's
`>>bos_Latn<<`/`>>hrv<<` gap has collapsed toward the base's 21.8 points — say
to single digits — the adapter has deafened the decoder to the only thing
separating Bosnian output from Croatian, and it does not ship whatever chrF2
says. Diagnose the recipe; do not raise the learning rate and try again.

**Both flat is a result and gets published as one.** `tc-base` is a smaller base
than the forward direction's, Helsinki publishes nothing big out of English into
this family, and a null here is what says whether a bigger base is the next move.
Nothing in any write-up may imply the two directions are of comparable quality;
on the evidence they are not, and the README already says so.

## Outcome — 8 September 2026

Run, fetched, re-measured, decided: every bar above holds (chrF2 60.00, BLEU
30.73, form rate 99.2%, label gap 22.5 points). The served build now carries
the adapter (`built.json` `fine_tuned: true`); the notebook's cell 0 is back
on `bs-en` / `fullft`. Numbers and controls: `training/PREREGISTRATION.md`,
"Outcome, 8 September 2026 — the LoRA fine-tune clears all four bars".
Publishing the bundle is the owner's step and has not been done.
