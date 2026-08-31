# Running Lilly as an agent team — the prompts

Companion to [`agent-teams.md`](agent-teams.md), which is the general reference.
This file is Lilly-specific: what shape of team fits this project, and the exact
text to paste.

Read `HANDOFF.md` and `SCOPE-V1.md` first — they say where the work actually is.

---

## The shape, and why

**v1 is finished and waiting on the owner** (the Hugging Face upload). A team
cannot help with that. Everything below is for **v2**, which is the nine open
tasks on the board in `scripts/team.py`.

Those nine tasks are already three independent lanes of three:

| lane | tasks | owns |
| :--- | :--- | :--- |
| `listen` | `listen-veri`, `listen-egitim`, `listen-olcum` | speech |
| `read` | `read-veri`, `read-egitim`, `read-olcum` | translation |
| `picture` | `picture-veri`, `picture-egitim`, `picture-olcum` | photographs |

**One teammate per lane, not per stage.** Within a lane the stages are
sequential — you cannot measure what has not trained. Across lanes there is no
dependency and no shared file. That is the exact condition agent teams need.

### The binding constraint is RAM, not tokens

This machine has 8 GB and **kernel-panicked on 27 August** from five concurrent
jobs. `python3 scripts/guard.py` reported **1.3 GB free** when this was written.

So the rule that makes a team survivable here:

> **Heavy compute goes to Kaggle. Local work is serialised through
> `scripts/guard.py`. Never more than one teammate holding model weights.**

Kaggle is off-machine, so three lanes can each have a GPU run in flight at no
local cost. Local evaluation is RAM-bound and must queue.

### Three pieces of project infrastructure the team runs on

| script | what it replaces |
| :--- | :--- |
| `scripts/team.py` | **the shared task list**, which does not exist on Opus 5 (see `agent-teams.md` §4). File-locked atomic claims. |
| `scripts/guard.py` | admission control. `claim(gb, name)` before any model load. |
| `scripts/state.py` | the definition of "done" — uncommitted, unpushed, untracked. |

---

## The prompt

Paste this as-is into a fresh session.

```text
Read HANDOFF.md, SCOPE-V1.md, RESUME.md and docs/agent-teams.md before you do
anything else.

v1 is done and waiting on me for the Hugging Face upload. Do not touch it.
This team is for v2 only - the nine open tasks on the board in scripts/team.py.

You are the lead. Spawn 3 teammates on Opus, one per lane, named listen, read
and picture:

  listen   owns listen-veri, listen-egitim, listen-olcum
  read     owns read-veri, read-egitim, read-olcum
  picture  owns picture-veri, picture-egitim, picture-olcum

Write each spawn prompt yourself, and put in it:
  - that lane's history: what is already measured, and what the dead ends were
  - the files it owns, and the files it must never touch
  - the finished deliverable, and that it reports to you by name
  - the eleven standing rules below, verbatim

STANDING RULES - copy into every spawn prompt:

1.  Claim work with `python3 scripts/team.py claim <task> <your-name>` before
    starting, and `done` or `block` it when you stop. That board is the only
    record of who has what - you have no shared task list.
2.  Run `python3 scripts/guard.py` before anything heavy, and call
    `claim(gb, name)` from scripts.guard before loading a model. This machine
    has 8 GB and has kernel-panicked from parallel jobs. If guard refuses,
    wait. Do not raise the ceiling and do not run it anyway.
3.  Heavy training goes to Kaggle, not this Mac. Local training was measured
    at 9.4 h per epoch.
4.  Report every result to the lead by message. Going idle tells the lead you
    stopped, not what you found.
5.  Commit and push after every step. Run `python3 scripts/state.py` before
    saying anything is done, and verify GitHub by commit SHA - never by
    raw.githubusercontent, which is CDN-cached and has served a stale file.
    Push before launching anything that clones the repo.
6.  Score through app.translate.Engine and app.ocr.scan, never the layer
    underneath. Both have been measured on the wrong path before.
7.  Touch only your own lane's files. Never touch data/clean/test.tsv,
    data/clean/valid.tsv, data/flores/, models/lilly/keep-2026-08-26/
    or CREDITS.md.
8.  Write your threshold into training/PREREGISTRATION.md before the run
    starts, and do not reinterpret it afterwards.
9.  No band-aids. Solve the cause or report the blocker. Never defer.
    A failed train, download, gate or measurement must stop the kernel
    (ERROR). Do not zip refused weights, skip WER to COMPLETE, skip
    OCR harvest and train synthetic-only, or treat CANCEL+zip as success.
    GitHub is the store: commit and push (never tokens). "Local only" is
    not a backup.
10. Signals, not conversation: task, acceptance test, result. Nothing else.
11. Spawn your own Sonnet subagents for downloading, counting and filtering.
    You are on Opus because the training and measurement judgement is yours.

HOW YOU RUN IT:

- Verify every number yourself before it goes anywhere. Re-measure it. Three
  times on this project a teammate reported a number and re-measurement said
  something else.
- One heavy local job at a time. When a step is finished or abandoned, kill its
  background job in the same turn - a stale job is taken straight out of
  whatever is measured next.
- Check `python3 scripts/team.py list` before assigning and
  `python3 scripts/guard.py` before allowing anything heavy to start.
- Do not bandage a failure so the pipeline COMPLETEs. Find the cause, fix it,
  push to GitHub, restart the work.
- Do not put any teammate in plan mode: their plans are auto-approved without
  your review.
- Report to me: what was done, what was achieved, which problems were solved,
  and which problems appeared. Do not omit the last one.
```

---

## Do and don't

The right-hand column is not hypothetical. Each one has cost this project a run,
a night, or a number.

| do | don't | why it bites here |
| :--- | :--- | :--- |
| **Own specific files.** One lane, one set of paths, written into the spawn prompt. | **Share the same file.** | Two teammates editing one file means overwrites, and the loser's work is gone with no error. `test.tsv` moving silently invalidates every published number. |
| **Define the output.** Name the artefact and the acceptance test: a number, a file, a table. | **Vague deliverables.** | A task with no acceptance criterion is a wish — `scripts/team.py add` refuses one for exactly this reason. |
| **Name recipients.** Say who each result goes to, by name. | **Assume the plan is understood.** | There is no broadcast: one message per recipient. A teammate that assumes someone else is watching the board reports to nobody, and its idle notification carries no output. |
| **Three to five teammates.** Three lanes, three teammates. | **Ten or more.** | Token cost scales linearly, coordination overhead scales worse. On 8 GB with 1.3 GB free, ten agents is the 27 August kernel panic with more steps. |
| **Give full context.** History, measured numbers, dead ends, in the spawn prompt. | **No history given.** | Teammates load CLAUDE.md, skills and MCP — but **not this conversation**. A lane without its history re-tries Mapillary, Panoramax and KartaView, all three already measured dead. |
| **Push after every step**, verified by commit SHA. | **Nothing pushed to GitHub.** | Kaggle notebooks clone from GitHub. An uncommitted rewrite killed a run 35 minutes in, and `truth.json` — the answer key behind the whole photograph measurement — sat untracked, one disk failure from taking it with it. |

---

## Why each rule is there

Every one of these prevents something that has already happened on this project.

| rule | the failure it prevents |
| :--- | :--- |
| 1 — claim on the board | Two agents killed each other's training run, believing it was a duplicate. GPU idle 20 minutes. |
| 2 — guard before loading | The 27 August kernel panic: 15 swapfiles, hard reboot. |
| 3 — Kaggle not the Mac | Local training measured at 0.101 s/sample = 9.4 h/epoch, worse under contention. |
| 4 — report by message | **Agent-teams-specific.** An idle notification carries no output. A lane that finishes silently has reported nothing. |
| 5 — push every step | A Kaggle run cloned a version of the downloader with no `--source` flag and died 35 minutes in. `truth.json` — the answer key behind the whole photograph measurement — was untracked. |
| 6 — score through the app | The translation number moved 3.36 chrF2 depending on which layer was measured. |
| 7 — file ownership | Two teammates on one file means overwrites. `test.tsv` moving silently invalidates every published number. |
| 8 — pre-register | The same fine-tune scored three ways, all three moving in the flattering direction. |
| 9 — no band-aids | A failed speech train still COMPLETEd; OCR zipped refused weights; harvest skipped and the notebook trained synthetic-only. |
| 11 — model per job | Data work on Opus is expensive and slow. Routing is by *detectability*: a bad download fails loudly, a bad benchmark stays plausible for months. |

### Two warnings specific to teams

**Do not put teammates in plan mode.** A teammate spawned while the lead is in
plan mode gets its plan **auto-approved by the lead without review**
(`agent-teams.md` §9). On a project whose standing rule is that the leader
verifies everything, that is the wrong default. Review plans by reading them.

**Teammate subagents run in the foreground only.** A teammate's background work
cannot outlive the lead's process, so rule 11's Sonnet subagents block their
lane while they run. Keep them short.

---

## When *not* to use a team here

- **Anything in v1.** It is done and waiting on the owner.
- **One heavy local measurement.** Three teammates cannot run three evaluations
  on 1.3 GB of free RAM. A single session and a queue is faster.
- **Same-file work.** Sequential edits to `app/`, `train_ocr.py`, or any one
  script — a team only adds overwrite risk.
- **When the answer is one number.** Spawn a subagent, get the result back. A
  teammate would tell you it stopped, not what it found.

The shape a team is genuinely good at on this project, and has already proved
twice: an **adversarial audit** — 42 findings narrowed to 8, then 45 to 6, each
verified before it was believed.

```text
Spawn 4 teammates to audit <target> from four independent angles: one on
correctness, one on measurement validity, one on whether the code path scored
is the code path served, one adversarially trying to disprove the other three.
Have them message each other directly. I verify every finding myself before it
counts.
```
