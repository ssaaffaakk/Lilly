# Lilly — four-agent run assignments (v1, 15 Sep 2026)

One task per agent, one file-lane per agent, one source of truth on GitHub. The
work here is drawn from the **measured** priorities (`docs/V4-PLAN.md`,
`docs/REPORT-what-would-raise-the-numbers-2026-09-13.md`,
`docs/what-moves-the-model.md`), not from the "10M-sentence / 42-BLEU" wish. Every
high-value payoff sits behind an **owner-gated launch** — agents build to
*launch-ready* and stop.

## Roster

| slot | tool | lane | status |
|---|---|---|---|
| Leader | **Claude — lilly-1b** (this session) | verify every push · own `PREREGISTRATION.md` + this doc · integrate | active |
| A | **Claude — session 8a8ed3** | Run B: en→bs back-translation | in progress |
| B | **Antigravity** | Whisper language-token gate (highest-value ready) | not started |
| C | **OpenCode** | Ekavica→ijekavica converter pilot | not started |
| D | **OpenClaw** | WikiMatrix alignment filter (Run C) | not started |

Lanes are keyed by **task**, not by session id: if you spin a different Claude for
slot A, the lane travels with the task.

## Rules for everyone — read before you touch a file

1. **`origin/main` is the only source of truth.** Push after every atomic step.
   A clone must be able to continue your work.
2. **One working directory per agent.** Preferred: your own clone or a
   `git worktree`. If two agents share one checkout, only one edits at a time.
   (The two Claude sessions currently share one tree — they serialize.)
3. **`git pull --rebase origin main` before every push.** Atomic commits: one
   file-set, one message. Never force-push.
4. **Stay in your lane. Never edit a file in another lane.**
5. **`training/PREREGISTRATION.md` is edited ONLY by the Leader.** Send the Leader
   your pre-registration text; the Leader commits it, serialized. This removes the
   one shared-file clobber risk.
6. **Bars before numbers.** Any new run is pre-registered *before* it runs. Only
   held-out numbers decide anything. Report the **count and the interval** beside
   every delta.
7. **Hard stop-lines — OWNER-GATED, never do these yourself:**
   - Do **not** launch any Kaggle run. Build the notebook/plumbing to
     *launch-ready*, then stop. (`scripts/preflight_kaggle.py` →
     `scripts/kaggle_train.py` is the owner's to run.)
   - Do **not** ship: no writes to `models/lilly/` served builds, no
     `publish_to_hf`, no change to `app/` served behaviour.
   - Do **not** touch tokens, `kaggle.json`, `.env`.
8. **Do-not-repeat (instant refuse):** OCR passes 8–19, Mapillary self-labels,
   whisper-large-v3 retrain, ParlaSpeech-HR as training data, the `-sla` base swap
   (measured and closed — untuned 31.71/60.89, below shipped 32.22/61.55). See
   `docs/kaggle-fail-stop.md` and `docs/OCR-ROADMAP.md`.
9. **Fail-stop.** A known failure is not an optimization. `run()` tees child
   stdout to `/kaggle/working/stdout.txt`. ERROR on a refused gate is the gate
   working — never bandage it to force COMPLETE.

---

## Lane A — Claude (8a8ed3): Run B, en→bs back-translation

**Why.** en→bs (reply) is the weak direction (32.22 BLEU / 61.55 chrF2). The
strongest known lever is target-side fluency from real Bosnian written by
Bosnians — back-translate MaCoCu-bs (730M words, CC0) with the shipped bs→en
model: synthetic English source, real Bosnian target.

**Status.** `765082f` pre-registration + `02f82c4` prep script & tests — verified
by Leader (6/6 green, leak guard confirmed).

**Own (edit/push):**
- `scripts/prepare_backtrans_bs.py` ✅ · `tests/test_prepare_backtrans.py` ✅
- `training/Lilly_Translation_Kaggle.ipynb` — add the back-translation phase
  (bs→en inference over the held-out MaCoCu-bs sample using the **shipped forward
  build**, its fingerprint pinned)
- `scripts/kaggle_train.py` — add the Run B job
- `scripts/preflight_kaggle.py` — add Run B `DIRECTION`/dataset checks (same commit
  as the notebook, per `.claude/CLAUDE.md`)

**Do:** finish notebook + launcher to launch-ready (`run()` tees stdout; 1:1
synthetic:real mix, real parallel up-sampled). Local smoke only: run the prep on a
few-hundred-line sample and confirm the holdout drop-count prints.

**STOP LINE:** the ~1M-line bs→en inference + the LoRA train are owner-gated
Kaggle. Hand the launch-ready notebook over; do not launch.

**Bars (locked, `765082f`, all must hold):** chrF2 > 61.55 (bootstrap CI excludes
0) · BLEU CI-low ≥ −0.30 · form rate ≥ 99.0% · label gap ≥ +15.

**Do not touch:** `app/`, `models/`, speech/OCR files, `PREREGISTRATION.md`
(Run B's section is already in; no further edits).

---

## Lane B — Antigravity: Whisper language-token gate

**Why.** The per-clip language-token fix already ran — **12.77% AFTER WER on 200
clips vs 34.9% baseline** — but its three real bars are UNMEASURED, because
`Lilly_Speech_Instrument_Kaggle.ipynb` is pinned to the large-v3 fingerprint and
cannot score the turbo candidate. Un-pin it so the candidate can face its gate.
This is the single highest-value *ready* item in the project.

**Own (edit/push):**
- `training/Lilly_Speech_Instrument_Kaggle.ipynb` — amend cell 5: compute + pin
  the candidate's fingerprint **after** training; set candidate base to
  `openai/whisper-large-v3-turbo`. Keep the split, normaliser, term list and the
  **three bars unchanged**.
- (only if the scoring path needs it) `training/evaluate_speech.py` /
  `speech_bench.py` — metric definitions must stay byte-identical.

**Do:**
1. Read first: `training/RESULTS-speech-langtoken.md`, the
   "v4 — listen — one language token per clip" section of `PREREGISTRATION.md`,
   `docs/kaggle-fail-stop.md`. Change **no** bar, split, or term.
2. Amend so ONLY the fingerprint pin + base id change — everything else identical
   (that limit is the gate working).
3. Write the pre-registration **amendment** text (candidate fingerprint pinned
   after training; base = turbo; bars unchanged) → send to **Leader** to commit.
4. Flag the dependency: the candidate weights (`lilly-listen.zip` / `-half2` from
   the 12 Sep Output) must live in a Kaggle **dataset**, not an expiring Output —
   tell the owner if promotion is needed.

**STOP LINE:** launching the 925-clip scoring run is owner-gated Kaggle. Make it
launch-ready; do not launch.

**Bars (the 3 pre-registered, unchanged, all must hold):** word error on 925 clips
(`--decode app`) strictly below `listen-previous` · Bosnian term recall not below
baseline · **Croatian substitution NOT above baseline** (this is the hypothesis).

**Do not touch:** `app/`, `models/`, translation/OCR files, `PREREGISTRATION.md`
(send text to Leader), whisper-large-v3 (closed by rule 3).

---

## Lane C — OpenCode: Ekavica→ijekavica converter pilot ("one spoonful")

**Why.** The only genuinely new HR/SR idea worth testing — convert Serbian
(ekavian) text to Bosnian (ijekavian) to widen source diversity. But `e→ije` is
**not** a blind find-replace (*selo, meso, more* keep their `e`). Measure the
**converter's precision on a human-checked sample BEFORE anyone trains on its
output** — a converter that poisons is worse than no converter.

**Own (edit/push):**
- `scripts/ekavica_to_ijekavica.py` [NEW] — use a curated yat-reflex lexicon
  and/or a morphological analyzer (CLASSLA/Stanza). **Not** a raw `e→ije` regex.
- `tests/test_ekavica.py` [NEW] — must include negatives that must NOT change
  (*selo→selo, meso→meso, more→more*) and positives (*vreme→vrijeme, reka→rijeka,
  dete→dijete*).
- `training/RESULTS-ekavica-precision.md` [NEW] — the precision measurement.

**Do:**
1. Build the converter. Draw a small sample (~300–500 Serbian sentences from a
   documented CC0 source; record source + licence).
2. Convert, then **hand-check a random 100** against correct Bosnian: report
   precision (correct / attempted) with the count and a Wilson interval.
3. **Pre-committed gate:** precision < 95% on the checked sample → STOP and write it
   up as closed. ≥ 95% → send the pre-registration text (small gated training arm;
   bar = held-out FLORES form-rate up, no BLEU collapse) to the **Leader**.

**STOP LINE:** no training. The converter + its precision number + the pre-reg is
the whole deliverable. Training is owner-gated and only if the gate passes.

**Do not touch:** `app/`, `models/`, speech/OCR, the main `train-mix`,
`PREREGISTRATION.md` (send to Leader).

---

## Lane D — OpenClaw: WikiMatrix alignment filter (Run C)

**Why.** WikiMatrix is ~56% of the bs↔en mix and ~1/6 of it is misaligned by the
project's own audit. Dropping the misaligned tail is a clean, no-launch
data-quality win for the strong (bs→en) direction.

**Own (edit/push):**
- `scripts/filter_wikimatrix_align.py` [NEW] — score each en–bs pair with a
  sentence-embedding aligner (LaBSE or LASER); output a cosine score per pair.
- `tests/test_filter_align.py` [NEW]
- `training/RESULTS-wikimatrix-align.md` [NEW] — score distribution, chosen
  threshold, drop count, and 15–20 example pairs that get dropped (proof they are
  genuinely misaligned).

**Do:**
1. Read first: `training/RESULTS-en-bs-hrv.md` and the data-audit note
   (WikiMatrix ~1/6 misaligned).
2. Build the scorer; run on a **sample** of the WikiMatrix portion locally
   (embedding a few thousand pairs is fine on CPU). Do NOT embed the full set
   locally if heavy — sample, report, and make the full pass a documented
   owner-gated step.
3. Pick a threshold from the distribution + examples; report the drop count. Send
   the pre-registration text (retrain the shipped bs→en recipe on the filtered
   mix; paired bootstrap on FLORES; bar = chrF2/BLEU not below shipped) to the
   **Leader**.

**STOP LINE:** the retrain is owner-gated Kaggle. The filter + drop stats + pre-reg
is the deliverable.

**Do not touch:** `app/`, `models/`, speech/OCR, `PREREGISTRATION.md` (send to
Leader).

---

## Leader (lilly-1b) — what I do

- Verify every push: read the diff, run the tests, before it counts as done.
- Own `training/PREREGISTRATION.md`: each executor sends me its pre-reg text; I
  commit it, serialized — the one shared file never clobbers.
- Own this doc.
- Keep the owner informed: what is launch-ready, what waits on an owner-gated
  launch, and what each launch costs.
- I do not launch Kaggle and I do not ship weights.

## The owner's gate (you)

Nothing ships or burns Kaggle without your explicit **"launch X"**. Launch-ready ≠
launched. When a lane hits its stop-line, the Leader tells you exactly what is
ready and what it costs.
