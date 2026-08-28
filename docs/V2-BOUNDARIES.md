# Lilly v2 — boundaries (28 August 2026)

This file is the contract for every v2 session. Read it before doing anything.
When a decision here conflicts with an older handoff, **this file wins for v2**.

Delete or amend it only when the owner explicitly changes direction.

---

## Why we are doing this

**Lilly** is a Bosnian-language tool: speak Bosnian, photograph Bosnian text,
translate Bosnian — not generic Serbo-Croatian output dressed as Bosnian.

v1 is finished and waiting on the owner for Hugging Face upload. **Do not touch v1**
(publish script, model card numbers bound to the shipped build, upload credentials).

v2 improves what users feel most: **listening** and **reading photographs**.
Translation fine-tuning is deprioritised (chrF2 barely moves; BosnianBench does not
support the central claim).

The owner is a third-year student with a small MacBook. The goal is a **clean,
defensible, impressive** project for senior developers and CEOs — honesty and a
working demo beat inflated metrics.

---

## What we are building (scope)

| In scope | Out of scope |
| :--- | :--- |
| Latin Bosnian OCR (č ć đ š ž) | **Cyrillic / Kiril** — dropped by owner 28 Aug |
| Whisper listener fine-tune (Kaggle) | Local model loads on the 8 GB Mac |
| More real + synthetic photo training | Mapillary, Panoramax, KartaView, Europeana |
| End-of-v2 measurement pass (one queue) | Constant interim scoring on the Mac |
| White paper documenting every source | Google Colab Pro / paid Google training |
| Push code, docs, notebooks to GitHub | Secrets, API keys, `.env`, credentials |

---

## How we work

1. **Train on Kaggle, not the Mac.** Heavy jobs use `scripts/kaggle_train.py`.
   The Mac commits, pushes, launches, and fetches — nothing else loads weights.
2. **Push before launch.** Kaggle clones GitHub. Dirty or unpushed trees have
   already killed a 35-minute run. Run `python3 scripts/state.py` first.
3. **Train first, measure last.** No 80-minute validity reruns between steps.
   When v2 training lands, one serial queue scores everything once.
4. **Plumbing checks only during training:** loss goes down, weights file exists,
   app path loads the right filename (`lilly.pth`, not `latin_g2.pth`). That is
   not “benchmarking”; it is not repeating the 27 August shipped-vs-trained trap.
5. **No band-aids.** Fix the cause or report the blocker.
6. **White paper at the end.** Every dataset, model, API, and photo source used
   in v2 is recorded in `docs/WHITE-PAPER.md` with licence and attribution.
   Nothing gets hand-waved at publication time.
7. **Push everything except security.** Commit and push code, docs, results
   markdown, and notebooks. Never commit: `.env`, HF tokens, Kaggle secrets,
   personal credentials, or anything under “Secrets” in `.gitignore`.

---

## Current v2 priorities (order)

1. **Speech:** `openai/whisper-large-v3` + LoRA on Kaggle (`Lilly_Speech_Kaggle.ipynb`).
   Baseline and fine-tune both use the same base so before/after is fair.
2. **OCR (Latin):** Long Kaggle or staged local train only after charset work is
   unnecessary (Cyrillic dropped). Focus: diacritics, real crops, synthetic scale.
3. **Translation:** Hold at v1 Arm B unless a Kaggle arm is explicitly reopened.

---

## Licensing (unchanged until owner rules otherwise)

- **voxpopuli_hr:** CC0 transcription + European Parliament attribution on
  recordings (checked 28 Aug — not “no conditions”).
- **ParlaSpeech-HR:** CC BY-SA — **not in training** until owner approves
  share-alike on published weights.
- **Wikimedia Commons:** only working photo source; CREDITS.tsv must stay clean.

---

## What “amazing” means here

- Model card and white paper state limits plainly.
- Demo works on a phone: speak, snap a sign, read the translation.
- Build is bound to numbers by content hash.
- A senior engineer sees reproducibility, not hype.

---

## Session anchor (human-readable)

> Kaggle-only training. Latin Bosnian only. Train hard, score once. Document
> every source in the white paper. Push all except secrets. v1 untouched.
