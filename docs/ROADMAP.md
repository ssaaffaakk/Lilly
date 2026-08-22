# Lilly Roadmap

Every phase ends with a push to GitHub, so the repo always shows exactly where we are.

## Phase 0 — Project setup ✅
Repo connected, folder structure, this roadmap.

## Phase 1 — Data: collect & clean Bosnian–English sentence pairs
The model can only be as Bosnian as the data we feed it. We pull from four public
sentence-pair collections:

- **News** — Balkan news, clean and formal Bosnian (~200k pairs)
- **Everyday sentences** — short, human-checked
- **Talk subtitles** — natural spoken style
- **Film & TV subtitles** — casual conversational Bosnian (big but noisy — we filter it)

Cleaning steps: remove Serbian/Croatian contamination where detectable, drop bad alignments,
normalize punctuation, deduplicate, split into train/validation/test.

**Done when:** one clean `train.bs-en` file + stats report, pushed.

## Phase 2 — Translation model (the heart)
- A South-Slavic-to-English sequence-to-sequence base we can deploy freely; we make it
  *exact* for Bosnian
- Method: LoRA fine-tuning on our Phase 1 data (fits free Colab T4 GPU)
- Measure quality with BLEU + chrF on held-out test set, compare against the base model
  so we can prove our version is better at Bosnian
- Export the trained weights + a small quantized version that runs on the Mac

**Done when:** our fine-tuned model beats the untuned base on the Bosnian test set,
training notebook + eval numbers pushed.

## Phase 3 — Speech input (listen)
- Speech recognition already understands Bosnian → we wire it in first (works day one)
- If accuracy on real Bosnian audio isn't good enough: try forcing Croatian
  (acoustically near-identical), step up a model size, or fine-tune on
  Croatian/Serbian parliamentary speech corpora (no usable Bosnian speech corpus exists)
- Two modes: Bosnian speech → Bosnian text → English, and straight speech → English

**Done when:** you can feed an audio file of spoken Bosnian and get correct English text.

## Phase 4 — Voice output (speak)
- A small, natural English voice reads the translation out loud
- Runs fully offline, real-time even on CPU

**Done when:** English translations can be played as natural-sounding audio.

## Phase 5 — Photo scan & translate
- Text recognition with Bosnian support, neural detector for phone photos
- Pipeline: photo → find & read the text → clean it up → translate → show result

**Done when:** a photo of a Bosnian sign/menu/document comes back as English text.

## Phase 6 — Web app
- Python backend serving translation + speech + voice + photo scan behind one API
- Mobile-friendly web page: text box, microphone button, camera button,
  play button for the English voice
- Design direction: glass, smooth, Apple-like. Calm and quick. Bosnian imagery
  (Mostar) as atmosphere. No emojis, no AI-slop styling.
- Works in the browser on any phone — no app store needed

**Done when:** you can open it on your phone and use all four ways to translate.

## Phase 7 — Correction button + database + retraining loop
- Every translation gets a **"This translation is wrong"** button
- Pressing it opens a small form: what's wrong, and the correct translation
- Corrections go into a database (SQLite to start) with a review step —
  only checked/approved corrections count
- Approved corrections are exported as extra training data and the model is
  retrained on them periodically, so Lilly improves from real usage

**Done when:** a correction made in the app ends up improving the next training run.

## Phase 8 — Polish & deploy
- Speed (quantized models), error handling, final design pass
- Deploy the server so it's reachable from anywhere
