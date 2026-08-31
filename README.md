# Lilly

![Lilly — Bosnian first](docs/images/lilly-hero.jpg)

<p align="center"><strong>A Bosnian → English translator you can type into, speak to, or point a camera at.</strong></p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.115+-009688.svg" alt="FastAPI"></a>
  <a href="docs/V3-PLAN.md"><img src="https://img.shields.io/badge/status-v3%20training-7ad3bd.svg" alt="Status"></a>
</p>

---

## Why I built it

I applied to a lot of internships and kept getting turned down for the same
reason: I did not know Bosnian.

I could not fix that in a week, so I built the thing I needed instead. Lilly is
a translator I made for myself. In class and at work I can press the microphone,
say what I hear, and get it back in English right away. I can point the camera
at a board or a sign and read it. I can type a sentence and have it spoken.

I tried the tools that already existed first. Google Translate and the others
kept getting my sentences wrong — close enough to look fine, wrong enough to
leave me lost in the room. So I trained models on real Bosnian, measured them
honestly, and published the numbers, including the ones that make me look bad.

Lilly is how I prove that I can work in Bosnian, and that I can build something
serious when I run into a wall.

Longer version: [`docs/STORY.md`](docs/STORY.md).

---

## What it does

![Type, Speak, Snap](docs/images/lilly-modes.jpg)

| | You do | Lilly does |
| --- | --- | --- |
| **Type** | Write Bosnian | Returns English and keeps č, ć, đ, š, ž intact |
| **Speak** | Talk into the microphone | Transcribes Bosnian speech, answers in spoken English |
| **Snap** | Photograph a sign, menu, or board | Reads the Bosnian in the image and translates it |

Everything runs locally. No network call at inference time.

There is also a correction button. When a translation is wrong, you say what it
should have been, the correction is verified, and it goes into the training
pool. The model gets better on the sentences people actually hit.

---

## Architecture

![Lilly offline architecture](docs/images/architecture.png)

```text
Phone / browser
    → FastAPI web UI (app/)
        → translate  (bs → en text)
        → listen     (speech → bs text)     Whisper large-v3
        → read       (photo → bs text)      EasyOCR + lilly.pth
        → speak      (en text → voice)
```

```python
from app.lilly import lilly

lilly.translate("Dobar dan")
lilly.listen("clip.m4a")
lilly.speak("Good day", "out.wav")
lilly.read("sign.jpg")
```

![Lilly stack illustration](docs/images/lilly-architecture.jpg)

Training runs on Kaggle GPUs. The Mac commits, launches, and fetches results; it
does not run multi-hour fine-tunes. The rules are in
[`docs/V2-BOUNDARIES.md`](docs/V2-BOUNDARIES.md).

---

## Results

| Task | Result |
| --- | --- |
| Translation | chrF2 **67.47** on 2,009 FLORES pairs |
| Speech | **34.9%** WER on held-out Bosnian; v3 trains whisper-large-v3 in two Kaggle halves |
| Photos | **54.7%** of words read correctly per photo on Commons signs with blind labels, up from 36% |

Thresholds are set before a run, builds are pinned to commit hashes, and the
numbers are published whether or not they improved.

---

## Run it locally

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/fetch_models.py
.venv/bin/python scripts/build_translator.py
.venv/bin/uvicorn app.server:app --port 8000
```

Open http://localhost:8000, allow the microphone, and try a photo with a **đ** in it.

---

## Training (v3)

![Kaggle training flow](docs/images/kaggle-flow.png)

| Lane | What it does | Gate before it ships a zip |
| --- | --- | --- |
| Speech half 1 | 1 epoch → `lilly-listen-half1.zip` | Training exits 0; no WER check here |
| Speech half 2 | `--resume` for epoch 2 → WER → `lilly-listen.zip` | WER must pass |
| OCR pass-7c | Harvest + real crops + synthetic → install gate → `lilly-read.zip` | Gate must pass |

A known failure stops the kernel. A COMPLETE status, a leftover zip, or the 12h
wall does not override a gate.

```bash
python3 scripts/preflight_kaggle.py
python3 scripts/kaggle_train.py speech          # half 1
python3 scripts/kaggle_train.py ocr             # parallel if a GPU slot is free
# after half 1 is COMPLETE with a fresh half1 zip:
python3 scripts/kaggle_train.py speech-half2
python3 scripts/kaggle_poll.py                  # CANCEL or ERROR counts as a failure
```

Live status: [`docs/V3-PLAN.md`](docs/V3-PLAN.md) ·
Notebook rules: [`docs/kaggle-notebooks.md`](docs/kaggle-notebooks.md) ·
Failure list: [`docs/kaggle-fail-stop.md`](docs/kaggle-fail-stop.md) ·
Docs index: [`docs/README.md`](docs/README.md)

---

## Repo map

| Path | What is in it |
| --- | --- |
| `app/` | FastAPI server and web UI |
| `models/lilly/` | Offline weights for translate, listen, read, speak |
| `training/` | Kaggle notebooks, training and evaluation scripts |
| `scripts/` | `kaggle_train.py`, preflight, poll, fetch |
| `docs/` | Plans, fail-stop rules, notebook rules, white paper |
| `data/` | Corpora and OCR crops (large, usually local only) |
| `space/` | Hugging Face Space packaging |

---

## Status

- [x] v1 — translate, listen, speak, read, web app, correction export
- [x] v3 operations — fail-stop, split speech training, required OCR harvest
- [ ] Hugging Face publish
- [ ] Kaggle runs in flight — speech half 1 and OCR; half 2 waits on half 1
- [ ] Final measurement pass and white paper

---

Built by [@ssaaffaakk](https://github.com/ssaaffaakk).
