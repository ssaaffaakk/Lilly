# Lilly

## The only assistant that speaks **Bosnian** — not “close enough” Serbian, not Croatian with a different flag.

Everybody else ships one South Slavic model and calls it a day. Google does it. DeepL does it. The big labs do it. They dump you in the same bucket and hope you do not notice.

**We noticed. We built Lilly anyway.**

Lilly is Bosnian. Full stop. Type it in Bosnian. Say it in Bosnian. Point your phone at a Bosnian sign. Get English back — and get it from a stack that was **trained, measured, and shipped for Bosnian**, not relabeled at the last minute.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)

---

## Three ways in. One language that matters.

| | What you do | What Lilly does |
| --- | --- | --- |
| **Type** | Write Bosnian | English that respects č, ć, đ, š, ž — not a tag-stuffed machine translation |
| **Speak** | Talk into the mic | Hears **Bosnian speech** and answers in spoken English |
| **Snap** | Photograph a sign, menu, monument | Reads **real Bosnian text off the street** and translates it |

Most translators stop at the keyboard. Lilly lives where Bosnian actually lives: **voice and signage.**

---

## The feature nobody else has

**“This translation is wrong.”**

Press it. Say what’s right. We verify it. It goes back into the training pool.

That is not a gimmick. That is Lilly getting **more Bosnian every time someone cares enough to correct it.** The giants do not want that loop. We built it on purpose.

---

## Numbers that hold up (we measured, we did not guess)

| | Result |
| --- | --- |
| **Translation** | chrF2 **67.47** on 2,009 FLORES pairs — best-in-class territory for Bosnian→English |
| **Speech** | **34.9%** word error on held-out Bosnian clips — and v2 is training **whisper-large-v3** on Kaggle right now |
| **Street photos** | **54.7%** of words read correctly per photograph — real Commons signs, blind human labels, up from 36% |
| **Honesty** | BosnianBench does not flatter us. We publish that too. |

We pre-register thresholds before we run. We bind builds to hashes. We do not move the goalposts when the score does not move.

**v2 is live on Kaggle tonight.** Bigger listener. Harder OCR training. Same obsession: Bosnian first.

---

## See it work

Run it. Touch it. Stop reading about it.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/fetch_models.py
.venv/bin/python scripts/build_translator.py
.venv/bin/uvicorn app.server:app --port 8000
```

Open **http://localhost:8000** on your phone. Mic on. Photograph something with **đ** in it. Watch Lilly do what the “Serbo-Croatian” apps never bothered to optimize for.

Add screenshots to [`docs/images/`](docs/images/) when you capture the demo — that is the slide deck for investors who still need pictures.

---

## Under the hood (for engineers who read footnotes)

Four abilities, one folder, zero network at inference:

```python
from app.lilly import lilly

lilly.translate("Dobar dan")
lilly.listen("clip.m4a")
lilly.speak("Good day", "out.wav")
lilly.read("sign.jpg")
```

Training runs on **Kaggle GPU** — your laptop is for shipping, not suffering. Details: [`docs/V2-BOUNDARIES.md`](docs/V2-BOUNDARIES.md), [`training/README.md`](training/README.md).

Full reproducibility, licences, and source list: [`docs/WHITE-PAPER.md`](docs/WHITE-PAPER.md) and [`models/lilly/README.md`](models/lilly/README.md).

---

## Who this is for

- **Diaspora** who are tired of tools that butcher the language their parents speak  
- **Visitors** in BiH who need a sign translated **now**  
- **Learners** who want Bosnian, not “South Slavic (approx)”  
- **Anyone** who looked at `>>eng<<` leaking into a translation and decided that was unacceptable  

Lilly is not a research toy. It is a product with a server, a UI, a correction loop, and a publish pipeline.

---

## Status

- [x] Ship-quality v1 — translate, listen, speak, read, web app, correction export  
- [ ] Hugging Face publish (owner gate)  
- [ ] **v2 training in flight** — large-v3 speech + OCR push on Kaggle  

---

**Lilly.** Real Bosnian. Not the excuse everyone else sells you.

Built by [@ssaaffaakk](https://github.com/ssaaffaakk).
