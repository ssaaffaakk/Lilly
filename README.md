# Lilly — Bosnian to English Translator

Lilly is our own Bosnian-to-English translation model and app. The goal is exact, real Bosnian — not generic "Serbo-Croatian" output — with three ways to use it:

- **Type it** — write Bosnian, get accurate English
- **Say it** — press the microphone, speak Bosnian, and Lilly answers back in English **voice**
- **Snap it** — take a picture of text (a sign, a menu, a document) and the translation appears in the chat

And one thing most translators don't have: a **"This translation is wrong" button**. Press it, say what's wrong, and after review the correction goes into our database — the model is retrained on real corrections and keeps getting better the more it is used.

## How it works

| Part | What it does |
|------|--------------|
| Translation | Our Bosnian-to-English model, trained on 337k real Bosnian-English sentence pairs so the Bosnian is exact |
| Listening | Turns spoken Bosnian into text |
| Speaking | Reads the English translation out loud, natural voice |
| Photo scan | Pulls the text out of the picture, then translates it |
| Corrections | Verified user corrections become new training data |
| App | Python API server + web frontend — works in the browser on any phone or computer |

Design direction for the app: glass, smooth, Apple-like. Calm, quick, easy to use. Bosnian imagery (Mostar) as atmosphere, not decoration. No emojis, no clutter.

Training runs on free Google Colab GPUs; the Mac is used for building the app and running the finished model.

## Project layout

```
Lilly/
├── data/        # scripts that download & clean Bosnian-English data (the data itself is not committed)
├── training/    # model training scripts (run on Colab GPU)
├── models/      # trained model weights land here (not committed — too big)
├── app/         # the web app: server + frontend
└── docs/        # roadmap and notes
```

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). Every finished step gets pushed to this repo.

## Status

- [x] Phase 0 — Project setup
- [x] Phase 1 — Collect & clean Bosnian-English data (337,790 clean pairs — see [data/STATS.md](data/STATS.md))
- [ ] Phase 2 — Fine-tune the translation model
- [x] Phase 3 — Speech input (listen to Bosnian)
- [x] Phase 4 — Voice output (speak the English)
- [x] Phase 5 — Photo scan & translate
- [x] Phase 6 — Web app (glass design, first version — design polish pending)
- [x] Phase 7 — Correction button + database (retraining export ready; periodic retraining pending)
- [ ] Phase 2 — Fine-tune the translation model (runs on Colab, base model live meanwhile)
- [ ] Phase 8 — Deploy

## Run it

```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/uvicorn app.server:app --port 8000
```

Then open http://localhost:8000 — first use of each feature downloads its model.
