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

Every part Lilly needs sits in one folder, `models/lilly/` — `translate`, `listen`, `speak`, `read` — and one object in [app/lilly.py](app/lilly.py) puts all four behind a single door:

```python
from app.lilly import lilly

lilly.translate("Dobar dan")          # Bosnian text   -> English text
lilly.listen("clip.m4a")              # spoken Bosnian -> Bosnian text
lilly.speak("Good day", "out.wav")    # English text   -> spoken English
lilly.read("sign.jpg")                # photo          -> Bosnian text
```

Nothing is fetched over the network while Lilly runs: the weights are read straight off this disk, so it works with the wifi off. Each ability loads the first time it is used, so starting up costs nothing.

Design direction for the app: glass, smooth, Apple-like. Calm, quick, easy to use. Bosnian imagery (Mostar) as atmosphere, not decoration. No emojis, no clutter.

Training runs on free Google Colab GPUs; the Mac is used for building the app and running the finished model.

## Project layout

```
Lilly/
├── data/        # scripts that download & clean Bosnian-English data (the data itself is not committed)
├── training/    # model training scripts (run on Colab GPU)
├── models/
│   └── lilly/   # every weight Lilly uses: translate, listen, speak, read, adapter
│                # (not committed — 1.3 GB, too big for git)
├── app/         # lilly.py (one object, four abilities) + server + web frontend
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

The weights are 1.3 GB, so they are not in git. Fetch them, then build the
translator the app serves from — that step quantises it and folds in the
fine-tuning, and has to be re-run after every training run:

```
.venv/bin/python scripts/fetch_models.py
.venv/bin/python scripts/build_translator.py
```

Then open http://localhost:8000. `python3 app/lilly.py` prints what is in the model
folder and what is missing.

## Deploying it

```
docker build -t lilly .
docker run -p 8000:8000 -v lilly-data:/data lilly
```

Three things decide whether a deployment works:

**Memory.** Measured on CPU, which is what a server is. Translation rests at 670 MB
and peaks at 975 MB on the largest input it accepts; reading a photo is the expensive
one and takes the process to about 3 GB. Give it **4 GB**, and one worker — a second
worker loads its own full copy of the weights. This rules out the common free tiers;
most give 512 MB.

**HTTPS.** The microphone is one of the three ways to use Lilly, and browsers only
hand it over on a secure origin. Served over plain http from anything other than
localhost, the button fails and blames the user. Put TLS in front of it.

**A volume for the corrections.** `data/feedback.db` is the only thing the app
writes, and everything a person ever corrected is in it. Anywhere inside the
deployment directory is wiped by the next deploy — mount a volume and point
`LILLY_DB` at it, as the Dockerfile does.
