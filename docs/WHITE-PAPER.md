# Lilly — white paper (v2, in progress)

**Status:** skeleton only. Filled at v2 completion, before public release.

Every model, dataset, API, and photograph source used in Lilly v2 is listed here
with licence, version, and how it was used. Nothing is asserted at publication
time that is not recorded here.

See also `docs/V2-BOUNDARIES.md` for scope and workflow rules.

---

## 1. Product

- **Name:** Lilly
- **Purpose:** Bosnian speech recognition, photograph reading (Latin), and
  Bosnian–English translation through one app.
- **Thesis:** Real Bosnian output, not generic Serbo-Croatian.

*(Expand: architecture diagram, app paths, deployment target.)*

---

## 2. Models

| Component | Base | Fine-tune | Where trained | Output path |
| :--- | :--- | :--- | :--- | :--- |
| Translation | `Helsinki-NLP/opus-mt-tc-big-zls-en` | Arm B (v1) | Kaggle T4 | `models/lilly/translator/` |
| Speech | *(fill after v2 run)* | LoRA | Kaggle GPU | `models/lilly/listen/` |
| OCR reader | EasyOCR `latin_g2` | CTC | *(fill)* | `models/lilly/read/lilly.pth` |

---

## 3. Datasets and sources

### Translation

- `data/clean/train.tsv`, `valid.tsv`, `test.tsv` — filtered WikiMatrix / SETIMES / TED mix
- `data/extra/extra-train.tsv` — wikimedia + NTREX (see `training/DATA-SOURCES.md`)
- FLORES-200 Bosnian–English — evaluation only

### Speech

- FLEURS Bosnian — train / test split
- `facebook/voxpopuli` hr — CC0 + EP attribution (see `data/scripts/download_extra_speech.py`)
- `google/fleurs` hr_hr — CC BY 4.0
- ParlaSpeech-HR — **not used unless owner approves CC BY-SA**

### Photographs

- Wikimedia Commons — 286 harvested photographs (`data/ocr/real-photos/harvested/CREDITS.tsv`)
- 40 scored test photographs — blind transcriptions (`truth.json`)
- Synthetic OCR crops — `data/scripts/generate_ocr_photos.py`

*(Fill: exact commit SHAs, row counts, download dates.)*

---

## 4. External services and tools

| Service | Use | Account / key |
| :--- | :--- | :--- |
| Kaggle | GPU training notebooks | Owner account — **not in repo** |
| Hugging Face | Model weights, datasets | Owner — **not in repo** |
| GitHub | Source of truth for Kaggle clone | Public repo |

*(Fill: Colab if ever used, any APIs.)*

---

## 5. Evaluation (end of v2)

*(Fill after the single measurement queue: WER, per-photo OCR, translation chrF2.
Pre-registered thresholds in `training/PREREGISTRATION.md`.)*

---

## 6. Known limits (honest)

- Cyrillic Bosnian text on photographs: **out of scope for v2** (owner decision 28 Aug).
- BosnianBench: does not support “understands Bosnian better than base” claim.
- Base Marian model was already trained on much of `data/clean/`.

---

## 7. Credits and attribution

- Commons photographers: see `CREDITS.tsv` / `CREDITS.md` (owner-maintained).
- European Parliament (voxpopuli recordings): attribution per EP legal notice.

*(Fill: full bibliography before release.)*

---

*Last updated: 28 August 2026 — skeleton only.*
