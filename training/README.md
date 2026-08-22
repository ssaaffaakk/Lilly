# Training Lilly's translation model

We start from a South-Slavic-to-English sequence-to-sequence base and **fine-tune it with
LoRA** on our 337k cleaned Bosnian–English pairs so the Bosnian side becomes *exact*.
LoRA means we train a small ~10-20 MB "adapter" on top of the frozen base: cheap to
train, easy to share, and it fits in this repo.

## How to run it (free, in the browser)

1. Go to <https://colab.research.google.com> and sign in with a Google account
2. **File → Open notebook → GitHub tab** → paste `ssaaffaakk/Lilly` → open
   `training/Lilly_Training_Colab.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**
   Colab has no copy of `models/lilly/translate` (git can't carry it), so upload that
   folder to Google Drive once as `MyDrive/lilly/translate` — cell 4 mounts Drive and
   points `LILLY_BASE` at it
4. Run the cells top to bottom. The real training cell takes ~2–3 hours — keep the tab open
5. The last cell downloads `lilly-adapter.zip` — that's the trained model. Unzip it to
   `models/lilly/adapter/` and the app picks it up on the next start

## What the scripts do

| File | Purpose |
|------|---------|
| `train_translation.py` | Fine-tunes the base model with LoRA on `data/clean/train.tsv` |
| `evaluate.py` | Scores the untuned base vs Lilly on the held-out test set (BLEU + chrF2) |
| `Lilly_Training_Colab.ipynb` | Runs the whole thing on a free Colab GPU |

`--quick-test` on the train script does a 3-minute miniature run to verify the pipeline
before committing hours to the real one.

## Success criterion

Lilly must beat the untuned base on BLEU and chrF2 on the 1,500-sentence held-out test set.
Results land in `RESULTS.md`.
