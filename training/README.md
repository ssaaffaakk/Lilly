# Training Lilly

Lilly does four things, and they are four separate models. Two of them can be
trained further; two cannot, because the people who built them never released the
code that trained them. That is not a gap we can close by trying harder — without
the original recipe there is nothing to continue from.

| Ability | Trainable | How |
|---------|-----------|-----|
| Translate — Bosnian to English | yes | `train_translation.py`, then `evaluate.py` |
| Listen — spoken Bosnian to text | yes | `train_speech.py`, then `evaluate_speech.py` |
| Read — photo to text | the reader only | `prepare_ocr_data.py`, then the upstream trainer |
| Speak — English voice | no | its authors published the model, not the training code |

Everything lands in `models/lilly/`, where the app picks it up on the next start.

---

## Translate

The heart of the project: our 337k cleaned Bosnian–English pairs, fine-tuned with
LoRA so the Bosnian side becomes exact. LoRA trains a small ~10-20 MB adapter on
top of the frozen base — cheap to train, easy to share, small enough to keep.

### Run it free, in the browser

1. Go to <https://colab.research.google.com> and sign in with a Google account
2. **File → Open notebook → GitHub tab** → paste `ssaaffaakk/Lilly` → open
   `training/Lilly_Training_Colab.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**
   Colab has no copy of `models/lilly/translate` (git can't carry it), so upload that
   folder to Google Drive once as `MyDrive/lilly/translate` — cell 4 mounts Drive and
   points `LILLY_BASE` at it
4. Run the cells top to bottom. The real training cell takes ~2–3 hours — keep the tab open
5. The last cell downloads `lilly-adapter.zip`. Unzip it to `models/lilly/adapter/` and
   the app picks it up on the next start

`--quick-test` on the train script does a 3-minute miniature run to verify the pipeline
before committing hours to the real one.

**Done when** Lilly beats the untuned base on BLEU and chrF2 over the 1,500-sentence
held-out test set. Results land in `RESULTS.md`.

---

## Listen

The app runs its listener in a fast inference format that cannot be trained, so
`train_speech.py` does the round trip for you: fine-tune the trainable checkpoint,
convert the result back, drop it into `models/lilly/listen/`. The listener it
replaces is kept next door as `listen-previous`, so you can measure against it.

```
python3 training/train_speech.py --data data/speech/train.tsv
python3 training/evaluate_speech.py --data data/speech/test.tsv
python3 training/evaluate_speech.py --data data/speech/test.tsv --model models/lilly/listen-previous
```

Data is a TSV of an audio path and what is actually said in it:

```
recordings/001.wav	Dobar dan, kako ste?
recordings/002.wav	Gdje je autobuska stanica?
```

**The hard part is the data, not the training.** There is no large public corpus of
transcribed Bosnian speech. Realistic options: record and transcribe your own, or
train on Croatian/Serbian speech as a proxy — close enough acoustically to help.

**Done when** word error rate on held-out clips is lower than the listener you
replaced. If the number does not move, the fine-tune did not work, however cleanly
it ran.

---

## Read

Two models: one finds where the text is, one reads it. Only the reader can be
retrained — the finder's training code was never released. Retraining the reader is
what fixes Bosnian letters like č ć đ š ž coming back wrong, which is the failure
you will actually hit: `Čaršija` read as `Caršija`, `Đačka` as `Backa`.

Labels have to come from a person, so it is two steps:

```
python3 training/prepare_ocr_data.py --photos data/ocr/photos
# open data/ocr/crops/labels.tsv, correct every wrong line by hand
python3 training/prepare_ocr_data.py --labels data/ocr/crops/labels.tsv
```

The first step cuts every text region out of your photos and fills in Lilly's own
reading as a first draft, with its confidence beside it — the low-confidence rows
are exactly the ones worth your attention. The second step writes `data/ocr/train/`
and `data/ocr/valid/` as a folder of images plus a `gt.txt`, which is the layout the
recogniser trainer expects, and reports how well your set covers each Bosnian letter.
A letter that barely appears will stay wrong.

The trainer itself lives upstream, in EasyOCR's `trainer/` directory and its
`custom_model.md` — that document also covers the `.yaml` and `.py` files a trained
recogniser needs before the app can load it. Aim for a few thousand crops; a few
dozen will not move anything.

---

## Speak

Not trainable here. The library its authors published is inference-only — the
training pipeline was never released — so there is no recipe to fine-tune from. If
the voice needs to change, the honest options are:

- use one of the other voices from the upstream model instead of the bundled one
- swap in a different TTS model that does publish a training recipe, and point
  `app/tts.py` at it

Writing a training script for it anyway would be guesswork dressed up as a pipeline.

---

## What has actually been checked

Worth being precise, because "the script runs" and "the model got better" are
different claims:

- **Verified end to end:** the translation pipeline (miniature run), and the speech
  pipeline — fine-tune, convert, load in the app's own runtime, transcribe correctly
  — plus both OCR data-prep steps on real images.
- **Not verified, and cannot be until the data exists:** whether any of these
  fine-tunes produce a *better* model. That is what `evaluate.py` and
  `evaluate_speech.py` are for. Run them before and after; the number is the answer.

None of these scripts reproduce how the original models were trained. They continue
training from released weights on new data, which is a different thing and does not
need the original recipe.
