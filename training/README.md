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

### Run it free, on Kaggle

Kaggle is the better of the two free options: **Save Version → Save & Run All (Commit)**
runs the notebook without you, so a 2-3 hour job survives closing the tab.

1. <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook** →
   upload `training/Lilly_Translation_Kaggle.ipynb`
2. Right-hand panel: **Accelerator → GPU** (P100 or T4) and **Internet → On**. The
   notebook pins itself to a single GPU on purpose: with two visible, the trainer splits
   each batch across both, which doubles the effective batch and halves the optimizer
   steps while the learning rate stays put — a different recipe than these numbers were
   set for
3. Upload `models/lilly/translate` once as a Kaggle Dataset (Datasets → New Dataset),
   then attach it here with **Input → Add Input → Datasets**. The weights are too big
   for git, so this is how they reach the machine — cell 4 finds them wherever they mount
4. **Save Version → Save & Run All (Commit)**, then close the tab
5. When it finishes, download `lilly-adapter.zip` from that version's **Output** tab and
   unzip it to `models/lilly/adapter/`

`Lilly_Training_Colab.ipynb` does the same job on Colab, reading the weights from Google
Drive instead. It works, but the tab has to stay open.

`--quick-test` does a 3-minute miniature run to verify the pipeline before committing
hours to the real one. It writes to `models/quicktest-adapter`, never to the real
adapter — a 200-pair toy sitting at the real path would be picked up by the app and
scored as the finished model.

**Done when** Lilly beats the untuned base on BLEU and chrF2 over the 1,500-sentence
held-out test set. Results land in `RESULTS.md`.

---

## Listen

The app runs its listener in a fast inference format that cannot be trained, so
`train_speech.py` does the round trip for you: fine-tune the trainable checkpoint,
convert the result back, drop it into `models/lilly/listen/`. The listener it
replaces is kept next door as `listen-previous`, so you can measure against it.

Fine-tuning wants a GPU, so the same Kaggle route applies — import
`training/Lilly_Speech_Kaggle.ipynb`, GPU on, internet on, **Save & Run All**. That
notebook needs no dataset upload: it fetches its own speech, measures the untrained
listener, trains, and measures again on the same held-out clips.

Locally, or once you have the data:

```
python3 data/scripts/download_speech_data.py
python3 training/train_speech.py --data data/speech/train.tsv
python3 training/evaluate_speech.py --data data/speech/test.tsv
python3 training/evaluate_speech.py --data data/speech/test.tsv --model models/lilly/listen-previous
```

Training watches `data/speech/valid.tsv` as it goes and keeps the epoch that scores best
on it rather than whichever one happened to be last — point `--valid` elsewhere to change
that. `--convert-only DIR` skips training and just converts a checkpoint, which is how
you get an untrained baseline to measure against, and how the Kaggle notebook separates
converting from training so a conversion failure cannot throw away the GPU hours.

The transcripts come with their capitals and punctuation intact. The dataset also ships a
stripped-down version of the same text, and training on that would teach the listener to
drop both — which then reaches the translator and the reader. Scoring normalises either
side, so the comparison stays fair regardless.

Data is a TSV of an audio path and what is actually said in it:

```
recordings/001.wav	Dobar dan, kako ste?
recordings/002.wav	Gdje je autobuska stanica?
```

**The data is small, and that is the real limit.** `download_speech_data.py` pulls a
read-speech set with human transcripts — 3,091 clips to train on, 925 held back to judge
with. That is tiny by speech standards. Croatian and Serbian are selectable as proxies
(`--lang hr_hr`, `--lang sr_rs`), and recording your own Bosnian is what would help most.

Measured baseline, so you know what you are aiming at: the listener the app ships with
scores **42% word error rate** on 50 of those held-out clips. The mistakes are not
acoustic — it hears the sound and guesses a word that does not exist, `pjevači` coming
back as `pivač`. That is the kind of error more Bosnian data fixes.

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
