# Lilly — project rules

## Fail-stop (speech, OCR, translation)

Notebook writing guide: `docs/kaggle-notebooks.md`.

A known failure is not an optimization. COMPLETE, a zip, or a 12h wall must not override the gate.

- `run()` must tee child stdout to `/kaggle/working/stdout.txt`. `check=True` alone is not enough — the Kaggle log never sees the child's fd.
- Do not zip refused, collapsed, or unmeasured weights into Output.
- Do not skip AFTER WER (speech half 2) or the OCR install gate to force COMPLETE.
- Do not launch OCR without `lilly-ocr-harvest` if the notebook is a harvest pass. Do not train "synthetic only" when harvest was required. Pass-8/9 mixed plates+human, pass-10 human-only, and pass-11 plates-then-human all refused the crop gate. Do not relaunch any of them. Do not mix plates and human in one `train_ocr`.
- Encoder 0-grad, NaN/Inf loss, missing split/source, interrupted download: `SystemExit` / exit 1. Not a warning.
- Watcher/poll: CANCEL and ERROR are failures even if a zip was recovered. Recovery is not success.
- GitHub is the store. Commit and push after the step (never tokens, `kaggle.json`, `.env`). Do not leave the real fix "local only."
- Fix the cause and relaunch. Do not bandage the pipeline so the next run can COMPLETE past the same hole.

## OCR

Shipped reader stands at 54.7% per-photo. The 1,294 human crops are exhausted. Do not spend more passes on them. Do not change exit codes on the crop gate (rule #5 on the do-not-repeat list). ERROR on a refused gate is the gate working.

**Mapillary Kaggle crops are usable.** 20,240 photos cropped on a T4 → 13,144 EasyOCR regions → 7,156 Cyrillic-city crops quarantined → 5,988 Latin → **915 keep** (`conf ≥ 0.85`, ≥3 letters, no dashcam OSD). Filter: `data/scripts/filter_mapillary_train.py`. Train list: `data/ocr/crops-kaggle/labels-train.tsv`. Images: `data/ocr/mapillary-train/`. Load with `train_ocr.py --train-dir data/ocr/mapillary-train` so `mly_*` never lands in `data/ocr/train` or `data/ocr/valid` (the gate would then score EasyOCR against itself). Do not dump the 20k photos or the 5,988 Latin rows. Do not mix these with sign-letter plates in one `train_ocr`. Do not relaunch pass-8/9/10/11.

**Product bar is meaning, not hats.** `kuca` → House is enough. A dropped č/ć/š/ž on a word the translator still gets is not a reason to throw the row out. The crop gate still counts letters and can ERROR; that is the notebook, not the product. Hats are not free on every word (`MUSKA` → Muscovy, `Carsija` → Empire) but they are not the keep/drop rule for this set.

## Push every step

Commit and push after every meaningful step. Verify on GitHub. A clone must be able to work before the pusher does.
