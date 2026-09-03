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

Read `docs/OCR-ROADMAP.md` before touching `training/` or `data/ocr/`. It is the queue and the do-not-repeat list; update its status board when a step lands.

Shipped reader: 54.7% words per photograph on the 40 Commons photographs (`training/RESULTS-ocr-restored.md`). Do not change exit codes on the crop gate (rule #5 on the do-not-repeat list). ERROR on a refused gate is the gate working.

**The Mapillary self-label line is closed.** Passes 14–19, five training sets, no gain (`training/RESULTS-ocr-pass19.md`). Do not relaunch any of pass-8 through pass-19. Do not relabel the crops with `recog_network="lilly"` and retry: the labels still come from the reader being trained, and the folded score never moved. The 20,240 photographs stay on Kaggle (`afaksrmeli/lilly-mapillary-photos`) and on the Mac; they are not training data until their labels come from a human or a non-EasyOCR vision model, and until the owner says shop signs are in scope.

**Labels never come from the model under training.** Any reader that writes labels goes through `app.ocr.read_regions`, never a bare `easyocr.Reader` — the stock `bs` character list cannot emit Č Ć Đ (`docs/crop-labels-were-crippled.md`).

**Only held-out numbers decide anything.** `data/ocr/crops/labels-human-latin.tsv` is 666/737 training-side (`training/ocr_split.is_valid_text`); it compares nothing. Report counts beside percentages and the interval beside the delta; 132 crops is ±8 points, 25 letters is ±19.

**Product bar is meaning, not hats.** `kuca` → House is enough. A dropped č/ć/š/ž on a word the translator still gets is not a reason to throw the row out. The crop gate still counts letters and can ERROR; that is the notebook, not the product.

## Push every step

Commit and push after every meaningful step. Verify on GitHub. A clone must be able to work before the pusher does.
