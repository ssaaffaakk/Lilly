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

Shipped reader since 5 Sep 2026: PaddleOCR PP-OCRv6, untrained, at confidence floor 0.9 (`app/ocr.py`, `DEFAULT_READER = "paddle"`). It finds 67.0% of a photograph's words with 65 invented on the 40 Commons photographs, and 57.8% with 450 invented on the 132 held-out photographs of test-v2 (`training/RESULTS-ocr-paddle-floor.md`; README.md carries the same pair). The fine-tuned EasyOCR reader's 54.7% on the 40 (`training/RESULTS-ocr-restored.md`; 54.5% / 182 invented under the pinned cv2 4.10.0) is the retired number, not the product number: it stays reachable as `LILLY_READER=easyocr`, the way back and the "before" the bake-off and floor runs measured against — on test-v2 it reads 34.6% with 2,071 invented. Do not change exit codes on the crop gate (rule #5 on the do-not-repeat list). ERROR on a refused gate is the gate working.

**The Mapillary self-label line is closed.** Passes 14–19, five training sets, no gain (`training/RESULTS-ocr-pass19.md`). Do not relaunch any of pass-8 through pass-19. Do not relabel the crops with `recog_network="lilly"` and retry: the labels still come from the reader being trained, and the folded score never moved. The 20,240 photographs stay on Kaggle (`afaksrmeli/lilly-mapillary-photos`) and on the Mac; they are not training data until their labels come from a human or a non-EasyOCR vision model, and until the owner says shop signs are in scope.

**Labels never come from the model under training.** Any reader that writes labels goes through `app.ocr.read_regions`, never a bare `easyocr.Reader` — the stock `bs` character list cannot emit Č Ć Đ (`docs/crop-labels-were-crippled.md`).

**Only held-out numbers decide anything.** `data/ocr/crops/labels-human-latin.tsv` is 666/737 training-side (`training/ocr_split.is_valid_text`); it compares nothing. Report counts beside percentages and the interval beside the delta; 132 crops is ±8 points, 25 letters is ±19.

**Product bar is meaning, not hats.** `kuca` → House is enough. A dropped č/ć/š/ž on a word the translator still gets is not a reason to throw the row out. The crop gate still counts letters and can ERROR; that is the notebook, not the product.

## Heavy compute goes to Kaggle

Training, benchmarking a large model, scoring thousands of sentences, reading
hundreds of photographs: these run on **Kaggle**, not on a laptop and not on a
cloud agent box. An agent that starts a six-hour CPU job "because it can" is
burning wall-clock the owner did not agree to spend and producing numbers on
hardware nobody else will reproduce.

Write the notebook, add the job to `scripts/kaggle_train.py`, update
`scripts/preflight_kaggle.py` in the same commit, push, and launch. The rules in
`docs/kaggle-notebooks.md` and `docs/kaggle-fail-stop.md` apply to every new
notebook, measurement jobs included.

What may still run locally: something that takes minutes, not hours — a smoke
test, an audit that loads no model, a scorer reading cached outputs, a
re-measurement small enough to finish while you wait. If it needs a GPU or more
than a few minutes of CPU, it is a Kaggle job.

## Push every step

Commit and push after every meaningful step. Verify on GitHub. A clone must be able to work before the pusher does.
