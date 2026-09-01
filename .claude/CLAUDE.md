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

## OCR — closed for v1

Shipped reader stands at 54.7% per-photo. The 1,294 human crops are exhausted. Do not spend more passes on them. Do not change exit codes on the crop gate (rule #5 on the do-not-repeat list). ERROR on a refused gate is the gate working.

## Push every step

Commit and push after every meaningful step. Verify on GitHub. A clone must be able to work before the pusher does.
