# Kaggle fail-stop

A known failure is not an optimization. COMPLETE, a zip, or the 12-hour wall
must not override the gate. Cursor agents also load
`.cursor/rules/kaggle-fail-stop.mdc` (always apply). Standing rule 9 in
`docs/agent-teams-lilly.md` is the same decision.

**Full notebook writing guide** (skeleton, wrong/right snippets, anti-patterns,
cell contracts, launcher/poller, preflight checklist — so the next agent does
not reopen these holes): [`docs/kaggle-notebooks.md`](kaggle-notebooks.md).

## What went wrong

These were done on purpose so a kernel could COMPLETE. Do not do them again.

1. Relaunch a recipe that already misses the 12-hour wall (speech: 2 epochs +
   AFTER WER in one session).
2. Clone Lilly into `/kaggle/working`. That floods Output; the zip never
   downloads. Clone to `/kaggle/temp`.
3. Zip merged Whisper large-v3 (~3 GB) as the speech artefact. Half 1 keeps
   the Trainer adapter (`--keep-adapter`). Half 2 `--resume`s it. Do not
   `--base` merged weights and call that epoch 2.
4. Encoder LoRA with 0 gradient: warn and keep training. Raise instead.
5. OCR: `check=False`, zip refused weights, then assert exit 0.
6. Skip a failed download split or shard and still `return 0`.
7. Skip AFTER WER (speech half 2) or the OCR install gate so the kernel can
   COMPLETE.
8. Launch while Kaggle already has 2/2 batch GPU sessions.
9. Register a GPU notebook on CPU so it ERRORs immediately.
10. Leave the real fix "local only." GitHub is the store (never tokens,
    `kaggle.json`, `.env`). Kaggle clones GitHub.
11. OCR: skip `lilly-ocr-harvest` and train human crops + synthetic only.
12. Treat CANCEL or ERROR as success because a zip was recovered. Recovery is
    not success.
13. Photo eval: skip a failed read and still print a score. A hole is a failed
    score.
14. Write `--keep-trained` / shippable names before the gate, then refuse.
    A refused run must not leave a file that looks installable.

## Gates

| Job | Must happen | Must not happen |
| :--- | :--- | :--- |
| Speech half 1 | Clone `/kaggle/temp`, 1 epoch, `--keep-adapter`, zip `lilly-listen-half1.zip` | BEFORE/AFTER WER, merge 3 GB, clone `/kaggle/working` |
| Speech half 2 | `--resume`, `SPEECH_EPOCHS = 2`, AFTER WER **then** zip `lilly-listen.zip` | `--base` merged weights, zip before WER |
| OCR | Harvest attached, `run(..., check=True)`, zip only after `train_ocr` exits 0 | `check=False`, synthetic-only fallback, zip refused `read-trained.pth` |

Launch: `python3 scripts/kaggle_train.py speech` / `speech-half2` / `ocr`.
Preflight: `python3 scripts/preflight_kaggle.py`.
