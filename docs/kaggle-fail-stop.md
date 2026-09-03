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
11. OCR: skip required data for this pass (pass-7c: harvest; pass-8: sign-letters
    + human crops) and train a thinner mix. Do not relaunch pass-7c harvest
    auto-crops (photographs worse), pass-8/9 mixed plates+human, pass-10
    human-only, or pass-11 plates-then-human — crop gate refused
    (`RESULTS-ocr-pass11.md`: words back to 41.7%, letters 64%→60%).
12. Treat CANCEL or ERROR as success because a zip was recovered. Recovery is
    not success.
13. Photo eval: skip a failed read and still print a score. A hole is a failed
    score.
14. Write `--keep-trained` / shippable names before the gate, then refuse.
    A refused run must not leave a file that looks installable.
15. Let a child trainer print to its own stdout and treat the Kaggle log as
    enough. The log never sees that fd. Tee to `/kaggle/working/stdout.txt`
    and write `experiment_log.json`. COMPLETE + zip is still not install.
16. OCR: compare a trained reader against the shipped one on crops the shipped
    one trained on. `labels-human-latin.tsv` is 666 of 737 training-side
    (passes 18–19 were scored on it). Held-out crops only, counts and interval
    beside the delta. `docs/OCR-ROADMAP.md`.
17. OCR: train on labels written by the reader being trained (passes 14–19,
    five sets, no gain). Labels come from a human or a blind-checked
    non-EasyOCR vision model, and any labelling reader goes through
    `app.ocr.read_regions` — the stock `bs` list cannot emit Č Ć Đ.
18. Launch another pass after the results doc has diagnosed the line as dead
    (`a12a2fb` diagnosed passes 14–17; passes 18 and 19 ran anyway). The step
    after a diagnosis is in `docs/OCR-ROADMAP.md`, not in a notebook. Do not
    relaunch pass-8 through pass-19.

## Gates

| Job | Must happen | Must not happen |
| :--- | :--- | :--- |
| Speech half 1 | Clone `/kaggle/temp`, 1 epoch, `--keep-adapter`, zip `lilly-listen-half1.zip`, tee + trainproof | BEFORE/AFTER WER, merge 3 GB, clone `/kaggle/working` |
| Speech half 2 | `--resume`, `SPEECH_EPOCHS = 2`, AFTER WER **then** zip `lilly-listen.zip` | `--base` merged weights, zip before WER |
| OCR | Attach what **this pass** requires, tee + photograph gate, zip `lilly-read.zip` only after the gate | `check=False`, skip required data, zip refused `read-trained.pth` as the app package, relaunch pass-8 through pass-19, train on EasyOCR-written labels, skip crop gate after stage 1 by installing |

Launch: `python3 scripts/kaggle_train.py speech` / `speech-half2` / `ocr`.
Preflight: `python3 scripts/preflight_kaggle.py`.
