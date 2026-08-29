# Overnight v2 — stopped (speech ERROR)

**When:** 2026-08-29 morning (owner woke / reported fail)  
**Status:** `afaksrmeli/lilly-speech` → **ERROR**

## What went wrong

Training **did finish**:

- **1221/1221** steps (~3 epochs), ~7h 22m train loop
- Log: `trained checkpoint: .../listen-trained`
- train_loss ≈ 0.217, final eval_loss ≈ 0.250

Then the notebook **assert** failed:

```text
AssertionError: nothing was trained
assert Path("models/lilly/listen-trained/model.safetensors").is_file()
```

**Cause:** whisper-large-v3 saves **sharded** files (`model-00001-of-0000N.safetensors`), not a single `model.safetensors`. Convert / after-eval / `lilly-listen.zip` never ran.

## Overnight pipeline

`overnight_v2.py` died without writing state/report (Mac sleep or process kill). Caffeinate alone did not keep the Python watcher alive.

## Weights recovery

Kaggle output is flooded with audio stubs; API hit **429 Too Many Requests** while searching for the ~3 GB checkpoint. Treat packaged `listen-trained` as **likely lost** from the dead worker.

## Fix + next action

1. Notebook assert fixed (any `model*.safetensors` / bins) + immediate zip of `listen-trained` after train.
2. Push + relaunch speech on Kaggle.
3. Re-arm overnight watcher after launch.
