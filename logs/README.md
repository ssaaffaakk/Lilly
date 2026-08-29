# Runtime logs (local only — not in git)

All machine-local output lives under `logs/<category>/`. Do not create `*.log` in the repo root.

## Layout

| Directory | Contents |
| :--- | :--- |
| `harvest/` | Pass-7 pipeline: `pass7.log`, `pass7.state.json`, `metrics.json`, `watch.nohup`, lock |
| `kaggle/` | Kernel poll/watch, fetch, launch (`kernels-watch.log`, `poll.state.json`, …) |
| `ocr/` | OCR train/eval, crops, screen, commons harvest |
| `speech/` | Speech baseline, SpeechBench, WER runs |
| `margin/` | Margin / rule sweeps |
| `bench/` | Benchmark harness output |
| `app/` | App-path eval, server |
| `eval/` | Generic eval runs (arm A/B, full eval) |
| `training/` | Overnight queue, night watch, measure passes |

Paths are defined in `scripts/logs_paths.sh` (bash) and `scripts/log_paths.py` (Python).

## Harvest

```bash
./scripts/harvest_watch.sh
.venv/bin/python3 scripts/harvest_report.py
tail -f logs/harvest/pass7.log
```

## Kaggle watch

```bash
./scripts/kaggle_watch_kernels.sh
tail -f logs/kaggle/kernels-watch.log
```
