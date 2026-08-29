#!/usr/bin/env bash
# One-time: move scattered root logs into logs/<category>/.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/logs_paths.sh

mv_if() {
  local src="$1" dst="$2"
  [[ -f "$src" ]] || return 0
  mkdir -p "$(dirname "$dst")"
  mv "$src" "$dst"
}

# Active harvest (flat logs/ → logs/harvest/)
mv_if logs/harvest-pass7.log "$LOG_HARVEST/pass7.log"
mv_if logs/harvest-pass7.state.json "$LOG_HARVEST/pass7.state.json"
mv_if logs/harvest-metrics.json "$LOG_HARVEST/metrics.json"
mv_if logs/harvest-pass7.nohup "$LOG_HARVEST/pass7.nohup"
mv_if logs/harvest-watch.nohup "$LOG_HARVEST/watch.nohup"
[[ -d logs/harvest-pass7.lock.d ]] && mv logs/harvest-pass7.lock.d "$LOG_HARVEST/pass7.lock.d"

# Kaggle
for f in kaggle-*.log; do
  [[ -f "$f" ]] || continue
  mv_if "$f" "$LOG_KAGGLE/${f#kaggle-}"
done
mv_if kaggle-poll.state.json "$LOG_KAGGLE/poll.state.json"
mv_if kaggle-launch.json "$LOG_KAGGLE/launch.json"

# OCR
for f in ocr-*.log crops*.log screen*.log commons-cats.log harvest-more.log synth-v2.log; do
  [[ -f "$f" ]] || continue
  mv_if "$f" "$LOG_OCR/$f"
done

# Speech
for f in speech-*.log speechbench-*.log wer-*.log; do
  [[ -f "$f" ]] || continue
  mv_if "$f" "$LOG_SPEECH/$f"
done

mv_if margin-sweep.log "$LOG_MARGIN/margin-sweep.log"

# Bench
mv_if bench-rerun.log "$LOG_BENCH/bench-rerun.log"
mv_if bench/run-full.log "$LOG_BENCH/run-full.log"

# App
for f in app-*.log; do
  [[ -f "$f" ]] || continue
  mv_if "$f" "$LOG_APP/$f"
done

# Eval
for f in eval-*.log build-arm*.log; do
  [[ -f "$f" ]] || continue
  mv_if "$f" "$LOG_EVAL/$f"
done

# Training / overnight
for f in training.log remaining.log significance.log speech-retry.log \
  overnight-v2.log night-watch.log train*.log v2-measure.log; do
  [[ -f "$f" ]] || continue
  mv_if "$f" "$LOG_TRAINING/$f"
done
mv_if training/RESULTS-v2-full.log "$LOG_TRAINING/measure-v2-full.log"

# State at repo root
mv_if overnight-v2.state.json "$LOG_TRAINING/overnight-v2.state.json"
mv_if harvest-pass7.state.json "$LOG_HARVEST/pass7.state.json"

echo "Log migration done. See logs/README.md"
