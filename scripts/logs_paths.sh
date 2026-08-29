#!/usr/bin/env bash
# Canonical runtime log directories (source from other scripts).
_LROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$_LROOT/logs"
LOG_HARVEST="$LOGS/harvest"
LOG_KAGGLE="$LOGS/kaggle"
LOG_OCR="$LOGS/ocr"
LOG_SPEECH="$LOGS/speech"
LOG_MARGIN="$LOGS/margin"
LOG_BENCH="$LOGS/bench"
LOG_APP="$LOGS/app"
LOG_EVAL="$LOGS/eval"
LOG_TRAINING="$LOGS/training"
mkdir -p "$LOG_HARVEST" "$LOG_KAGGLE" "$LOG_OCR" "$LOG_SPEECH" \
  "$LOG_MARGIN" "$LOG_BENCH" "$LOG_APP" "$LOG_EVAL" "$LOG_TRAINING"
