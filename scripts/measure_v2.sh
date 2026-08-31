#!/bin/bash
# v2 end-of-training measurement — one job at a time, serial queue.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=logs_paths.sh
source "$(dirname "$0")/logs_paths.sh"
PY=.venv/bin/python
LOG="$LOG_TRAINING/measure-v2-full.log"
REPORT=training/RESULTS-v2-full.md

: > "$LOG"
echo "# Lilly v2 — full measurement pass" > "$REPORT"
echo "Started: $(date)" >> "$REPORT"
echo "" >> "$REPORT"

section() {
  echo ""
  echo "========== $1 ==========" | tee -a "$LOG"
  echo "" >> "$REPORT"
  echo "## $1" >> "$REPORT"
  echo '```' >> "$REPORT"
}

end_section() { echo '```' >> "$REPORT"; echo "" >> "$REPORT"; }

section "DINLEME — WER (yeni model, 200 clip)"
$PY training/evaluate_speech.py --data data/speech/test.tsv --limit 200 --show 3 2>&1 | tee -a "$LOG" >> "$REPORT"
end_section

section "DINLEME — WER (önceki model, 200 clip)"
$PY training/evaluate_speech.py --data data/speech/test.tsv --limit 200 \
  --model models/lilly/listen-previous --show 3 2>&1 | tee -a "$LOG" >> "$REPORT"
end_section

section "DINLEME — SpeechBench (before vs after, first200)"
$PY training/speech_bench.py --clips first200 --fresh \
  --model models/lilly/listen.before-training \
  --model models/lilly/listen 2>&1 | tee -a "$LOG" >> "$REPORT"
end_section

section "OKUMA — gerçek fotoğraflar"
$PY training/evaluate_ocr.py --out training/RESULTS-ocr-v2-final.md 2>&1 | tee -a "$LOG"
end_section
cat training/RESULTS-ocr-v2-final.md >> "$REPORT"

section "YAZMA — çeviri (FLORES devtest)"
$PY training/evaluate_app.py --split devtest --fresh 2>&1 | tee -a "$LOG" >> "$REPORT"
end_section
cat training/RESULTS-product.md >> "$REPORT"

section "YAZMA — BosnianBench"
$PY training/bosnian_bench.py --fresh 2>&1 | tee -a "$LOG" >> "$REPORT"
end_section

section "KONUŞMA — TTS smoke (v1, değişmedi)"
$PY app/tts.py "Hello from Lilly." /tmp/lilly-speak-smoke.wav 2>&1 | tee -a "$LOG" >> "$REPORT"
ls -lah /tmp/lilly-speak-smoke.wav 2>&1 | tee -a "$LOG" >> "$REPORT"
end_section

echo "Finished: $(date)" >> "$REPORT"
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
