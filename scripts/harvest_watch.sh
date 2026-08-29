#!/usr/bin/env bash
# Every 5 min: report harvest progress + restart if stalled.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
ORCH=./scripts/harvest_pass7.sh
mkdir -p logs

while true; do
  out=$($PY scripts/harvest_report.py 2>&1)
  printf '%s\n' "$out"
  if printf '%s\n' "$out" | grep -q "status=COMPLETE"; then
    exit 0
  fi
  if printf '%s\n' "$out" | grep -q "status=STALLED"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog restart" >>logs/harvest-pass7.log
    nohup "$ORCH" >>logs/harvest-pass7.nohup 2>&1 &
  fi
  sleep 300
done
