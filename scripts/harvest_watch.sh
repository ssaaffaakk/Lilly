#!/usr/bin/env bash
# Every 5 min: honest progress report + restart only when truly stalled.
set -u
cd "$(dirname "$0")/.."
# shellcheck source=harvest_lock.sh
source "$(dirname "$0")/harvest_lock.sh"
PY=.venv/bin/python3
ORCH=./scripts/harvest_pass7.sh
mkdir -p logs

while true; do
  out=$($PY scripts/harvest_report.py 2>&1)
  rc=$?
  printf '%s\n' "$out"
  if [[ $rc -eq 2 ]]; then
    exit 0
  fi
  if [[ $rc -eq 1 ]]; then
    if harvest_lock_held; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog skip — lock held" >>logs/harvest-pass7.log
    else
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog restart (stalled)" >>logs/harvest-pass7.log
      nohup "$ORCH" >>logs/harvest-pass7.nohup 2>&1 &
    fi
  fi
  sleep 300
done
