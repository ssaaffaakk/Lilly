#!/usr/bin/env bash
# Every 5 min: honest progress report + restart only when truly stalled.
set -u
cd "$(dirname "$0")/.."
# shellcheck source=logs_paths.sh
source "$(dirname "$0")/logs_paths.sh"
PY=.venv/bin/python3
if [[ "${LILLY_HARVEST_ROLE:-}" != "watch" ]]; then
  exec "$PY" scripts/harvest_detach.py --role watch --log "$LOG_HARVEST/watch.nohup" -- "$PWD/scripts/harvest_watch.sh" "$@"
fi
# shellcheck source=harvest_lock.sh
source "$(dirname "$0")/harvest_lock.sh"
ORCH=./scripts/harvest_pass7.sh

while true; do
  out=$($PY scripts/harvest_report.py 2>&1)
  rc=$?
  printf '%s\n' "$out"
  if [[ $rc -eq 2 ]]; then
    exit 0
  fi
  if [[ $rc -eq 3 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog skip — step failed (not a stall)" >>"$LOG_HARVEST/pass7.log"
    sleep 300
    continue
  fi
  if [[ $rc -eq 1 ]]; then
    if harvest_lock_held; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog skip — lock held" >>"$LOG_HARVEST/pass7.log"
    else
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog restart (stalled)" >>"$LOG_HARVEST/pass7.log"
      "$PY" scripts/harvest_detach.py --role orch --log "$LOG_HARVEST/pass7.nohup" -- "$PWD/scripts/harvest_pass7.sh"
    fi
  fi
  sleep 300
done
