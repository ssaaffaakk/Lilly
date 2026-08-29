#!/usr/bin/env bash
# Watch lilly-speech on Kaggle every N seconds. Logs to kaggle-speech-watch.log.
# Emits AGENT_LOOP_TICK_kaggle for the agent when status changes or on ERROR/COMPLETE.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
INTERVAL="${1:-300}"
LOG="$ROOT/kaggle-speech-watch.log"
PY="$ROOT/.venv/bin/python"
LAST=""

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

status_line() {
  local attempt line
  for attempt in 1 2 3; do
    line="$("$PY" scripts/kaggle_train.py speech --status 2>&1 | tail -1)" && {
      echo "$line"
      return 0
    }
    sleep 10
  done
  return 1
}

tick() {
  local reason="$1"
  echo "AGENT_LOOP_TICK_kaggle {\"prompt\":\"${reason}\",\"log\":\"$LOG\"}"
}

log "watch started (interval ${INTERVAL}s)"

while true; do
  sleep "$INTERVAL"
  if ! line="$(status_line)"; then
    log "status check failed after 3 retries (Kaggle API?) — will retry next interval"
    continue
  fi
  state="$(echo "$line" | sed -n 's/.*status "\([^"]*\)".*/\1/p')"
  [ -z "$state" ] && state="UNKNOWN"
  log "$line"

  if [ "$state" = "$LAST" ] && [ "$state" = "KernelWorkerStatus.RUNNING" ]; then
    continue
  fi

  case "$state" in
    *ERROR*)
      tick "Kaggle lilly-speech FAILED. Fetch lilly-speech.log, diagnose root cause, fix repo, push, relaunch speech."
      mkdir -p /tmp/lilly-speech-out
      "$ROOT/.venv/bin/kaggle" kernels output afaksrmeli/lilly-speech -p /tmp/lilly-speech-out 2>>"$LOG" || true
      ;;
    *COMPLETE*)
      tick "Kaggle lilly-speech COMPLETE. Tell user to run speech --fetch and install listen weights."
      ;;
    *RUNNING*)
      tick "Kaggle lilly-speech still RUNNING. Brief progress note only."
      ;;
    *)
      tick "Kaggle lilly-speech status: ${state}. Investigate."
      ;;
  esac
  LAST="$state"
done
