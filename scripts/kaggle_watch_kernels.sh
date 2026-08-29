#!/usr/bin/env bash
# Watch lilly-speech and lilly-ocr on Kaggle every N seconds.
# macOS ships bash 3.2 — no associative arrays; kaggle CLI for status (not python).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
INTERVAL="${1:-300}"
LOG="$ROOT/kaggle-kernels-watch.log"
KAGGLE="$ROOT/.venv/bin/kaggle"
USER="${KAGGLE_USER:-afaksrmeli}"
LAST_speech=""
LAST_ocr=""

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

status_line() {
  local slug="$1" attempt line
  for attempt in 1 2 3; do
    line="$("$KAGGLE" kernels status "$slug" 2>&1)" && {
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

watch_job() {
  local job="$1" slug="$2" prev state line
  case "$job" in
    speech) prev="$LAST_speech" ;;
    ocr)    prev="$LAST_ocr" ;;
  esac

  if ! line="$(status_line "$slug")"; then
    log "$job: status check failed after 3 retries"
    return
  fi
  state="$(echo "$line" | sed -n 's/.*status "\([^"]*\)".*/\1/p')"
  [ -z "$state" ] && state="UNKNOWN"
  log "[$job] $line"

  if [ "$state" = "$prev" ] && [ "$state" = "KernelWorkerStatus.RUNNING" ]; then
    return
  fi

  case "$state" in
    *ERROR*)
      tick "Kaggle lilly-${job} FAILED. Fetch logs, diagnose, fix, push, relaunch."
      mkdir -p "/tmp/lilly-${job}-out"
      "$KAGGLE" kernels output "$slug" -p "/tmp/lilly-${job}-out" 2>>"$LOG" || true
      ;;
    *COMPLETE*)
      tick "Kaggle lilly-${job} COMPLETE. Run kaggle_train.py ${job} --fetch and install."
      ;;
    *RUNNING*)
      tick "Kaggle lilly-${job} still RUNNING."
      ;;
    *NONE*|*"no recent"*)
      : # not launched yet
      ;;
    *)
      tick "Kaggle lilly-${job} status: ${state}. Investigate."
      ;;
  esac

  case "$job" in
    speech) LAST_speech="$state" ;;
    ocr)    LAST_ocr="$state" ;;
  esac
}

log "watch started (interval ${INTERVAL}s) — speech + ocr"

while true; do
  watch_job speech "${USER}/lilly-speech"
  watch_job ocr "${USER}/lilly-ocr"
  sleep "$INTERVAL"
done
