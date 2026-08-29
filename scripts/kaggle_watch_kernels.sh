#!/usr/bin/env bash
# Watch lilly-speech and lilly-ocr on Kaggle every N seconds.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
INTERVAL="${1:-300}"
LOG="$ROOT/kaggle-kernels-watch.log"
PY="$ROOT/.venv/bin/python"
KAGGLE="$ROOT/.venv/bin/kaggle"
declare -A LAST=()

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

status_line() {
  local job="$1" attempt line
  for attempt in 1 2 3; do
    line="$("$PY" scripts/kaggle_train.py "$job" --status 2>&1 | tail -1)" && {
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

log "watch started (interval ${INTERVAL}s) — speech + ocr"

while true; do
  sleep "$INTERVAL"
  for job in speech ocr; do
    slug="afaksrmeli/lilly-${job}"
    if ! line="$(status_line "$job")"; then
      log "$job: status check failed after 3 retries"
      continue
    fi
    state="$(echo "$line" | sed -n 's/.*status "\([^"]*\)".*/\1/p')"
    [ -z "$state" ] && state="UNKNOWN"
    prev="${LAST[$job]:-}"
    log "[$job] $line"

    if [ "$state" = "$prev" ] && [ "$state" = "KernelWorkerStatus.RUNNING" ]; then
      continue
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
    LAST[$job]="$state"
  done
done
