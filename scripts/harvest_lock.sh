#!/usr/bin/env bash
# Portable single-instance lock (macOS has no flock). Source from harvest scripts.
set -u
HARVEST_LOCKDIR="${HARVEST_LOCKDIR:-logs/harvest-pass7.lock.d}"

harvest_lock_stale() {
  [[ ! -d "$HARVEST_LOCKDIR" ]] && return 0
  local pid=""
  [[ -f "$HARVEST_LOCKDIR/pid" ]] && pid=$(cat "$HARVEST_LOCKDIR/pid" 2>/dev/null || true)
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    rm -rf "$HARVEST_LOCKDIR"
    return 0
  fi
  return 1
}

harvest_lock_held() {
  harvest_lock_stale
  [[ -d "$HARVEST_LOCKDIR" ]]
}

harvest_acquire_lock() {
  mkdir -p "$(dirname "$HARVEST_LOCKDIR")"
  harvest_lock_stale
  if ! mkdir "$HARVEST_LOCKDIR" 2>/dev/null; then
    harvest_lock_stale && mkdir "$HARVEST_LOCKDIR" 2>/dev/null || return 1
  fi
  echo $$ >"$HARVEST_LOCKDIR/pid"
  trap 'harvest_release_lock' EXIT INT TERM
  return 0
}

harvest_release_lock() {
  rm -rf "$HARVEST_LOCKDIR"
}
