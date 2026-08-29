#!/usr/bin/env bash
# OCR pass-7 data collection — serial, resumable, watchdog-friendly.
# Logs: logs/harvest-pass7.log  State: logs/harvest-pass7.state.json
# White-paper manifest: data/ocr/HARVEST-MANIFEST.tsv (git-tracked)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/harvest-pass7.log
STATE=logs/harvest-pass7.state.json
LOCK=logs/harvest-pass7.lock
MANIFEST=data/ocr/HARVEST-MANIFEST.tsv
PY=.venv/bin/python3
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# One orchestrator at a time — duplicate restarts were re-running OSM from scratch.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(ts)] SKIP already running (lock $(basename "$LOCK"))" | tee -a "$LOG"
  exit 0
fi

init_state() {
  if [[ ! -f "$STATE" ]]; then
    cat >"$STATE" <<EOF
{"started":"$(ts)","heartbeat":"$(ts)","step":"commons_cats","status":"running","steps":{}}
EOF
  fi
}

mark_step() {
  local step="$1" status="$2"
  "$PY" - "$STATE" "$step" "$status" "$(ts)" <<'PY'
import json, sys
path, step, status, hb = sys.argv[1:5]
data = json.loads(open(path, encoding="utf-8").read())
data.setdefault("steps", {})[step] = {"status": status, "at": hb}
data["step"] = step
data["status"] = status
data["heartbeat"] = hb
json.dump(data, open(path, "w", encoding="utf-8"), indent=2)
PY
}

manifest_row() {
  local source="$1" script="$2" count="$3" credits="$4" note="$5"
  mkdir -p "$(dirname "$MANIFEST")"
  [[ -f "$MANIFEST" ]] || printf "recorded_utc\tsource\tscript\tcount\tcredits_path\tnote\n" >"$MANIFEST"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(ts)" "$source" "$script" "$count" "$credits" "$note" >>"$MANIFEST"
}

count_lines() { [[ -f "$1" ]] && wc -l <"$1" | tr -d ' ' || echo 0; }

run_step() {
  local step="$1"; shift
  mark_step "$step" "running"
  echo "[$(ts)] START [$step] $*" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "[$(ts)] DONE  [$step] $*" | tee -a "$LOG"
    mark_step "$step" "done"
    return 0
  fi
  echo "[$(ts)] FAIL  [$step] exit=$? $*" | tee -a "$LOG"
  mark_step "$step" "failed"
  return 1
}

wait_worker() {
  local pattern="$1" step="$2"
  echo "[$(ts)] WAIT [$step] ($pattern already running)" | tee -a "$LOG"
  mark_step "$step" "running"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    heartbeat
    sleep 30
  done
}

heartbeat() {
  "$PY" - "$STATE" "$(ts)" <<'PY'
import json, sys
path, hb = sys.argv[1:3]
data = json.loads(open(path, encoding="utf-8").read())
data["heartbeat"] = hb
json.dump(data, open(path, "w", encoding="utf-8"), indent=2)
PY
}

step_done() {
  local step="$1"
  "$PY" -c "import json; d=json.load(open('$STATE')); print(d.get('steps',{}).get('$step',{}).get('status','')=='done')"
}

init_state
echo "[$(ts)] === pass-7 harvest (resume-aware) ===" | tee -a "$LOG"

# --- 1. Commons category trees ---
if [[ "$(step_done commons_cats)" != "True" ]]; then
  if pgrep -f "fetch_commons_categories.py" >/dev/null 2>&1; then
    wait_worker "fetch_commons_categories.py" commons_cats
  fi
  if [[ "$(step_done commons_cats)" != "True" ]]; then
    run_step commons_cats "$PY" data/scripts/fetch_commons_categories.py
  fi
  if [[ "$(step_done commons_cats)" == "True" ]]; then
    n=$(($(count_lines data/ocr/real-photos/commons-cats/CREDITS.tsv) - 1))
    manifest_row "Wikimedia Commons categories" "fetch_commons_categories.py" "$n" \
      "data/ocr/real-photos/commons-cats/CREDITS.tsv" "CC/PD per row; NC/ND skipped"
  fi
fi

# --- 2. OSM sign strings (checkpoint per city; do not restart mid-run) ---
if [[ "$(step_done osm_signs)" != "True" ]]; then
  if pgrep -f "harvest_sign_text.py" >/dev/null 2>&1; then
    wait_worker "harvest_sign_text.py" osm_signs
  fi
  if [[ "$(step_done osm_signs)" != "True" ]]; then
    run_step osm_signs "$PY" data/scripts/harvest_sign_text.py --resume
  fi
  n=$(count_lines data/signs/sign-text.tsv)
  manifest_row "OpenStreetMap sign text" "harvest_sign_text.py" "$n" \
    "data/signs/sign-text.tsv" "ODbL; attribution required"
fi

# --- 3. Synthetic v2 ---
if [[ "$(step_done synth_flat)" != "True" ]]; then
  run_step synth_flat "$PY" data/scripts/generate_ocr_data.py --count 50000
  n=$(ls data/ocr/synthetic 2>/dev/null | wc -l | tr -d ' ')
  manifest_row "Synthetic OCR crops" "generate_ocr_data.py" "$n" \
    "data/ocr/synthetic/" "labels by construction; đ oversampled"
fi

# --- 4. Commons photo harvest (EasyOCR screen) ---
if [[ "$(step_done commons_harvest)" != "True" ]]; then
  run_step commons_harvest "$PY" data/scripts/harvest_sign_photos.py \
    --target 500 --max-screen 2500 --workers 1 --max-crawl-calls 400
  n=$(($(count_lines data/ocr/real-photos/harvested/CREDITS.tsv) - 1))
  manifest_row "Wikimedia Commons photos" "harvest_sign_photos.py" "$n" \
    "data/ocr/real-photos/harvested/CREDITS.tsv" "text-on-sign filter; scored set excluded"
fi

# --- 5. Photo-style synthetic ---
if [[ "$(step_done synth_photo)" != "True" ]]; then
  run_step synth_photo "$PY" data/scripts/generate_ocr_photos.py --count 20000
  n=$(ls data/ocr/train/photo*.png 2>/dev/null | wc -l | tr -d ' ')
  manifest_row "Photographic synthetic" "generate_ocr_photos.py" "$n" \
    "data/ocr/train/" "degradation-matched to real failure modes"
fi

mark_step complete done
echo "[$(ts)] === pipeline complete ===" | tee -a "$LOG"
