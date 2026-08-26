#!/bin/bash
# Everything after the translation run, in order, unattended.
#
#     nohup bash scripts/run_remaining.sh > remaining.log 2>&1 &
#
# Watch:  tail -f remaining.log
# Stop:   pkill -f run_remaining
#
# Every stage checks what it produced rather than trusting an exit code, and the
# run ends with a plain list of what worked and what did not. The previous
# version printed "everything done" after skipping the speech training entirely
# and crashing the photo training — which is worse than failing.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FAILED=()
DONE=()

note_ok()   { DONE+=("$1");   echo "  OK: $1"; }
note_fail() { FAILED+=("$1"); echo "  FAILED: $1"; }

echo "=== $(date '+%H:%M:%S')  waiting for anything still running ==="
# Match the python invocations, not the shell script names. A watcher process
# with "run_significance" in its own command line matches a loose pattern and
# the wait never ends — which cost an hour of doing nothing once already.
busy() {
  pgrep -f "training/train_translation.py" > /dev/null && return 0
  pgrep -f "training/evaluate.py" > /dev/null && return 0
  return 1
}
while busy; do sleep 60; done

# ---------------------------------------------------------------- listening
echo
echo "=== $(date '+%H:%M:%S')  speech 1/4  fetching the clips ==="
# Retry: this is 3 GB over the network and the first attempt lost the train split.
for attempt in 1 2 3; do
  $PY data/scripts/download_speech_data.py && break
  echo "  attempt $attempt failed, retrying"
  sleep 30
done

missing=""
for split in train valid test; do
  if [ ! -s "data/speech/$split.tsv" ]; then missing="$missing $split"; fi
done

if [ -n "$missing" ]; then
  note_fail "speech data ($missing never arrived)"
else
  note_ok "speech data: $(wc -l < data/speech/train.tsv) train, $(wc -l < data/speech/test.tsv) test"

  echo "=== $(date '+%H:%M:%S')  speech 2/4  before ==="
  $PY training/evaluate_speech.py --data data/speech/test.tsv --limit 200 --show 3

  echo "=== $(date '+%H:%M:%S')  speech 3/4  training ==="
  if $PY training/train_speech.py --data data/speech/train.tsv --no-convert \
     && [ -f models/lilly/listen-trained/model.safetensors ]; then
    echo "=== $(date '+%H:%M:%S')  speech 4/4  converting and scoring ==="
    if $PY training/train_speech.py --base models/lilly/listen-trained \
                                    --convert-only models/lilly/listen \
       && [ -f models/lilly/listen/model.bin ]; then
      $PY training/evaluate_speech.py --data data/speech/test.tsv --limit 200 --show 3
      note_ok "speech trained and installed"
    else
      note_fail "speech conversion (the trained checkpoint is still in models/lilly/listen-trained)"
    fi
  else
    note_fail "speech training"
  fi
fi

# ------------------------------------------------------------- photo reading
echo
echo "=== $(date '+%H:%M:%S')  photo 1/2  generating Bosnian text images ==="
if $PY data/scripts/generate_ocr_data.py --count 20000 && [ -s data/ocr/train/gt.txt ]; then
  note_ok "photo data: $(wc -l < data/ocr/train/gt.txt) images"
  echo "=== $(date '+%H:%M:%S')  photo 2/2  training the reader ==="
  if $PY training/train_ocr.py; then
    note_ok "photo reader trained"
  else
    note_fail "photo reader training"
  fi
else
  note_fail "photo data generation"
fi

# --------------------------------------------------------------------- report
echo
echo "=== $(date '+%H:%M:%S')  summary ==="
for d in "${DONE[@]}";   do echo "  worked:  $d"; done
for f in "${FAILED[@]}"; do echo "  FAILED:  $f"; done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "RUN INCOMPLETE — ${#FAILED[@]} stage(s) failed"
  exit 1
fi
echo "all stages completed"
