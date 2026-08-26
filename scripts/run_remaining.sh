#!/bin/bash
# Everything after the translation run, in order, unattended.
#
# Waits for the translation training to finish, then trains the listening, then
# the photo reading. Each one measures before and after, and refuses to replace
# a shipped model that it made worse.
#
#     nohup bash scripts/run_remaining.sh > remaining.log 2>&1 &
#
# Watch:  tail -f remaining.log
# Stop:   pkill -f run_remaining
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== $(date '+%H:%M:%S')  waiting for the translation run ==="
# Wait for the whole run_training.sh, not just the training process: it goes on to
# merge and quantise the model and then score it, and starting a second training
# on top of that is what turns a two-hour job into an eight-hour one.
while pgrep -f "run_training.sh" > /dev/null || pgrep -f train_translation > /dev/null; do
  sleep 60
done
echo "=== $(date '+%H:%M:%S')  translation pipeline finished ==="
[ -f training/RESULTS.md ] && cat training/RESULTS.md

# ---------------------------------------------------------------- listening
echo
echo "=== $(date '+%H:%M:%S')  speech 1/3  fetching the clips ==="
# The clips are 3 GB; only fetch what is missing.
$PY data/scripts/download_speech_data.py || echo "  speech data failed — skipping speech"

if [ -f data/speech/train.tsv ] && [ -f data/speech/test.tsv ]; then
  echo "=== $(date '+%H:%M:%S')  speech 2/3  before ==="
  $PY training/evaluate_speech.py --data data/speech/test.tsv --limit 200 --show 3 || true

  echo "=== $(date '+%H:%M:%S')  speech 3/3  training ==="
  $PY training/train_speech.py --data data/speech/train.tsv --no-convert \
    && $PY training/train_speech.py --base models/lilly/listen-trained \
                                    --convert-only models/lilly/listen \
    && $PY training/evaluate_speech.py --data data/speech/test.tsv --limit 200 --show 3
else
  echo "  no speech data on disk — skipped"
fi

# ------------------------------------------------------------- photo reading
echo
echo "=== $(date '+%H:%M:%S')  photo 1/2  generating Bosnian text images ==="
$PY data/scripts/generate_ocr_data.py --count 20000

echo "=== $(date '+%H:%M:%S')  photo 2/2  training the reader ==="
$PY training/train_ocr.py

echo
echo "=== $(date '+%H:%M:%S')  everything done ==="
echo "translation:"; cat training/RESULTS.md 2>/dev/null | head -12
