#!/bin/bash
# The whole Phase 2 run, start to finished table, detached so it survives the
# terminal closing. Start it with:
#
#     nohup bash scripts/run_training.sh > training.log 2>&1 &
#
# Watch it with:   tail -f training.log
# Stop it with:    pkill -f train_translation
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== $(date '+%H:%M:%S')  1/3  training ==="
# batch 8 x accum 8 keeps the effective batch at 64 while halving the peak, which
# matters on a machine this size. group_by_length is what makes it finish.
$PY training/train_translation.py --batch-size 8 --grad-accum 8

echo "=== $(date '+%H:%M:%S')  2/3  building the served model ==="
# A quantised model cannot take an adapter later, so the merge happens here.
$PY scripts/build_translator.py

echo "=== $(date '+%H:%M:%S')  3/3  scoring against the base ==="
$PY training/evaluate.py --adapter models/lilly/adapter

echo "=== $(date '+%H:%M:%S')  done ==="
cat training/RESULTS.md
