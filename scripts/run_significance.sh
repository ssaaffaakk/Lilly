#!/bin/bash
# The p-value for the translation result, once the machine is free.
#
# The run that produced RESULTS.md could not compute it — sacrebleu's paired
# test takes a list of (name, outputs) pairs, not a dict, and its submodule is
# not pulled in by the top-level import. Both fixed; this re-scores to fill the
# number in. It also leaves the translations in training/hypotheses.json, so
# this never costs an hour again.
cd "$(dirname "$0")/.."
while pgrep -f run_remaining > /dev/null; do sleep 120; done
echo "=== $(date '+%H:%M:%S')  rescoring for the p-value ==="
.venv/bin/python training/evaluate.py --adapter models/lilly/adapter
echo "=== $(date '+%H:%M:%S')  done ==="
grep -A3 "Is the difference real" training/RESULTS.md
