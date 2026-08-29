#!/usr/bin/env bash
# Back-compat wrapper — watches speech AND ocr (see kaggle_watch_kernels.sh).
exec "$(cd "$(dirname "$0")" && pwd)/kaggle_watch_kernels.sh" "$@"
