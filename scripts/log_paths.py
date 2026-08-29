"""Canonical paths for local runtime logs (all under logs/)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"
HARVEST = LOGS / "harvest"
KAGGLE = LOGS / "kaggle"
OCR = LOGS / "ocr"
SPEECH = LOGS / "speech"
MARGIN = LOGS / "margin"
BENCH = LOGS / "bench"
APP = LOGS / "app"
EVAL = LOGS / "eval"
TRAINING = LOGS / "training"

for _d in (HARVEST, KAGGLE, OCR, SPEECH, MARGIN, BENCH, APP, EVAL, TRAINING):
    _d.mkdir(parents=True, exist_ok=True)
