#!/usr/bin/env python3
"""Kaggle Output contract for Lilly notebooks.

Taken from agentic-kaggle-skill `references/kaggle-offload.md` (experiment_log,
metrics.jsonl, artifacts_manifest, stdout tee) and from trainproof (a log that
cannot be judged is not a pass). Lilly's end is the photograph / WER gate, not
a competition submission. COMPLETE still does not mean install.

Call after the clone so `training/` is on sys.path. The tee itself lives in
the notebook's first cell — that cell runs before git clone.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

WORKING = Path("/kaggle/working")
STDOUT = WORKING / "stdout.txt"
LOG = WORKING / "experiment_log.json"
METRICS = WORKING / "metrics.jsonl"
MANIFEST = WORKING / "artifacts_manifest.json"

# Phrases the trainers already print when they stop. If they appear in the tee
# and the process still returned 0, that is a hole — raise here.
TRAINPROOF_STOP = (
    "loss became non-finite",
    "the encoder is not learning",
    "FiniteLossCheck",
    "training collapsed",
)


def git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def scan_trainproof(path: Path = STDOUT) -> None:
    """Fail if the tee contains a trainer death the kernel still treated as 0."""
    if not path.is_file():
        raise SystemExit(
            f"no tee at {path} — the run cannot be judged (trainproof exit 2)")
    text = path.read_text(encoding="utf-8", errors="replace")
    for needle in TRAINPROOF_STOP:
        if needle in text:
            raise SystemExit(
                f"trainproof: {needle!r} in {path.name} — not packaging")


def require_wer(path: Path) -> dict:
    """Half-2 trainsafe analog: a listener that scored nothing is not shippable.

    Language probes from the trainsafe package are Arabic/Chinese defaults.
    Lilly's check is Bosnian WER on held-out clips, written as JSON.
    """
    if not path.is_file():
        raise SystemExit(f"no WER json at {path} — not packaging")
    got = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(got, dict) or "wer" not in got:
        raise SystemExit(f"{path} is not a WER record")
    if int(got.get("words") or 0) < 1:
        raise SystemExit("WER scored 0 reference words — not packaging")
    return got


class Offload:
    """Write the four Output files the offload skill requires."""

    def __init__(self, job: str, run_id: str):
        WORKING.mkdir(parents=True, exist_ok=True)
        self.body = {
            "job": job,
            "run_id": run_id,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": git_sha(),
            "status": "running",
            "error": None,
            "metrics": {},
            "artifacts": [],
        }
        self.flush()

    def flush(self) -> None:
        LOG.write_text(json.dumps(self.body, indent=2) + "\n", encoding="utf-8")

    def hardware(self, name: str) -> None:
        self.body["hardware"] = name
        self.flush()

    def metric(self, name: str, value, *, stage: str = "", step: int | None = None) -> None:
        rec = {
            "metric": name,
            "value": value,
            "stage": stage,
            "step": step,
            "ts": time.time(),
        }
        with METRICS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.body["metrics"][name] = value
        self.flush()

    def artifacts(self, paths: list[str]) -> None:
        self.body["artifacts"] = paths
        MANIFEST.write_text(json.dumps({"files": paths}, indent=2) + "\n",
                            encoding="utf-8")
        self.flush()

    def fail(self, error: str) -> None:
        self.body["status"] = "failed"
        self.body["error"] = error
        self.body["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.flush()

    def check_trainproof(self, path: Path = STDOUT) -> None:
        try:
            scan_trainproof(path)
        except SystemExit as exc:
            self.fail(str(exc))
            raise

    def finish(self, status: str, artifacts: list[str] | None = None) -> None:
        if artifacts is not None:
            self.artifacts(artifacts)
        self.body["status"] = status
        self.body["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.flush()
