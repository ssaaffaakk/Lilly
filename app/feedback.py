#!/usr/bin/env python3
"""Correction database — the "this translation is wrong" loop.

SQLite, one file at data/feedback.db. Corrections wait for review; only
approved ones are exported as extra training data.

Flow: user presses "wrong" -> row saved (status=pending) -> we review ->
approved rows are exported by export_approved() and joined to the training set.

Both translation directions land in this one table and are told apart by the
`direction` column, because a correction is a correction whichever way the user
was working. They separate at export: each direction trains its own model and
gets its own file.
"""
import os
import re
import sqlite3
import time
from pathlib import Path

# Deployed, this has to point at a mounted volume: anywhere inside the
# deployment directory is wiped on the next restart, taking every correction
# anyone ever submitted with it.
DB_PATH = Path(os.environ.get("LILLY_DB")
               or Path(__file__).resolve().parents[1] / "data" / "feedback.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source_text TEXT NOT NULL,          -- what the user typed
    model_output TEXT NOT NULL,         -- what Lilly said
    user_complaint TEXT,                -- what the user says is wrong
    suggested_translation TEXT,         -- the user's proposed correction
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected
    direction TEXT NOT NULL DEFAULT 'bs-en'  -- bs-en | en-bs
);
"""

DIRECTIONS = ("bs-en", "en-bs")


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(corrections)")}
    if "direction" not in cols:
        conn.execute(
            "ALTER TABLE corrections ADD COLUMN direction TEXT NOT NULL DEFAULT 'bs-en'")
    return conn


def _flatten(text):
    """One line, no tabs.

    Approved corrections are exported as tab-separated training pairs, and both
    of these fields arrive from a stranger's POST body. A tab or a newline left
    in them would split one row into several and shift every column after it,
    quietly poisoning the training set.
    """
    return re.sub(r"\s+", " ", (text or "")).strip()


def add_correction(source_text, model_output, user_complaint="",
                   suggested_translation="", direction="bs-en"):
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, not {direction!r}")
    source_text = _flatten(source_text)
    model_output = _flatten(model_output)
    user_complaint = _flatten(user_complaint)
    suggested_translation = _flatten(suggested_translation)
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO corrections (created_at, source_text, model_output,"
            " user_complaint, suggested_translation, direction)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), source_text, model_output,
             user_complaint, suggested_translation, direction))
        return cur.lastrowid


def list_corrections(status="pending"):
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM corrections WHERE status = ? ORDER BY id", (status,))
        return [dict(r) for r in rows]


def set_status(correction_id, status):
    assert status in ("pending", "approved", "rejected")
    with _connect() as conn:
        conn.execute("UPDATE corrections SET status = ? WHERE id = ?",
                     (status, correction_id))


def export_approved(out_path=None, direction="bs-en"):
    """Write one direction's approved corrections as training pairs (TSV).

    One direction per call, never both. The TSV is source-then-target and each
    direction trains its own model, so a reverse pair in a forward file would
    teach that model to translate the wrong way round. Both directions collect
    corrections in the same table; this is where they part.

    Returns the path and the number of pairs actually written — not the number
    of approved rows, which is larger whenever the other direction has any.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, not {direction!r}")
    out_path = out_path or DB_PATH.parent / "clean" / f"corrections-{direction}.tsv"
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in list_corrections("approved"):
            if (r.get("direction") or "bs-en") != direction:
                continue
            if not r["suggested_translation"]:
                continue
            f.write(f"UserCorrections\t{r['source_text']}\t{r['suggested_translation']}\n")
            written += 1
    return str(out_path), written


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        # `export` alone writes bs-en, the direction that has a fine-tune;
        # `export en-bs` writes the reverse pool for when that one is trained.
        direction = sys.argv[2] if len(sys.argv) > 2 else "bs-en"
        path, n = export_approved(direction=direction)
        print(f"exported {n} approved {direction} corrections to {path}")
    else:
        for row in list_corrections():
            print(row)
