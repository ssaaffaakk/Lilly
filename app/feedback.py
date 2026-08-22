#!/usr/bin/env python3
"""Correction database — the "this translation is wrong" loop.

SQLite, one file at data/feedback.db. Corrections wait for review; only
approved ones are exported as extra training data.

Flow: user presses "wrong" -> row saved (status=pending) -> we review ->
approved rows are exported by export_approved() and joined to the training set.
"""
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "feedback.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source_text TEXT NOT NULL,          -- the Bosnian input
    model_output TEXT NOT NULL,         -- what Lilly said
    user_complaint TEXT,                -- what the user says is wrong
    suggested_translation TEXT,         -- the user's proposed correct English
    status TEXT NOT NULL DEFAULT 'pending'  -- pending | approved | rejected
);
"""


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def add_correction(source_text, model_output, user_complaint="", suggested_translation=""):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO corrections (created_at, source_text, model_output,"
            " user_complaint, suggested_translation) VALUES (?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), source_text, model_output,
             user_complaint, suggested_translation))
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


def export_approved(out_path=None):
    """Write approved corrections as extra training pairs (same TSV format)."""
    out_path = out_path or DB_PATH.parent / "clean" / "corrections.tsv"
    rows = list_corrections("approved")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            if r["suggested_translation"]:
                f.write(f"UserCorrections\t{r['source_text']}\t{r['suggested_translation']}\n")
    return str(out_path), len(rows)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        path, n = export_approved()
        print(f"exported {n} approved corrections to {path}")
    else:
        for row in list_corrections():
            print(row)
