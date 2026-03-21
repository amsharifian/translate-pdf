from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from translator.crypto import encrypt, decrypt

DB_PATH = Path("jobs/queue.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_name TEXT,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                provider TEXT NOT NULL,
                model TEXT,
                base_url TEXT,
                api_key TEXT,
                font_path TEXT,
                input_files TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                progress_current INTEGER DEFAULT 0,
                progress_total INTEGER DEFAULT 0,
                error TEXT
            )
            """
        )
        # Best-effort migration if table already exists
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN job_name TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        # New columns for extended features
        for col_def in [
            "target_lang TEXT DEFAULT 'fa'",
            "glossary TEXT DEFAULT '{}'",
            "page_range TEXT",
            "side_by_side INTEGER DEFAULT 0",
            "webhook_url TEXT",
            "job_log TEXT",
            "custom_prompt TEXT DEFAULT ''",
        ]:
            try:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass
    conn.close()


def create_job(job: Dict[str, Any]) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, job_name, status, priority, provider, model, base_url, api_key, font_path,
                input_files, output_dir, created_at, updated_at,
                progress_current, progress_total, error,
                target_lang, glossary, page_range, side_by_side, webhook_url, custom_prompt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["id"],
                job.get("job_name"),
                job["status"],
                job.get("priority", 0),
                job["provider"],
                job.get("model"),
                job.get("base_url"),
                encrypt(job.get("api_key") or ""),
                job.get("font_path"),
                json.dumps(job["input_files"]),
                job["output_dir"],
                job["created_at"],
                job["updated_at"],
                job.get("progress_current", 0),
                job.get("progress_total", 0),
                job.get("error"),
                job.get("target_lang", "fa"),
                json.dumps(job.get("glossary") or {}),
                json.dumps(job.get("page_range")) if job.get("page_range") else None,
                1 if job.get("side_by_side") else 0,
                job.get("webhook_url"),
                job.get("custom_prompt") or "",
            ),
        )
    conn.close()


def _decrypt_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decrypt sensitive fields in a job row."""
    row = dict(row)
    if row.get("api_key"):
        row["api_key"] = decrypt(row["api_key"])
    return row


def list_jobs() -> List[Dict[str, Any]]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY status IN ('queued','running','paused') DESC, priority DESC, created_at DESC"
    ).fetchall()
    conn.close()
    return [_decrypt_row(row) for row in rows]


def fetch_next_job() -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM jobs WHERE status = 'queued' ORDER BY priority DESC, created_at ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return _decrypt_row(row) if row else None


def update_job_status(job_id: str, status: str, error: str | None = None) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, updated_at = datetime('now') WHERE id = ?",
            (status, error, job_id),
        )
    conn.close()


def get_job_status(job_id: str) -> Optional[str]:
    conn = _connect()
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return row["status"] if row else None


def update_priority(job_id: str, priority: int) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            "UPDATE jobs SET priority = ?, updated_at = datetime('now') WHERE id = ?",
            (priority, job_id),
        )
    conn.close()


def delete_job(job_id: str) -> None:
    conn = _connect()
    with conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.close()


def update_progress(job_id: str, current: int, total: int) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
            UPDATE jobs
            SET progress_current = ?, progress_total = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (current, total, job_id),
        )
    conn.close()
