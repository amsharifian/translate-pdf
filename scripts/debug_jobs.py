#!/usr/bin/env python3
"""Quick debug script to check job queue and worker status."""
import os
import sqlite3
import json
from pathlib import Path

DB = Path("jobs/queue.db")
PID_FILE = Path("jobs/.worker.pid")

print("=" * 60)
print("WORKER STATUS")
print("=" * 60)
if PID_FILE.exists():
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)
        print(f"  Worker PID {pid}: ALIVE")
    except OSError:
        print(f"  Worker PID {pid}: DEAD (stale PID file)")
else:
    print("  No PID file — worker was never started")

print()
print("=" * 60)
print("JOB QUEUE")
print("=" * 60)
if not DB.exists():
    print("  No database found.")
else:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, job_name, status, progress_current, progress_total, error, created_at "
        "FROM jobs ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    if not rows:
        print("  No jobs in queue.")
    for r in rows:
        name = (r["job_name"] or "unnamed")[:40]
        err = f" ERROR: {r['error'][:80]}" if r["error"] else ""
        print(f"  [{r['status']:10s}] {r['progress_current']}/{r['progress_total']} pages | {name}{err}")
        print(f"             id={r['id']}")
        print(f"             created={r['created_at']}")
        print()
    conn.close()

print("=" * 60)
print("LOG FILES")
print("=" * 60)
logs = list(Path("jobs").rglob("*.log.jsonl"))
if not logs:
    print("  No log files found (worker hasn't written page logs yet)")
else:
    for l in logs:
        lines = l.read_text().strip().splitlines()
        print(f"  {l}: {len(lines)} entries")
