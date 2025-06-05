import os
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(os.getenv("PROMPT_DB_PATH", Path(__file__).resolve().with_name("prompts.db")))


def init_db() -> None:
    """Create required tables if they don't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                submitter TEXT,
                api TEXT,
                model TEXT,
                time TEXT,
                status TEXT
            )
            """
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS prs (number INTEGER PRIMARY KEY, status TEXT)"
        )
        conn.commit()


def add_prompt(text: str, submitter: str, api: str, model: str, *, status: str = "queued") -> int:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO prompts (text, submitter, api, model, time, status) VALUES (?, ?, ?, ?, ?, ?)",
            (text, submitter, api, model, datetime.utcnow().isoformat(), status),
        )
        conn.commit()
        return cur.lastrowid


def update_status(prompt_id: int, status: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE prompts SET status = ? WHERE id = ?", (status, prompt_id))
        conn.commit()


def get_seen_prs() -> dict[int, str]:
    """Return a mapping of PR numbers to their status."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT number, status FROM prs")
        return {int(row[0]): row[1] for row in cur.fetchall()}


def update_pr_status(number: int, status: str) -> None:
    """Insert or update a PR's status."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO prs (number, status) VALUES (?, ?)",
            (number, status),
        )
        conn.commit()
