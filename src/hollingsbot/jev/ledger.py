"""Jev's reply log and daily spend, in the bot's shared SQLite DB.

Every reply (finished or interrupted) is one row with its cost and the full
per-word trace, so an odd reply can be explained later:

    SELECT reply, stop_reason, cost, steps_json FROM jev_replies ORDER BY id DESC LIMIT 5;
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from hollingsbot import prompt_db

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from hollingsbot.jev.writer import ChatLine, Reply

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jev_replies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    day TEXT NOT NULL,
    channel_id INTEGER,
    message_id INTEGER,
    reply TEXT NOT NULL,
    stop_reason TEXT NOT NULL,
    words INTEGER NOT NULL,
    calls INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    cost REAL NOT NULL,
    seconds REAL NOT NULL,
    model TEXT,
    chat_json TEXT,
    steps_json TEXT
)
"""


def _utc_day(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


class JevLedger:
    """``db_path`` defaults to the bot DB (``prompt_db.DB_PATH``), read at call time."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = db_path
        self._ready_for: str | None = None

    @contextlib.contextmanager
    def _connect(self):
        path = str(self._db_path or prompt_db.DB_PATH)
        with contextlib.closing(sqlite3.connect(path, timeout=30.0)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            if self._ready_for != path:
                conn.execute(_SCHEMA)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_jev_replies_day ON jev_replies(day)")
                self._ready_for = path
            yield conn
            conn.commit()

    def spent_today(self, now: datetime | None = None) -> float:
        day = _utc_day(now or datetime.now(timezone.utc))
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(SUM(cost), 0) FROM jev_replies WHERE day = ?", (day,)).fetchone()
        return float(row[0])

    def record(
        self,
        reply: Reply,
        chat: Sequence[ChatLine],
        *,
        channel_id: int | None,
        message_id: int | None,
        stop_reason: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Log one reply. ``stop_reason`` overrides the writer's (e.g. "interrupted")."""
        now = now or datetime.now(timezone.utc)
        u = reply.usage
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jev_replies (
                    created_at, day, channel_id, message_id, reply, stop_reason, words, calls,
                    input_tokens, cost, seconds, model, chat_json, steps_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now.isoformat(),
                    _utc_day(now),
                    channel_id,
                    message_id,
                    reply.text,
                    stop_reason or reply.stop_reason,
                    len(reply.words),
                    u.calls,
                    u.input_tokens,
                    u.cost,
                    u.seconds,
                    reply.model,
                    json.dumps([asdict(line) for line in chat]),
                    json.dumps([asdict(step) for step in reply.steps]),
                ),
            )
