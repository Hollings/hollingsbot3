"""Channels Jev was spawned into with `!spawn jev N`, and how many replies it has left in each.

A spawned channel works like one in JEV_BOT_CHANNELS until Jev has posted its N
replies there (or someone runs `!despawn jev`). The visits live in the bot's
shared SQLite DB, so a restart (every deploy) doesn't silently end one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from hollingsbot.jev.db import connect

if TYPE_CHECKING:
    from pathlib import Path

_TABLE = """
CREATE TABLE IF NOT EXISTS jev_spawns (
    channel_id INTEGER PRIMARY KEY,
    replies_left INTEGER NOT NULL,
    spawned_by TEXT,
    spawned_at TEXT NOT NULL
)
"""
_SCHEMA = (_TABLE,)


class JevSpawns:
    """``db_path`` defaults to the bot DB (``prompt_db.DB_PATH``), read at call time."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = db_path

    def _connect(self):
        return connect(self._db_path, _SCHEMA)

    def active(self) -> dict[int, int]:
        """Every current visit: channel ID -> replies left."""
        with self._connect() as conn:
            return dict(conn.execute("SELECT channel_id, replies_left FROM jev_spawns WHERE replies_left > 0"))

    def start(self, channel_id: int, replies: int, *, by: str, now: datetime | None = None) -> None:
        """Start a visit of ``replies`` replies (spawning again resets the count)."""
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jev_spawns (channel_id, replies_left, spawned_by, spawned_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    replies_left = excluded.replies_left,
                    spawned_by = excluded.spawned_by,
                    spawned_at = excluded.spawned_at
                """,
                (channel_id, replies, by, stamp),
            )

    def use_reply(self, channel_id: int) -> int | None:
        """Jev posted a reply here: how many it has left (0 ends the visit), None if it isn't visiting."""
        with self._connect() as conn:
            row = conn.execute("SELECT replies_left FROM jev_spawns WHERE channel_id = ?", (channel_id,)).fetchone()
            if row is None:
                return None
            left = max(0, row[0] - 1)
            if left:
                conn.execute("UPDATE jev_spawns SET replies_left = ? WHERE channel_id = ?", (left, channel_id))
            else:
                conn.execute("DELETE FROM jev_spawns WHERE channel_id = ?", (channel_id,))
            return left

    def end(self, channel_id: int) -> bool:
        """End the visit early; False if there was none."""
        with self._connect() as conn:
            return conn.execute("DELETE FROM jev_spawns WHERE channel_id = ?", (channel_id,)).rowcount > 0
