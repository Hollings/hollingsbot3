"""Channels a Jev was spawned into with `!spawn jev N` (or `!spawn jev2 N`), and how many replies it has left.

A spawned channel works like one in JEV_BOT_CHANNELS until that Jev has posted
its N replies there (or someone runs `!despawn jev`). Each Jev (Jev, and the
spawn-only copies like Jev2) has its own visits, so two can be in one channel.
The visits live in the bot's shared SQLite DB, so a restart (every deploy)
doesn't silently end one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from hollingsbot.jev.db import connect

if TYPE_CHECKING:
    from pathlib import Path

_TABLE = """
CREATE TABLE IF NOT EXISTS jev_visits (
    channel_id INTEGER NOT NULL,
    bot TEXT NOT NULL,
    replies_left INTEGER NOT NULL,
    spawned_by TEXT,
    spawned_at TEXT NOT NULL,
    PRIMARY KEY (channel_id, bot)
)
"""
# The first layout (2026-09-26) had one Jev per channel in `jev_spawns`: carry its visits over as Jev's.
_FROM_ONE_JEV = (
    """
    CREATE TABLE IF NOT EXISTS jev_spawns (
        channel_id INTEGER PRIMARY KEY, replies_left INTEGER NOT NULL, spawned_by TEXT, spawned_at TEXT NOT NULL
    )
    """,
    """
    INSERT OR IGNORE INTO jev_visits (channel_id, bot, replies_left, spawned_by, spawned_at)
    SELECT channel_id, 'Jev', replies_left, spawned_by, spawned_at FROM jev_spawns
    """,
    "DROP TABLE jev_spawns",
)
_SCHEMA = (_TABLE, *_FROM_ONE_JEV)


class JevSpawns:
    """The visits of the Jev named ``bot``. ``db_path`` defaults to the bot DB, read at call time."""

    def __init__(self, db_path: str | Path | None = None, *, bot: str = "Jev") -> None:
        self._db_path = db_path
        self.bot = bot

    def _connect(self):
        return connect(self._db_path, _SCHEMA)

    def active(self) -> dict[int, int]:
        """Every current visit: channel ID -> replies left."""
        with self._connect() as conn:
            return dict(
                conn.execute(
                    "SELECT channel_id, replies_left FROM jev_visits WHERE bot = ? AND replies_left > 0", (self.bot,)
                )
            )

    def start(self, channel_id: int, replies: int, *, by: str, now: datetime | None = None) -> None:
        """Start a visit of ``replies`` replies (spawning again resets the count)."""
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jev_visits (channel_id, bot, replies_left, spawned_by, spawned_at) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(channel_id, bot) DO UPDATE SET
                    replies_left = excluded.replies_left,
                    spawned_by = excluded.spawned_by,
                    spawned_at = excluded.spawned_at
                """,
                (channel_id, self.bot, replies, by, stamp),
            )

    def use_reply(self, channel_id: int) -> int | None:
        """It posted a reply here: how many it has left (0 ends the visit), None if it isn't visiting."""
        key = (channel_id, self.bot)
        with self._connect() as conn:
            row = conn.execute("SELECT replies_left FROM jev_visits WHERE channel_id = ? AND bot = ?", key).fetchone()
            if row is None:
                return None
            left = max(0, row[0] - 1)
            if left:
                conn.execute("UPDATE jev_visits SET replies_left = ? WHERE channel_id = ? AND bot = ?", (left, *key))
            else:
                conn.execute("DELETE FROM jev_visits WHERE channel_id = ? AND bot = ?", key)
            return left

    def end(self, channel_id: int) -> bool:
        """End the visit early; False if there was none."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM jev_visits WHERE channel_id = ? AND bot = ?", (channel_id, self.bot))
            return cursor.rowcount > 0
