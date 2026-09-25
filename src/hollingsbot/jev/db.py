"""SQLite access for Jev's tables, in the bot's shared DB (``prompt_db.DB_PATH``)."""

from __future__ import annotations

import contextlib
import sqlite3
from typing import TYPE_CHECKING

from hollingsbot import prompt_db

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

_ready: set[tuple[str, int]] = set()  # (db path, schema) pairs already created this process


@contextlib.contextmanager
def connect(db_path: str | Path | None, schema: Sequence[str]) -> Iterator[sqlite3.Connection]:
    """Open the DB, create ``schema`` (CREATE ... IF NOT EXISTS statements) once, commit on success.

    ``db_path`` None means the bot DB, read at call time so tests can point it elsewhere.
    """
    path = str(db_path or prompt_db.DB_PATH)
    with contextlib.closing(sqlite3.connect(path, timeout=30.0)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        key = (path, hash(tuple(schema)))
        if key not in _ready:
            for statement in schema:
                conn.execute(statement)
            _ready.add(key)
        yield conn
        conn.commit()
