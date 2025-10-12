# prompt_db.py
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

DEFAULT_DB = Path(__file__).resolve().with_name("prompts.db")
DB_PATH = Path(os.getenv("PROMPT_DB_PATH", str(DEFAULT_DB))).expanduser()


def init_db() -> None:
    """Create required tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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
        # Per-user model preference (scoped to guild)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_prefs (
                guild_id INTEGER NOT NULL,
                user_id  INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model    TEXT NOT NULL,
                PRIMARY KEY (guild_id, user_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS starboard_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                source_channel_id INTEGER NOT NULL,
                starboard_channel_id INTEGER NOT NULL,
                original_message_id INTEGER NOT NULL,
                starboard_message_id INTEGER NOT NULL,
                reactor_user_id INTEGER NOT NULL,
                reaction_emoji TEXT,
                original_author_id INTEGER NOT NULL,
                original_author_name TEXT,
                jump_url TEXT,
                content TEXT,
                attachment_urls TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


class RateLimitError(RuntimeError):
    """Raised when a prompt reservation would exceed the configured limit."""

    def __init__(
        self,
        *,
        limit: int,
        used: int,
        requested: int,
        window_start: str | None,
    ) -> None:
        self.limit = limit
        self.used = used
        self.requested = requested
        self.window_start = window_start
        super().__init__(
            f"Daily limit exceeded: requested {requested} with {used}/{limit} already used"
        )


def add_prompt(text: str, submitter: str, api: str, model: str, *, status: str = "queued") -> int:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO prompts (text, submitter, api, model, time, status) VALUES (?, ?, ?, ?, ?, ?)",
            (text, submitter, api, model, datetime.utcnow().isoformat(), status),
        )
        conn.commit()
        return cur.lastrowid


def bulk_add_prompts(
    prompts: Iterable[str],
    submitter: str,
    api: str,
    model: str,
    *,
    status: str = "queued",
    daily_limit: int | None = None,
    window_start: str | None = None,
) -> list[int]:
    """Insert multiple prompts, enforcing an optional per-day limit.

    Parameters
    ----------
    prompts:
        Iterable of prompt texts to record.
    submitter / api / model:
        Metadata stored alongside each prompt.
    status:
        Initial status to store for each prompt (default: ``queued``).
    daily_limit:
        Maximum prompts allowed in the current window. When ``None``, no limit
        is enforced.
    window_start:
        ISO8601 timestamp representing the inclusive lower bound of the current
        window (typically today's UTC midnight). Required when ``daily_limit``
        is provided.

    Returns
    -------
    list[int]
        The row IDs for the inserted prompt records, in the same order as the
        ``prompts`` iterable.

    Raises
    ------
    RateLimitError
        If inserting the supplied prompts would exceed ``daily_limit``.
    """

    prompts = [p for p in prompts]
    if not prompts:
        return []

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        if daily_limit is not None:
            if window_start is None:
                raise ValueError("window_start must be provided when enforcing a daily_limit")
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                SELECT COUNT(*)
                FROM prompts
                WHERE submitter = ?
                  AND model = ?
                  AND time >= ?
                  AND status NOT LIKE 'failed:%'
                """,
                (submitter, model, window_start),
            )
            used = int(cur.fetchone()[0])
            requested = len(prompts)
            if used + requested > daily_limit:
                raise RateLimitError(
                    limit=daily_limit,
                    used=used,
                    requested=requested,
                    window_start=window_start,
                )
        else:
            conn.execute("BEGIN")

        row_ids: list[int] = []
        for prompt_text in prompts:
            cur = conn.execute(
                "INSERT INTO prompts (text, submitter, api, model, time, status) VALUES (?, ?, ?, ?, ?, ?)",
                (prompt_text, submitter, api, model, datetime.utcnow().isoformat(), status),
            )
            row_ids.append(int(cur.lastrowid))
        conn.commit()
        return row_ids


def update_status(prompt_id: int, status: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE prompts SET status = ? WHERE id = ?", (status, prompt_id))
        conn.commit()


def log_starboard_post(
    *,
    guild_id: int,
    source_channel_id: int,
    starboard_channel_id: int,
    original_message_id: int,
    starboard_message_id: int,
    reactor_user_id: int,
    reaction_emoji: str | None,
    original_author_id: int,
    original_author_name: str,
    jump_url: str,
    content: str,
    attachment_urls: str,
) -> None:
    """Persist metadata for a starboard repost event.

    ``attachment_urls`` should be a JSON-encoded list containing both direct
    attachment URLs and any available proxied CDN URLs so downstream consumers
    can fetch media even if the original link expires.
    """

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO starboard_posts (
                guild_id,
                source_channel_id,
                starboard_channel_id,
                original_message_id,
                starboard_message_id,
                reactor_user_id,
                reaction_emoji,
                original_author_id,
                original_author_name,
                jump_url,
                content,
                attachment_urls
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                guild_id,
                source_channel_id,
                starboard_channel_id,
                original_message_id,
                starboard_message_id,
                reactor_user_id,
                reaction_emoji,
                original_author_id,
                original_author_name,
                jump_url,
                content,
                attachment_urls,
            ),
        )
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


# ------------------------- Model preferences -------------------------

def set_model_pref(guild_id: int, user_id: int, provider: str, model: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO model_prefs (guild_id, user_id, provider, model) VALUES (?, ?, ?, ?)",
            (guild_id, user_id, provider, model),
        )
        conn.commit()


def get_model_pref(guild_id: int, user_id: int) -> tuple[str, str] | None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT provider, model FROM model_prefs WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else None
