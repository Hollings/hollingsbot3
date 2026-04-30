# temp_bot_db.py
"""SQLite helpers for the temp_bots feature.

The `temp_bots` table schema lives in `prompt_db.init_db()` (intentionally
left there because `init_db()` is the single source of truth for all bot
tables). This module owns all CRUD/query helpers that operate on that table
and on the related `cached_messages` rows used to reconstruct temp-bot
session context.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from hollingsbot.prompt_db import DB_PATH, init_db

# ---------------------------------------------------------------------------
# Row-to-dict helpers
# ---------------------------------------------------------------------------
#
# Every SELECT against `temp_bots` returns the same first 9 columns in the
# same order. The only thing that varies between queries is which trailing
# columns each call site needs (created_at, deactivated_at, is_active,
# conversation_summary). These helpers centralize the column lists and the
# dict construction so the SELECTs and the row unpacking can never drift
# apart.

_BASE_COLUMNS = (
    "id, channel_id, webhook_id, name, avatar_url, spawn_prompt, replies_remaining, spawn_message_id, avatar_bytes"
)


def _base_row_to_dict(row: tuple) -> dict:
    """Convert a row containing the 9 base columns to a dict."""
    return {
        "id": row[0],
        "channel_id": row[1],
        "webhook_id": row[2],
        "name": row[3],
        "avatar_url": row[4],
        "spawn_prompt": row[5],
        "replies_remaining": row[6],
        "spawn_message_id": row[7],
        "avatar_bytes": row[8],
    }


def _row_with_created(row: tuple) -> dict:
    """Base columns + created_at."""
    return {**_base_row_to_dict(row), "created_at": row[9]}


def _row_with_summary(row: tuple) -> dict:
    """Base columns + created_at + conversation_summary."""
    return {**_row_with_created(row), "conversation_summary": row[10]}


def _row_with_deactivated(row: tuple) -> dict:
    """Base columns + created_at + deactivated_at."""
    return {**_row_with_created(row), "deactivated_at": row[10]}


def _row_with_active_state(row: tuple) -> dict:
    """Base columns + created_at + deactivated_at + is_active."""
    return {**_row_with_deactivated(row), "is_active": bool(row[11])}


# ---------------------------------------------------------------------------
# Create / read / soft-delete
# ---------------------------------------------------------------------------


def create_temp_bot(
    channel_id: int,
    webhook_id: int,
    name: str,
    avatar_url: str | None,
    spawn_prompt: str,
    replies_remaining: int,
    spawn_message_id: int | None = None,
    avatar_bytes: bytes | None = None,
) -> int:
    """Create a new temporary bot record."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO temp_bots (
                channel_id, webhook_id, name, avatar_url, avatar_bytes,
                spawn_prompt, replies_remaining, spawn_message_id,
                is_active, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                channel_id,
                webhook_id,
                name,
                avatar_url,
                avatar_bytes,
                spawn_prompt,
                replies_remaining,
                spawn_message_id,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_temp_bots_for_channel(channel_id: int) -> list[dict]:
    """Get all active temp bots for a specific channel."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"""
            SELECT {_BASE_COLUMNS}, created_at
            FROM temp_bots
            WHERE channel_id = ? AND is_active = 1
            """,
            (channel_id,),
        )
        return [_row_with_created(row) for row in cur.fetchall()]


def get_temp_bot_by_webhook_id(webhook_id: int) -> dict | None:
    """Get a temp bot by its webhook ID."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"""
            SELECT {_BASE_COLUMNS}, created_at
            FROM temp_bots
            WHERE webhook_id = ?
            """,
            (webhook_id,),
        )
        row = cur.fetchone()
        return _row_with_created(row) if row else None


def deactivate_temp_bot(webhook_id: int) -> None:
    """Mark a temp bot as inactive (soft delete)."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE temp_bots
            SET is_active = 0, deactivated_at = ?
            WHERE webhook_id = ?
            """,
            (datetime.utcnow().isoformat(), webhook_id),
        )
        conn.commit()


def delete_temp_bot(webhook_id: int) -> None:
    """Mark a temp bot as inactive (soft delete).

    Alias for `deactivate_temp_bot` retained for backwards compatibility:
    several callers (notably `temp_bot_commands.py` and the temp_bot cog)
    import `delete_temp_bot` directly, and renaming them is out of scope
    for this DB-layer extraction.
    """
    deactivate_temp_bot(webhook_id)


def get_depleted_temp_bots() -> list[dict]:
    """Get all active temp bots that have run out of replies."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"""
            SELECT {_BASE_COLUMNS}, created_at, conversation_summary
            FROM temp_bots
            WHERE replies_remaining <= 0 AND is_active = 1
            """
        )
        return [_row_with_summary(row) for row in cur.fetchall()]


def get_historical_temp_bots(
    channel_id: int | None = None,
    limit: int = 50,
) -> list[dict]:
    """Get historical (inactive) temp bots, optionally filtered by channel.

    Returns bots ordered by most recently deactivated first.
    """
    init_db()
    select = f"SELECT {_BASE_COLUMNS}, created_at, deactivated_at FROM temp_bots"
    with sqlite3.connect(DB_PATH) as conn:
        if channel_id is not None:
            cur = conn.execute(
                f"""
                {select}
                WHERE channel_id = ? AND is_active = 0
                ORDER BY deactivated_at DESC
                LIMIT ?
                """,
                (channel_id, limit),
            )
        else:
            cur = conn.execute(
                f"""
                {select}
                WHERE is_active = 0
                ORDER BY deactivated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [_row_with_deactivated(row) for row in cur.fetchall()]


def get_temp_bot_by_name(name: str, channel_id: int | None = None) -> dict | None:
    """Find a temp bot by name (case-insensitive), preferring the most recent.

    If channel_id is provided, searches that channel first.
    Returns the most recently created/deactivated bot with that name.
    """
    init_db()
    select = f"SELECT {_BASE_COLUMNS}, created_at, deactivated_at, is_active FROM temp_bots"
    with sqlite3.connect(DB_PATH) as conn:
        # Try channel-specific search first
        if channel_id is not None:
            cur = conn.execute(
                f"""
                {select}
                WHERE channel_id = ? AND LOWER(name) = LOWER(?)
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (channel_id, name),
            )
            row = cur.fetchone()
            if row:
                return _row_with_active_state(row)

        # Global search
        cur = conn.execute(
            f"""
            {select}
            WHERE LOWER(name) = LOWER(?)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (name,),
        )
        row = cur.fetchone()
        return _row_with_active_state(row) if row else None


def search_temp_bots(query: str, limit: int = 10) -> list[dict]:
    """Search temp bots by name or spawn_prompt (case-insensitive).

    Returns matching bots ordered by most recent first.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"""
            SELECT {_BASE_COLUMNS}, created_at, deactivated_at, is_active
            FROM temp_bots
            WHERE LOWER(name) LIKE LOWER(?) OR LOWER(spawn_prompt) LIKE LOWER(?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", limit),
        )
        return [_row_with_active_state(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Cached-message lookups used by the temp_bot cog
# ---------------------------------------------------------------------------


def get_temp_bot_previous_messages(channel_id: int, bot_name: str, limit: int = 5) -> list[dict]:
    """Get a temp bot's most recent messages from their last session.

    Args:
        channel_id: The channel to search in
        bot_name: The bot's display name
        limit: Max number of messages to return

    Returns:
        List of message dicts with author_name, content, timestamp
        Ordered chronologically (oldest first)
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT author_name, content, timestamp
            FROM cached_messages
            WHERE channel_id = ? AND author_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (channel_id, bot_name, limit),
        )
        messages = [
            {
                "author_name": row[0],
                "content": row[1],
                "timestamp": row[2],
            }
            for row in cur.fetchall()
        ]
        # Return in chronological order (oldest first)
        return list(reversed(messages))


def get_messages_since_bot_left(channel_id: int, bot_name: str) -> int:
    """Count how many messages have been sent since a bot was last active.

    Args:
        channel_id: The channel to check
        bot_name: The bot's display name

    Returns:
        Number of messages since the bot's last message, or 0 if not found
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        # Find the bot's last message timestamp
        cur = conn.execute(
            """
            SELECT MAX(timestamp)
            FROM cached_messages
            WHERE channel_id = ? AND author_name = ?
            """,
            (channel_id, bot_name),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return 0

        last_timestamp = row[0]

        # Count messages after that timestamp
        cur = conn.execute(
            """
            SELECT COUNT(*)
            FROM cached_messages
            WHERE channel_id = ? AND timestamp > ?
            """,
            (channel_id, last_timestamp),
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Atomic reply-counter mutations
# ---------------------------------------------------------------------------


def _adjust_replies_remaining(webhook_id: int, delta: int) -> int:
    """Atomically add `delta` to `replies_remaining` for an active temp bot.

    Uses BEGIN IMMEDIATE for a write lock plus RETURNING for atomicity, so
    concurrent decrement/increment calls cannot lose updates.

    Returns:
        The new `replies_remaining` value, or -1 if no active row matched.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                """
                UPDATE temp_bots
                SET replies_remaining = replies_remaining + ?
                WHERE webhook_id = ? AND is_active = 1
                RETURNING replies_remaining
                """,
                (delta, webhook_id),
            )
            row = cur.fetchone()
            conn.commit()
            return -1 if row is None else int(row[0])
        except Exception:
            conn.rollback()
            raise


def decrement_temp_bot_replies(webhook_id: int) -> tuple[int, bool]:
    """Atomically decrement replies_remaining and return (new_value, should_cleanup).

    Returns:
        tuple: (remaining_replies, should_cleanup)
        - remaining_replies: The new value after decrement (-1 if bot not found)
        - should_cleanup: True if bot should be cleaned up (remaining <= 0)
    """
    remaining = _adjust_replies_remaining(webhook_id, -1)
    if remaining == -1:
        return (-1, False)
    return (remaining, remaining <= 0)


def increment_temp_bot_replies(webhook_id: int) -> int:
    """Atomically increment replies_remaining (refund a cancelled reply).

    Returns:
        The new value after increment, or -1 if bot not found.
    """
    return _adjust_replies_remaining(webhook_id, 1)
