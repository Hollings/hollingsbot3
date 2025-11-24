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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS temp_bots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                webhook_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                avatar_url TEXT,
                spawn_prompt TEXT NOT NULL,
                replies_remaining INTEGER NOT NULL,
                spawn_message_id INTEGER
            )
            """
        )
        # Add spawn_message_id column to existing tables
        try:
            conn.execute("ALTER TABLE temp_bots ADD COLUMN spawn_message_id INTEGER")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                message_id INTEGER,
                channel_id INTEGER,
                bot_name TEXT,
                conversation_json TEXT NOT NULL,
                response_text TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS message_history (
                message_id INTEGER PRIMARY KEY,
                channel_id INTEGER NOT NULL,
                guild_id INTEGER,
                timestamp TEXT NOT NULL,
                author_id INTEGER NOT NULL,
                author_nickname TEXT,
                is_bot INTEGER DEFAULT 0,
                is_webhook INTEGER DEFAULT 0,
                content TEXT,
                attachment_urls TEXT,
                reply_to_id INTEGER,
                reactions TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_history_channel_time ON message_history(channel_id, timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_message_history_author ON message_history(author_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                request_description TEXT NOT NULL,
                status TEXT NOT NULL,
                questions_message_id INTEGER,
                conversation_log TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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


# ------------------------- Temp bot management -------------------------

def create_temp_bot(
    channel_id: int,
    webhook_id: int,
    name: str,
    avatar_url: str | None,
    spawn_prompt: str,
    replies_remaining: int,
    spawn_message_id: int | None = None,
) -> int:
    """Create a new temporary bot record."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO temp_bots (channel_id, webhook_id, name, avatar_url, spawn_prompt, replies_remaining, spawn_message_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (channel_id, webhook_id, name, avatar_url, spawn_prompt, replies_remaining, spawn_message_id),
        )
        conn.commit()
        return cur.lastrowid


def get_temp_bots_for_channel(channel_id: int) -> list[dict]:
    """Get all active temp bots for a specific channel."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT id, channel_id, webhook_id, name, avatar_url, spawn_prompt, replies_remaining, spawn_message_id
            FROM temp_bots
            WHERE channel_id = ?
            """,
            (channel_id,),
        )
        return [
            {
                "id": row[0],
                "channel_id": row[1],
                "webhook_id": row[2],
                "name": row[3],
                "avatar_url": row[4],
                "spawn_prompt": row[5],
                "replies_remaining": row[6],
                "spawn_message_id": row[7],
            }
            for row in cur.fetchall()
        ]


def get_temp_bot_by_webhook_id(webhook_id: int) -> dict | None:
    """Get a temp bot by its webhook ID."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT id, channel_id, webhook_id, name, avatar_url, spawn_prompt, replies_remaining, spawn_message_id
            FROM temp_bots
            WHERE webhook_id = ?
            """,
            (webhook_id,),
        )
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "channel_id": row[1],
                "webhook_id": row[2],
                "name": row[3],
                "avatar_url": row[4],
                "spawn_prompt": row[5],
                "replies_remaining": row[6],
                "spawn_message_id": row[7],
            }
        return None


def delete_temp_bot(webhook_id: int) -> None:
    """Delete a temp bot record."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM temp_bots WHERE webhook_id = ?", (webhook_id,))
        conn.commit()


def get_depleted_temp_bots() -> list[dict]:
    """Get all temp bots that have run out of replies."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT id, channel_id, webhook_id, name, avatar_url, spawn_prompt, replies_remaining, spawn_message_id
            FROM temp_bots
            WHERE replies_remaining <= 0
            """
        )
        return [
            {
                "id": row[0],
                "channel_id": row[1],
                "webhook_id": row[2],
                "name": row[3],
                "avatar_url": row[4],
                "spawn_prompt": row[5],
                "replies_remaining": row[6],
                "spawn_message_id": row[7],
            }
            for row in cur.fetchall()
        ]


def decrement_temp_bot_replies(webhook_id: int) -> int:
    """Decrement replies_remaining for a temp bot and return the new value."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE temp_bots
            SET replies_remaining = replies_remaining - 1
            WHERE webhook_id = ?
            """,
            (webhook_id,),
        )
        cur = conn.execute(
            "SELECT replies_remaining FROM temp_bots WHERE webhook_id = ?",
            (webhook_id,),
        )
        row = cur.fetchone()
        conn.commit()
        return int(row[0]) if row else 0


# ------------------------- LLM API logging -------------------------

def log_llm_api_call(
    provider: str,
    model: str,
    conversation_json: str,
    response_text: str,
    status: str = "success",
    error_message: str | None = None,
    message_id: int | None = None,
    channel_id: int | None = None,
    bot_name: str | None = None,
) -> int:
    """Log an LLM API call for debugging purposes."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO llm_api_logs (
                timestamp, provider, model, message_id, channel_id, bot_name,
                conversation_json, response_text, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                provider,
                model,
                message_id,
                channel_id,
                bot_name,
                conversation_json,
                response_text,
                status,
                error_message,
            ),
        )
        conn.commit()
        return cur.lastrowid


# ------------------------- Message history -------------------------

def add_message_to_history(
    message_id: int,
    channel_id: int,
    timestamp: str,
    author_id: int,
    content: str | None = None,
    *,
    guild_id: int | None = None,
    author_nickname: str | None = None,
    is_bot: bool = False,
    is_webhook: bool = False,
    attachment_urls: list[str] | None = None,
    reply_to_id: int | None = None,
    reactions: list[dict] | None = None,
) -> None:
    """Add or update a message in the history."""
    import json

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO message_history (
                message_id, channel_id, guild_id, timestamp, author_id, author_nickname,
                is_bot, is_webhook, content, attachment_urls, reply_to_id, reactions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                channel_id,
                guild_id,
                timestamp,
                author_id,
                author_nickname,
                1 if is_bot else 0,
                1 if is_webhook else 0,
                content,
                json.dumps(attachment_urls) if attachment_urls else None,
                reply_to_id,
                json.dumps(reactions) if reactions else None,
            ),
        )
        conn.commit()


def search_message_history(
    channel_id: int,
    query: str | None = None,
    *,
    author_id: int | None = None,
    after_timestamp: str | None = None,
    before_timestamp: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Search message history with optional filters.

    Returns messages matching the search criteria, ordered by timestamp (newest first).
    """
    import json

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conditions = ["channel_id = ?"]
        params: list = [channel_id]

        if query:
            conditions.append("content LIKE ?")
            params.append(f"%{query}%")

        if author_id:
            conditions.append("author_id = ?")
            params.append(author_id)

        if after_timestamp:
            conditions.append("timestamp > ?")
            params.append(after_timestamp)

        if before_timestamp:
            conditions.append("timestamp < ?")
            params.append(before_timestamp)

        where_clause = " AND ".join(conditions)

        cur = conn.execute(
            f"""
            SELECT message_id, channel_id, guild_id, timestamp, author_id, author_nickname,
                   is_bot, is_webhook, content, attachment_urls, reply_to_id, reactions
            FROM message_history
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params + [limit],
        )

        results = []
        for row in cur.fetchall():
            results.append({
                "message_id": row[0],
                "channel_id": row[1],
                "guild_id": row[2],
                "timestamp": row[3],
                "author_id": row[4],
                "author_nickname": row[5],
                "is_bot": bool(row[6]),
                "is_webhook": bool(row[7]),
                "content": row[8],
                "attachment_urls": json.loads(row[9]) if row[9] else [],
                "reply_to_id": row[10],
                "reactions": json.loads(row[11]) if row[11] else [],
            })

        return results


def get_latest_message_timestamp(channel_id: int) -> str | None:
    """Get the timestamp of the most recent message in a channel."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT MAX(timestamp) FROM message_history WHERE channel_id = ?",
            (channel_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def bulk_add_messages(messages: list[dict]) -> int:
    """Bulk insert messages (for migration script).

    Returns the number of messages inserted.
    """
    import json

    if not messages:
        return 0

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("BEGIN")
        for msg in messages:
            conn.execute(
                """
                INSERT OR REPLACE INTO message_history (
                    message_id, channel_id, guild_id, timestamp, author_id, author_nickname,
                    is_bot, is_webhook, content, attachment_urls, reply_to_id, reactions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg["message_id"],
                    msg["channel_id"],
                    msg.get("guild_id"),
                    msg["timestamp"],
                    msg["author_id"],
                    msg.get("author_nickname"),
                    1 if msg.get("is_bot", False) else 0,
                    1 if msg.get("is_webhook", False) else 0,
                    msg.get("content"),
                    json.dumps(msg.get("attachment_urls", [])),
                    msg.get("reply_to_id"),
                    json.dumps(msg.get("reactions", [])),
                ),
            )
        conn.commit()
        return len(messages)


# ------------------------- Feature request automation -------------------------

def create_feature_request(
    channel_id: int,
    message_id: int,
    author_id: int,
    request_description: str,
    status: str = "pending_questions",
) -> int:
    """Create a new feature request record."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO feature_requests (
                channel_id, message_id, author_id, request_description, status
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (channel_id, message_id, author_id, request_description, status),
        )
        conn.commit()
        return cur.lastrowid


def update_feature_request(
    request_id: int,
    *,
    status: str | None = None,
    questions_message_id: int | None = None,
    conversation_log: str | None = None,
) -> None:
    """Update a feature request record."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if questions_message_id is not None:
            updates.append("questions_message_id = ?")
            params.append(questions_message_id)

        if conversation_log is not None:
            updates.append("conversation_log = ?")
            params.append(conversation_log)

        if updates:
            updates.append("updated_at = ?")
            params.append(datetime.utcnow().isoformat())

            conn.execute(
                f"UPDATE feature_requests SET {', '.join(updates)} WHERE id = ?",
                params + [request_id],
            )
            conn.commit()


def get_feature_request_by_id(request_id: int) -> dict | None:
    """Get a feature request by its ID."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT id, channel_id, message_id, author_id, request_description,
                   status, questions_message_id, conversation_log, created_at, updated_at
            FROM feature_requests
            WHERE id = ?
            """,
            (request_id,),
        )
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "channel_id": row[1],
                "message_id": row[2],
                "author_id": row[3],
                "request_description": row[4],
                "status": row[5],
                "questions_message_id": row[6],
                "conversation_log": row[7],
                "created_at": row[8],
                "updated_at": row[9],
            }
        return None


def get_feature_request_by_questions_message_id(questions_message_id: int) -> dict | None:
    """Get a feature request by its questions message ID."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT id, channel_id, message_id, author_id, request_description,
                   status, questions_message_id, conversation_log, created_at, updated_at
            FROM feature_requests
            WHERE questions_message_id = ?
            """,
            (questions_message_id,),
        )
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "channel_id": row[1],
                "message_id": row[2],
                "author_id": row[3],
                "request_description": row[4],
                "status": row[5],
                "questions_message_id": row[6],
                "conversation_log": row[7],
                "created_at": row[8],
                "updated_at": row[9],
            }
        return None
