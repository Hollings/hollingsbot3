# prompt_db.py
import os
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path("/data/hollingsbot.db")
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
        conn.execute("CREATE TABLE IF NOT EXISTS prs (number INTEGER PRIMARY KEY, status TEXT)")
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
                avatar_bytes BLOB,
                spawn_prompt TEXT NOT NULL,
                replies_remaining INTEGER NOT NULL,
                spawn_message_id INTEGER,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                deactivated_at TEXT,
                conversation_summary TEXT
            )
            """
        )
        # Migration: Add new columns to existing temp_bots tables
        for col, col_type, default in [
            ("spawn_message_id", "INTEGER", None),
            ("is_active", "INTEGER", "1"),
            ("created_at", "TEXT", None),
            ("deactivated_at", "TEXT", None),
            ("avatar_bytes", "BLOB", None),
            ("conversation_summary", "TEXT", None),
        ]:
            try:
                if default is not None:
                    conn.execute(f"ALTER TABLE temp_bots ADD COLUMN {col} {col_type} DEFAULT {default}")
                else:
                    conn.execute(f"ALTER TABLE temp_bots ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_message_history_author ON message_history(author_id)")
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

        # ==================== User Tokens Table ====================
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_tokens (
                user_id INTEGER PRIMARY KEY,
                tokens INTEGER DEFAULT 0,
                last_received_at TEXT
            )
        """)

        # ==================== Cost Tracking Tables ====================
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_daily_costs (
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                total_cost REAL DEFAULT 0.0,
                free_budget_used REAL DEFAULT 0.0,
                credits_used REAL DEFAULT 0.0,
                generation_count INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_credits (
                user_id INTEGER PRIMARY KEY,
                balance REAL DEFAULT 0.0,
                lifetime_spent REAL DEFAULT 0.0,
                last_updated TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_hourly_budget (
                user_id INTEGER PRIMARY KEY,
                current_budget REAL DEFAULT 0.0,
                last_tick_minute TIMESTAMP NOT NULL
            )
        """)

        # ==================== Summary Cache Tables ====================
        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                level INTEGER NOT NULL,
                start_message_id INTEGER NOT NULL,
                end_message_id INTEGER NOT NULL,
                summary_text TEXT,
                message_count INTEGER NOT NULL,
                start_timestamp INTEGER,
                end_timestamp INTEGER,
                created_at INTEGER NOT NULL,
                UNIQUE(channel_id, level, start_message_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_groups_channel_level
            ON message_groups(channel_id, level)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_groups_channel_messages
            ON message_groups(channel_id, start_message_id, end_message_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cached_messages (
                channel_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                author_id INTEGER,
                author_name TEXT,
                content TEXT,
                timestamp INTEGER NOT NULL,
                has_images BOOLEAN DEFAULT 0,
                has_attachments BOOLEAN DEFAULT 0,
                PRIMARY KEY (channel_id, message_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_channel_time
            ON cached_messages(channel_id, timestamp)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_clear_points (
                channel_id INTEGER PRIMARY KEY,
                clear_after_message_id INTEGER NOT NULL,
                cleared_at INTEGER NOT NULL
            )
        """)

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
        super().__init__(f"Daily limit exceeded: requested {requested} with {used}/{limit} already used")


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

    prompts = list(prompts)
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
#
# The temp_bots schema lives in `init_db()` above (kept colocated with all
# other table definitions). The actual CRUD/query helpers were extracted
# into `hollingsbot.temp_bot_db` to keep this file focused, and are
# re-exported below so existing imports
# (`from hollingsbot.prompt_db import create_temp_bot, ...`) keep working.
from hollingsbot import temp_bot_db as _temp_bot_db

create_temp_bot = _temp_bot_db.create_temp_bot
deactivate_temp_bot = _temp_bot_db.deactivate_temp_bot
decrement_temp_bot_replies = _temp_bot_db.decrement_temp_bot_replies
delete_temp_bot = _temp_bot_db.delete_temp_bot
get_depleted_temp_bots = _temp_bot_db.get_depleted_temp_bots
get_historical_temp_bots = _temp_bot_db.get_historical_temp_bots
get_messages_since_bot_left = _temp_bot_db.get_messages_since_bot_left
get_temp_bot_by_name = _temp_bot_db.get_temp_bot_by_name
get_temp_bot_by_webhook_id = _temp_bot_db.get_temp_bot_by_webhook_id
get_temp_bot_previous_messages = _temp_bot_db.get_temp_bot_previous_messages
get_temp_bots_for_channel = _temp_bot_db.get_temp_bots_for_channel
increment_temp_bot_replies = _temp_bot_db.increment_temp_bot_replies
search_temp_bots = _temp_bot_db.search_temp_bots


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


def get_daily_llm_call_count(exclude_haiku: bool = True) -> int:
    """Get the count of LLM API calls made today (UTC).

    Args:
        exclude_haiku: If True, exclude haiku calls (temp bots) from count.
                       The taper only applies to main bot, not temp bots.

    Returns:
        Number of successful LLM calls today.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        if exclude_haiku:
            cur = conn.execute(
                """
                SELECT COUNT(*) FROM llm_api_logs
                WHERE date(timestamp) = date('now')
                AND status = 'success'
                AND model != 'claude-haiku-4-5'
                """
            )
        else:
            cur = conn.execute(
                """
                SELECT COUNT(*) FROM llm_api_logs
                WHERE date(timestamp) = date('now')
                AND status = 'success'
                """
            )
        return cur.fetchone()[0]


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
            [*params, limit],
        )

        results = []
        for row in cur.fetchall():
            results.append(
                {
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
                }
            )

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
                [*params, request_id],
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


# ------------------------- User tokens -------------------------


def give_user_token(user_id: int) -> int:
    """Give one token to a user. Returns the user's new token balance."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO user_tokens (user_id, tokens, last_received_at)
            VALUES (?, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                tokens = tokens + 1,
                last_received_at = excluded.last_received_at
            """,
            (user_id, datetime.utcnow().isoformat()),
        )
        cur = conn.execute(
            "SELECT tokens FROM user_tokens WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        conn.commit()
        return row[0] if row else 0


def get_user_token_balance(user_id: int) -> int:
    """Get a user's token balance."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT tokens FROM user_tokens WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        return row[0] if row else 0


def deduct_user_tokens(user_id: int, amount: int) -> tuple[bool, int]:
    """Deduct tokens from a user. Returns (success, new_balance).

    Returns (False, current_balance) if user doesn't have enough tokens.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT tokens FROM user_tokens WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        current_balance = row[0] if row else 0

        if current_balance < amount:
            return (False, current_balance)

        conn.execute(
            "UPDATE user_tokens SET tokens = tokens - ? WHERE user_id = ?",
            (amount, user_id),
        )
        conn.commit()
        return (True, current_balance - amount)


def get_token_leaderboard(limit: int = 10) -> list[tuple[int, int]]:
    """Get the top token holders.

    Returns:
        List of (user_id, tokens) tuples, sorted by tokens descending.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT user_id, tokens FROM user_tokens
            WHERE tokens > 0
            ORDER BY tokens DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


def resolve_user_by_display_name(display_name: str, channel_id: int | None = None) -> int | None:
    """Try to resolve a display name to a user ID from cached messages.

    Searches for exact match (case-insensitive) in author_name.
    If channel_id provided, prefers matches from that channel.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        # Try channel-specific search first if channel_id provided
        if channel_id:
            cur = conn.execute(
                """
                SELECT author_id FROM cached_messages
                WHERE channel_id = ? AND LOWER(author_name) = LOWER(?)
                ORDER BY timestamp DESC LIMIT 1
                """,
                (channel_id, display_name),
            )
            row = cur.fetchone()
            if row:
                return row[0]

        # Fall back to global search
        cur = conn.execute(
            """
            SELECT author_id FROM cached_messages
            WHERE LOWER(author_name) = LOWER(?)
            ORDER BY timestamp DESC LIMIT 1
            """,
            (display_name,),
        )
        row = cur.fetchone()
        return row[0] if row else None
