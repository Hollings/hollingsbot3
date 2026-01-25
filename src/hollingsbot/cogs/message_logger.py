"""Standalone message logger cog - logs all messages from whitelisted guilds.

This cog is designed to be self-contained and portable between Discord bots.
It maintains its own database schema and requires no external dependencies
beyond discord.py.

Configuration via environment variables:
    MESSAGE_LOGGER_GUILDS: Comma-separated guild IDs to log (required)
    MESSAGE_LOGGER_DB_PATH: Path to SQLite database (default: /data/hollingsbot.db)

Schema (message_history table):
    message_id INTEGER PRIMARY KEY
    channel_id INTEGER NOT NULL
    guild_id INTEGER
    timestamp TEXT NOT NULL (ISO 8601)
    author_id INTEGER NOT NULL
    author_nickname TEXT
    is_bot INTEGER DEFAULT 0
    is_webhook INTEGER DEFAULT 0
    content TEXT
    attachment_urls TEXT (JSON array)
    reply_to_id INTEGER
    reactions TEXT (JSON array)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

import discord
from discord.ext import commands

_LOG = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("/data/hollingsbot.db")


class MessageLoggerCog(commands.Cog):
    """Logs all messages from whitelisted guilds to SQLite."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.db_path = Path(os.getenv("MESSAGE_LOGGER_DB_PATH", str(DEFAULT_DB_PATH)))
        self.allowed_guilds = self._parse_guild_ids()

        if not self.allowed_guilds:
            _LOG.warning("MESSAGE_LOGGER_GUILDS not set - message logging disabled")
        else:
            self._init_db()
            _LOG.info(
                "MessageLoggerCog initialized with %d guild(s): %s",
                len(self.allowed_guilds),
                ", ".join(str(g) for g in self.allowed_guilds)
            )

    def _parse_guild_ids(self) -> set[int]:
        """Parse MESSAGE_LOGGER_GUILDS env var into a set of guild IDs."""
        raw = os.getenv("MESSAGE_LOGGER_GUILDS", "")
        if not raw.strip():
            return set()

        guild_ids = set()
        for part in raw.split(","):
            part = part.strip()
            if part:
                try:
                    guild_ids.add(int(part))
                except ValueError:
                    _LOG.warning("Invalid guild ID in MESSAGE_LOGGER_GUILDS: %s", part)
        return guild_ids

    def _init_db(self) -> None:
        """Initialize the SQLite database schema.

        Handles migration from older schemas by adding missing columns.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Create table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_history (
                    message_id INTEGER PRIMARY KEY,
                    channel_id INTEGER NOT NULL,
                    guild_id INTEGER,
                    timestamp TEXT NOT NULL,
                    author_id INTEGER,
                    author_nickname TEXT,
                    is_bot INTEGER DEFAULT 0,
                    is_webhook INTEGER DEFAULT 0,
                    content TEXT,
                    attachment_urls TEXT,
                    reply_to_id INTEGER,
                    reactions TEXT
                )
            """)

            # Migration: Add columns that might be missing in older schemas
            # These are added with defaults that work for existing data
            migration_columns = [
                ("guild_id", "INTEGER"),
                ("author_id", "INTEGER"),
                ("is_bot", "INTEGER DEFAULT 0"),
                ("is_webhook", "INTEGER DEFAULT 0"),
                ("reply_to_id", "INTEGER"),
            ]

            for col_name, col_type in migration_columns:
                try:
                    conn.execute(f"ALTER TABLE message_history ADD COLUMN {col_name} {col_type}")
                    _LOG.info("Added column %s to message_history", col_name)
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Create indexes (IF NOT EXISTS handles existing indexes)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_history_channel_time
                ON message_history(channel_id, timestamp)
            """)
            # Only create author index if author_id exists
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_message_history_author
                    ON message_history(author_id)
                """)
            except sqlite3.OperationalError:
                pass  # Index creation failed, column might not exist yet
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_message_history_guild
                    ON message_history(guild_id)
                """)
            except sqlite3.OperationalError:
                pass  # Index creation failed
            conn.commit()

        _LOG.info("Message logger database initialized at %s", self.db_path)

    def _log_message(self, message: discord.Message) -> None:
        """Log a message to the database."""
        try:
            # Build attachment URLs list
            attachment_urls = [att.url for att in message.attachments] if message.attachments else []

            # Get reply target if this is a reply
            reply_to_id = None
            if message.reference and message.reference.message_id:
                reply_to_id = message.reference.message_id

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO message_history (
                        message_id, channel_id, guild_id, timestamp,
                        author_id, author_nickname, is_bot, is_webhook,
                        content, attachment_urls, reply_to_id, reactions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    message.channel.id,
                    message.guild.id if message.guild else None,
                    message.created_at.isoformat(),
                    message.author.id,
                    message.author.display_name,
                    1 if message.author.bot else 0,
                    1 if message.webhook_id else 0,
                    message.content,
                    json.dumps(attachment_urls) if attachment_urls else None,
                    reply_to_id,
                    None,  # reactions populated on edit/reaction events
                ))
                conn.commit()
        except Exception as e:
            _LOG.exception("Failed to log message %s: %s", message.id, e)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Log incoming messages from whitelisted guilds."""
        # Skip if no guilds configured
        if not self.allowed_guilds:
            return

        # Skip DMs
        if not message.guild:
            return

        # Check guild whitelist
        if message.guild.id not in self.allowed_guilds:
            return

        # Log the message
        self._log_message(message)

    @commands.Cog.listener()
    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        """Update message content when edited."""
        if not self.allowed_guilds:
            return

        # Check if we have guild_id (may not be present for DMs)
        if not payload.guild_id or payload.guild_id not in self.allowed_guilds:
            return

        # Only update if we have new content
        data = payload.data
        if "content" not in data:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE message_history
                    SET content = ?
                    WHERE message_id = ?
                """, (data["content"], payload.message_id))
                conn.commit()
        except Exception as e:
            _LOG.exception("Failed to update edited message %s: %s", payload.message_id, e)

    @commands.Cog.listener()
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        """Handle message deletion - currently does nothing (keeps record)."""
        # We keep deleted messages in the archive for historical record.
        # If you want to mark them as deleted, add a 'deleted_at' column and update here.
        pass


async def setup(bot: commands.Bot) -> None:
    """Discord.py extension setup function."""
    await bot.add_cog(MessageLoggerCog(bot))
