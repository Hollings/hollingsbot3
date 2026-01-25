"""Summary cache database operations for hierarchical message-count summarization.

Tables are created by prompt_db.init_db() - this module just uses them.
"""

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from hollingsbot.prompt_db import DB_PATH

# Cutoff timestamp for summarization - messages before this won't be summarized
# This prevents huge summarization jobs when migrating old data
# Format: Unix timestamp (seconds since epoch)
# Default: 2025-11-01 00:00:00 UTC (before temp bot system was active)
SUMMARIZATION_CUTOFF_TIMESTAMP = int(os.getenv("SUMMARIZATION_CUTOFF_TIMESTAMP", "1730419200"))


@dataclass
class MessageGroup:
    """Represents a group of messages with an optional summary.

    Level 1: Covers exactly 5 raw messages
    Level 2: Covers exactly 5 level-1 groups (25 messages)
    Level 3+: Each covers 5 groups from the level below
    """

    id: int | None
    channel_id: int
    level: int
    start_message_id: int  # First message ID in this group
    end_message_id: int  # Last message ID in this group
    summary_text: str | None  # None until summarized
    message_count: int
    start_timestamp: int | None = None  # Timestamp of first message
    end_timestamp: int | None = None  # Timestamp of last message
    created_at: int | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = int(time.time())


@dataclass
class CachedMessage:
    """Represents a cached Discord message."""

    channel_id: int
    message_id: int
    author_id: int
    author_name: str
    content: str
    timestamp: int
    has_images: bool = False
    has_attachments: bool = False


# Constants for the hierarchical system
GROUP_SIZE = 5  # Messages per level-1 group, groups per higher-level group
RAW_MESSAGE_COUNT = 7  # Number of raw messages to show in context (2-message overlap with summaries)


class SummaryCache:
    """Database interface for hierarchical message-count summarization."""

    def __init__(self):
        """Initialize cache using shared DB_PATH from prompt_db."""
        self.db_path = str(DB_PATH)
        # Tables are created by prompt_db.init_db() which runs at bot startup

    # ==================== Database Connection ====================

    @contextmanager
    def _get_connection(self, row_factory: bool = False) -> Iterator[sqlite3.Connection]:
        """Get a database connection with optional row factory.

        Args:
            row_factory: If True, set conn.row_factory to sqlite3.Row for
                         dictionary-style row access.

        Yields:
            SQLite connection that auto-commits on success.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            if row_factory:
                conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ==================== Message Caching ====================

    def cache_message(self, message: CachedMessage) -> None:
        """Cache a Discord message."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cached_messages
                (channel_id, message_id, author_id, author_name, content,
                 timestamp, has_images, has_attachments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.channel_id,
                    message.message_id,
                    message.author_id,
                    message.author_name,
                    message.content,
                    message.timestamp,
                    message.has_images,
                    message.has_attachments,
                ),
            )

    def get_message_count(self, channel_id: int) -> int:
        """Get total number of cached messages for a channel."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM cached_messages WHERE channel_id = ?",
                (channel_id,),
            )
            return cursor.fetchone()[0]

    def get_all_messages_ordered(self, channel_id: int, include_old: bool = False) -> list[CachedMessage]:
        """Get all cached messages for a channel, ordered by message_id (chronological).

        Args:
            channel_id: The Discord channel ID.
            include_old: If False (default), only returns messages after
                        SUMMARIZATION_CUTOFF_TIMESTAMP to prevent summarizing old data.
        """
        with self._get_connection(row_factory=True) as conn:
            if include_old:
                cursor = conn.execute(
                    """
                    SELECT * FROM cached_messages
                    WHERE channel_id = ?
                    ORDER BY message_id ASC
                    """,
                    (channel_id,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM cached_messages
                    WHERE channel_id = ? AND timestamp >= ?
                    ORDER BY message_id ASC
                    """,
                    (channel_id, SUMMARIZATION_CUTOFF_TIMESTAMP),
                )
            return [self._row_to_cached_message(row) for row in cursor.fetchall()]

    def get_messages_by_ids(self, channel_id: int, start_id: int, end_id: int) -> list[CachedMessage]:
        """Get cached messages between two message IDs (inclusive)."""
        with self._get_connection(row_factory=True) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM cached_messages
                WHERE channel_id = ? AND message_id >= ? AND message_id <= ?
                ORDER BY message_id ASC
                """,
                (channel_id, start_id, end_id),
            )
            return [self._row_to_cached_message(row) for row in cursor.fetchall()]

    def get_recent_messages(self, channel_id: int, count: int) -> list[CachedMessage]:
        """Get the N most recent cached messages for a channel.

        Returns messages in chronological order (oldest first).
        """
        with self._get_connection(row_factory=True) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM cached_messages
                WHERE channel_id = ?
                ORDER BY message_id DESC
                LIMIT ?
                """,
                (channel_id, count),
            )
            # Reverse to get chronological order (oldest first)
            messages = [self._row_to_cached_message(row) for row in cursor.fetchall()]
            return list(reversed(messages))

    def _row_to_cached_message(self, row: sqlite3.Row) -> CachedMessage:
        """Convert a database row to a CachedMessage."""
        return CachedMessage(
            channel_id=row["channel_id"],
            message_id=row["message_id"],
            author_id=row["author_id"],
            author_name=row["author_name"],
            content=row["content"],
            timestamp=row["timestamp"],
            has_images=bool(row["has_images"]),
            has_attachments=bool(row["has_attachments"]),
        )

    # ==================== Message Groups ====================

    def save_message_group(self, group: MessageGroup) -> int:
        """Save a message group and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO message_groups
                (channel_id, level, start_message_id, end_message_id,
                 summary_text, message_count, start_timestamp, end_timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group.channel_id,
                    group.level,
                    group.start_message_id,
                    group.end_message_id,
                    group.summary_text,
                    group.message_count,
                    group.start_timestamp,
                    group.end_timestamp,
                    group.created_at,
                ),
            )
            return cursor.lastrowid

    def update_group_summary(self, group_id: int, summary_text: str) -> None:
        """Update the summary text for a message group."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE message_groups SET summary_text = ? WHERE id = ?",
                (summary_text, group_id),
            )

    def get_groups_by_level(self, channel_id: int, level: int) -> list[MessageGroup]:
        """Get all message groups at a specific level for a channel."""
        return self._query_groups(channel_id, level, summary_filter=None)

    def get_unsummarized_groups(self, channel_id: int, level: int) -> list[MessageGroup]:
        """Get groups that don't have summaries yet at a specific level."""
        return self._query_groups(channel_id, level, summary_filter="unsummarized")

    def get_summarized_groups(self, channel_id: int, level: int) -> list[MessageGroup]:
        """Get groups that have summaries at a specific level."""
        return self._query_groups(channel_id, level, summary_filter="summarized")

    def _query_groups(
        self,
        channel_id: int,
        level: int,
        summary_filter: Literal["summarized", "unsummarized"] | None,
    ) -> list[MessageGroup]:
        """Query message groups with optional summary filter.

        Args:
            channel_id: The Discord channel ID.
            level: The hierarchy level (1, 2, etc.).
            summary_filter: None for all groups, "summarized" for groups with summaries,
                           "unsummarized" for groups without summaries.

        Returns:
            List of MessageGroup objects ordered by start_message_id ascending.
        """
        # Build WHERE clause based on filter
        if summary_filter == "summarized":
            summary_clause = "AND summary_text IS NOT NULL"
        elif summary_filter == "unsummarized":
            summary_clause = "AND summary_text IS NULL"
        else:
            summary_clause = ""

        with self._get_connection(row_factory=True) as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM message_groups
                WHERE channel_id = ? AND level = ? {summary_clause}
                ORDER BY start_message_id ASC
                """,
                (channel_id, level),
            )
            return [self._row_to_message_group(row) for row in cursor.fetchall()]

    def get_latest_group(self, channel_id: int, level: int) -> MessageGroup | None:
        """Get the most recent message group at a specific level."""
        with self._get_connection(row_factory=True) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM message_groups
                WHERE channel_id = ? AND level = ?
                ORDER BY end_message_id DESC
                LIMIT 1
                """,
                (channel_id, level),
            )
            row = cursor.fetchone()
            return self._row_to_message_group(row) if row else None

    def group_exists(self, channel_id: int, level: int, start_message_id: int) -> bool:
        """Check if a group already exists."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 1 FROM message_groups
                WHERE channel_id = ? AND level = ? AND start_message_id = ?
                """,
                (channel_id, level, start_message_id),
            )
            return cursor.fetchone() is not None

    def _row_to_message_group(self, row: sqlite3.Row) -> MessageGroup:
        """Convert a database row to a MessageGroup."""
        return MessageGroup(
            id=row["id"],
            channel_id=row["channel_id"],
            level=row["level"],
            start_message_id=row["start_message_id"],
            end_message_id=row["end_message_id"],
            summary_text=row["summary_text"],
            message_count=row["message_count"],
            start_timestamp=row.get("start_timestamp", None),
            end_timestamp=row.get("end_timestamp", None),
            created_at=row["created_at"],
        )

    # ==================== Context Building ====================

    def get_hierarchical_context(self, channel_id: int, raw_message_count: int | None = None) -> dict:
        """Get hierarchical context for LLM: raw messages + summaries at each level.

        Respects clear points - if a clear point is set, only returns messages
        and summaries after that point.

        Args:
            channel_id: The channel to get context for
            raw_message_count: Number of raw messages to include (default: RAW_MESSAGE_COUNT)

        Returns:
            Dictionary with keys:
                - raw_messages: list[CachedMessage] - Recent messages (count determined by param)
                - level_1_groups: list[MessageGroup] - Up to 5 level-1 summary groups
                - level_2_groups: list[MessageGroup] - Up to 5 level-2 summary groups
                - total_message_coverage: int - Total messages represented
        """
        clear_point = self.get_clear_point(channel_id)

        # Get raw messages, filtering by clear point
        count = raw_message_count if raw_message_count is not None else RAW_MESSAGE_COUNT
        raw_messages = self.get_recent_messages(channel_id, count)
        if clear_point:
            raw_messages = [m for m in raw_messages if m.message_id > clear_point]

        # Extract level-1 groups (most recent GROUP_SIZE, allow overlap with raw)
        level_1_all = self.get_summarized_groups(channel_id, 1)
        level_1_groups = self._extract_groups(
            level_1_all[-GROUP_SIZE:],  # Take most recent, not oldest!
            exclude_end_ids=set(),  # Don't exclude - overlap is intentional
            limit=GROUP_SIZE,
            min_start_id=clear_point,
        )

        # Extract level-2 groups (most recent GROUP_SIZE)
        level_2_all = self.get_summarized_groups(channel_id, 2)
        level_2_groups = self._extract_groups(
            level_2_all[-GROUP_SIZE:],
            exclude_end_ids=set(),
            limit=GROUP_SIZE,
            min_start_id=clear_point,
        )

        level_1_coverage = sum(g.message_count for g in level_1_groups)
        level_2_coverage = sum(g.message_count for g in level_2_groups)
        total_coverage = len(raw_messages) + level_1_coverage + level_2_coverage

        return {
            "raw_messages": raw_messages,
            "level_1_groups": level_1_groups,
            "level_2_groups": level_2_groups,
            "total_message_coverage": total_coverage,
        }

    def _extract_groups(
        self,
        groups: list[MessageGroup],
        exclude_end_ids: set[int],
        limit: int,
        min_start_id: int | None = None,
    ) -> list[MessageGroup]:
        """Extract message groups that pass filtering criteria.

        Args:
            groups: List of MessageGroup objects to filter.
            exclude_end_ids: Set of message IDs - skip groups whose end_message_id
                            is in this set.
            limit: Maximum number of groups to return.
            min_start_id: If set, skip groups whose start_message_id <= this value
                         (used for clear point filtering).

        Returns:
            List of MessageGroup objects that pass filters.
        """
        result = []

        for group in groups:
            # Skip groups before clear point
            if min_start_id and group.start_message_id <= min_start_id:
                continue
            if group.end_message_id in exclude_end_ids:
                continue
            if group.summary_text:
                result.append(group)
            if len(result) >= limit:
                break

        return result

    # ==================== Clear Points ====================

    def set_clear_point(self, channel_id: int, message_id: int) -> None:
        """Set a clear point for a channel (soft-delete).

        All messages and summaries with IDs <= message_id will be ignored
        when building context for this channel.

        Args:
            channel_id: The Discord channel ID.
            message_id: The message ID to clear up to (inclusive).
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO channel_clear_points
                (channel_id, clear_after_message_id, cleared_at)
                VALUES (?, ?, ?)
                """,
                (channel_id, message_id, int(time.time())),
            )

    def get_clear_point(self, channel_id: int) -> int | None:
        """Get the clear point message ID for a channel.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            The message ID of the clear point, or None if no clear point set.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT clear_after_message_id FROM channel_clear_points WHERE channel_id = ?",
                (channel_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    # ==================== Summarization Logic Helpers ====================

    def get_messages_needing_level1_summary(self, channel_id: int) -> list[list[CachedMessage]]:
        """Get groups of 5 messages that need level-1 summaries.

        Returns message groups that:
        1. Are not in the summarization buffer (last GROUP_SIZE messages)
        2. Don't already have a level-1 summary

        Note: We use GROUP_SIZE (5) as the buffer, not RAW_MESSAGE_COUNT (7),
        to create a 2-message overlap between summaries and raw messages.

        Returns:
            List of message groups, each containing exactly GROUP_SIZE CachedMessages.
        """
        all_messages = self.get_all_messages_ordered(channel_id)

        if len(all_messages) <= GROUP_SIZE:
            return []  # Not enough messages to summarize

        # Messages available for summarization (excluding last GROUP_SIZE)
        # This creates overlap: raw shows 7, but we summarize up to last 5
        available = all_messages[:-GROUP_SIZE]

        # Build set of already-covered message ranges
        existing_groups = self.get_groups_by_level(channel_id, 1)
        covered_ranges = {(g.start_message_id, g.end_message_id) for g in existing_groups}

        return self._find_uncovered_chunks(available, covered_ranges)

    def get_level1_groups_needing_level2(self, channel_id: int) -> list[list[MessageGroup]]:
        """Get groups of 5 level-1 summaries that need level-2 summarization.

        Returns:
            List of level-1 group lists, each containing exactly GROUP_SIZE MessageGroups.
        """
        level_1_groups = self.get_summarized_groups(channel_id, 1)

        if len(level_1_groups) < GROUP_SIZE:
            return []

        # Build set of start IDs already covered by level-2 groups
        existing_level_2 = self.get_groups_by_level(channel_id, 2)
        covered_start_ids = {g.start_message_id for g in existing_level_2}

        # Group level-1 summaries into chunks of GROUP_SIZE
        groups_to_summarize = []
        for i in range(0, len(level_1_groups) - GROUP_SIZE + 1, GROUP_SIZE):
            chunk = level_1_groups[i : i + GROUP_SIZE]
            if len(chunk) == GROUP_SIZE:
                start_id = chunk[0].start_message_id
                if start_id not in covered_start_ids:
                    groups_to_summarize.append(chunk)

        return groups_to_summarize

    def _find_uncovered_chunks(
        self,
        messages: list[CachedMessage],
        covered_ranges: set[tuple[int, int]],
    ) -> list[list[CachedMessage]]:
        """Find message chunks of GROUP_SIZE that aren't already covered.

        Args:
            messages: List of messages to chunk.
            covered_ranges: Set of (start_id, end_id) tuples already summarized.

        Returns:
            List of message chunks, each of exactly GROUP_SIZE messages.
        """
        chunks = []
        for i in range(0, len(messages) - GROUP_SIZE + 1, GROUP_SIZE):
            chunk = messages[i : i + GROUP_SIZE]
            if len(chunk) == GROUP_SIZE:
                start_id = chunk[0].message_id
                end_id = chunk[-1].message_id
                if (start_id, end_id) not in covered_ranges:
                    chunks.append(chunk)
        return chunks
