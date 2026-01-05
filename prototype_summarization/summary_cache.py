"""Summary cache database operations."""

import sqlite3
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Summary:
    """Represents a cached conversation summary."""

    channel_id: int
    start_time: int
    end_time: int
    summary_level: int
    summary_text: str
    message_count: int
    created_at: Optional[int] = None

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


class SummaryCache:
    """Database interface for storing and retrieving summaries and cached messages."""

    def __init__(self, db_path: str):
        """Initialize cache with database path."""
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    channel_id INTEGER NOT NULL,
                    start_time INTEGER NOT NULL,
                    end_time INTEGER NOT NULL,
                    summary_level INTEGER NOT NULL,
                    summary_text TEXT NOT NULL,
                    message_count INTEGER,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (channel_id, start_time, end_time, summary_level)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_channel_time
                ON conversation_summaries(channel_id, start_time, end_time)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_level
                ON conversation_summaries(channel_id, summary_level)
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

    def save_summary(self, summary: Summary) -> None:
        """Save a single summary to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversation_summaries
                (channel_id, start_time, end_time, summary_level, summary_text, message_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.channel_id,
                summary.start_time,
                summary.end_time,
                summary.summary_level,
                summary.summary_text,
                summary.message_count,
                summary.created_at,
            ))

    def save_summaries_batch(self, summaries: list[Summary]) -> None:
        """Save multiple summaries atomically."""
        with sqlite3.connect(self.db_path) as conn:
            # This will automatically be a transaction
            for summary in summaries:
                conn.execute("""
                    INSERT INTO conversation_summaries
                    (channel_id, start_time, end_time, summary_level, summary_text, message_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    summary.channel_id,
                    summary.start_time,
                    summary.end_time,
                    summary.summary_level,
                    summary.summary_text,
                    summary.message_count,
                    summary.created_at,
                ))

    def get_summary(
        self,
        channel_id: int,
        start_time: int,
        end_time: int,
        summary_level: int,
    ) -> Optional[Summary]:
        """Retrieve a specific summary."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM conversation_summaries
                WHERE channel_id = ? AND start_time = ? AND end_time = ? AND summary_level = ?
            """, (channel_id, start_time, end_time, summary_level))

            row = cursor.fetchone()
            if row is None:
                return None

            return Summary(
                channel_id=row['channel_id'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                summary_level=row['summary_level'],
                summary_text=row['summary_text'],
                message_count=row['message_count'],
                created_at=row['created_at'],
            )

    def get_summaries_for_channel(self, channel_id: int) -> list[Summary]:
        """Get all summaries for a channel, sorted by start_time."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM conversation_summaries
                WHERE channel_id = ?
                ORDER BY start_time ASC
            """, (channel_id,))

            summaries = []
            for row in cursor.fetchall():
                summaries.append(Summary(
                    channel_id=row['channel_id'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    summary_level=row['summary_level'],
                    summary_text=row['summary_text'],
                    message_count=row['message_count'],
                    created_at=row['created_at'],
                ))

            return summaries

    def get_summaries_by_level(self, channel_id: int, level: int) -> list[Summary]:
        """Get all summaries for a channel at a specific level."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM conversation_summaries
                WHERE channel_id = ? AND summary_level = ?
                ORDER BY start_time ASC
            """, (channel_id, level))

            summaries = []
            for row in cursor.fetchall():
                summaries.append(Summary(
                    channel_id=row['channel_id'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    summary_level=row['summary_level'],
                    summary_text=row['summary_text'],
                    message_count=row['message_count'],
                    created_at=row['created_at'],
                ))

            return summaries

    def get_summaries_for_context(
        self,
        channel_id: int,
        before_timestamp: Optional[int] = None,
        char_limit: int = 10000,
    ) -> list[Summary]:
        """
        Get summaries for LLM context, respecting char limit.

        Summaries are returned sorted by start_time (oldest first).
        If char limit would be exceeded, oldest summaries are dropped.

        Args:
            channel_id: Channel to get summaries for
            before_timestamp: Only include summaries ending before this time
            char_limit: Maximum total characters in summaries

        Returns:
            List of summaries that fit within char limit
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if before_timestamp is not None:
                cursor = conn.execute("""
                    SELECT * FROM conversation_summaries
                    WHERE channel_id = ? AND end_time < ?
                    ORDER BY start_time ASC
                """, (channel_id, before_timestamp))
            else:
                cursor = conn.execute("""
                    SELECT * FROM conversation_summaries
                    WHERE channel_id = ?
                    ORDER BY start_time ASC
                """, (channel_id,))

            all_summaries = []
            for row in cursor.fetchall():
                all_summaries.append(Summary(
                    channel_id=row['channel_id'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    summary_level=row['summary_level'],
                    summary_text=row['summary_text'],
                    message_count=row['message_count'],
                    created_at=row['created_at'],
                ))

        # Apply char limit by dropping oldest summaries
        result = []
        total_chars = 0

        # Work backwards (newest to oldest) to keep most recent summaries
        for summary in reversed(all_summaries):
            summary_chars = len(summary.summary_text)
            if total_chars + summary_chars <= char_limit:
                result.insert(0, summary)  # Insert at beginning to maintain order
                total_chars += summary_chars
            else:
                # Can't fit any more
                break

        return result

    def cache_message(self, message: CachedMessage) -> None:
        """Cache a Discord message."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cached_messages
                (channel_id, message_id, author_id, author_name, content, timestamp, has_images, has_attachments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.channel_id,
                message.message_id,
                message.author_id,
                message.author_name,
                message.content,
                message.timestamp,
                message.has_images,
                message.has_attachments,
            ))

    def get_cached_messages(
        self,
        channel_id: int,
        start_time: int,
        end_time: int,
    ) -> list[CachedMessage]:
        """Get cached messages in a time range, sorted by timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM cached_messages
                WHERE channel_id = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (channel_id, start_time, end_time))

            messages = []
            for row in cursor.fetchall():
                messages.append(CachedMessage(
                    channel_id=row['channel_id'],
                    message_id=row['message_id'],
                    author_id=row['author_id'],
                    author_name=row['author_name'],
                    content=row['content'],
                    timestamp=row['timestamp'],
                    has_images=bool(row['has_images']),
                    has_attachments=bool(row['has_attachments']),
                ))

            return messages

    def get_latest_cached_message_id(self, channel_id: int) -> Optional[int]:
        """Get the message_id of the most recent cached message for a channel."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT message_id FROM cached_messages
                WHERE channel_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (channel_id,))

            row = cursor.fetchone()
            return row[0] if row else None

    def get_context_windows(
        self,
        channel_id: int,
        current_time: int,
        chunk_size: int,
        chunks_per_level: int,
        chunk_size_multiplier: int,
    ) -> dict:
        """
        Get hierarchical context windows for LLM context building.

        Returns a dictionary with raw messages and summaries at each level,
        structured to provide maximum detail for recent messages and
        progressively coarser summaries for older periods.

        Args:
            channel_id: Channel to get context for
            current_time: The "now" timestamp (usually current Unix timestamp)
            chunk_size: Base chunk size in seconds (e.g., 1800 = 30 minutes)
            chunks_per_level: How many chunks per level (e.g., 4)
            chunk_size_multiplier: How much to multiply chunk size per level (e.g., 2)

        Returns:
            Dictionary with structure:
            {
                "raw_messages": [CachedMessage, ...],  # Level 0 (current + buffer)
                "level_1": [Summary, ...],
                "level_2": [Summary, ...],
                "level_3": [Summary, ...],
                ...
                "config": {
                    "total_depth_seconds": int,
                    "chunks_per_level": int,
                    "chunk_size": int,
                    "chunk_size_multiplier": int,
                }
            }
        """
        # Calculate time boundaries for each level
        boundaries = self._calculate_level_boundaries(
            current_time, chunk_size, chunks_per_level, chunk_size_multiplier
        )

        result = {
            "raw_messages": [],
            "config": {
                "total_depth_seconds": boundaries["total_depth"],
                "chunks_per_level": chunks_per_level,
                "chunk_size": chunk_size,
                "chunk_size_multiplier": chunk_size_multiplier,
            }
        }

        # Level 0: Raw messages (current window + overlap buffer)
        # Current window: current_time - chunk_size to current_time
        # Buffer window: current_time - (2 * chunk_size) to current_time - chunk_size
        level_0_start = current_time - (2 * chunk_size)
        level_0_end = current_time

        result["raw_messages"] = self.get_cached_messages(
            channel_id, level_0_start, level_0_end
        )

        # Level 1+: Summaries at each level
        for level_num, level_info in boundaries["levels"].items():
            summaries = self._get_summaries_in_range(
                channel_id,
                level_info["start"],
                level_info["end"],
                level_num
            )
            result[f"level_{level_num}"] = summaries

        return result

    def _calculate_level_boundaries(
        self,
        current_time: int,
        chunk_size: int,
        chunks_per_level: int,
        multiplier: int,
    ) -> dict:
        """
        Calculate time boundaries for each summary level.

        Returns:
            {
                "total_depth": int (total seconds of history),
                "levels": {
                    1: {"start": int, "end": int, "chunk_size": int},
                    2: {"start": int, "end": int, "chunk_size": int},
                    ...
                }
            }
        """
        boundaries = {"levels": {}}

        # Level 0: current + buffer (2 chunks)
        level_0_depth = 2 * chunk_size

        # Start from after the overlap buffer
        current_boundary = current_time - level_0_depth
        total_depth = level_0_depth

        # Calculate each summary level
        level = 1
        current_chunk_size = chunk_size

        # We'll calculate a few levels (up to level 5 should be plenty)
        for level in range(1, 6):
            level_depth = chunks_per_level * current_chunk_size
            level_start = current_boundary - level_depth
            level_end = current_boundary - 1  # Non-overlapping with previous level

            boundaries["levels"][level] = {
                "start": level_start,
                "end": level_end,
                "chunk_size": current_chunk_size,
            }

            current_boundary = level_start
            total_depth += level_depth
            current_chunk_size *= multiplier

        boundaries["total_depth"] = total_depth
        return boundaries

    def _get_summaries_in_range(
        self,
        channel_id: int,
        start_time: int,
        end_time: int,
        level: int,
    ) -> list[Summary]:
        """Get all summaries of a specific level that overlap with a time range."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Get summaries that overlap with the time range:
            # Summary overlaps if: summary.start <= range.end AND summary.end >= range.start
            cursor = conn.execute("""
                SELECT * FROM conversation_summaries
                WHERE channel_id = ?
                  AND summary_level = ?
                  AND start_time <= ?
                  AND end_time >= ?
                ORDER BY start_time ASC
            """, (channel_id, level, end_time, start_time))

            summaries = []
            for row in cursor.fetchall():
                summaries.append(Summary(
                    channel_id=row['channel_id'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    summary_level=row['summary_level'],
                    summary_text=row['summary_text'],
                    message_count=row['message_count'],
                    created_at=row['created_at'],
                ))

            return summaries
