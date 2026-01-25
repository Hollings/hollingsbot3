"""Background summarization worker for hierarchical message-count summarization.

This module provides a worker that generates hierarchical summaries of Discord
messages in the background. Level-1 summaries cover groups of raw messages,
and level-2 summaries cover groups of level-1 summaries.
"""

import asyncio
import logging
from typing import Union

from .summarizer import Summarizer
from .summary_cache import CachedMessage, MessageGroup, SummaryCache

_LOG = logging.getLogger(__name__)


class SummaryWorker:
    """Handles background summarization with message-count-based grouping.

    This worker generates hierarchical summaries asynchronously:
    - Level-1: Summarizes groups of raw messages
    - Level-2: Summarizes groups of level-1 summaries

    Uses per-channel locking to prevent concurrent summarization of the same channel.
    """

    def __init__(self, cache: SummaryCache, summarizer: Summarizer) -> None:
        """Initialize the summary worker.

        Args:
            cache: Summary cache for database operations.
            summarizer: Summarizer instance for generating summary text.
        """
        self.cache = cache
        self.summarizer = summarizer
        self._job_locks: dict[int, asyncio.Lock] = {}

    def _get_lock(self, channel_id: int) -> asyncio.Lock:
        """Get or create a lock for a specific channel.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            An asyncio.Lock instance for the channel.
        """
        if channel_id not in self._job_locks:
            self._job_locks[channel_id] = asyncio.Lock()
        return self._job_locks[channel_id]

    async def _generate_summary_for_group(
        self,
        channel_id: int,
        level: int,
        items: Union[list[CachedMessage], list[MessageGroup]],
    ) -> None:
        """Generate and save a summary for a group of messages or lower-level groups.

        Args:
            channel_id: The Discord channel ID.
            level: The summary level (1 for messages, 2 for level-1 groups).
            items: Either a list of CachedMessage (level 1) or MessageGroup (level 2+).
        """
        if level == 1:
            messages = items
            summary_text = await self.summarizer.summarize_messages(messages)
            start_id = messages[0].message_id
            end_id = messages[-1].message_id
            start_timestamp = messages[0].timestamp
            end_timestamp = messages[-1].timestamp
            message_count = len(messages)
        else:
            groups = items
            summary_text = await self.summarizer.summarize_groups(groups)
            start_id = groups[0].start_message_id
            end_id = groups[-1].end_message_id
            start_timestamp = groups[0].start_timestamp
            end_timestamp = groups[-1].end_timestamp
            message_count = sum(g.message_count for g in groups)

        group = MessageGroup(
            id=None,
            channel_id=channel_id,
            level=level,
            start_message_id=start_id,
            end_message_id=end_id,
            summary_text=summary_text,
            message_count=message_count,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        self.cache.save_message_group(group)

        _LOG.info(
            "Generated level-%d summary for channel %d: messages %d-%d (%d total)",
            level,
            channel_id,
            start_id,
            end_id,
            message_count,
        )

    async def _generate_level_1_summaries(self, channel_id: int) -> int:
        """Generate level-1 summaries for message groups that need them.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            Number of summaries generated.
        """
        groups_to_summarize = self.cache.get_messages_needing_level1_summary(channel_id)
        if not groups_to_summarize:
            return 0

        for messages in groups_to_summarize:
            await self._generate_summary_for_group(channel_id, level=1, items=messages)

        return len(groups_to_summarize)

    async def _generate_level_2_summaries(self, channel_id: int) -> int:
        """Generate level-2 summaries for level-1 groups that need them.

        Args:
            channel_id: The Discord channel ID.

        Returns:
            Number of summaries generated.
        """
        groups_to_summarize = self.cache.get_level1_groups_needing_level2(channel_id)
        if not groups_to_summarize:
            return 0

        for level_1_groups in groups_to_summarize:
            await self._generate_summary_for_group(
                channel_id, level=2, items=level_1_groups
            )

        return len(groups_to_summarize)

    async def _run_summarization(self, channel_id: int) -> dict[str, int]:
        """Run summarization for a channel at all levels.

        Generates level-1 summaries first, then level-2 summaries if enough
        level-1 summaries exist.

        Args:
            channel_id: The Discord channel ID to summarize.

        Returns:
            Dictionary with counts: {"level_1": int, "level_2": int}.
        """
        result = {
            "level_1": await self._generate_level_1_summaries(channel_id),
            "level_2": await self._generate_level_2_summaries(channel_id),
        }

        if result["level_1"] > 0 or result["level_2"] > 0:
            _LOG.info(
                "Summarization complete for channel %d: %d level-1, %d level-2",
                channel_id,
                result["level_1"],
                result["level_2"],
            )

        return result

    async def trigger_summarization(self, channel_id: int) -> None:
        """Trigger background summarization for a channel.

        If summarization is already running for this channel, this method
        returns immediately without doing anything.

        Args:
            channel_id: The Discord channel ID to summarize.
        """
        lock = self._get_lock(channel_id)

        # Use non-blocking acquire to skip if lock is already held
        # This is atomic and avoids the race condition of check-then-lock
        if lock.locked():
            return

        # Try to acquire lock, but another task might have acquired it
        # between our check and now, so use try_lock pattern
        try:
            async with lock:
                try:
                    await self._run_summarization(channel_id)
                except Exception:
                    _LOG.exception(
                        "Error during summarization for channel %d", channel_id
                    )
        except RuntimeError:
            # Lock was acquired by another task between our check and acquire
            pass
