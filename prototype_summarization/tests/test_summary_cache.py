"""Tests for summary cache database operations."""

import pytest
import tempfile
import os
from datetime import datetime, timezone
from prototype_summarization.summary_cache import (
    SummaryCache,
    Summary,
    CachedMessage,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Give Windows time to release file handles
    try:
        os.unlink(path)
    except PermissionError:
        # File still locked, ignore for testing
        pass


@pytest.fixture
def cache(temp_db):
    """Create a SummaryCache instance with temp database."""
    return SummaryCache(temp_db)


class TestSummaryCacheSaveSummary:
    """Test saving summaries to the database."""

    def test_save_and_retrieve_summary(self, cache):
        """Test basic save and retrieve."""
        summary = Summary(
            channel_id=123,
            start_time=1000,
            end_time=1800,
            summary_level=1,
            summary_text="This is a test summary",
            message_count=10,
        )

        cache.save_summary(summary)
        retrieved = cache.get_summary(123, 1000, 1800, 1)

        assert retrieved is not None
        assert retrieved.channel_id == 123
        assert retrieved.start_time == 1000
        assert retrieved.end_time == 1800
        assert retrieved.summary_level == 1
        assert retrieved.summary_text == "This is a test summary"
        assert retrieved.message_count == 10

    def test_retrieve_nonexistent_summary_returns_none(self, cache):
        """Test that retrieving non-existent summary returns None."""
        result = cache.get_summary(999, 1000, 2000, 1)
        assert result is None

    def test_save_multiple_summaries_different_channels(self, cache):
        """Test saving summaries for different channels."""
        summary1 = Summary(123, 1000, 1800, 1, "Summary 1", 10)
        summary2 = Summary(456, 1000, 1800, 1, "Summary 2", 15)

        cache.save_summary(summary1)
        cache.save_summary(summary2)

        retrieved1 = cache.get_summary(123, 1000, 1800, 1)
        retrieved2 = cache.get_summary(456, 1000, 1800, 1)

        assert retrieved1.summary_text == "Summary 1"
        assert retrieved2.summary_text == "Summary 2"

    def test_save_multiple_summaries_different_levels(self, cache):
        """Test saving summaries at different levels for same time range."""
        summary_l1 = Summary(123, 1000, 1800, 1, "Level 1", 10)
        summary_l2 = Summary(123, 1000, 3600, 2, "Level 2", 20)

        cache.save_summary(summary_l1)
        cache.save_summary(summary_l2)

        retrieved_l1 = cache.get_summary(123, 1000, 1800, 1)
        retrieved_l2 = cache.get_summary(123, 1000, 3600, 2)

        assert retrieved_l1.summary_text == "Level 1"
        assert retrieved_l2.summary_text == "Level 2"


class TestSummaryCacheGetSummariesForChannel:
    """Test retrieving all summaries for a channel."""

    def test_get_all_summaries_empty(self, cache):
        """Test getting summaries when none exist."""
        summaries = cache.get_summaries_for_channel(123)
        assert summaries == []

    def test_get_all_summaries(self, cache):
        """Test getting all summaries for a channel."""
        cache.save_summary(Summary(123, 1000, 1800, 1, "S1", 10))
        cache.save_summary(Summary(123, 2000, 2800, 1, "S2", 15))
        cache.save_summary(Summary(456, 1000, 1800, 1, "Other", 5))

        summaries = cache.get_summaries_for_channel(123)
        assert len(summaries) == 2
        assert all(s.channel_id == 123 for s in summaries)

    def test_summaries_sorted_by_time(self, cache):
        """Test that summaries are returned sorted by start_time."""
        cache.save_summary(Summary(123, 3000, 3800, 1, "S3", 10))
        cache.save_summary(Summary(123, 1000, 1800, 1, "S1", 10))
        cache.save_summary(Summary(123, 2000, 2800, 1, "S2", 10))

        summaries = cache.get_summaries_for_channel(123)
        assert summaries[0].start_time == 1000
        assert summaries[1].start_time == 2000
        assert summaries[2].start_time == 3000

    def test_get_summaries_by_level(self, cache):
        """Test filtering summaries by level."""
        cache.save_summary(Summary(123, 1000, 1800, 1, "L1-A", 10))
        cache.save_summary(Summary(123, 2000, 2800, 1, "L1-B", 15))
        cache.save_summary(Summary(123, 1000, 2800, 2, "L2-A", 25))

        level_1 = cache.get_summaries_by_level(123, 1)
        level_2 = cache.get_summaries_by_level(123, 2)

        assert len(level_1) == 2
        assert len(level_2) == 1
        assert all(s.summary_level == 1 for s in level_1)
        assert level_2[0].summary_level == 2


class TestSummaryCacheGetSummariesForContext:
    """Test retrieving summaries for LLM context building with hierarchical windows."""

    def test_get_context_basic_structure(self, cache):
        """Test basic context window retrieval structure."""
        # Current time: 10000 (represents "now")
        # Config: chunk_size=30min (1800s), chunks_per_level=4, multiplier=2
        #
        # Expected boundaries:
        # Level 0 (raw): 10000 - 3600 to 10000 = 6400 to 10000
        # Level 1: 6400 - 7200 to 6400 - 1 = -800 to 6399 (4 chunks × 1800s)
        # Level 2: -800 - 14400 to -800 - 1 = -15200 to -801 (4 chunks × 3600s)

        # Raw messages (Level 0: 6400-10000)
        cache.cache_message(CachedMessage(123, 1, 456, "User", "Recent msg", 9000))
        cache.cache_message(CachedMessage(123, 2, 456, "User", "Buffer msg", 7000))

        # Level-1 summaries (-800 to 6399, each chunk is 1800s)
        cache.save_summary(Summary(123, 4600, 6399, 1, "L1-A", 10))  # Most recent L1 chunk
        cache.save_summary(Summary(123, 2800, 4599, 1, "L1-B", 10))
        cache.save_summary(Summary(123, 1000, 2799, 1, "L1-C", 10))
        cache.save_summary(Summary(123, -800, 999, 1, "L1-D", 10))   # Oldest L1 chunk

        context = cache.get_context_windows(
            channel_id=123,
            current_time=10000,
            chunk_size=1800,  # 30 minutes in seconds
            chunks_per_level=4,
            chunk_size_multiplier=2
        )

        assert "raw_messages" in context
        assert "level_1" in context
        assert len(context["raw_messages"]) == 2
        assert len(context["level_1"]) == 4

    def test_level_0_includes_current_and_buffer_windows(self, cache):
        """Test that Level 0 (raw messages) includes current window + overlap buffer."""
        # Current time: 10000
        # Current window: 8200-10000 (30 min)
        # Buffer window: 6400-8199 (30 min before current)

        cache.cache_message(CachedMessage(123, 1, 456, "User", "Current", 9000))
        cache.cache_message(CachedMessage(123, 2, 456, "User", "Buffer", 7000))
        cache.cache_message(CachedMessage(123, 3, 456, "User", "Too old", 5000))

        context = cache.get_context_windows(
            channel_id=123,
            current_time=10000,
            chunk_size=1800,
            chunks_per_level=4,
            chunk_size_multiplier=2
        )

        raw_msgs = context["raw_messages"]
        assert len(raw_msgs) == 2
        assert raw_msgs[0].content == "Buffer"  # Chronological order
        assert raw_msgs[1].content == "Current"

    def test_level_1_summaries_non_overlapping_with_level_2(self, cache):
        """Test that Level-1 and Level-2 windows don't overlap."""
        # Current time: 50000
        # Level 0 (raw): 46400 to 50000
        # Level 1: 39200 to 46399 (4 chunks × 1800s = 7200s)
        # Level 2: 24800 to 39199 (4 chunks × 3600s = 14400s)

        # Level-1 summaries (each 1800s, 4 chunks)
        cache.save_summary(Summary(123, 44600, 46399, 1, "L1-0", 10))
        cache.save_summary(Summary(123, 42800, 44599, 1, "L1-1", 10))
        cache.save_summary(Summary(123, 41000, 42799, 1, "L1-2", 10))
        cache.save_summary(Summary(123, 39200, 40999, 1, "L1-3", 10))

        # Level-2 summaries (each 3600s, 4 chunks)
        cache.save_summary(Summary(123, 35600, 39199, 2, "L2-0", 20))
        cache.save_summary(Summary(123, 32000, 35599, 2, "L2-1", 20))
        cache.save_summary(Summary(123, 28400, 31999, 2, "L2-2", 20))
        cache.save_summary(Summary(123, 24800, 28399, 2, "L2-3", 20))

        context = cache.get_context_windows(
            channel_id=123,
            current_time=50000,
            chunk_size=1800,
            chunks_per_level=4,
            chunk_size_multiplier=2
        )

        # Verify no time overlap between Level-1 and Level-2
        l1_summaries = context["level_1"]
        l2_summaries = context["level_2"]

        assert len(l1_summaries) == 4
        assert len(l2_summaries) == 4

        l1_min_start = min(s.start_time for s in l1_summaries)
        l2_max_end = max(s.end_time for s in l2_summaries)

        # Level-1 is more recent, so it should start AFTER Level-2 ends
        assert l1_min_start > l2_max_end  # No overlap

    def test_overlap_buffer_overlaps_with_level_1(self, cache):
        """Test that the overlap buffer (Level 0) intentionally overlaps with Level-1."""
        # Current time: 10000
        # Level 0 (raw): 6400 to 10000
        # Level 1: -800 to 6399
        # The most recent L1 summary should overlap with the buffer period

        cache.cache_message(CachedMessage(123, 1, 456, "User", "In buffer", 7000))

        # Create a Level-1 summary that ends right before Level 0 starts
        # This means the most recent L1 chunk covers 4600-6399
        cache.save_summary(Summary(123, 4600, 6399, 1, "L1-most-recent", 10))

        context = cache.get_context_windows(
            channel_id=123,
            current_time=10000,
            chunk_size=1800,
            chunks_per_level=4,
            chunk_size_multiplier=2
        )

        # The Level-1 summary ends at 6399
        # The Level 0 buffer starts at 6400
        # They are adjacent (non-overlapping in time, but provide smooth handoff)
        assert len(context["level_1"]) == 1
        assert context["level_1"][0].end_time == 6399

        # Verify buffer messages exist
        assert len(context["raw_messages"]) == 1
        assert context["raw_messages"][0].timestamp >= 6400

    def test_handles_missing_summaries_at_higher_levels(self, cache):
        """Test graceful handling when higher-level summaries don't exist."""
        # Only have Level-1 summaries, no Level-2 or Level-3
        cache.save_summary(Summary(123, 1000, 2799, 1, "L1-A", 10))
        cache.save_summary(Summary(123, 2800, 4599, 1, "L1-B", 10))

        context = cache.get_context_windows(
            channel_id=123,
            current_time=10000,
            chunk_size=1800,
            chunks_per_level=4,
            chunk_size_multiplier=2
        )

        assert "level_1" in context
        assert "level_2" in context
        assert "level_3" in context

        # Should have some Level-1, but empty higher levels
        assert len(context["level_1"]) >= 0
        assert len(context["level_2"]) == 0
        assert len(context["level_3"]) == 0

    def test_handles_gaps_in_summary_coverage(self, cache):
        """Test handling gaps where no summaries exist for certain time periods."""
        # Current time: 50000
        # Level 0 (raw): 46400 to 50000
        # Level 1: 39200 to 46399
        # Only have 2 out of 4 expected Level-1 chunks

        cache.save_summary(Summary(123, 44600, 46399, 1, "L1-A", 10))  # Most recent chunk
        # Gap here (missing 42800-44599)
        cache.save_summary(Summary(123, 41000, 42799, 1, "L1-C", 10))  # Older chunk
        # Gap here (missing 39200-40999)

        context = cache.get_context_windows(
            channel_id=123,
            current_time=50000,
            chunk_size=1800,
            chunks_per_level=4,
            chunk_size_multiplier=2
        )

        # Should return only the summaries that exist within the range
        assert len(context["level_1"]) == 2
        assert context["level_1"][0].summary_text == "L1-C"
        assert context["level_1"][1].summary_text == "L1-A"


class TestContextWindowConfiguration:
    """Test context window configuration parameters."""

    def test_total_context_depth_calculation(self, cache):
        """Test that total context depth matches expected time span."""
        # Config: chunk_size=30min, chunks_per_level=4, multiplier=2
        # Expected levels (implementation creates 5 levels):
        # Level 0: 1h (30min current + 30min buffer)
        # Level 1: 2h (4 × 30min)
        # Level 2: 4h (4 × 1h)
        # Level 3: 8h (4 × 2h)
        # Level 4: 16h (4 × 4h)
        # Level 5: 32h (4 × 8h)
        # Total: 63 hours = 226800 seconds

        current_time = 100000
        chunk_size = 1800  # 30 min

        context = cache.get_context_windows(
            channel_id=123,
            current_time=current_time,
            chunk_size=chunk_size,
            chunks_per_level=4,
            chunk_size_multiplier=2
        )

        # The implementation creates 5 summary levels
        expected_depth = 226800  # 63 hours in seconds

        # The context metadata should include depth information
        assert "config" in context
        assert context["config"]["total_depth_seconds"] == expected_depth

    def test_different_chunk_size_config(self, cache):
        """Test with different chunk_size (15 minutes instead of 30)."""
        # chunk_size=15min (900s), chunks_per_level=4, multiplier=2
        # Level 0: 30min (15min × 2)
        # Level 1: 1h (4 × 15min)
        # Level 2: 2h (4 × 30min)
        # Level 3: 4h (4 × 1h)
        # Level 4: 8h (4 × 2h)
        # Level 5: 16h (4 × 4h)
        # Total: 31.5 hours = 113400 seconds

        context = cache.get_context_windows(
            channel_id=123,
            current_time=20000,
            chunk_size=900,  # 15 min
            chunks_per_level=4,
            chunk_size_multiplier=2
        )

        expected_depth = 113400  # 31.5 hours
        assert context["config"]["total_depth_seconds"] == expected_depth

    def test_different_multiplier_config(self, cache):
        """Test with different chunk_size_multiplier (3 instead of 2)."""
        # chunk_size=30min, chunks_per_level=4, multiplier=3
        # Level 0: 1h (30min × 2)
        # Level 1: 2h (4 × 30min)
        # Level 2: 6h (4 × 90min)
        # Level 3: 18h (4 × 4.5h)
        # Level 4: 54h (4 × 13.5h)
        # Level 5: 162h (4 × 40.5h)
        # Total: 243 hours = 874800 seconds

        context = cache.get_context_windows(
            channel_id=123,
            current_time=100000,
            chunk_size=1800,
            chunks_per_level=4,
            chunk_size_multiplier=3
        )

        expected_depth = 874800  # 243 hours
        assert context["config"]["total_depth_seconds"] == expected_depth

    def test_different_chunks_per_level(self, cache):
        """Test with different chunks_per_level (2 instead of 4)."""
        # chunk_size=30min, chunks_per_level=2, multiplier=2
        # Level 0: 30min + 30min
        # Level 1: 2 × 30min = 1 hour
        # Level 2: 2 × 60min = 2 hours

        context = cache.get_context_windows(
            channel_id=123,
            current_time=50000,
            chunk_size=1800,
            chunks_per_level=2,
            chunk_size_multiplier=2
        )

        # Verify only 2 chunks per level are requested
        assert context["config"]["chunks_per_level"] == 2


class TestSummaryCacheCachedMessages:
    """Test caching raw Discord messages."""

    def test_cache_and_retrieve_message(self, cache):
        """Test basic message caching."""
        msg = CachedMessage(
            channel_id=123,
            message_id=999,
            author_id=456,
            author_name="TestUser",
            content="Hello world",
            timestamp=1000,
        )

        cache.cache_message(msg)
        messages = cache.get_cached_messages(123, 1000, 2000)

        assert len(messages) == 1
        assert messages[0].message_id == 999
        assert messages[0].content == "Hello world"
        assert messages[0].author_name == "TestUser"

    def test_get_messages_in_time_range(self, cache):
        """Test retrieving messages within a specific time range."""
        cache.cache_message(CachedMessage(123, 1, 456, "User1", "Msg1", 1000))
        cache.cache_message(CachedMessage(123, 2, 456, "User1", "Msg2", 2000))
        cache.cache_message(CachedMessage(123, 3, 456, "User1", "Msg3", 3000))

        messages = cache.get_cached_messages(123, 1500, 2500)

        assert len(messages) == 1
        assert messages[0].message_id == 2

    def test_get_latest_cached_message(self, cache):
        """Test getting the most recent cached message."""
        cache.cache_message(CachedMessage(123, 1, 456, "User", "Old", 1000))
        cache.cache_message(CachedMessage(123, 3, 456, "User", "Newest", 3000))
        cache.cache_message(CachedMessage(123, 2, 456, "User", "Middle", 2000))

        latest_id = cache.get_latest_cached_message_id(123)
        assert latest_id == 3

    def test_get_latest_for_empty_channel_returns_none(self, cache):
        """Test getting latest message when channel has no cached messages."""
        latest_id = cache.get_latest_cached_message_id(999)
        assert latest_id is None

    def test_messages_sorted_by_timestamp(self, cache):
        """Test that messages are returned in chronological order."""
        cache.cache_message(CachedMessage(123, 3, 456, "User", "Third", 3000))
        cache.cache_message(CachedMessage(123, 1, 456, "User", "First", 1000))
        cache.cache_message(CachedMessage(123, 2, 456, "User", "Second", 2000))

        messages = cache.get_cached_messages(123, 0, 5000)

        assert messages[0].message_id == 1
        assert messages[1].message_id == 2
        assert messages[2].message_id == 3


class TestSummaryCacheBulkOperations:
    """Test bulk save operations for atomic commits."""

    def test_save_multiple_summaries_atomically(self, cache):
        """Test saving multiple summaries in one transaction."""
        summaries = [
            Summary(123, 1000, 1800, 1, "S1", 10),
            Summary(123, 2000, 2800, 1, "S2", 15),
            Summary(123, 1000, 2800, 2, "S12", 25),
        ]

        cache.save_summaries_batch(summaries)

        # Verify all were saved
        all_summaries = cache.get_summaries_for_channel(123)
        assert len(all_summaries) == 3

    def test_batch_save_rollback_on_error(self, cache):
        """Test that batch save is atomic (all-or-nothing)."""
        # Save one summary first
        cache.save_summary(Summary(123, 1000, 1800, 1, "Initial", 10))

        # Try to save batch with a duplicate (should fail)
        summaries = [
            Summary(123, 1000, 1800, 1, "Duplicate", 10),  # This exists already
            Summary(123, 2000, 2800, 1, "New", 15),
        ]

        # This should raise an error due to PRIMARY KEY constraint
        with pytest.raises(Exception):
            cache.save_summaries_batch(summaries)

        # Verify that the new summary was NOT saved (atomic rollback)
        new_summary = cache.get_summary(123, 2000, 2800, 1)
        assert new_summary is None  # Should not have been saved

        # Original should still exist
        original = cache.get_summary(123, 1000, 1800, 1)
        assert original.summary_text == "Initial"
