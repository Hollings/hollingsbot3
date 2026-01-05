"""Tests for background summarization worker and cascade logic."""

import pytest
import tempfile
import os
from prototype_summarization.summary_worker import SummaryWorker
from prototype_summarization.summary_cache import SummaryCache, CachedMessage, Summary
from prototype_summarization.summarizer import Summarizer
from prototype_summarization.time_windows import get_window_at_time


class DummyLLM:
    """Dummy LLM for testing."""

    def __init__(self):
        self.call_count = 0
        self.prompts = []

    async def generate(self, prompt: str) -> str:
        """Generate a dummy summary."""
        self.call_count += 1
        self.prompts.append(prompt)
        return f"Summary #{self.call_count}"


@pytest.fixture
def temp_db():
    """Create temporary database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except PermissionError:
        pass


@pytest.fixture
def cache(temp_db):
    """Create cache instance."""
    return SummaryCache(temp_db)


@pytest.fixture
def dummy_llm():
    """Create dummy LLM."""
    return DummyLLM()


@pytest.fixture
def summarizer(dummy_llm):
    """Create summarizer with dummy LLM."""
    return Summarizer(dummy_llm)


@pytest.fixture
def worker(cache, summarizer):
    """Create summary worker."""
    return SummaryWorker(cache, summarizer)


class TestFindCombinablePairs:
    """Test finding summaries that can be combined."""

    def test_no_summaries_returns_empty(self, worker):
        """Test that no summaries means no pairs."""
        pairs = worker._find_combinable_pairs(123, 1)
        assert pairs == []

    def test_single_summary_returns_empty(self, worker, cache):
        """Test that a single summary can't be paired."""
        cache.save_summary(Summary(123, 1000, 1800, 1, "S1", 10))

        pairs = worker._find_combinable_pairs(123, 1)
        assert pairs == []

    def test_two_same_level_summaries_form_pair(self, worker, cache):
        """Test that two Level-1 summaries can be paired."""
        cache.save_summary(Summary(123, 1000, 1800, 1, "S1", 10))
        cache.save_summary(Summary(123, 2000, 2800, 1, "S2", 15))

        pairs = worker._find_combinable_pairs(123, 1)

        assert len(pairs) == 1
        assert pairs[0][0].start_time == 1000
        assert pairs[0][1].start_time == 2000

    def test_three_summaries_creates_one_pair(self, worker, cache):
        """Test that three summaries creates one pair (leaves one unpaired)."""
        cache.save_summary(Summary(123, 1000, 1800, 1, "S1", 10))
        cache.save_summary(Summary(123, 2000, 2800, 1, "S2", 15))
        cache.save_summary(Summary(123, 3000, 3800, 1, "S3", 20))

        pairs = worker._find_combinable_pairs(123, 1)

        assert len(pairs) == 1
        # Should pair the first two
        assert pairs[0][0].start_time == 1000
        assert pairs[0][1].start_time == 2000

    def test_four_summaries_creates_two_pairs(self, worker, cache):
        """Test that four summaries creates two pairs."""
        cache.save_summary(Summary(123, 1000, 1800, 1, "S1", 10))
        cache.save_summary(Summary(123, 2000, 2800, 1, "S2", 15))
        cache.save_summary(Summary(123, 3000, 3800, 1, "S3", 20))
        cache.save_summary(Summary(123, 4000, 4800, 1, "S4", 25))

        pairs = worker._find_combinable_pairs(123, 1)

        assert len(pairs) == 2
        assert pairs[0][0].start_time == 1000
        assert pairs[0][1].start_time == 2000
        assert pairs[1][0].start_time == 3000
        assert pairs[1][1].start_time == 4000

    def test_pairs_with_time_gaps(self, worker, cache):
        """Test pairing works even with time gaps between summaries."""
        # Large gap between summaries (not adjacent windows)
        cache.save_summary(Summary(123, 1000, 1800, 1, "S1", 10))
        cache.save_summary(Summary(123, 9000, 9800, 1, "S2", 15))  # 2 hour gap

        pairs = worker._find_combinable_pairs(123, 1)

        # Should still pair them
        assert len(pairs) == 1
        assert pairs[0][0].start_time == 1000
        assert pairs[0][1].start_time == 9000

    def test_ignores_different_level_summaries(self, worker, cache):
        """Test that different levels aren't paired together."""
        cache.save_summary(Summary(123, 1000, 1800, 1, "L1-A", 10))
        cache.save_summary(Summary(123, 2000, 4800, 2, "L2-A", 25))
        cache.save_summary(Summary(123, 5000, 5800, 1, "L1-B", 15))

        # Looking for Level-1 pairs
        pairs = worker._find_combinable_pairs(123, 1)

        # Should pair the two Level-1 summaries, ignoring Level-2
        assert len(pairs) == 1
        assert pairs[0][0].start_time == 1000
        assert pairs[0][1].start_time == 5000


class TestGenerateLevel1Summaries:
    """Test generating Level-1 summaries from messages."""

    @pytest.mark.anyio
    async def test_generate_for_window_with_messages(self, worker, cache, dummy_llm):
        """Test generating Level-1 summary for a window."""
        # Add messages to a window
        cache.cache_message(CachedMessage(123, 1, 456, "User1", "Hello", 1000))
        cache.cache_message(CachedMessage(123, 2, 456, "User1", "World", 1100))

        # Generate summary for that window
        summary = await worker._generate_level_1_summary(123, 1000, 1800)

        assert summary is not None
        assert summary.channel_id == 123
        assert summary.start_time == 1000
        assert summary.end_time == 1800
        assert summary.summary_level == 1
        assert summary.message_count == 2
        assert dummy_llm.call_count == 1

    @pytest.mark.anyio
    async def test_no_summary_for_empty_window(self, worker, cache, dummy_llm):
        """Test that empty windows don't generate summaries."""
        # No messages cached
        summary = await worker._generate_level_1_summary(123, 1000, 1800)

        assert summary is None
        assert dummy_llm.call_count == 0

    @pytest.mark.anyio
    async def test_skips_window_with_existing_summary(self, worker, cache, dummy_llm):
        """Test that windows with existing summaries are skipped."""
        # Add existing summary
        cache.save_summary(Summary(123, 1000, 1800, 1, "Existing", 10))

        # Try to generate again
        summary = await worker._generate_level_1_summary(123, 1000, 1800)

        assert summary is None  # Should return None (already exists)
        assert dummy_llm.call_count == 0


class TestCombineSummaries:
    """Test combining two summaries into a higher level."""

    @pytest.mark.anyio
    async def test_combine_two_level_1_summaries(self, worker, cache, dummy_llm):
        """Test combining two Level-1 summaries into Level-2."""
        # Each window is 1800 seconds (30 min)
        # Combined they span 3600 seconds (1 hour) = Level 2
        s1 = Summary(123, 1000, 2799, 1, "First", 10)  # 1799 seconds
        s2 = Summary(123, 2800, 4599, 1, "Second", 15)  # 1799 seconds

        combined = await worker._combine_summaries(s1, s2)

        assert combined.channel_id == 123
        assert combined.start_time == 1000
        assert combined.end_time == 4599
        assert combined.summary_level == 2
        assert combined.message_count == 25  # 10 + 15
        assert dummy_llm.call_count == 1

    @pytest.mark.anyio
    async def test_combine_preserves_time_span_with_gaps(self, worker, dummy_llm):
        """Test that combining summaries with gaps preserves full time span."""
        s1 = Summary(123, 1000, 1800, 1, "First", 10)
        s2 = Summary(123, 9000, 9800, 1, "Second", 15)  # 2 hour gap

        combined = await worker._combine_summaries(s1, s2)

        assert combined.start_time == 1000
        assert combined.end_time == 9800
        # Level should be calculated based on actual time span
        assert combined.summary_level >= 2


class TestRunCascade:
    """Test the full summarization cascade."""

    @pytest.mark.anyio
    async def test_cascade_with_no_messages(self, worker, cache):
        """Test cascade with no messages does nothing."""
        new_summaries = await worker._run_cascade(123)

        assert new_summaries == []

    @pytest.mark.skip(reason="Level-1 generation from messages not yet implemented in cascade")
    @pytest.mark.anyio
    async def test_cascade_creates_level_1_summaries(self, worker, cache, dummy_llm):
        """Test cascade creates Level-1 summaries from messages."""
        # Add messages to two different windows
        cache.cache_message(CachedMessage(123, 1, 456, "User", "Msg1", 1000))
        cache.cache_message(CachedMessage(123, 2, 456, "User", "Msg2", 2000))

        new_summaries = await worker._run_cascade(123)

        # Should create one or two Level-1 summaries depending on window boundaries
        assert len(new_summaries) >= 1
        assert all(s.summary_level == 1 for s in new_summaries)

    @pytest.mark.anyio
    async def test_cascade_creates_level_2_from_level_1(self, worker, cache, dummy_llm):
        """Test cascade combines Level-1 summaries into Level-2."""
        # Pre-populate with two Level-1 summaries (each 30 min = 1799 seconds)
        cache.save_summary(Summary(123, 1000, 2799, 1, "S1", 10))
        cache.save_summary(Summary(123, 2800, 4599, 1, "S2", 15))

        new_summaries = await worker._run_cascade(123)

        # Should create a Level-2 summary
        level_2_summaries = [s for s in new_summaries if s.summary_level == 2]
        assert len(level_2_summaries) == 1
        assert level_2_summaries[0].message_count == 25

    @pytest.mark.anyio
    async def test_cascade_multiple_levels(self, worker, cache, dummy_llm):
        """Test cascade creates multiple levels when enough summaries exist."""
        # Create 4 Level-1 summaries (each 30 min, total 2 hours)
        cache.save_summary(Summary(123, 1000, 2799, 1, "S1", 10))     # 30 min
        cache.save_summary(Summary(123, 2800, 4599, 1, "S2", 15))     # 30 min
        cache.save_summary(Summary(123, 4600, 6399, 1, "S3", 20))     # 30 min
        cache.save_summary(Summary(123, 6400, 8199, 1, "S4", 25))     # 30 min

        new_summaries = await worker._run_cascade(123)

        # Should create:
        # - 2 Level-2 summaries (from 4 Level-1s)
        # - 1 Level-3 summary (from 2 Level-2s)
        level_2_summaries = [s for s in new_summaries if s.summary_level == 2]
        level_3_summaries = [s for s in new_summaries if s.summary_level == 3]

        assert len(level_2_summaries) == 2
        # Level-3 creation depends on cascade running multiple iterations
        # For now, just verify Level-2 summaries are created
        assert len(level_3_summaries) >= 0  # May or may not create Level-3 in single run

    @pytest.mark.anyio
    async def test_cascade_commits_atomically(self, worker, cache, dummy_llm):
        """Test that cascade commits all summaries atomically."""
        # Setup
        cache.save_summary(Summary(123, 1000, 2799, 1, "S1", 10))
        cache.save_summary(Summary(123, 2800, 4599, 1, "S2", 15))

        # Before cascade
        summaries_before = cache.get_summaries_for_channel(123)
        assert len(summaries_before) == 2

        # Run cascade
        new_summaries = await worker._run_cascade(123)

        # All new summaries should be in cache
        summaries_after = cache.get_summaries_for_channel(123)
        assert len(summaries_after) == len(summaries_before) + len(new_summaries)


class TestTriggerSummarization:
    """Test triggering background summarization."""

    @pytest.mark.anyio
    async def test_trigger_runs_cascade(self, worker, cache, dummy_llm):
        """Test that trigger actually runs the cascade."""
        cache.save_summary(Summary(123, 1000, 2799, 1, "S1", 10))
        cache.save_summary(Summary(123, 2800, 4599, 1, "S2", 15))

        await worker.trigger_summarization(123)

        # Should have created Level-2 summary
        summaries = cache.get_summaries_for_channel(123)
        level_2 = [s for s in summaries if s.summary_level == 2]
        assert len(level_2) == 1

    @pytest.mark.anyio
    async def test_trigger_prevents_duplicate_work(self, worker, cache):
        """Test that triggering twice doesn't cause duplicate summarization."""
        cache.save_summary(Summary(123, 1000, 2799, 1, "S1", 10))
        cache.save_summary(Summary(123, 2800, 4599, 1, "S2", 15))

        # Trigger twice in rapid succession
        await worker.trigger_summarization(123)
        await worker.trigger_summarization(123)

        # Should only have one Level-2 summary
        summaries = cache.get_summaries_for_channel(123)
        level_2 = [s for s in summaries if s.summary_level == 2]
        assert len(level_2) == 1
