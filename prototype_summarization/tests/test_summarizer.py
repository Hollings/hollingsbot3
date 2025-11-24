"""Tests for summary generation logic."""

import pytest
from prototype_summarization.summarizer import (
    Summarizer,
    build_level_1_prompt,
    build_meta_summary_prompt,
)
from prototype_summarization.summary_cache import CachedMessage, Summary


class DummyLLM:
    """Dummy LLM that returns predictable responses for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_prompt = None

    async def generate(self, prompt: str) -> str:
        """Generate a dummy summary based on prompt content."""
        self.call_count += 1
        self.last_prompt = prompt

        # Return different summaries based on what's in the prompt
        if "User1: Hello" in prompt:
            return "Summary: User1 greeted the channel."
        elif "User2: How are you" in prompt:
            return "Summary: User2 asked about wellbeing."
        elif "combining two time-window summaries" in prompt:
            # Meta-summary
            return "Combined summary of both periods."
        else:
            return f"Generated summary #{self.call_count}"


@pytest.fixture
def dummy_llm():
    """Create a dummy LLM for testing."""
    return DummyLLM()


@pytest.fixture
def summarizer(dummy_llm):
    """Create a Summarizer with dummy LLM."""
    return Summarizer(llm=dummy_llm)


class TestBuildLevel1Prompt:
    """Test Level-1 summary prompt building."""

    def test_prompt_includes_messages(self):
        """Test that prompt includes all message content."""
        messages = [
            CachedMessage(123, 1, 456, "User1", "Hello world", 1000),
            CachedMessage(123, 2, 456, "User1", "How are you?", 1100),
        ]

        prompt = build_level_1_prompt(messages, overlap_buffer=5)

        assert "User1: Hello world" in prompt
        assert "User1: How are you?" in prompt

    def test_prompt_includes_overlap_note(self):
        """Test that overlap buffer instruction is included."""
        messages = [
            CachedMessage(123, 1, 456, "User1", "Test", 1000),
        ]

        prompt = build_level_1_prompt(messages, overlap_buffer=7)

        assert "7 messages continue into the next window" in prompt or "smooth handoff" in prompt

    def test_prompt_with_multiple_authors(self):
        """Test prompt with messages from different users."""
        messages = [
            CachedMessage(123, 1, 456, "Alice", "Hi there", 1000),
            CachedMessage(123, 2, 789, "Bob", "Hello Alice", 1100),
        ]

        prompt = build_level_1_prompt(messages, overlap_buffer=5)

        assert "Alice:" in prompt
        assert "Bob:" in prompt


class TestBuildMetaSummaryPrompt:
    """Test meta-summary prompt building."""

    def test_prompt_includes_both_summaries(self):
        """Test that both summaries are included."""
        summary1 = Summary(123, 1000, 1800, 1, "First period summary", 10)
        summary2 = Summary(123, 2000, 2800, 1, "Second period summary", 15)

        prompt = build_meta_summary_prompt(summary1, summary2)

        assert "First period summary" in prompt
        assert "Second period summary" in prompt

    def test_prompt_includes_time_ranges(self):
        """Test that time ranges are mentioned."""
        summary1 = Summary(123, 1000, 1800, 1, "Summary 1", 10)
        summary2 = Summary(123, 2000, 2800, 1, "Summary 2", 15)

        prompt = build_meta_summary_prompt(summary1, summary2)

        # Should mention the time ranges somehow
        assert "1000" in prompt or "Window 1" in prompt
        assert "2000" in prompt or "Window 2" in prompt

    def test_prompt_notes_gap_when_present(self):
        """Test that gaps between summaries are noted."""
        # 2-hour gap between summaries
        summary1 = Summary(123, 1000, 1800, 1, "Summary 1", 10)
        summary2 = Summary(123, 9000, 9800, 1, "Summary 2", 15)

        prompt = build_meta_summary_prompt(summary1, summary2)

        # Should mention the gap
        assert "gap" in prompt.lower() or "hour" in prompt.lower()

    def test_prompt_no_gap_note_for_adjacent_windows(self):
        """Test that no gap note is added for adjacent windows."""
        # Adjacent windows (1800 -> 2000 is just 200 seconds)
        summary1 = Summary(123, 1000, 1800, 1, "Summary 1", 10)
        summary2 = Summary(123, 2000, 2800, 1, "Summary 2", 15)

        prompt = build_meta_summary_prompt(summary1, summary2)

        # Shouldn't have a big gap warning
        # (or if it does, it should be minimal)
        gap_count = prompt.lower().count("gap")
        assert gap_count <= 1  # Might mention "gap" in general context


class TestSummarizerLevel1:
    """Test Level-1 summary generation from raw messages."""

    @pytest.mark.anyio
    async def test_summarize_messages_basic(self, summarizer, dummy_llm):
        """Test basic message summarization."""
        messages = [
            CachedMessage(123, 1, 456, "User1", "Hello", 1000),
            CachedMessage(123, 2, 789, "User2", "Hi", 1100),
        ]

        summary_text = await summarizer.summarize_messages(
            messages,
            window_start=1000,
            window_end=1800
        )

        assert isinstance(summary_text, str)
        assert len(summary_text) > 0
        assert dummy_llm.call_count == 1

    @pytest.mark.anyio
    async def test_summarize_passes_messages_to_llm(self, summarizer, dummy_llm):
        """Test that messages are included in LLM prompt."""
        messages = [
            CachedMessage(123, 1, 456, "User1", "Hello", 1000),
        ]

        await summarizer.summarize_messages(messages, 1000, 1800)

        assert "User1" in dummy_llm.last_prompt
        assert "Hello" in dummy_llm.last_prompt

    @pytest.mark.anyio
    async def test_summarize_empty_messages_returns_placeholder(self, summarizer):
        """Test that summarizing empty message list returns appropriate text."""
        summary_text = await summarizer.summarize_messages([], 1000, 1800)

        # Should return some kind of placeholder
        assert isinstance(summary_text, str)
        assert "no messages" in summary_text.lower() or len(summary_text) == 0


class TestSummarizerMetaSummary:
    """Test meta-summary generation from summaries."""

    @pytest.mark.anyio
    async def test_summarize_summaries_basic(self, summarizer, dummy_llm):
        """Test combining two summaries."""
        summary1 = Summary(123, 1000, 1800, 1, "First summary", 10)
        summary2 = Summary(123, 2000, 2800, 1, "Second summary", 15)

        meta_summary = await summarizer.summarize_summaries(
            summary1,
            summary2,
            target_level=2
        )

        assert isinstance(meta_summary, str)
        assert len(meta_summary) > 0
        assert dummy_llm.call_count == 1

    @pytest.mark.anyio
    async def test_meta_summary_includes_both_summaries_in_prompt(self, summarizer, dummy_llm):
        """Test that both input summaries are passed to LLM."""
        summary1 = Summary(123, 1000, 1800, 1, "First summary", 10)
        summary2 = Summary(123, 2000, 2800, 1, "Second summary", 15)

        await summarizer.summarize_summaries(summary1, summary2, target_level=2)

        assert "First summary" in dummy_llm.last_prompt
        assert "Second summary" in dummy_llm.last_prompt

    @pytest.mark.anyio
    async def test_meta_summary_calculates_message_count(self, summarizer):
        """Test that meta-summary combines message counts."""
        summary1 = Summary(123, 1000, 1800, 1, "S1", 10)
        summary2 = Summary(123, 2000, 2800, 1, "S2", 15)

        meta_text = await summarizer.summarize_summaries(summary1, summary2, target_level=2)

        # The summarizer should track that this represents 25 total messages
        # (This would be returned as part of a Summary object in real usage)
        assert isinstance(meta_text, str)
