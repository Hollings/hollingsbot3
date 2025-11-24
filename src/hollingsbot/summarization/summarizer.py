"""Summary generation logic."""

from datetime import datetime, timezone
from typing import Protocol
from .summary_cache import CachedMessage, Summary


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


def build_level_1_prompt(messages: list[CachedMessage], overlap_buffer: int = 7) -> str:
    """
    Build prompt for Level-1 summary from raw messages.

    Args:
        messages: List of cached messages to summarize
        overlap_buffer: Number of messages that continue into next window

    Returns:
        Formatted prompt for LLM
    """
    formatted_messages = []
    for msg in messages:
        formatted_messages.append(f"{msg.author_name}: {msg.content}")

    messages_text = "\n".join(formatted_messages)

    prompt = f"""You are summarizing a 30-minute window of a Discord conversation for context retention.

Summarize the key points, decisions, and important context from these messages.
Focus on information that would be useful for continuing the conversation later.

Note: The last {overlap_buffer} messages continue into the next window, so ensure
your summary provides smooth handoff and context continuity.

Messages:
{messages_text}

Provide a concise summary (2-4 paragraphs)."""

    return prompt


def build_meta_summary_prompt(summary1: Summary, summary2: Summary) -> str:
    """
    Build prompt for meta-summary from two summaries.

    Args:
        summary1: First summary (earlier in time)
        summary2: Second summary (later in time)

    Returns:
        Formatted prompt for LLM
    """
    # Calculate gap between summaries
    gap_seconds = summary2.start_time - summary1.end_time
    gap_hours = gap_seconds / 3600

    if gap_hours > 1:
        gap_note = f"Note: There is a {gap_hours:.1f} hour gap between these summaries with no messages."
    else:
        gap_note = ""

    # Format timestamps
    def format_timestamp(ts: int) -> str:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    prompt = f"""You are creating a higher-level summary by combining two time-window summaries.

Previous summaries cover:
- Window 1: {format_timestamp(summary1.start_time)} to {format_timestamp(summary1.end_time)}
- Window 2: {format_timestamp(summary2.start_time)} to {format_timestamp(summary2.end_time)}

{gap_note}

Combine these into a single coherent summary that captures the essential context
from both time periods.

Summary 1:
{summary1.summary_text}

Summary 2:
{summary2.summary_text}

Provide a concise combined summary (2-4 paragraphs)."""

    return prompt


class Summarizer:
    """Handles summary generation using an LLM."""

    def __init__(self, llm: LLMProtocol, overlap_buffer: int = 7):
        """
        Initialize summarizer.

        Args:
            llm: LLM instance that implements generate() method
            overlap_buffer: Number of messages to note as continuing into next window
        """
        self.llm = llm
        self.overlap_buffer = overlap_buffer

    async def summarize_messages(
        self,
        messages: list[CachedMessage],
        window_start: int,
        window_end: int,
    ) -> str:
        """
        Generate Level-1 summary from raw messages.

        Args:
            messages: Messages to summarize
            window_start: Start timestamp of window
            window_end: End timestamp of window

        Returns:
            Summary text
        """
        if not messages:
            return "[No messages in this window]"

        prompt = build_level_1_prompt(messages, self.overlap_buffer)
        summary_text = await self.llm.generate(prompt)

        return summary_text.strip()

    async def summarize_summaries(
        self,
        summary1: Summary,
        summary2: Summary,
        target_level: int,
    ) -> str:
        """
        Generate meta-summary from two lower-level summaries.

        Args:
            summary1: First summary (earlier in time)
            summary2: Second summary (later in time)
            target_level: Target level for the combined summary

        Returns:
            Combined summary text
        """
        prompt = build_meta_summary_prompt(summary1, summary2)
        meta_summary_text = await self.llm.generate(prompt)

        return meta_summary_text.strip()
