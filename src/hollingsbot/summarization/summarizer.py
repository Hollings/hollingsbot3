"""Summary generation logic for hierarchical message-count summarization."""

from typing import Protocol

from .summary_cache import CachedMessage, MessageGroup


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


def build_level_1_prompt(messages: list[CachedMessage]) -> str:
    """
    Build prompt for Level-1 summary from raw messages.

    Args:
        messages: List of cached messages to summarize (typically 5)

    Returns:
        Formatted prompt for LLM
    """
    messages_text = "\n".join(f"{msg.author_name}: {msg.content}" for msg in messages)

    prompt = f"""Write a ONE sentence note about this conversation from Wendy's perspective (first person as Wendy). Be extremely brief. Example: "I talked with Hollings about X" or "We discussed Y". Never say "I talked with Wendy" - you ARE Wendy.

Messages:
{messages_text}

Wendy's one-sentence note:"""

    return prompt


def build_level_2_prompt(summaries: list[str]) -> str:
    """
    Build prompt for Level-2 summary from 5 Level-1 summaries.

    Args:
        summaries: List of exactly 5 summary strings

    Returns:
        Formatted prompt for LLM
    """
    summaries_text = "\n".join(f"- {s}" for s in summaries)

    prompt = f"""Combine these notes into exactly 2 sentences from Wendy's perspective (first person as Wendy). Merge related topics, drop minor details.

Wendy's previous notes:
{summaries_text}

Wendy's two-sentence summary:"""

    return prompt


class Summarizer:
    """Handles summary generation using an LLM."""

    def __init__(self, llm: LLMProtocol):
        """
        Initialize summarizer.

        Args:
            llm: LLM instance that implements generate() method
        """
        self.llm = llm

    async def summarize_messages(self, messages: list[CachedMessage]) -> str:
        """
        Generate Level-1 summary from 5 raw messages.

        Args:
            messages: Exactly 5 messages to summarize

        Returns:
            Summary text (1 sentence, highly compressed)
        """
        if not messages:
            return "[No messages]"

        prompt = build_level_1_prompt(messages)
        summary_text = await self.llm.generate(prompt)

        return summary_text.strip()

    async def summarize_groups(self, groups: list[MessageGroup]) -> str:
        """
        Generate Level-2 summary from 5 Level-1 groups.

        Args:
            groups: Exactly 5 MessageGroups with summaries

        Returns:
            Combined summary text (2 sentences, compressed)
        """
        summaries = [g.summary_text for g in groups if g.summary_text]

        if not summaries:
            return "[No summaries to combine]"

        prompt = build_level_2_prompt(summaries)
        meta_summary_text = await self.llm.generate(prompt)

        return meta_summary_text.strip()
