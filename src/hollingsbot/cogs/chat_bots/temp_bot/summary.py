"""Conversation summarization for departed temp bots.

When a temp bot leaves, we ask Haiku to write a short summary of what happened
during its session and persist it on the ``temp_bots`` row so a future
``!recall`` can rehydrate the bot's memory.

These helpers are intentionally side-effecting (sqlite + Anthropic API) but
have no dependency on :class:`TempBotManager`, so they live as module-level
async functions.
"""

from __future__ import annotations

import logging
import sqlite3

from hollingsbot.prompt_db import DB_PATH
from hollingsbot.text_generators import AnthropicTextGenerator

_LOG = logging.getLogger(__name__)

# Skip summary if the conversation transcript is shorter than this many chars.
_MIN_CONVERSATION_LENGTH = 50
# Truncate transcripts longer than this before sending to the LLM.
_MAX_CONVERSATION_LENGTH = 8000
# Truncate any single message in the transcript past this length.
_MAX_MESSAGE_LENGTH = 500


def _format_transcript(messages: list[tuple[str, str]], bot_name: str) -> str:
    """Render ``[(author, content), ...]`` rows into a single transcript string."""
    lines: list[str] = []
    for author, content in messages:
        if not content:
            continue
        # Strip the <Author>: prefix if present
        if content.startswith(f"<{author}>:"):
            content = content[len(f"<{author}>:") :].strip()
        if len(content) > _MAX_MESSAGE_LENGTH:
            content = content[:_MAX_MESSAGE_LENGTH] + "..."
        # Mark the temp bot's messages
        if author == bot_name:
            lines.append(f"[{bot_name}]: {content}")
        else:
            lines.append(f"{author}: {content}")
    return "\n".join(lines)


def _fetch_conversation_messages(channel_id: int, bot_name: str) -> list[tuple[str, str]]:
    """Pull the cached message rows that bracket this bot's session."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT MIN(timestamp), MAX(timestamp)
            FROM cached_messages
            WHERE channel_id = ? AND author_name = ?
            """,
            (channel_id, bot_name),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return []

        first_ts, last_ts = row
        cur = conn.execute(
            """
            SELECT author_name, content
            FROM cached_messages
            WHERE channel_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (channel_id, first_ts, last_ts),
        )
        return cur.fetchall()


def _build_summary_prompt(bot_name: str, spawn_prompt: str, conversation: str) -> str:
    return f"""Summarize this Discord conversation involving a temporary bot named "{bot_name}".

The bot was spawned with this personality/purpose: "{spawn_prompt}"

Conversation:
{conversation}

Write a 3-5 sentence summary of what happened in this conversation. Focus on:
- What the bot's personality/character was like
- Key interactions or memorable moments
- How the conversation ended

Be concise and capture the essence of the bot's time in the chat."""


async def generate_conversation_summary(channel_id: int, bot_name: str, spawn_prompt: str) -> str | None:
    """Generate a conversation summary for a temp bot using Haiku.

    Returns the summary text, or ``None`` if the conversation was too short,
    no messages were found, or the LLM call failed.
    """
    try:
        messages = _fetch_conversation_messages(channel_id, bot_name)
        if not messages:
            _LOG.warning("No messages found for bot '%s' in channel %s", bot_name, channel_id)
            return None

        conversation = _format_transcript(messages, bot_name)
        if len(conversation) < _MIN_CONVERSATION_LENGTH:
            _LOG.info("Conversation too short for bot '%s', skipping summary", bot_name)
            return None

        if len(conversation) > _MAX_CONVERSATION_LENGTH:
            conversation = conversation[:_MAX_CONVERSATION_LENGTH] + "\n... (truncated)"

        prompt = _build_summary_prompt(bot_name, spawn_prompt, conversation)
        generator = AnthropicTextGenerator(model="claude-haiku-4-5")
        summary = await generator.generate(prompt, temperature=0.7)
        return summary.strip() if summary else None

    except Exception:
        _LOG.exception("Failed to generate summary for bot '%s'", bot_name)
        return None


async def save_conversation_summary(webhook_id: int, summary: str, append: bool = False) -> None:
    """Save or append a conversation summary to the ``temp_bots`` table.

    Args:
        webhook_id: The bot's webhook ID.
        summary: The summary text to save.
        append: If True, append to existing summary (for recalled bots).
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            if append:
                cur = conn.execute(
                    "SELECT conversation_summary FROM temp_bots WHERE webhook_id = ?",
                    (webhook_id,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    summary = f"{row[0]}\n\n---\n\n**Recalled session:**\n{summary}"

            conn.execute(
                "UPDATE temp_bots SET conversation_summary = ? WHERE webhook_id = ?",
                (summary, webhook_id),
            )
            conn.commit()
            _LOG.info("Saved conversation summary for webhook %s", webhook_id)
    except Exception:
        _LOG.exception("Failed to save summary for webhook %s", webhook_id)
