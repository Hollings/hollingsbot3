"""Search message history tool for LLM."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from hollingsbot.prompt_db import search_message_history

_LOG = logging.getLogger(__name__)


def search_messages(query: str = "", author_id: str = "") -> str:
    """Search message history in the current channel, excluding messages in the bot's current context window.

    Args:
        query: Text to search for in message content (optional)
        author_id: Filter by Discord user ID (optional)

    Returns:
        Formatted search results with message content, author, and timestamps (5 most recent matches older than context)
    """
    try:
        # This will be set by the executor to the current channel
        from hollingsbot.tools.parser import get_current_context
        import sqlite3

        context = get_current_context()
        if not context or not context.get("channel_id"):
            return "Error: No channel context available"

        channel_id = context["channel_id"]

        # Parse parameters
        limit_int = 5  # Always return 5 most recent (outside context window)
        author_id_int = int(author_id) if author_id.isdigit() else None

        # Get the timestamp of the 20th most recent message to exclude context window
        # The bot's context window is typically 20 messages
        from hollingsbot.prompt_db import DB_PATH
        before_timestamp = None
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                """
                SELECT timestamp FROM message_history
                WHERE channel_id = ?
                ORDER BY timestamp DESC
                LIMIT 1 OFFSET 20
                """,
                (channel_id,)
            )
            row = cur.fetchone()
            if row:
                before_timestamp = row[0]

        # Search (excluding recent messages in context window)
        results = search_message_history(
            channel_id=channel_id,
            query=query or None,
            author_id=author_id_int,
            before_timestamp=before_timestamp,
            limit=limit_int,
        )

        if not results:
            search_desc = []
            if query:
                search_desc.append(f"query='{query}'")
            if author_id_int:
                search_desc.append(f"author_id={author_id_int}")
            return f"No messages found matching: {', '.join(search_desc)}"

        # Format results
        lines = [f"Found {len(results)} message(s):", ""]

        for i, msg in enumerate(results, 1):
            timestamp = msg["timestamp"]
            # Parse ISO timestamp and format nicely
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                formatted_time = timestamp

            author = msg.get("author_nickname") or f"User {msg['author_id']}"
            content = msg.get("content") or "[no text content]"

            # Truncate long messages to 500 characters
            if len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"{i}. [{formatted_time}] {author}:")
            lines.append(f"   {content}")

            # Include attachment info if present
            if msg.get("attachment_urls"):
                num_attachments = len(msg["attachment_urls"])
                lines.append(f"   [+{num_attachments} attachment(s)]")

            lines.append("")

        return "\n".join(lines)

    except Exception as exc:
        _LOG.exception("Failed to search message history")
        return f"Error searching message history: {exc}"
