#!/usr/bin/env python3
"""Generate conversation summaries for temp bots using Claude Haiku.

This script:
1. Finds all temp bots without a conversation_summary
2. Retrieves messages between their first and last appearance
3. Uses Claude Haiku to generate a 3-5 sentence summary
4. Saves the summary to the temp_bots table

Usage:
    python scripts/summarize_temp_bot_conversations.py [--dry-run] [--limit N]
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hollingsbot.prompt_db import DB_PATH, init_db

# Try to import anthropic
try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)


def get_temp_bots_needing_summary(conn: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    """Get temp bots that don't have a conversation summary yet."""
    query = """
        SELECT id, channel_id, name, spawn_prompt, created_at
        FROM temp_bots
        WHERE conversation_summary IS NULL
        ORDER BY created_at DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cur = conn.execute(query)
    return [
        {
            "id": row[0],
            "channel_id": row[1],
            "name": row[2],
            "spawn_prompt": row[3],
            "created_at": row[4],
        }
        for row in cur.fetchall()
    ]


def get_conversation_messages(conn: sqlite3.Connection, channel_id: int, bot_name: str) -> list[dict]:
    """Get all messages in the conversation window for a temp bot.

    Finds the first and last message from the bot, then retrieves all messages
    in that time range.
    """
    # Find the bot's first and last message timestamps
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

    # Get all messages in that time range
    cur = conn.execute(
        """
        SELECT author_name, content, timestamp
        FROM cached_messages
        WHERE channel_id = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
        """,
        (channel_id, first_ts, last_ts),
    )

    return [
        {
            "author": row[0],
            "content": row[1],
            "timestamp": row[2],
        }
        for row in cur.fetchall()
    ]


def format_conversation_for_summary(messages: list[dict], bot_name: str) -> str:
    """Format messages into a readable conversation for the LLM."""
    lines = []
    for msg in messages:
        author = msg["author"]
        content = msg["content"] or ""

        # Strip the <Author>: prefix if present
        if content.startswith(f"<{author}>:"):
            content = content[len(f"<{author}>:") :].strip()

        # Truncate long messages
        if len(content) > 500:
            content = content[:500] + "..."

        # Mark the temp bot's messages
        if author == bot_name:
            lines.append(f"[{bot_name}]: {content}")
        else:
            lines.append(f"{author}: {content}")

    return "\n".join(lines)


def generate_summary(client: anthropic.Anthropic, bot_name: str, spawn_prompt: str, conversation: str) -> str:
    """Generate a 3-5 sentence summary using Claude Haiku."""
    prompt = f"""Summarize this Discord conversation involving a temporary bot named "{bot_name}".

The bot was spawned with this personality/purpose: "{spawn_prompt}"

Conversation:
{conversation}

Write a 3-5 sentence summary of what happened in this conversation. Focus on:
- What the bot's personality/character was like
- Key interactions or memorable moments
- How the conversation ended

Be concise and capture the essence of the bot's time in the chat."""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def save_summary(conn: sqlite3.Connection, bot_id: int, summary: str) -> None:
    """Save the conversation summary to the database."""
    conn.execute(
        "UPDATE temp_bots SET conversation_summary = ? WHERE id = ?",
        (summary, bot_id),
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Generate conversation summaries for temp bots")
    parser.add_argument("--dry-run", action="store_true", help="Don't save summaries, just print them")
    parser.add_argument("--limit", type=int, help="Limit number of bots to process")
    parser.add_argument("--bot-id", type=int, help="Process a specific bot by ID")
    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Database: {DB_PATH}")
    print(f"Dry run: {args.dry_run}")
    print()

    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        if args.bot_id:
            # Process specific bot
            cur = conn.execute(
                "SELECT id, channel_id, name, spawn_prompt, created_at FROM temp_bots WHERE id = ?",
                (args.bot_id,),
            )
            row = cur.fetchone()
            if not row:
                print(f"Bot with ID {args.bot_id} not found")
                return
            bots = [
                {
                    "id": row[0],
                    "channel_id": row[1],
                    "name": row[2],
                    "spawn_prompt": row[3],
                    "created_at": row[4],
                }
            ]
        else:
            bots = get_temp_bots_needing_summary(conn, limit=args.limit)

        print(f"Found {len(bots)} temp bot(s) to summarize")
        print()

        summarized = 0
        skipped = 0
        errors = 0

        for i, bot in enumerate(bots, 1):
            bot_id = bot["id"]
            bot_name = bot["name"]
            channel_id = bot["channel_id"]
            spawn_prompt = bot["spawn_prompt"] or "(unknown)"

            print(f"[{i}/{len(bots)}] {bot_name}...")

            # Get conversation messages
            messages = get_conversation_messages(conn, channel_id, bot_name)
            if not messages:
                print("  No messages found, skipping")
                skipped += 1
                continue

            print(f"  Found {len(messages)} messages in conversation")

            # Format conversation
            conversation = format_conversation_for_summary(messages, bot_name)
            if len(conversation) < 50:
                print("  Conversation too short, skipping")
                skipped += 1
                continue

            # Truncate if too long (keep it reasonable for Haiku)
            if len(conversation) > 8000:
                conversation = conversation[:8000] + "\n... (truncated)"

            try:
                # Generate summary
                summary = generate_summary(client, bot_name, spawn_prompt, conversation)
                print(f"  Summary: {summary[:100]}...")

                if not args.dry_run:
                    save_summary(conn, bot_id, summary)
                    print("  Saved!")
                else:
                    print("  [DRY RUN] Would save summary")

                summarized += 1

                # Rate limiting - be nice to the API
                time.sleep(0.5)

            except Exception as e:
                print(f"  Error: {e}")
                errors += 1

            print()

        print("=" * 50)
        print(f"Summarized: {summarized}")
        print(f"Skipped: {skipped}")
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
