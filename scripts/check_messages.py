#!/usr/bin/env python3
"""Fetch recent messages from the database for a channel.

Usage: python check_messages.py <channel_id> [--since <message_id>] [--limit <n>]

Returns JSON array of recent messages, newest first.
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Default database path
DB_PATH = os.getenv("PROMPT_DB_PATH", "/data/hollingsbot.db")


def get_recent_messages(channel_id: int, since_id: int | None = None, limit: int = 10) -> list[dict]:
    """Fetch recent messages from the cached_messages table."""
    db_path = Path(DB_PATH)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        if since_id:
            # Get messages newer than since_id
            query = """
                SELECT message_id, channel_id, author_name, content, timestamp
                FROM cached_messages
                WHERE channel_id = ? AND message_id > ?
                ORDER BY message_id DESC
                LIMIT ?
            """
            rows = conn.execute(query, (channel_id, since_id, limit)).fetchall()
        else:
            # Get last N messages
            query = """
                SELECT message_id, channel_id, author_name, content, timestamp
                FROM cached_messages
                WHERE channel_id = ?
                ORDER BY message_id DESC
                LIMIT ?
            """
            rows = conn.execute(query, (channel_id, limit)).fetchall()

        messages = []
        for row in rows:
            messages.append({
                "message_id": row["message_id"],
                "author": row["author_name"],
                "content": row["content"][:500],  # Truncate long messages
                "timestamp": row["timestamp"],
            })

        # Return in chronological order (oldest first)
        return list(reversed(messages))

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Fetch recent channel messages")
    parser.add_argument("channel_id", type=int, help="Discord channel ID")
    parser.add_argument("--since", type=int, help="Only messages after this message ID")
    parser.add_argument("--limit", type=int, default=10, help="Max messages to return")

    args = parser.parse_args()

    messages = get_recent_messages(args.channel_id, args.since, args.limit)
    print(json.dumps(messages, indent=2))


if __name__ == "__main__":
    main()
