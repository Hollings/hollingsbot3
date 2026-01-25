#!/usr/bin/env python3
"""Fetch recent messages from the database for a channel.

Usage: python check_messages.py <channel_id> [--since <message_id>] [--limit <n>]

Returns JSON array of recent messages, newest first.
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path

# Default database path
DB_PATH = os.getenv("PROMPT_DB_PATH", "/data/hollingsbot.db")
STATE_FILE = Path("/data/wendy/message_check_state.json")


def get_last_seen(channel_id: int) -> int | None:
    """Get the last seen message_id for a channel."""
    if not STATE_FILE.exists():
        return None
    try:
        state = json.loads(STATE_FILE.read_text())
        return state.get("last_seen", {}).get(str(channel_id))
    except (OSError, json.JSONDecodeError):
        return None


def update_last_seen(channel_id: int, message_id: int) -> None:
    """Update the last seen message_id for a channel."""
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            state = {}

    if "last_seen" not in state:
        state["last_seen"] = {}

    state["last_seen"][str(channel_id)] = message_id
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def find_attachments_for_message(message_id: int) -> list[str]:
    """Find attachment files for a message ID.

    Looks in /data/wendy/attachments/ for files named: msg_{id}_{idx}_{filename}
    """
    attachments_dir = Path("/data/wendy/attachments")
    if not attachments_dir.exists():
        return []

    matching = []
    for att_file in attachments_dir.glob(f"msg_{message_id}_*"):
        matching.append(str(att_file))

    return sorted(matching)


def get_recent_messages(channel_id: int, since_id: int | None = None, limit: int = 10) -> list[dict]:
    """Fetch recent messages from the cached_messages table, with local image paths."""
    db_path = Path(DB_PATH)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        if since_id:
            # Get messages newer than since_id (excluding Wendy's own messages and commands)
            query = """
                SELECT message_id, channel_id, author_name, content, timestamp, has_images
                FROM cached_messages
                WHERE channel_id = ? AND message_id > ?
                AND LOWER(author_name) NOT LIKE '%wendy%'
                AND LOWER(author_name) NOT LIKE '%hollingsbot%'
                AND content NOT LIKE '!spawn%'
                AND content NOT LIKE '-%'
                ORDER BY message_id DESC
                LIMIT ?
            """
            rows = conn.execute(query, (channel_id, since_id, limit)).fetchall()
        else:
            # Get last N messages (excluding Wendy's own messages and commands)
            query = """
                SELECT message_id, channel_id, author_name, content, timestamp, has_images
                FROM cached_messages
                WHERE channel_id = ?
                AND LOWER(author_name) NOT LIKE '%wendy%'
                AND LOWER(author_name) NOT LIKE '%hollingsbot%'
                AND content NOT LIKE '!spawn%'
                AND content NOT LIKE '-%'
                ORDER BY message_id DESC
                LIMIT ?
            """
            rows = conn.execute(query, (channel_id, limit)).fetchall()

        messages = []
        for row in rows:
            msg = {
                "message_id": row["message_id"],
                "author": row["author_name"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            }
            # Find local attachment files for this message
            attachments = find_attachments_for_message(row["message_id"])
            if attachments:
                msg["attachments"] = attachments
            messages.append(msg)

        # Return in chronological order (oldest first)
        return list(reversed(messages))

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Fetch recent channel messages")
    parser.add_argument("channel_id", type=int, help="Discord channel ID")
    parser.add_argument("--since", type=int, help="Only messages after this message ID")
    parser.add_argument("--limit", type=int, default=10, help="Max messages to return")
    parser.add_argument("--all", action="store_true", help="Get all recent messages ignoring last_seen")

    args = parser.parse_args()

    # Use stored last_seen if no --since provided (unless --all flag)
    since_id = args.since
    if since_id is None and not args.all:
        since_id = get_last_seen(args.channel_id)

    messages = get_recent_messages(args.channel_id, since_id, args.limit)
    print(json.dumps(messages, indent=2))

    # Update last_seen with the newest message_id
    if messages:
        newest_id = max(m["message_id"] for m in messages)
        update_last_seen(args.channel_id, newest_id)


if __name__ == "__main__":
    main()
