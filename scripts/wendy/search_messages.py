#!/usr/bin/env python3
"""Search message history in the database.

Usage:
    python search_messages.py <channel_id> --search "keyword"
    python search_messages.py <channel_id> --author "username"
    python search_messages.py <channel_id> --recent 50
    python search_messages.py --all-channels --search "keyword"

Examples:
    python search_messages.py 1050900592031178752 --search "hello"
    python search_messages.py 1050900592031178752 --author "John" --limit 20
    python search_messages.py --all-channels --search "important" --limit 100
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime

DB_PATH = "/data/hollingsbot.db"


def search_messages(channel_id=None, search=None, author=None, recent=None, limit=50):
    """Search messages with various filters."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    conditions = []
    params = []

    if channel_id:
        conditions.append("channel_id = ?")
        params.append(channel_id)

    if search:
        conditions.append("content LIKE ?")
        params.append(f"%{search}%")

    if author:
        conditions.append("author_name LIKE ?")
        params.append(f"%{author}%")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT message_id, channel_id, author_name, content, timestamp,
               datetime(timestamp/1000, 'unixepoch') as time_str
        FROM cached_messages
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT ?
    """
    params.append(limit)

    try:
        cursor = conn.execute(query, params)
        results = []
        for row in cursor:
            results.append({
                "message_id": row["message_id"],
                "channel_id": row["channel_id"],
                "author": row["author_name"],
                "content": row["content"][:500] if row["content"] else "",
                "time": row["time_str"],
            })
        return results
    finally:
        conn.close()


def get_channel_stats():
    """Get message counts per channel."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT channel_id, COUNT(*) as count,
               MIN(datetime(timestamp/1000, 'unixepoch')) as oldest,
               MAX(datetime(timestamp/1000, 'unixepoch')) as newest
        FROM cached_messages
        GROUP BY channel_id
        ORDER BY count DESC
    """

    try:
        cursor = conn.execute(query)
        return [dict(row) for row in cursor]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Search message history")
    parser.add_argument("channel_id", nargs="?", type=int, help="Channel ID to search")
    parser.add_argument("--search", "-s", help="Search for keyword in message content")
    parser.add_argument("--author", "-a", help="Filter by author name")
    parser.add_argument("--recent", "-r", type=int, help="Get N most recent messages")
    parser.add_argument("--limit", "-l", type=int, default=50, help="Max results (default 50)")
    parser.add_argument("--all-channels", action="store_true", help="Search all channels")
    parser.add_argument("--stats", action="store_true", help="Show channel statistics")

    args = parser.parse_args()

    if args.stats:
        stats = get_channel_stats()
        print(json.dumps(stats, indent=2))
        return

    if not args.channel_id and not args.all_channels:
        print("Error: Provide channel_id or use --all-channels")
        parser.print_help()
        sys.exit(1)

    channel_id = None if args.all_channels else args.channel_id
    limit = args.recent if args.recent else args.limit

    results = search_messages(
        channel_id=channel_id,
        search=args.search,
        author=args.author,
        limit=limit,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
