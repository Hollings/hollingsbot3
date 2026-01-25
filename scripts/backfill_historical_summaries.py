#!/usr/bin/env python3
"""Backfill historical message summaries using Claude Haiku.

Creates L3 summaries for old messages (before SUMMARIZATION_CUTOFF_TIMESTAMP)
that aren't covered by the normal summarization system.

Run inside Docker:
    docker compose exec bot python scripts/backfill_historical_summaries.py --limit 4
    docker compose exec bot python scripts/backfill_historical_summaries.py --dry-run
    docker compose exec bot python scripts/backfill_historical_summaries.py

"""

import argparse
import asyncio
import sqlite3

# Add src to path for imports
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hollingsbot.prompt_db import DB_PATH
from hollingsbot.summarization.summary_cache import (
    SUMMARIZATION_CUTOFF_TIMESTAMP,
)
from hollingsbot.text_generators import AnthropicTextGenerator

CHUNK_SIZE = 250
MODEL = "claude-haiku-4-5"


def get_uncovered_old_messages(conn: sqlite3.Connection, channel_id: int) -> list[dict]:
    """Get messages before cutoff not covered by existing L3 groups.

    Returns messages in chronological order (oldest first).
    """
    # Get all existing L3 group ranges for this channel
    cur = conn.execute(
        """
        SELECT start_message_id, end_message_id
        FROM message_groups
        WHERE channel_id = ? AND level = 3
        """,
        (channel_id,),
    )
    covered_ranges = [(row[0], row[1]) for row in cur.fetchall()]

    # Get old messages (before cutoff)
    cur = conn.execute(
        """
        SELECT message_id, author_name, content, timestamp
        FROM cached_messages
        WHERE channel_id = ? AND timestamp < ?
        ORDER BY message_id ASC
        """,
        (channel_id, SUMMARIZATION_CUTOFF_TIMESTAMP),
    )

    messages = []
    for row in cur.fetchall():
        msg_id = row[0]
        # Check if this message is covered by any existing L3 group
        is_covered = any(start <= msg_id <= end for start, end in covered_ranges)
        if not is_covered:
            messages.append(
                {
                    "message_id": msg_id,
                    "author_name": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                }
            )

    return messages


def chunk_messages(messages: list[dict], size: int = CHUNK_SIZE) -> list[list[dict]]:
    """Split messages into chunks of specified size."""
    chunks = []
    for i in range(0, len(messages), size):
        chunk = messages[i : i + size]
        if len(chunk) >= size // 2:  # Only include chunks that are at least half-full
            chunks.append(chunk)
    return chunks


def format_chunk_for_summary(messages: list[dict]) -> str:
    """Format messages as chat log text."""
    lines = []
    for msg in messages:
        author = msg["author_name"]
        content = msg["content"] or ""

        # Strip the <Author>: prefix if present
        if content.startswith(f"<{author}>:"):
            content = content[len(f"<{author}>:") :].strip()

        # Truncate very long messages
        if len(content) > 400:
            content = content[:400] + "..."

        lines.append(f"{author}: {content}")

    return "\n".join(lines)


async def summarize_with_haiku(generator: AnthropicTextGenerator, chat_text: str) -> str:
    """Use claude-haiku-4-5 to summarize."""
    prompt = f"""Write a 3 sentence summary of the most important events in this Discord chat log. Be concise and focus on key topics, memorable moments, and who was involved.

Chat log:
{chat_text}

Summary:"""
    return await generator.generate(prompt, temperature=0.7)


def insert_l3_summary(
    conn: sqlite3.Connection,
    channel_id: int,
    start_id: int,
    end_id: int,
    summary: str,
    msg_count: int,
    start_ts: int,
    end_ts: int,
) -> int:
    """Insert L3 summary into message_groups. Returns the new group ID."""
    cur = conn.execute(
        """
        INSERT INTO message_groups (
            channel_id, level, start_message_id, end_message_id,
            summary_text, message_count, start_timestamp, end_timestamp, created_at
        ) VALUES (?, 3, ?, ?, ?, ?, ?, ?, ?)
        """,
        (channel_id, start_id, end_id, summary, msg_count, start_ts, end_ts, int(time.time())),
    )
    conn.commit()
    return cur.lastrowid


def get_all_channels_with_old_messages(conn: sqlite3.Connection) -> list[int]:
    """Get all channel IDs that have messages before the cutoff."""
    cur = conn.execute(
        """
        SELECT DISTINCT channel_id
        FROM cached_messages
        WHERE timestamp < ?
        ORDER BY channel_id
        """,
        (SUMMARIZATION_CUTOFF_TIMESTAMP,),
    )
    return [row[0] for row in cur.fetchall()]


async def main():
    parser = argparse.ArgumentParser(description="Backfill historical message summaries")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--limit", type=int, help="Limit number of chunks to process")
    parser.add_argument("--channel", type=int, help="Process a specific channel only")
    args = parser.parse_args()

    print(f"Database: {DB_PATH}")
    print(f"Cutoff timestamp: {SUMMARIZATION_CUTOFF_TIMESTAMP}")
    print(f"Chunk size: {CHUNK_SIZE} messages")
    print(f"Model: {MODEL}")
    print(f"Dry run: {args.dry_run}")
    print(f"Limit: {args.limit or 'None'}")
    print()

    generator = AnthropicTextGenerator(model=MODEL)

    with sqlite3.connect(DB_PATH) as conn:
        # Get channels to process
        if args.channel:
            channels = [args.channel]
        else:
            channels = get_all_channels_with_old_messages(conn)

        print(f"Found {len(channels)} channel(s) with old messages")
        print()

        total_chunks = 0
        total_messages = 0
        chunks_processed = 0
        errors = 0

        for channel_id in channels:
            # Get uncovered old messages
            messages = get_uncovered_old_messages(conn, channel_id)
            if not messages:
                continue

            chunks = chunk_messages(messages, CHUNK_SIZE)
            if not chunks:
                continue

            print(f"Channel {channel_id}: {len(messages)} uncovered messages -> {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                if args.limit and chunks_processed >= args.limit:
                    print(f"\nReached limit of {args.limit} chunks, stopping.")
                    break

                start_id = chunk[0]["message_id"]
                end_id = chunk[-1]["message_id"]
                start_ts = chunk[0]["timestamp"]
                end_ts = chunk[-1]["timestamp"]
                msg_count = len(chunk)

                print(f"  Chunk {i + 1}/{len(chunks)}: {msg_count} messages ({start_id} -> {end_id})")

                # Format for summary
                chat_text = format_chunk_for_summary(chunk)

                # Truncate if too long (keep it reasonable for Haiku)
                if len(chat_text) > 15000:
                    chat_text = chat_text[:15000] + "\n... (truncated)"

                try:
                    # Generate summary
                    summary = await summarize_with_haiku(generator, chat_text)
                    print(f"    Summary: {summary[:100]}...")

                    if not args.dry_run:
                        group_id = insert_l3_summary(
                            conn, channel_id, start_id, end_id, summary, msg_count, start_ts, end_ts
                        )
                        print(f"    Saved as group ID {group_id}")
                    else:
                        print("    [DRY RUN] Would save summary")

                    chunks_processed += 1
                    total_messages += msg_count

                    # Small delay to be nice to the API
                    await asyncio.sleep(0.3)

                except Exception as e:
                    print(f"    Error: {e}")
                    errors += 1

                total_chunks += 1

            if args.limit and chunks_processed >= args.limit:
                break

            print()

        print("=" * 60)
        print(f"Chunks processed: {chunks_processed}")
        print(f"Messages covered: {total_messages}")
        print(f"Errors: {errors}")


if __name__ == "__main__":
    asyncio.run(main())
