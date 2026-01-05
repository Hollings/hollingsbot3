#!/usr/bin/env python3
"""Migrate historical temp bot data from message_history.

This script scans message_history for temp bot arrival/departure patterns
and reconstructs historical temp_bot records.

Pattern examples:
- "*[Veiled Cipher arrives for 10 messages]*"
- "*[Hollow Echo departs]*"
- "*[Silent Whisper has depleted all replies and fades away]*"

Usage:
    python scripts/migrate_temp_bot_history.py [--dry-run]
"""

import argparse
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hollingsbot.prompt_db import DB_PATH, init_db


# Patterns for detecting temp bot messages
ARRIVAL_PATTERN = re.compile(
    r'\*\[(.+?)\s+arrives\s+for\s+(\d+)\s+messages?\]\*',
    re.IGNORECASE
)
DEPART_PATTERN = re.compile(
    r'\*\[(.+?)\s+departs\]\*',
    re.IGNORECASE
)
DEPLETE_PATTERN = re.compile(
    r'\*\[(.+?)\s+has\s+depleted\s+all\s+replies\s+and\s+fades\s+away\]\*',
    re.IGNORECASE
)


def find_temp_bot_events(conn: sqlite3.Connection) -> list[dict]:
    """Find all temp bot arrival/departure events from message_history."""
    events = []

    # Find arrival messages (webhook messages with arrival pattern)
    cur = conn.execute("""
        SELECT message_id, channel_id, guild_id, timestamp, author_id,
               author_nickname, content, is_webhook
        FROM message_history
        WHERE is_webhook = 1
          AND (content LIKE '%arrives for%messages%'
               OR content LIKE '%departs%'
               OR content LIKE '%fades away%')
        ORDER BY timestamp ASC
    """)

    for row in cur.fetchall():
        msg_id, channel_id, guild_id, timestamp, author_id, nickname, content, is_webhook = row

        if not content:
            continue

        # Check for arrival
        arrival_match = ARRIVAL_PATTERN.search(content)
        if arrival_match:
            bot_name = arrival_match.group(1).strip()
            reply_count = int(arrival_match.group(2))

            # Extract spawn_prompt from the rest of the message (after arrival announcement)
            remaining = content[arrival_match.end():].strip()
            spawn_prompt = remaining if remaining else f"(unknown - reconstructed from {bot_name})"

            events.append({
                "type": "arrival",
                "bot_name": bot_name,
                "reply_count": reply_count,
                "spawn_prompt": spawn_prompt,
                "channel_id": channel_id,
                "message_id": msg_id,
                "timestamp": timestamp,
                "webhook_id": author_id,  # For webhooks, author_id is the webhook ID
            })
            continue

        # Check for departure
        depart_match = DEPART_PATTERN.search(content)
        if depart_match:
            bot_name = depart_match.group(1).strip()
            events.append({
                "type": "departure",
                "bot_name": bot_name,
                "channel_id": channel_id,
                "message_id": msg_id,
                "timestamp": timestamp,
            })
            continue

        # Check for depletion
        deplete_match = DEPLETE_PATTERN.search(content)
        if deplete_match:
            bot_name = deplete_match.group(1).strip()
            events.append({
                "type": "depletion",
                "bot_name": bot_name,
                "channel_id": channel_id,
                "message_id": msg_id,
                "timestamp": timestamp,
            })

    return events


def reconstruct_temp_bots(events: list[dict]) -> list[dict]:
    """Match arrivals with departures to create temp bot records."""
    temp_bots = []

    # Track active bots by (channel_id, bot_name)
    active_bots: dict[tuple[int, str], dict] = {}

    for event in events:
        key = (event["channel_id"], event["bot_name"])

        if event["type"] == "arrival":
            # New bot spawned
            active_bots[key] = {
                "channel_id": event["channel_id"],
                "webhook_id": event.get("webhook_id", 0),
                "name": event["bot_name"],
                "spawn_prompt": event["spawn_prompt"],
                "replies_remaining": 0,  # Already used up
                "spawn_message_id": event["message_id"],
                "created_at": event["timestamp"],
                "deactivated_at": None,
                "is_active": False,
                "original_reply_count": event["reply_count"],
            }
        elif event["type"] in ("departure", "depletion"):
            # Bot left
            if key in active_bots:
                active_bots[key]["deactivated_at"] = event["timestamp"]
                temp_bots.append(active_bots.pop(key))
            else:
                # Departure without matching arrival - create partial record
                temp_bots.append({
                    "channel_id": event["channel_id"],
                    "webhook_id": 0,
                    "name": event["bot_name"],
                    "spawn_prompt": f"(unknown - only departure found for {event['bot_name']})",
                    "replies_remaining": 0,
                    "spawn_message_id": None,
                    "created_at": None,
                    "deactivated_at": event["timestamp"],
                    "is_active": False,
                    "original_reply_count": None,
                })

    # Any remaining active bots (no departure found) - still add them
    for bot in active_bots.values():
        temp_bots.append(bot)

    return temp_bots


def insert_historical_bots(conn: sqlite3.Connection, bots: list[dict], dry_run: bool = False) -> int:
    """Insert reconstructed temp bots into database."""
    inserted = 0

    for bot in bots:
        # Check if already exists (by name + channel + approximate time)
        cur = conn.execute("""
            SELECT id FROM temp_bots
            WHERE channel_id = ? AND LOWER(name) = LOWER(?)
              AND (created_at = ? OR spawn_message_id = ?)
        """, (
            bot["channel_id"],
            bot["name"],
            bot["created_at"],
            bot["spawn_message_id"],
        ))

        if cur.fetchone():
            print(f"  Skipping duplicate: {bot['name']} in channel {bot['channel_id']}")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would insert: {bot['name']} (channel={bot['channel_id']}, prompt={bot['spawn_prompt'][:50]}...)")
            inserted += 1
            continue

        conn.execute("""
            INSERT INTO temp_bots (
                channel_id, webhook_id, name, avatar_url, avatar_bytes,
                spawn_prompt, replies_remaining, spawn_message_id,
                is_active, created_at, deactivated_at
            ) VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, 0, ?, ?)
        """, (
            bot["channel_id"],
            bot["webhook_id"],
            bot["name"],
            bot["spawn_prompt"],
            bot["replies_remaining"],
            bot["spawn_message_id"],
            bot["created_at"],
            bot["deactivated_at"],
        ))
        inserted += 1
        print(f"  Inserted: {bot['name']} (channel={bot['channel_id']})")

    if not dry_run:
        conn.commit()

    return inserted


def main():
    parser = argparse.ArgumentParser(description="Migrate historical temp bot data")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually insert, just show what would be done")
    args = parser.parse_args()

    print(f"Database path: {DB_PATH}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Initialize database (ensures tables exist with new columns)
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        # Find events
        print("Scanning message_history for temp bot events...")
        events = find_temp_bot_events(conn)
        print(f"Found {len(events)} events")

        # Group by type
        arrivals = [e for e in events if e["type"] == "arrival"]
        departures = [e for e in events if e["type"] in ("departure", "depletion")]
        print(f"  - {len(arrivals)} arrivals")
        print(f"  - {len(departures)} departures/depletions")
        print()

        # Reconstruct bots
        print("Reconstructing temp bot records...")
        bots = reconstruct_temp_bots(events)
        print(f"Reconstructed {len(bots)} temp bot records")
        print()

        # Insert
        print("Inserting into database...")
        inserted = insert_historical_bots(conn, bots, dry_run=args.dry_run)
        print()
        print(f"{'Would insert' if args.dry_run else 'Inserted'} {inserted} records")

        # Show summary
        if not args.dry_run:
            cur = conn.execute("SELECT COUNT(*) FROM temp_bots WHERE is_active = 0")
            total_inactive = cur.fetchone()[0]
            print(f"Total inactive temp bots in database: {total_inactive}")


if __name__ == "__main__":
    main()
