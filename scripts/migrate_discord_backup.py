#!/usr/bin/env python3
"""Migrate messages from discord_backup.db into hollingsbot.db message_history table.

This script imports historical Discord messages from a DiscordChatExporter backup
into the hollingsbot message_history table for archival and searchability.

Usage:
    python scripts/migrate_discord_backup.py <backup_db_path> [--dry-run]

Example:
    python scripts/migrate_discord_backup.py "/mnt/c/Users/jhol/iCloudDrive/discord_backup.db" --dry-run
    python scripts/migrate_discord_backup.py "/mnt/c/Users/jhol/iCloudDrive/discord_backup.db"
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict

# Target database path - use backup since Docker owns the original
TARGET_DB = "/mnt/c/Users/jhol/hollingsbot3/hollingsbot.db.backup-20260117"
GUILD_ID = 1020445108707000331
BATCH_SIZE = 10000


def get_user_info(backup_conn: sqlite3.Connection) -> dict:
    """Build lookup table: user_id -> (nickname, name, is_bot)."""
    cursor = backup_conn.execute(
        "SELECT id, name, nickname, is_bot FROM users"
    )
    users = {}
    for row in cursor.fetchall():
        user_id = int(row[0])
        name = row[1]
        nickname = row[2] or name  # Use nickname if available, else name
        is_bot = row[3] or 0
        users[user_id] = (nickname, name, is_bot)
    return users


def get_attachments(backup_conn: sqlite3.Connection) -> dict:
    """Build lookup table: message_id -> [attachment_urls]."""
    cursor = backup_conn.execute(
        "SELECT message_id, url FROM attachments"
    )
    attachments = defaultdict(list)
    for row in cursor.fetchall():
        message_id = int(row[0])
        url = row[1]
        attachments[message_id].append(url)
    return attachments


def get_reactions(backup_conn: sqlite3.Connection) -> dict:
    """Build lookup table: message_id -> [reaction_objects]."""
    cursor = backup_conn.execute(
        "SELECT message_id, emoji_id, emoji_name, count FROM reactions"
    )
    reactions = defaultdict(list)
    for row in cursor.fetchall():
        message_id = int(row[0])
        emoji_id = row[1]
        emoji_name = row[2]
        count = row[3]
        reactions[message_id].append({
            "emoji": emoji_name,
            "emoji_id": emoji_id,
            "count": count,
        })
    return reactions


def get_existing_message_ids(target_conn: sqlite3.Connection) -> set:
    """Get set of message IDs already in message_history."""
    cursor = target_conn.execute("SELECT message_id FROM message_history")
    return {row[0] for row in cursor.fetchall()}


def migrate(backup_db_path: str, dry_run: bool = False) -> int:
    """Migrate messages from backup to hollingsbot.db."""

    print(f"Source: {backup_db_path}")
    print(f"Target: {TARGET_DB}")
    print(f"Guild ID: {GUILD_ID}")
    print(f"Dry run: {dry_run}")
    print()

    # Connect to databases
    backup_conn = sqlite3.connect(backup_db_path)
    target_conn = sqlite3.connect(TARGET_DB)

    # Get existing message IDs to skip duplicates
    print("Loading existing message IDs from target...")
    existing_ids = get_existing_message_ids(target_conn)
    print(f"  Found {len(existing_ids)} existing messages")

    # Build lookup tables
    print("Building user lookup table...")
    users = get_user_info(backup_conn)
    print(f"  Found {len(users)} users")

    print("Building attachments lookup table...")
    attachments = get_attachments(backup_conn)
    print(f"  Found {len(attachments)} messages with attachments")

    print("Building reactions lookup table...")
    reactions = get_reactions(backup_conn)
    print(f"  Found {len(reactions)} messages with reactions")

    # Get channel info for logging
    print("Getting channel info...")
    cursor = backup_conn.execute("SELECT id, name FROM channels")
    channels = {int(row[0]): row[1] for row in cursor.fetchall()}
    print(f"  Channels: {channels}")

    # Count messages per channel
    print("\nCounting messages per channel in backup...")
    cursor = backup_conn.execute(
        "SELECT channel_id, COUNT(*) FROM messages GROUP BY channel_id"
    )
    channel_counts = {}
    for row in cursor.fetchall():
        channel_id = int(row[0])
        count = row[1]
        channel_name = channels.get(channel_id, "unknown")
        channel_counts[channel_id] = count
        print(f"  {channel_name}: {count} messages")

    # Get total count
    total_in_backup = backup_conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"\nTotal messages in backup: {total_in_backup}")

    # Count new messages (not in existing)
    new_count = 0
    cursor = backup_conn.execute("SELECT id FROM messages")
    for row in cursor.fetchall():
        msg_id = int(row[0])
        if msg_id not in existing_ids:
            new_count += 1
    print(f"New messages to import: {new_count}")

    if dry_run:
        print("\n[DRY RUN] Would import these messages. Run without --dry-run to proceed.")
        backup_conn.close()
        target_conn.close()
        return 0

    # Process messages in batches
    print(f"\nImporting messages in batches of {BATCH_SIZE}...")

    cursor = backup_conn.execute("""
        SELECT id, channel_id, author_id, timestamp, content,
               is_pinned, type, reference_message_id
        FROM messages
        ORDER BY timestamp
    """)

    imported = 0
    skipped = 0
    batch = []

    for row in cursor.fetchall():
        msg_id = int(row[0])

        # Skip if already exists
        if msg_id in existing_ids:
            skipped += 1
            continue

        channel_id = int(row[1])
        author_id = int(row[2])
        timestamp = row[3]
        content = row[4]
        # is_pinned = row[5]  # Not used in target schema
        msg_type = row[6]
        reference_message_id = int(row[7]) if row[7] else None

        # Get user info
        nickname, name, is_bot = users.get(author_id, (None, None, 0))
        author_nickname = nickname or name or str(author_id)

        # Detect webhook (message type contains "Webhook" or author has no user entry)
        is_webhook = 1 if "Webhook" in (msg_type or "") else 0

        # Get attachments and reactions
        msg_attachments = attachments.get(msg_id, [])
        msg_reactions = reactions.get(msg_id, [])

        batch.append((
            msg_id,
            channel_id,
            GUILD_ID,
            timestamp,
            author_id,
            author_nickname,
            is_bot,
            is_webhook,
            content,
            json.dumps(msg_attachments) if msg_attachments else "[]",
            reference_message_id,
            json.dumps(msg_reactions) if msg_reactions else "[]",
        ))

        if len(batch) >= BATCH_SIZE:
            target_conn.executemany("""
                INSERT OR IGNORE INTO message_history
                (message_id, channel_id, guild_id, timestamp, author_id, author_nickname,
                 is_bot, is_webhook, content, attachment_urls, reply_to_id, reactions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            target_conn.commit()
            imported += len(batch)
            print(f"  Imported {imported} messages...")
            batch = []

    # Insert remaining batch
    if batch:
        target_conn.executemany("""
            INSERT OR IGNORE INTO message_history
            (message_id, channel_id, guild_id, timestamp, author_id, author_nickname,
             is_bot, is_webhook, content, attachment_urls, reply_to_id, reactions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        target_conn.commit()
        imported += len(batch)

    print("\nMigration complete!")
    print(f"  Imported: {imported}")
    print(f"  Skipped (already existed): {skipped}")

    # Verify counts
    print("\nVerifying message counts per channel in target...")
    cursor = target_conn.execute("""
        SELECT channel_id, COUNT(*)
        FROM message_history
        WHERE guild_id = ?
        GROUP BY channel_id
    """, (GUILD_ID,))
    for row in cursor.fetchall():
        channel_id = row[0]
        count = row[1]
        channel_name = channels.get(channel_id, "unknown")
        print(f"  {channel_name}: {count} messages")

    total_in_target = target_conn.execute(
        "SELECT COUNT(*) FROM message_history WHERE guild_id = ?",
        (GUILD_ID,)
    ).fetchone()[0]
    print(f"\nTotal messages in target for guild: {total_in_target}")

    backup_conn.close()
    target_conn.close()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Migrate discord_backup.db into hollingsbot.db message_history"
    )
    parser.add_argument(
        "backup_db_path",
        help="Path to the discord_backup.db file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without actually importing"
    )
    args = parser.parse_args()

    return migrate(args.backup_db_path, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
