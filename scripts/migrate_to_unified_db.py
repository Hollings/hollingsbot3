#!/usr/bin/env python3
"""Migrate summaries.db tables into the unified hollingsbot.db.

Run this INSIDE the Docker container after renaming prompts.db to hollingsbot.db.

Usage:
    docker compose exec bot python scripts/migrate_to_unified_db.py
"""

import os
import sqlite3
import sys

OLD_SUMMARY_DB = "/app/data/summaries.db"
NEW_DB = os.getenv("PROMPT_DB_PATH", "/data/hollingsbot.db")


def migrate():
    """Copy summary tables from old summaries.db into unified database."""

    # Check if old summary DB exists
    if not os.path.exists(OLD_SUMMARY_DB):
        print(f"Old summary DB not found at {OLD_SUMMARY_DB}, skipping migration")
        return 0

    # Check if new DB exists
    if not os.path.exists(NEW_DB):
        print(f"New DB not found at {NEW_DB}")
        print("Make sure to rename prompts.db to hollingsbot.db first!")
        return 1

    print(f"Migrating from {OLD_SUMMARY_DB} to {NEW_DB}")

    old_conn = sqlite3.connect(OLD_SUMMARY_DB)
    new_conn = sqlite3.connect(NEW_DB)

    tables = ["message_groups", "cached_messages", "channel_clear_points"]
    total_migrated = 0

    for table in tables:
        try:
            # Check if table exists in old DB
            cursor = old_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if not cursor.fetchone():
                print(f"  Table {table} not found in old DB, skipping")
                continue

            # Get row count
            count = old_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count == 0:
                print(f"  Table {table}: 0 rows, skipping")
                continue

            # Get column info
            cursor = old_conn.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            col_str = ", ".join(columns)
            placeholders = ", ".join("?" * len(columns))

            # Copy rows
            rows = old_conn.execute(f"SELECT {col_str} FROM {table}").fetchall()

            # Insert into new DB (ignore conflicts)
            for row in rows:
                try:
                    new_conn.execute(f"INSERT OR IGNORE INTO {table} ({col_str}) VALUES ({placeholders})", row)
                except sqlite3.IntegrityError:
                    pass  # Skip duplicates

            new_conn.commit()
            print(f"  Table {table}: migrated {count} rows")
            total_migrated += count

        except Exception as e:
            print(f"  Error migrating {table}: {e}")

    old_conn.close()
    new_conn.close()

    print(f"\nMigration complete! Total rows: {total_migrated}")
    print(f"\nYou can now move {OLD_SUMMARY_DB} to old_dbs/")
    return 0


if __name__ == "__main__":
    sys.exit(migrate())
