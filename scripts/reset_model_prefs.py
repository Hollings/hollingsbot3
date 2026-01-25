#!/usr/bin/env python3
"""Reset all user model preferences to use the default model.

NOTE: This script should be run when the bot is stopped, as the database
may be locked if the bot is actively using it.
"""

import sqlite3
from pathlib import Path


def reset_model_prefs():
    """Clear all user model preferences from the database."""
    db_path = Path(__file__).parent.parent / "src" / "hollingsbot" / "prompts.db"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM model_prefs")
        rows_deleted = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"✓ Cleared {rows_deleted} user model preferences")
        print("All users will now use the default model (claude-sonnet-4-5-20250929)")
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
        print("\nThe database appears to be locked. Please:")
        print("1. Stop the bot (docker-compose down)")
        print("2. Run this script again")
        print("3. Restart the bot (docker-compose up -d)")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    reset_model_prefs()
