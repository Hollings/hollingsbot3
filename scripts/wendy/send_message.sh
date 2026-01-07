#!/bin/bash
# Send a message to a Discord channel
# Usage: ./send_message.sh <channel_id> <message> [file_path]
# The message will be picked up by the bot and sent to Discord
#
# NOTE: This script will block if there are new messages since your last
# check_messages call. Run check_messages.sh first, or retry after seeing
# the new messages.

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./send_message.sh <channel_id> <message> [file_path]"
    exit 1
fi

export CHANNEL_ID="$1"
export MESSAGE="$2"
export FILE_PATH="${3:-}"
export OUTBOX_DIR="/data/wendy/outbox"

# Check for new messages since last check_messages call
python3 << 'CHECKEOF'
import json
import os
import sqlite3
import sys
from pathlib import Path

channel_id = int(os.environ.get('CHANNEL_ID', '0'))
STATE_FILE = Path("/data/wendy/message_check_state.json")
DB_PATH = os.getenv("PROMPT_DB_PATH", "/data/hollingsbot.db")

# Read last_seen for this channel
last_seen = None
if STATE_FILE.exists():
    try:
        state = json.loads(STATE_FILE.read_text())
        last_seen = state.get("last_seen", {}).get(str(channel_id))
    except (json.JSONDecodeError, IOError):
        pass

# If no last_seen, allow the send (first message to this channel)
if last_seen is None:
    sys.exit(0)

# Check for newer messages
db_path = Path(DB_PATH)
if not db_path.exists():
    sys.exit(0)  # Fail open if DB unavailable

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

try:
    query = """
        SELECT message_id, author_name, content
        FROM cached_messages
        WHERE channel_id = ? AND message_id > ?
        ORDER BY message_id ASC
    """
    rows = conn.execute(query, (channel_id, last_seen)).fetchall()

    if rows:
        print("ERROR: New messages since you last checked:", file=sys.stderr)
        print("", file=sys.stderr)
        for row in rows:
            print(f"  {row['author_name']}: {row['content']}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Prefer sending ONE message that responds to all users at once, rather than one message per person.", file=sys.stderr)
        print("Edit your reply to address everyone, then retry.", file=sys.stderr)

        # Auto-acknowledge: update last_seen so retry succeeds
        newest_id = max(r["message_id"] for r in rows)
        state = {}
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                state = {}
        if "last_seen" not in state:
            state["last_seen"] = {}
        state["last_seen"][str(channel_id)] = newest_id
        STATE_FILE.write_text(json.dumps(state, indent=2))

        sys.exit(1)

finally:
    conn.close()

sys.exit(0)
CHECKEOF

# Exit if check failed (new messages found)
if [ $? -ne 0 ]; then
    exit 1
fi

# Use Python to safely write JSON (handles all escaping properly)
python3 << 'PYEOF'
import json
import os
import sys
import time

channel_id = int(os.environ.get('CHANNEL_ID', '0'))
message = os.environ.get('MESSAGE', '')
file_path = os.environ.get('FILE_PATH', '')
outbox_dir = os.environ.get('OUTBOX_DIR', '/data/wendy/outbox')

# Ensure directory exists
os.makedirs(outbox_dir, exist_ok=True)

# Build JSON data
timestamp = str(int(time.time() * 1e9))
data = {
    "channel_id": channel_id,
    "message": message,
    "timestamp": timestamp
}
if file_path:
    data["file_path"] = file_path

# Write to file
filename = f"{channel_id}_{timestamp}.json"
filepath = os.path.join(outbox_dir, filename)
with open(filepath, 'w') as f:
    json.dump(data, f)

print(f"Message queued for channel {channel_id}")
PYEOF
