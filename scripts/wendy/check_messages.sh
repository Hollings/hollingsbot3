#!/bin/bash
# Check recent messages for a channel
# Usage: ./check_messages.sh <channel_id> [--since <message_id>]

if [ -z "$1" ]; then
    echo "Usage: ./check_messages.sh <channel_id> [--since <message_id>]"
    exit 1
fi

CHANNEL_ID="$1"
SINCE_ARG=""

if [ "$2" == "--since" ] && [ -n "$3" ]; then
    SINCE_ARG="--since $3"
fi

# Use local copy or fallback to /app/scripts
SCRIPT_DIR="$(dirname "$0")"
if [ -f "$SCRIPT_DIR/check_messages.py" ]; then
    python3 "$SCRIPT_DIR/check_messages.py" "$CHANNEL_ID" $SINCE_ARG --limit 10
else
    python3 /app/scripts/check_messages.py "$CHANNEL_ID" $SINCE_ARG --limit 10
fi
