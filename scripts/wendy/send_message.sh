#!/bin/bash
# Send a message to a Discord channel
# Usage: ./send_message.sh <channel_id> <message>
# The message will be picked up by the bot and sent to Discord

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./send_message.sh <channel_id> <message>"
    exit 1
fi

CHANNEL_ID="$1"
MESSAGE="$2"
OUTBOX_DIR="/data/wendy/outbox"
TIMESTAMP=$(date +%s%N)

# Ensure outbox directory exists
mkdir -p "$OUTBOX_DIR"

# Write message to outbox as JSON
cat > "$OUTBOX_DIR/${CHANNEL_ID}_${TIMESTAMP}.json" << EOF
{
    "channel_id": $CHANNEL_ID,
    "message": $(echo "$MESSAGE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
    "timestamp": "$TIMESTAMP"
}
EOF

echo "Message queued for channel $CHANNEL_ID"
