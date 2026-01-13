#!/bin/bash
# game_logs.sh - Get logs from a deployed game server
#
# Usage: game_logs.sh <game-name> [lines]
#
# Examples:
#   game_logs.sh multiplayer-demo        # last 100 lines
#   game_logs.sh multiplayer-demo 50     # last 50 lines

set -euo pipefail

GAME_NAME="${1:-}"
LINES="${2:-100}"
PROXY_URL="${WENDY_PROXY_URL:-http://wendy_proxy:8080}"

if [[ -z "$GAME_NAME" ]]; then
    echo "Usage: game_logs.sh <game-name> [lines]"
    echo ""
    echo "Examples:"
    echo "  game_logs.sh multiplayer-demo"
    echo "  game_logs.sh multiplayer-demo 50"
    exit 1
fi

# Fetch logs
RESPONSE=$(curl -s "${PROXY_URL}/api/game_logs/${GAME_NAME}?lines=${LINES}")

# Extract and display logs
echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"=== Logs for {data.get('name', 'unknown')} ===\")
    print(data.get('logs', 'No logs found'))
except:
    print('Failed to parse response')
"
