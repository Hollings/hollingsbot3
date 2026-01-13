#!/bin/bash
# deploy.sh - Deploy a website or game server from wendys_folder to wendy.monster
#
# Usage: deploy.sh <project-path> [target-url]
#
# Arguments:
#   project-path  - Path to project folder (relative to wendys_folder/)
#   target-url    - Optional. URL path on wendy.monster. Defaults to folder name.
#
# Auto-detection:
#   - Has server.ts  -> Game server deployment to wendy.monster/game/<name>/
#   - Has index.html -> Static site deployment to wendy.monster/<name>/
#
# Examples:
#   deploy.sh landing              # -> wendy.monster/landing/ (site)
#   deploy.sh landing my-site      # -> wendy.monster/my-site/ (site)
#   deploy.sh snake-game           # -> wendy.monster/game/snake-game/ (game)
#   deploy.sh games/pong pong      # -> wendy.monster/game/pong/ (game)

set -euo pipefail

PROJECT_PATH="${1:-}"
DEFAULT_TARGET=$(basename "$PROJECT_PATH")
TARGET_URL="${2:-$DEFAULT_TARGET}"
PROJECT_DIR="/data/wendy/wendys_folder/${PROJECT_PATH}"
PROXY_URL="${WENDY_PROXY_URL:-http://wendy_proxy:8080}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

success() {
    echo -e "${GREEN}$1${NC}"
}

info() {
    echo -e "${YELLOW}$1${NC}"
}

# Show usage
if [[ -z "$PROJECT_PATH" ]]; then
    echo "Usage: deploy.sh <project-path> [target-url]"
    echo ""
    echo "Arguments:"
    echo "  project-path  - Path to project folder (relative to wendys_folder/)"
    echo "  target-url    - Optional. URL path on wendy.monster. Defaults to folder name."
    echo ""
    echo "Auto-detection:"
    echo "  - Has server.ts  -> Game server at wendy.monster/game/<name>/"
    echo "  - Has index.html -> Static site at wendy.monster/<name>/"
    echo ""
    echo "Examples:"
    echo "  deploy.sh landing              # site -> wendy.monster/landing/"
    echo "  deploy.sh landing my-site      # site -> wendy.monster/my-site/"
    echo "  deploy.sh snake-game           # game -> wendy.monster/game/snake-game/"
    exit 1
fi

# Check project directory exists
if [[ ! -d "$PROJECT_DIR" ]]; then
    error "Project path '$PROJECT_PATH' not found in wendys_folder/"
fi

# Validate target URL
if [[ ! "$TARGET_URL" =~ ^[a-z0-9][a-z0-9-]{0,30}[a-z0-9]$|^[a-z0-9]$ ]]; then
    error "Target URL '$TARGET_URL' must be lowercase alphanumeric with hyphens, 1-32 chars, cannot start/end with hyphen"
fi

# Auto-detect deployment type
HAS_SERVER_TS=false
HAS_INDEX_HTML=false

[[ -f "$PROJECT_DIR/server.ts" ]] && HAS_SERVER_TS=true
[[ -f "$PROJECT_DIR/index.html" ]] && HAS_INDEX_HTML=true

if $HAS_SERVER_TS && $HAS_INDEX_HTML; then
    error "Ambiguous project - has both server.ts and index.html. Remove one to clarify deployment type."
fi

if ! $HAS_SERVER_TS && ! $HAS_INDEX_HTML; then
    error "Project needs either index.html (for static site) or server.ts (for game server)"
fi

# Set deployment type and limits
if $HAS_SERVER_TS; then
    DEPLOY_TYPE="game"
    API_ENDPOINT="/api/deploy_game"
    MAX_SIZE=10485760  # 10 MB
    MAX_SIZE_HUMAN="10MB"
    FINAL_URL="wendy.monster/game/${TARGET_URL}/"
else
    DEPLOY_TYPE="site"
    API_ENDPOINT="/api/deploy_site"
    MAX_SIZE=52428800  # 50 MB
    MAX_SIZE_HUMAN="50MB"
    FINAL_URL="wendy.monster/${TARGET_URL}/"
fi

info "Detected: ${DEPLOY_TYPE} (found $(if $HAS_SERVER_TS; then echo 'server.ts'; else echo 'index.html'; fi))"

# Create tarball
TMP_TAR="/tmp/deploy_${TARGET_URL}_$$.tar.gz"
echo "Creating tarball of $PROJECT_DIR..."
tar -czf "$TMP_TAR" -C "$PROJECT_DIR" .

# Get size
SIZE=$(stat -f%z "$TMP_TAR" 2>/dev/null || stat -c%s "$TMP_TAR" 2>/dev/null)
SIZE_KB=$((SIZE / 1024))
echo "Tarball size: ${SIZE} bytes (~${SIZE_KB}KB)"

if [[ $SIZE -gt $MAX_SIZE ]]; then
    rm -f "$TMP_TAR"
    error "Project too large (max ${MAX_SIZE_HUMAN}). Current size: ${SIZE_KB}KB"
fi

# Deploy via proxy
echo "Deploying '$PROJECT_PATH' to ${FINAL_URL}..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${PROXY_URL}${API_ENDPOINT}" \
    -F "name=${TARGET_URL}" \
    -F "files=@${TMP_TAR}")

# Parse response
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

# Clean up
rm -f "$TMP_TAR"

# Check result
if [[ "$HTTP_CODE" != "200" ]]; then
    error "Deployment failed (HTTP $HTTP_CODE): $BODY"
fi

# Extract URLs from JSON response (|| true to handle no match)
URL=$(echo "$BODY" | grep -o '"url"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/"url"[[:space:]]*:[[:space:]]*"\([^"]*\)"/\1/' || true)
WS_URL=$(echo "$BODY" | grep -o '"ws"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/"ws"[[:space:]]*:[[:space:]]*"\([^"]*\)"/\1/' || true)

success "Deployment successful!"
echo ""
echo "  Type:   ${DEPLOY_TYPE}"
echo "  Source: wendys_folder/${PROJECT_PATH}/"
if [[ -n "$URL" ]]; then
    echo "  URL:    ${URL}"
else
    echo "  URL:    https://${FINAL_URL}"
fi
if [[ -n "$WS_URL" ]]; then
    echo "  WebSocket: ${WS_URL}"
fi
echo ""
echo "Double-check: Is this what you expected to deploy?"
