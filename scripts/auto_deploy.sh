#!/bin/bash
# Auto-deploy script - polls GitHub for changes and deploys if needed
# Run via cron or systemd timer every few minutes

set -e

REPO_DIR="/home/ubuntu/hollingsbot3"
BRANCH="main"
LOG_FILE="/var/log/hollingsbot-deploy.log"

cd "$REPO_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Fetch latest from remote
git fetch origin "$BRANCH" --quiet

# Check if we're behind
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    # No changes, exit silently
    exit 0
fi

log "New commits detected: $LOCAL -> $REMOTE"
log "Pulling changes..."

git pull origin "$BRANCH"

log "Restarting containers..."
docker compose -f docker-compose.orange-pi.yml down
docker compose -f docker-compose.orange-pi.yml up -d --build

log "Deploy complete!"
