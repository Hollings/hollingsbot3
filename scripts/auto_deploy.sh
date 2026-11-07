#!/bin/bash
# Poll main and retry until the target is both running and ready.
set -euo pipefail

REPO_DIR="${HOLLINGSBOT_DEPLOY_REPO:-/home/ubuntu/hollingsbot3}"
BRANCH=main
LOG_FILE="${HOLLINGSBOT_DEPLOY_LOG:-/var/log/hollingsbot-deploy.log}"
READY_TIMEOUT="${HOLLINGSBOT_DEPLOY_READY_TIMEOUT:-120}"
COMPOSE_FILE=docker-compose.orange-pi.yml

cd "$REPO_DIR"
REPO_DIR="$PWD"
GIT_DIR=$(git rev-parse --absolute-git-dir)
HEALTHY_SHA="$GIT_DIR/hollingsbot-healthy-sha"
exec 9>"$GIT_DIR/hollingsbot-deploy.lock"
if flock -n -E 75 9; then
    :
else
    result=$?
    if (( result == 75 )); then exit 0; fi
    exit "$result"
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

git fetch origin "$BRANCH" --quiet
LOCAL=$(git rev-parse HEAD)
TARGET=$(git rev-parse "origin/$BRANCH")
if [[ "$LOCAL" == "$TARGET" && -f "$HEALTHY_SHA" && $(<"$HEALTHY_SHA") == "$TARGET" ]]; then
    exit 0
fi

if ! git merge-base --is-ancestor "$LOCAL" "$TARGET"; then
    log "Cannot deploy $TARGET: checkout is not a fast-forward."
    exit 1
fi

compose=(docker compose)
if [[ -f "$REPO_DIR/.env" ]]; then
    # Preserve interpolation (including build args) without copying runtime
    # credentials into the candidate build context.
    compose+=(--env-file "$REPO_DIR/.env")
fi
PROJECT_NAME=$("${compose[@]}" -f "$COMPOSE_FILE" config --format json \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["name"])')
compose+=(--project-name "$PROJECT_NAME" -f "$COMPOSE_FILE")
candidate=$(mktemp -d)
stamp=''
cleanup() {
    rm -rf -- "$candidate"
    if [[ -n "$stamp" ]]; then rm -f -- "$stamp"; fi
}
trap cleanup EXIT

log "Building $TARGET before changing the running checkout ($LOCAL)."
# The running services bind-mount the checkout, and the bot watches Python
# files there. Do not move HEAD until the candidate images have built.
git archive "$TARGET" | tar -xf - -C "$candidate"
# Keep any Pi-local Compose settings that also apply to the live checkout.
overrides=$(git diff --binary HEAD -- "$COMPOSE_FILE")
if [[ -n "$overrides" ]] && ! printf '%s\n' "$overrides" | (cd "$candidate"; git apply -); then
    log "Local Compose overrides do not apply to $TARGET; leaving the checkout unchanged."
    exit 1
fi
# Compose needs the env_file present when it loads the build configuration;
# a build does not need runtime secrets in its context.
touch "$candidate/.env"
if ! (cd "$candidate"; "${compose[@]}" build bot celery_text celery_image); then
    log "Build failed for $TARGET; the running checkout was not changed."
    exit 1
fi

if ! git merge --ff-only "$TARGET"; then
    log "Could not fast-forward the live checkout; local changes were kept."
    exit 1
fi
log "Starting $TARGET; leaving Redis running if unchanged."
"${compose[@]}" up -d --no-build redis
# Recreate the application services even on a retry of the same SHA, which
# may otherwise leave an unhealthy container in place. Do not restart Redis.
"${compose[@]}" up -d --no-build --no-deps --force-recreate bot celery_text celery_image

service_ready() {
    local service=$1 marker=$2 container started
    container=$(timeout 10s "${compose[@]}" ps -q "$service") && [[ -n "$container" ]] || return 1
    started=$(timeout 10s docker inspect --format '{{if .State.Running}}{{.State.StartedAt}}{{end}}' "$container") \
        && [[ -n "$started" ]] || return 1
    timeout 10s docker logs --since "$started" "$container" 2>&1 | grep -F "$marker" >/dev/null
}

ready() {
    local redis
    redis=$(timeout 10s "${compose[@]}" exec -T redis redis-cli ping) && [[ "$redis" == PONG ]] \
        && service_ready bot 'Bot is ready. Logged in as' \
        && service_ready celery_text ' ready.' \
        && service_ready celery_image ' ready.'
}

deadline=$((SECONDS + READY_TIMEOUT))
until ready; do
    if (( SECONDS >= deadline )); then
        log "Readiness failed for $TARGET; the next timer tick will retry."
        exit 1
    fi
    sleep 2
done

stamp=$(mktemp "$GIT_DIR/hollingsbot-healthy-sha.XXXXXX")
printf '%s\n' "$TARGET" >"$stamp"
mv -f -- "$stamp" "$HEALTHY_SHA"
log "Deploy complete and ready: $TARGET"
