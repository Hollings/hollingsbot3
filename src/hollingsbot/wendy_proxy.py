"""Wendy Proxy API - Sandboxed endpoints for send_message, check_messages, and deploy_site.

This service acts as a proxy so Wendy (running in Claude CLI) can send messages,
check for new messages, and deploy sites without having direct access to the
Discord token or other sensitive environment variables.
"""

import json
import os
import sqlite3
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Wendy Proxy API")

# Configuration
DB_PATH = os.getenv("PROMPT_DB_PATH", "/data/hollingsbot.db")
OUTBOX_DIR = Path("/data/wendy/outbox")
STATE_FILE = Path("/data/wendy/message_check_state.json")
ATTACHMENTS_DIR = Path("/data/wendy/attachments")


class SendMessageRequest(BaseModel):
    channel_id: str
    content: str | None = None  # Alias for message
    message: str | None = None  # Legacy field name
    attachment: str | None = None


class SendMessageResponse(BaseModel):
    success: bool
    message: str


class MessageInfo(BaseModel):
    message_id: int
    author: str
    content: str
    timestamp: int | str
    attachments: list[str] | None = None


# ==================== State Management ====================


def get_last_seen(channel_id: int) -> int | None:
    """Get the last seen message_id for a channel."""
    if not STATE_FILE.exists():
        return None
    try:
        state = json.loads(STATE_FILE.read_text())
        return state.get("last_seen", {}).get(str(channel_id))
    except (OSError, json.JSONDecodeError):
        return None


def update_last_seen(channel_id: int, message_id: int) -> None:
    """Update the last seen message_id for a channel."""
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            state = {}

    if "last_seen" not in state:
        state["last_seen"] = {}

    state["last_seen"][str(channel_id)] = message_id
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ==================== Attachments ====================


def find_attachments_for_message(message_id: int) -> list[str]:
    """Find attachment files for a message ID."""
    if not ATTACHMENTS_DIR.exists():
        return []

    matching = []
    for att_file in ATTACHMENTS_DIR.glob(f"msg_{message_id}_*"):
        matching.append(str(att_file))

    return sorted(matching)


# ==================== Endpoints ====================


@app.post("/api/send_message", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    """Send a message to a Discord channel via the outbox."""
    try:
        OUTBOX_DIR.mkdir(parents=True, exist_ok=True)

        # Validate attachment path if provided
        if request.attachment:
            att_path = Path(request.attachment)
            # Only allow attachments from Wendy's directories
            allowed_prefixes = ["/data/wendy/", "/tmp/"]
            if not any(request.attachment.startswith(p) for p in allowed_prefixes):
                raise HTTPException(
                    status_code=400, detail=f"Attachment must be in /data/wendy/ or /tmp/, got: {request.attachment}"
                )
            if not att_path.exists():
                raise HTTPException(status_code=400, detail=f"Attachment file not found: {request.attachment}")

        # Create outbox message
        timestamp_ns = time.time_ns()
        filename = f"{request.channel_id}_{timestamp_ns}.json"

        # Use content or message field (content preferred)
        msg_text = request.content or request.message or ""

        message_data = {
            "channel_id": request.channel_id,
            "message": msg_text,  # outbox expects "message" not "content"
        }
        if request.attachment:
            message_data["file_path"] = request.attachment  # outbox expects "file_path"

        outbox_path = OUTBOX_DIR / filename
        outbox_path.write_text(json.dumps(message_data))

        return SendMessageResponse(success=True, message=f"Message queued: {filename}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/check_messages/{channel_id}")
async def check_messages(channel_id: int, limit: int = 10, all_messages: bool = False) -> list[MessageInfo]:
    """Check for new messages in a channel."""
    try:
        db_path = Path(DB_PATH)
        if not db_path.exists():
            return []

        # Get since_id from state unless all_messages is True
        since_id = None if all_messages else get_last_seen(channel_id)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            if since_id:
                query = """
                    SELECT message_id, channel_id, author_name, content, timestamp, has_images
                    FROM cached_messages
                    WHERE channel_id = ? AND message_id > ?
                    AND LOWER(author_name) NOT LIKE '%wendy%'
                    AND LOWER(author_name) NOT LIKE '%hollingsbot%'
                    AND content NOT LIKE '!spawn%'
                    AND content NOT LIKE '-%'
                    ORDER BY message_id DESC
                    LIMIT ?
                """
                rows = conn.execute(query, (channel_id, since_id, limit)).fetchall()
            else:
                query = """
                    SELECT message_id, channel_id, author_name, content, timestamp, has_images
                    FROM cached_messages
                    WHERE channel_id = ?
                    AND LOWER(author_name) NOT LIKE '%wendy%'
                    AND LOWER(author_name) NOT LIKE '%hollingsbot%'
                    AND content NOT LIKE '!spawn%'
                    AND content NOT LIKE '-%'
                    ORDER BY message_id DESC
                    LIMIT ?
                """
                rows = conn.execute(query, (channel_id, limit)).fetchall()

            messages = []
            for row in rows:
                attachments = find_attachments_for_message(row["message_id"])
                msg = MessageInfo(
                    message_id=row["message_id"],
                    author=row["author_name"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    attachments=attachments if attachments else None,
                )
                messages.append(msg)

            # Return in chronological order (oldest first)
            messages = list(reversed(messages))

            # Update last_seen with the newest message_id
            if messages:
                newest_id = max(m.message_id for m in messages)
                update_last_seen(channel_id, newest_id)

            return messages

        finally:
            conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ==================== Site Deployment ====================

# Configuration for wendy-sites service
WENDY_SITES_URL = os.getenv("WENDY_SITES_URL", "http://100.120.250.100:8910")
WENDY_DEPLOY_TOKEN = os.getenv("WENDY_DEPLOY_TOKEN", "")

# Configuration for wendy-games service
WENDY_GAMES_URL = os.getenv("WENDY_GAMES_URL", "http://100.120.250.100:8920")
WENDY_GAMES_TOKEN = os.getenv("WENDY_GAMES_TOKEN", "")


class DeploySiteResponse(BaseModel):
    success: bool
    url: str | None = None
    message: str


@app.post("/api/deploy_site", response_model=DeploySiteResponse)
async def deploy_site(
    name: str = Form(...),
    files: UploadFile = File(...),
):
    """Deploy a site to wendy.monster.

    This endpoint proxies the request to the wendy-sites service,
    adding the deploy token from the environment (never exposed to Claude CLI).

    Args:
        name: Site name (lowercase alphanumeric with hyphens)
        files: Tarball (.tar.gz) of site files
    """
    if not WENDY_DEPLOY_TOKEN:
        raise HTTPException(status_code=500, detail="WENDY_DEPLOY_TOKEN not configured on server")

    try:
        # Read the uploaded file
        content = await files.read()

        # Forward to wendy-sites service with auth token
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{WENDY_SITES_URL}/api/deploy",
                data={"name": name},
                files={"files": ("site.tar.gz", content, "application/gzip")},
                headers={"Authorization": f"Bearer {WENDY_DEPLOY_TOKEN}"},
            )

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("detail", error_detail)
            except Exception:
                pass
            raise HTTPException(status_code=response.status_code, detail=f"Deploy failed: {error_detail}")

        result = response.json()
        return DeploySiteResponse(
            success=True,
            url=result.get("url"),
            message=result.get("message", "Site deployed successfully"),
        )

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to connect to wendy-sites service: {e!s}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Game Deployment ====================


class DeployGameResponse(BaseModel):
    success: bool
    url: str | None = None
    ws: str | None = None
    port: int | None = None
    message: str


@app.get("/api/game_logs/{name}")
async def get_game_logs(name: str, lines: int = 100):
    """Get logs from a game server."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{WENDY_GAMES_URL}/api/games/{name}/logs",
                params={"lines": lines},
                headers={"Authorization": f"Bearer {WENDY_GAMES_TOKEN}"},
            )

        if response.status_code == 404:
            return {"name": name, "logs": f"Game '{name}' not found"}

        if response.status_code != 200:
            return {"name": name, "logs": f"Error: {response.text}"}

        return response.json()

    except httpx.RequestError as e:
        return {"name": name, "logs": f"Connection error: {e!s}"}


@app.post("/api/deploy_game", response_model=DeployGameResponse)
async def deploy_game(
    name: str = Form(...),
    files: UploadFile = File(...),
):
    """Deploy a multiplayer game backend to wendy.monster.

    This endpoint proxies the request to the wendy-games manager service,
    adding the deploy token from the environment (never exposed to Claude CLI).

    Args:
        name: Game name (lowercase alphanumeric with hyphens)
        files: Tarball (.tar.gz) containing server.ts
    """
    if not WENDY_GAMES_TOKEN:
        raise HTTPException(status_code=500, detail="WENDY_GAMES_TOKEN not configured on server")

    try:
        # Read the uploaded file
        content = await files.read()

        # Forward to wendy-games service with auth token
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{WENDY_GAMES_URL}/api/deploy",
                data={"name": name},
                files={"files": ("game.tar.gz", content, "application/gzip")},
                headers={"Authorization": f"Bearer {WENDY_GAMES_TOKEN}"},
            )

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("detail", error_detail)
            except Exception:
                pass
            raise HTTPException(status_code=response.status_code, detail=f"Deploy failed: {error_detail}")

        result = response.json()
        return DeployGameResponse(
            success=True,
            url=result.get("url"),
            ws=result.get("ws"),
            port=result.get("port"),
            message=result.get("message", "Game deployed successfully"),
        )

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to connect to wendy-games service: {e!s}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
