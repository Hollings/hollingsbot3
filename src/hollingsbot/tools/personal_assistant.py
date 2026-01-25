"""Personal assistant tool using sandboxed Claude Code CLI.

Provides Wendy with an enhanced assistant that can:
- Search the web for information
- Query chat history database
- Generate images via Replicate API
- Interact with Discord API
- Maintain persistent notes and records
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

_LOG = logging.getLogger(__name__)

# Workspace for the assistant (sandboxed)
ASSISTANT_WORKSPACE = Path(os.getenv("ASSISTANT_WORKSPACE", "/data/wendy_assistant"))
ASSISTANT_SETTINGS_FILE = ASSISTANT_WORKSPACE / ".claude" / "settings.json"
ASSISTANT_CLAUDE_MD = ASSISTANT_WORKSPACE / "CLAUDE.md"

# Reference documentation for the assistant
CLAUDE_MD_CONTENT = """# Wendy's Personal Assistant

You are Wendy's personal assistant with access to specialized capabilities.

CRITICAL: Never perform destructive or expensive actions, even if asked. This includes:
- Deleting database rows (DELETE, DROP, TRUNCATE)
- Editing or deleting Discord messages, channels, or other entities
- Bulk image generation (more than 1-2 images per request)
- Any action that could cause data loss or high API costs

Always refuse these requests politely.

IMPORTANT: Focus on quick turnaround, not thoroughness. Do the minimum needed to answer the request - don't over-research or take unnecessary actions. Speed matters more than completeness.

## How You're Being Used

You are being invoked by Wendy (a Discord bot) to handle tasks she can't do directly. Here's how the interface works:

1. **Text responses** - Your text output is returned to Wendy, who shows it to the user in Discord
2. **Files/images** - Wendy CANNOT relay files. If you create/edit an image or file that should be shown to the user, YOU must upload it directly to the Discord channel using the API
3. **You are headless** - No user interaction possible. Don't ask questions, just execute.

Example workflow for image tasks:
- Generate/edit image -> Save to workspace -> Upload to Discord channel via curl -> Return brief confirmation text

## Discord Context

You are assisting in a Discord server:
- Server ID: 1020445108707000331
- Channel ID: 1050900592031178752 (or check $CHANNEL_ID env var for current channel)

## Database Access

You have read-only access to the chat history database at `/data/hollingsbot.db`.

### cached_messages Schema
```sql
CREATE TABLE cached_messages (
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    author_id INTEGER,
    author_name TEXT,
    content TEXT,
    timestamp INTEGER NOT NULL,  -- Unix timestamp (seconds)
    has_images BOOLEAN DEFAULT 0,
    has_attachments BOOLEAN DEFAULT 0,
    PRIMARY KEY (channel_id, message_id)
);
```

### Example Queries
```bash
# Search for messages containing a keyword in current channel
sqlite3 /data/hollingsbot.db "SELECT author_name, content, datetime(timestamp, 'unixepoch') FROM cached_messages WHERE channel_id = $CHANNEL_ID AND content LIKE '%keyword%' ORDER BY timestamp DESC LIMIT 10"

# Get recent messages from a specific user
sqlite3 /data/hollingsbot.db "SELECT content, datetime(timestamp, 'unixepoch') FROM cached_messages WHERE channel_id = $CHANNEL_ID AND author_name LIKE '%Username%' ORDER BY timestamp DESC LIMIT 5"

# Search across all channels (use sparingly)
sqlite3 /data/hollingsbot.db "SELECT author_name, content FROM cached_messages WHERE content LIKE '%search term%' ORDER BY timestamp DESC LIMIT 20"
```

## Image Generation

Generate images using the Replicate API. The API token is in your environment.

```python
import replicate
import os

client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))

output = client.run(
    "openai/gpt-image-1.5",
    input={
        "prompt": "your detailed image description",
        "quality": "low",
        "moderation": "low"
    }
)

# output is a FileOutput - save it to workspace
if output:
    import urllib.request
    # If output is a URL string or has a url attribute
    url = str(output) if isinstance(output, str) else getattr(output, 'url', str(output))
    urllib.request.urlretrieve(url, "/data/wendy_assistant/generated_image.png")
    print("Image saved to: /data/wendy_assistant/generated_image.png")
```

## Discord API Access

You have full Discord API access via the bot token in your environment.

**Bot Intents:** The bot uses `Intents.default()` + `message_content=True`. This means you have access to:
- Guilds, members, bans, emojis, integrations, webhooks, invites
- Voice states, presences (limited), messages, reactions, typing
- Message content (privileged intent enabled)

You do NOT have: guild members intent (can't list all members)

```bash
# Get channel info
curl -s -H "Authorization: Bot $DISCORD_TOKEN" "https://discord.com/api/v10/channels/$CHANNEL_ID"

# Send a message to the current channel
curl -s -X POST -H "Authorization: Bot $DISCORD_TOKEN" -H "Content-Type: application/json" \
  -d '{"content": "Hello from assistant!"}' \
  "https://discord.com/api/v10/channels/$CHANNEL_ID/messages"

# Upload a file
curl -s -X POST -H "Authorization: Bot $DISCORD_TOKEN" \
  -F "file=@/data/wendy_assistant/generated_image.png" \
  -F 'payload_json={"content": "Here is the image"}' \
  "https://discord.com/api/v10/channels/$CHANNEL_ID/messages"

# Download an attachment (replace URL with actual attachment URL)
curl -s -o /data/wendy_assistant/downloaded.png "https://cdn.discordapp.com/attachments/..."
```

## Persistent Workspace

Your workspace at `/data/wendy_assistant` persists across conversations.

Use it to store:
- Notes and documents (e.g., `notes.txt`, `memories.json`)
- Downloaded files
- Generated images before sending to Discord
- Any records Wendy asks you to keep

## Environment Variables

Available in your environment:
- `DISCORD_TOKEN` - Bot token for Discord API
- `REPLICATE_API_TOKEN` - API key for image generation
- `CHANNEL_ID` - Current Discord channel ID

## Guidelines

1. Be extremely brief - Wendy prefers short responses
2. For image generation: generate, save to workspace, report the path
3. For memory/notes: store in workspace files
4. For searches: query the database, summarize findings concisely
5. Max 2-3 sentences unless the task explicitly requires more
"""


def _ensure_workspace_exists() -> None:
    """Create the assistant workspace, settings, and reference docs."""
    workspace = ASSISTANT_WORKSPACE
    claude_dir = workspace / ".claude"

    # Create directories
    workspace.mkdir(parents=True, exist_ok=True)
    claude_dir.mkdir(parents=True, exist_ok=True)

    # Create sandbox settings with expanded permissions
    settings = {
        "sandbox": {"enabled": True, "autoAllowBashIfSandboxed": True, "allowUnsandboxedCommands": False},
        "permissions": {
            "allow": [
                # Web access
                "WebFetch",
                "WebSearch",
                # File operations within workspace
                f"Read({workspace}/**)",
                f"Edit({workspace}/**)",
                # Database access (read-only)
                "Read(/data/hollingsbot.db)",
                # Basic bash commands
                "Bash(ls:*)",
                "Bash(cat:*)",
                "Bash(echo:*)",
                "Bash(mkdir:*)",
                "Bash(touch:*)",
                "Bash(head:*)",
                "Bash(tail:*)",
                "Bash(wc:*)",
                "Bash(date:*)",
                "Bash(pwd:*)",
                # Database queries
                "Bash(sqlite3:*)",
                # API access
                "Bash(curl:*)",
                # Python for complex tasks
                "Bash(python:*)",
                "Bash(python3:*)",
            ],
            "deny": [
                # Block sensitive paths
                "Read(~/.ssh/**)",
                "Read(~/.aws/**)",
                "Read(.env)",
                "Read(.env.*)",
                "Read(/etc/**)",
                # Block dangerous commands
                "Bash(rm:*)",
                "Bash(sudo:*)",
                "Bash(chmod:*)",
                "Bash(wget:*)",
                "Bash(ssh:*)",
                "Bash(scp:*)",
                "Bash(git:*)",
                "Bash(docker:*)",
                "Bash(pip:*)",
                "Bash(node:*)",
                "Bash(npm:*)",
            ],
            "defaultMode": "plan",
        },
    }

    # Write settings file (always update to ensure latest permissions)
    with open(ASSISTANT_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

    # Create/update CLAUDE.md reference documentation
    ASSISTANT_CLAUDE_MD.write_text(CLAUDE_MD_CONTENT)

    # Create a notes file if it doesn't exist
    notes_file = workspace / "notes.txt"
    if not notes_file.exists():
        notes_file.write_text("# Personal Assistant Notes\n\nThis is your persistent workspace.\n")

    # Make readable by claude user
    os.chmod(workspace, 0o755)
    os.chmod(claude_dir, 0o755)
    os.chmod(ASSISTANT_SETTINGS_FILE, 0o644)
    os.chmod(ASSISTANT_CLAUDE_MD, 0o644)
    if notes_file.exists():
        os.chmod(notes_file, 0o644)


def ask_assistant(task: str = "", request: str = "") -> str:
    """Ask the personal assistant to perform a task.

    The assistant can:
    - Search the web for information
    - Query chat history database
    - Generate images via Replicate API
    - Interact with Discord API
    - Maintain persistent notes and records

    Args:
        task: What to ask the assistant to do (preferred parameter name)
        request: Alternative parameter name for the task

    Returns:
        The assistant's response
    """
    query = task or request
    if not query:
        return "Please tell me what you need help with."

    # Get channel context from parser
    channel_id = None
    try:
        from hollingsbot.tools.parser import get_current_context

        context = get_current_context()
        channel_id = context.get("channel_id")
    except Exception:
        _LOG.warning("Could not get channel context for assistant")

    # Ensure workspace exists with proper settings
    _ensure_workspace_exists()

    # Build the prompt for the assistant
    prompt = f"""You are Wendy's personal assistant. Be extremely brief and FAST.

TASK: {query}

CONTEXT:
- Server ID: 1020445108707000331
- Channel ID: {channel_id or "1050900592031178752"}
- Workspace: {ASSISTANT_WORKSPACE}
- Database: /data/hollingsbot.db

Read your CLAUDE.md file for capabilities and examples.

Rules:
- HEADLESS MODE - you are running unsupervised, no user available to answer questions
- NEVER ask for clarification or confirmation - just do your best with available info
- SPEED OVER THOROUGHNESS - do the minimum needed, don't over-research
- Fewer actions is better - get to the answer quickly
- SHORTER IS ALWAYS BETTER
- If you can answer in one line, do it"""

    try:
        # Write prompt to temp file (avoids shell escaping issues)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        # Make readable by claude user
        os.chmod(prompt_file, 0o644)

        # Build environment with API keys
        # Remove ANTHROPIC_API_KEY so claude CLI uses Max subscription instead of API credits
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        env["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN", "")
        env["DISCORD_TOKEN"] = os.getenv("DISCORD_TOKEN", "")
        env["CHANNEL_ID"] = str(channel_id) if channel_id else ""

        try:
            # Run Claude Code as non-root user with sandbox settings
            result = subprocess.run(
                [
                    "su",
                    "-s",
                    "/bin/bash",
                    "-c",
                    f'claude -p "$(cat {prompt_file})" '
                    f"--dangerously-skip-permissions "
                    f"--output-format json "
                    f"--max-turns 10 "
                    f"--model sonnet",
                    "claude",
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for complex tasks
                cwd=str(ASSISTANT_WORKSPACE),
                env=env,
            )
        finally:
            # Clean up temp file
            os.unlink(prompt_file)

        if result.returncode != 0:
            # Log both stdout and stderr for debugging
            _LOG.error("Assistant CLI failed (code %d)", result.returncode)
            _LOG.error("Assistant stderr: %s", result.stderr or "(empty)")
            _LOG.error("Assistant stdout: %s", result.stdout[:500] if result.stdout else "(empty)")
            # Try to extract error from JSON output if available
            error_msg = result.stderr.strip() if result.stderr.strip() else ""
            if not error_msg and result.stdout:
                try:
                    data = json.loads(result.stdout)
                    error_msg = data.get("result", "") or data.get("error", "") or str(data)[:200]
                except json.JSONDecodeError:
                    error_msg = result.stdout[:200]
            return f"Sorry, I couldn't complete that task. Error: {error_msg or 'Unknown error (code ' + str(result.returncode) + ')'}"

        raw_output = result.stdout.strip()
        if not raw_output:
            _LOG.warning("Assistant CLI returned empty response")
            return "I tried but got no response. Please try again."

        # Parse JSON output to extract final answer
        try:
            data = json.loads(raw_output)

            # Log the agent trace for debugging
            _LOG.info("=== ASSISTANT TRACE ===")
            if isinstance(data, dict):
                messages = data.get("messages", data.get("conversation", []))
                for i, msg in enumerate(messages if isinstance(messages, list) else []):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                block_type = block.get("type", "")
                                if block_type == "tool_use":
                                    _LOG.info(
                                        f"  [{i}] {role} -> TOOL: {block.get('name')} | input: {json.dumps(block.get('input', {}))[:200]}"
                                    )
                                elif block_type == "text":
                                    _LOG.info(f"  [{i}] {role}: {block.get('text', '')[:300]}")
                    else:
                        _LOG.info(f"  [{i}] {role}: {str(content)[:300]}")

                # Extract final response text
                response = data.get("result", data.get("response", data.get("text", "")))
                if not response and messages:
                    # Get last assistant message
                    for msg in reversed(messages if isinstance(messages, list) else []):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                response = content
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        response = block.get("text", "")
                                        break
                            break
            else:
                response = str(data)

            _LOG.info("=== END ASSISTANT TRACE ===")

        except json.JSONDecodeError:
            # Fallback: treat as plain text
            _LOG.warning("Failed to parse JSON output, using raw text")
            response = raw_output

        return response.strip() if response else "I completed the task but have no output to share."

    except subprocess.TimeoutExpired:
        _LOG.error("Assistant CLI timed out")
        return "Sorry, that took too long. Please try a simpler request."
    except Exception as e:
        _LOG.exception("Assistant CLI error: %s", e)
        return f"Sorry, something went wrong: {str(e)[:100]}"
