"""Text-generation backend using Claude CLI for subscription-based usage.

This generator invokes the `claude` CLI command instead of the Anthropic API,
allowing use of subscription usage instead of API credits for cost savings.

Uses --resume for persistent per-channel sessions, with simple nudge prompts
to trigger Wendy to check messages via her tools.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from .base import TextGeneratorAPI

_LOG = logging.getLogger(__name__)

SESSION_STATE_FILE = Path("/data/wendy/session_state.json")
SESSION_DIR = Path("/root/.claude/projects/-data-wendy")
STREAM_LOG_FILE = Path("/data/wendy/stream.jsonl")
MAX_DISCORD_MESSAGES = 50  # Actual Discord messages, not API turns
MAX_STREAM_LOG_LINES = 5000  # Rolling log limit


def _count_discord_messages_in_tool_result(content: str) -> int:
    """Count Discord messages in a check_messages tool result.

    The check_messages API returns JSON like:
    [{"message_id":123,"author":"user","content":"...","timestamp":123}]

    Returns the number of messages in the response, or 0 if not a check_messages result.
    """
    try:
        # Try to parse as JSON array of messages
        data = json.loads(content)
        if isinstance(data, list):
            # Check if it looks like Discord messages (has message_id and author)
            if data and isinstance(data[0], dict) and "message_id" in data[0] and "author" in data[0]:
                return len(data)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass
    return 0


def _count_discord_messages(messages: list) -> int:
    """Count actual Discord messages in a session by parsing check_messages tool results."""
    count = 0
    for msg in messages:
        if msg.get("type") != "user":
            continue
        # Look for tool_result content
        content_list = msg.get("message", {}).get("content", [])
        if not isinstance(content_list, list):
            continue
        for content_item in content_list:
            if content_item.get("type") == "tool_result":
                result_content = content_item.get("content", "")
                count += _count_discord_messages_in_tool_result(result_content)
    return count


class ClaudeCliError(Exception):
    """Base exception for Claude CLI errors."""

    pass


class ClaudeCliTextGenerator(TextGeneratorAPI):
    """Generate text using Claude CLI (subscription-based).

    This generator invokes the `claude` CLI command instead of the API,
    allowing use of subscription usage instead of API credits.

    Accepts either:
    - str: Single user message
    - Sequence[dict]: Conversation with role/text/images keys

    Images are saved to temp files and referenced in the prompt.
    """

    def __init__(self, model: str = "sonnet") -> None:
        self.model = model
        self.cli_path = self._find_cli_path()
        self.timeout = int(os.getenv("CLAUDE_CLI_TIMEOUT", "300"))
        self._temp_dir: Path | None = None
        self._temp_files: List[Path] = []

    def _load_session_state(self) -> Dict[str, Any]:
        """Load session state from file."""
        if SESSION_STATE_FILE.exists():
            try:
                return json.loads(SESSION_STATE_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_session_state(self, state: Dict[str, Any]) -> None:
        """Save session state to file."""
        SESSION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_STATE_FILE.write_text(json.dumps(state, indent=2))

    def _get_channel_session(self, channel_id: int) -> Dict[str, Any] | None:
        """Get session info for a channel."""
        state = self._load_session_state()
        return state.get(str(channel_id))

    def _create_channel_session(self, channel_id: int) -> str:
        """Create a new session for a channel, return session_id."""
        session_id = str(uuid.uuid4())
        state = self._load_session_state()
        state[str(channel_id)] = {
            "session_id": session_id,
            "created_at": int(time.time()),
            "message_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_create_tokens": 0,
        }
        self._save_session_state(state)
        _LOG.info("Created new session %s for channel %d", session_id, channel_id)
        return session_id

    def _update_session_stats(self, channel_id: int, usage: Dict[str, Any]) -> None:
        """Update session stats after a run."""
        state = self._load_session_state()
        channel_key = str(channel_id)
        if channel_key not in state:
            return

        state[channel_key]["message_count"] += 1
        state[channel_key]["total_input_tokens"] += usage.get("input_tokens", 0)
        state[channel_key]["total_output_tokens"] += usage.get("output_tokens", 0)
        state[channel_key]["total_cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)
        state[channel_key]["total_cache_create_tokens"] += usage.get("cache_creation_input_tokens", 0)
        state[channel_key]["last_used_at"] = int(time.time())
        self._save_session_state(state)

        # Check if session needs truncation
        self._truncate_session_if_needed(state[channel_key]["session_id"])

    def _truncate_session_if_needed(self, session_id: str) -> None:
        """Truncate session if Discord messages exceed MAX_DISCORD_MESSAGES."""
        session_file = SESSION_DIR / f"{session_id}.jsonl"
        if not session_file.exists():
            return

        try:
            # Read all messages
            messages = []
            with open(session_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            # Count actual Discord messages (from check_messages tool results)
            discord_msg_count = _count_discord_messages(messages)

            if discord_msg_count <= MAX_DISCORD_MESSAGES:
                return  # No truncation needed

            _LOG.info(
                "Session %s has %d Discord messages (max %d), truncating...",
                session_id[:8], discord_msg_count, MAX_DISCORD_MESSAGES
            )

            # Find cutoff point by walking backwards and counting Discord messages
            # We want to keep the last MAX_DISCORD_MESSAGES Discord messages
            discord_msgs_seen = 0
            cutoff_idx = len(messages)

            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if msg.get("type") == "user":
                    # Count Discord messages in this tool result
                    content_list = msg.get("message", {}).get("content", [])
                    if isinstance(content_list, list):
                        for content_item in content_list:
                            if content_item.get("type") == "tool_result":
                                result_content = content_item.get("content", "")
                                discord_msgs_seen += _count_discord_messages_in_tool_result(result_content)

                    if discord_msgs_seen >= MAX_DISCORD_MESSAGES:
                        cutoff_idx = i
                        break

            # IMPORTANT: Make sure we don't start with a tool_result
            # This would break the session because tool_result needs a preceding tool_use
            while cutoff_idx < len(messages):
                msg = messages[cutoff_idx]
                # Check if this is a user message with tool_result content
                if msg.get("type") == "user":
                    content = msg.get("message", {}).get("content", [])
                    if isinstance(content, list) and content:
                        # If first content item is a tool_result, back up
                        if content[0].get("type") == "tool_result":
                            cutoff_idx += 1
                            continue
                break

            # If we backed up too far, just keep everything
            if cutoff_idx >= len(messages):
                _LOG.warning("Session %s: couldn't find clean truncation point", session_id[:8])
                return

            truncated = messages[cutoff_idx:]
            removed_count = len(messages) - len(truncated)

            # Write truncated session
            with open(session_file, "w") as f:
                for msg in truncated:
                    f.write(json.dumps(msg) + "\n")

            _LOG.info(
                "Truncated session %s: removed %d entries, kept %d (with %d Discord messages)",
                session_id[:8], removed_count, len(truncated),
                _count_discord_messages(truncated)
            )

        except Exception as e:
            _LOG.error("Failed to truncate session %s: %s", session_id[:8], e)

    def get_session_stats(self, channel_id: int) -> Dict[str, Any] | None:
        """Get session stats for a channel (used by !context command)."""
        return self._get_channel_session(channel_id)

    def reset_channel_session(self, channel_id: int) -> str:
        """Reset a channel's session, return new session_id."""
        return self._create_channel_session(channel_id)

    def _find_cli_path(self) -> str:
        """Find the claude CLI executable."""
        # Check env var override first
        cli_path = os.getenv("CLAUDE_CLI_PATH")
        if cli_path and Path(cli_path).exists():
            return cli_path

        # Check common locations
        candidates = [
            str(Path.home() / ".local" / "bin" / "claude"),
            str(Path.home() / ".claude" / "local" / "claude"),
            shutil.which("claude"),
        ]

        for path in candidates:
            if path and Path(path).exists():
                return path

        raise ClaudeCliError(
            "Claude CLI not found. Install it or set CLAUDE_CLI_PATH env var."
        )

    def _get_recent_cli_error(self) -> str | None:
        """Read the most recent Claude CLI debug file to extract error messages."""
        debug_dir = Path.home() / ".claude" / "debug"
        if not debug_dir.exists():
            return None

        try:
            # Get most recent .txt debug file
            debug_files = sorted(debug_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not debug_files:
                return None

            # Read the most recent one and look for errors
            content = debug_files[0].read_text(errors="replace")

            # Look for OAuth errors specifically
            if "OAuth token has expired" in content:
                return "OAuth token has expired"

            # Look for authentication errors
            if "authentication_error" in content:
                # Try to extract the message
                import re
                match = re.search(r'"message":\s*"([^"]+)"', content)
                if match:
                    return match.group(1)
                return "authentication error"

            # Look for any [ERROR] lines in the last few lines
            lines = content.strip().split("\n")
            for line in reversed(lines[-20:]):
                if "[ERROR]" in line:
                    # Extract just the error part
                    if "Error:" in line:
                        return line.split("Error:", 1)[-1].strip()[:200]
                    return line.split("[ERROR]", 1)[-1].strip()[:200]

            return None
        except Exception as e:
            _LOG.warning("Failed to read CLI debug files: %s", e)
            return None

    def _ensure_temp_dir(self) -> Path:
        """Create temp directory for images if needed."""
        if self._temp_dir is None or not self._temp_dir.exists():
            self._temp_dir = Path(tempfile.mkdtemp(prefix="claude_cli_"))
        return self._temp_dir

    def _save_images_to_temp(self, images: List[Dict[str, Any]]) -> List[Path]:
        """Save base64 images to Wendy's images folder."""
        paths = []
        # Save to persistent location
        images_dir = Path("/data/wendy/images")
        images_dir.mkdir(parents=True, exist_ok=True)

        _LOG.info("Processing %d images for CLI", len(images))

        for i, img in enumerate(images):
            data_url = img.get("data_url", "")
            _LOG.info("Image %d: data_url length=%d, starts_with=%s",
                      i, len(data_url) if data_url else 0,
                      data_url[:50] if data_url else "None")

            if not data_url or not data_url.startswith("data:"):
                _LOG.warning("Image %d: Invalid or missing data_url", i)
                continue

            try:
                # Parse data URL: data:image/jpeg;base64,/9j/4AAQ...
                header, b64_data = data_url.split(",", 1)
                media_type = header.split(";")[0].replace("data:", "")
                ext = media_type.split("/")[-1]
                if ext == "jpeg":
                    ext = "jpg"

                _LOG.info("Image %d: media_type=%s, b64_data_len=%d", i, media_type, len(b64_data))

                image_bytes = base64.b64decode(b64_data)
                image_path = images_dir / f"img_{i}_{int(time.time() * 1000)}.{ext}"
                image_path.write_bytes(image_bytes)
                paths.append(image_path)
                _LOG.info("Saved image to %s (%d bytes)", image_path, len(image_bytes))
            except Exception as e:
                _LOG.exception("Failed to save image %d: %s", i, e)

        return paths

    def _format_image_references(self, images: List[Dict[str, Any]]) -> str:
        """Format image references for the prompt."""
        paths = self._save_images_to_temp(images)
        if not paths:
            return ""

        refs = [f"[Image: {p}]" for p in paths]
        return "\n".join(refs)

    def _format_conversation(
        self, conversation: List[Dict[str, Any]]
    ) -> tuple[str, str]:
        """Convert conversation to (system_prompt, user_prompt) for CLI.

        Returns:
            system_prompt: The system message text (or empty string)
            user_prompt: The formatted conversation history + current message
        """
        system_prompt = ""
        formatted_parts = []

        for turn in conversation:
            role = str(turn.get("role", "user")).lower()
            text = str(turn.get("text", turn.get("content", "")))
            images = turn.get("images") or []

            if role == "system":
                system_prompt = text
                continue

            # Format image references
            if images:
                image_refs = self._format_image_references(images)
                text = f"{text}\n{image_refs}" if text else image_refs

            # Format role marker
            if role == "user":
                formatted_parts.append(f"[User]: {text}")
            elif role == "assistant":
                formatted_parts.append(f"[Assistant]: {text}")

        # Construct the prompt
        prompt_text = "\n\n".join(formatted_parts)

        # Add instruction to continue
        prompt_text += "\n\n[Assistant]:"

        return system_prompt, prompt_text

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary image files."""
        for path in self._temp_files:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_files.clear()

        # Remove temp directory if empty
        if self._temp_dir and self._temp_dir.exists():
            try:
                self._temp_dir.rmdir()
            except OSError:
                pass  # Not empty or other error

    def _setup_wendy_scripts(self, wendy_dir: Path) -> None:
        """Ensure Wendy's shell scripts are available in her directory."""
        scripts_src = Path("/app/scripts/wendy")

        # Copy scripts if they exist
        if scripts_src.exists():
            # Copy shell scripts
            for script in scripts_src.glob("*.sh"):
                dest = wendy_dir / script.name
                if not dest.exists() or dest.stat().st_mtime < script.stat().st_mtime:
                    import shutil
                    shutil.copy2(script, dest)
                    dest.chmod(0o755)  # Make executable
            # Copy Python scripts
            for script in scripts_src.glob("*.py"):
                dest = wendy_dir / script.name
                if not dest.exists() or dest.stat().st_mtime < script.stat().st_mtime:
                    import shutil
                    shutil.copy2(script, dest)

        # Also copy check_messages.py from parent scripts dir
        check_messages_src = Path("/app/scripts/check_messages.py")
        if check_messages_src.exists():
            check_messages_dest = wendy_dir / "check_messages.py"
            if not check_messages_dest.exists():
                import shutil
                shutil.copy2(check_messages_src, check_messages_dest)

        # Create outbox directory
        (wendy_dir / "outbox").mkdir(exist_ok=True)

        # Create Wendy's personal folder (she can read/write here)
        (wendy_dir / "wendys_folder").mkdir(exist_ok=True)

        # Create uploads directory for Discord attachments
        (wendy_dir / "uploads").mkdir(exist_ok=True)

    def _get_wendys_notes(self) -> str:
        """Load Wendy's self-editable notes from her personal CLAUDE.md."""
        notes_path = Path("/data/wendy/wendys_folder/CLAUDE.md")
        if not notes_path.exists():
            return ""
        try:
            content = notes_path.read_text().strip()
            if content:
                return f"\n\n---\nYOUR PERSONAL NOTES (from wendys_folder/CLAUDE.md - you can edit this!):\n{content}\n---"
            return ""
        except Exception as e:
            _LOG.warning("Failed to read Wendy's notes: %s", e)
            return ""

    def _get_tool_instructions(self, channel_id: int) -> str:
        """Get instructions for Wendy's API tools."""
        return f"""

---
REAL-TIME CHANNEL TOOLS (Channel ID: {channel_id})

CRITICAL: You are running in HEADLESS MODE. Your final output is NOT sent to Discord.
You MUST use the send_message API to respond - this is the ONLY way users will see your messages!

RESPONSE EXPECTATIONS:
- You should ALMOST ALWAYS respond. Users expect you to participate in conversation.
- If you don't call send_message, users see NOTHING - it looks like you ignored them.
- Only skip responding if users EXPLICITLY say they don't want your input (e.g., "wendy stop", "shut up", "go away").
- In ambiguous situations, neutral chats, or when unsure: RESPOND. Err on the side of engaging.
- Even a brief acknowledgment ("gotcha!", "nice", "haha") is better than silence.

1. SEND A MESSAGE (REQUIRED to respond):
   curl -X POST http://wendy_proxy:8080/api/send_message -H "Content-Type: application/json" -d '{{"channel_id": "{channel_id}", "content": "your message here"}}'

   With attachment (file must be in /data/wendy/):
   curl -X POST http://wendy_proxy:8080/api/send_message -H "Content-Type: application/json" -d '{{"channel_id": "{channel_id}", "content": "check this out", "attachment": "/data/wendy/uploads/file.png"}}'

   This is the ONLY way to send messages to users. Your final output goes nowhere.

2. CHECK FOR NEW MESSAGES (optional, use before responding):
   curl -s http://wendy_proxy:8080/api/check_messages/{channel_id}

   Shows the last 10 messages to see if anyone sent new messages while you were thinking.
   Note: Always use -s flag with curl for cleaner output.

WORKFLOW:
1. Read/process the user's request
2. Do any work needed (read files, search, etc.)
3. ALWAYS call the send_message API to reply (unless explicitly told not to)
4. You can send multiple messages if needed

LONG TASKS:
Before doing something that might take a while (writing code, researching, reading multiple files, etc.), send a quick message to let users know. Otherwise they might think you froze or crashed. Send a quick "gimme a sec..." then do the work, then send your actual response.

ATTACHMENTS:
When users upload files (images, documents, code, etc.), the check_messages response includes an "attachments" array with file paths:
  {{"author": "someone", "content": "look at this", "attachments": ["/data/wendy/attachments/msg_123_0_photo.jpg"]}}
- You CANNOT see attachments without using the Read tool on the file path. The path is just a reference.
- If a message has an "attachments" array, you MUST call Read on each path to actually see the content.
- Do NOT describe or comment on files you haven't actually Read - you will hallucinate.
- Always check for the "attachments" field in message JSON when users seem to be sharing something.

SENDING FILES TO DISCORD:
To upload/share a file with users, you MUST:
1. First WRITE the file to /data/wendy/uploads/ using the Write tool (this is the ONLY directory you can upload from)
2. Then use send_message API with the "attachment" field pointing to that file

Example workflow to share a GIF:
  1. Write the file: Use Write tool to save to /data/wendy/uploads/my_image.gif
  2. Send it: curl -X POST ... -d '{{"channel_id": "...", "attachment": "/data/wendy/uploads/my_image.gif"}}'

IMPORTANT: You can ONLY write to /data/wendy/wendys_folder/** and /data/wendy/uploads/**. Any other path will fail.

PERSONAL FOLDER:
You have a personal folder at /data/wendy/wendys_folder/ where you can save notes or files. This persists between conversations.

SELF-CUSTOMIZATION:
You can edit /data/wendy/wendys_folder/CLAUDE.md to customize your own behavior. Anything you write there becomes part of your system instructions on the next message. Use this to remember things, set personal preferences, or adjust how you behave. Changes take effect immediately - no restart needed.

MESSAGE HISTORY DATABASE:
You have full read access to the message history at /data/hollingsbot.db. Use query_db.py to search messages, check past conversations, or find old content. Don't wait to be asked - if someone asks about history or past messages, just query it!

Usage:
  python3 /data/wendy/query_db.py "SELECT * FROM message_history WHERE content LIKE '%keyword%' LIMIT 20"
  python3 /data/wendy/query_db.py --schema    # Show all tables

Key tables:
- message_history: Full raw messages (message_id, channel_id, author_nickname, content, timestamp, reactions, attachment_urls)
  - message_id is the Discord message ID - you can make jump links: https://discord.com/channels/{{guild_id}}/{{channel_id}}/{{message_id}}
- cached_messages: Recent messages used for LLM context (lighter schema)
- message_groups: Conversation summaries at different time levels
"""

    def _parse_stream_json(self, output: str, channel_id: int | None = None) -> str:
        """Parse stream-json output from Claude CLI and save debug log.

        Stream-json format is one JSON object per line with types:
        - system: init info
        - assistant: model messages, tool uses, thinking
        - result: final result with usage stats
        """
        events = []
        result_text = ""

        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                events.append(event)

                # Extract result text from the result event
                if event.get("type") == "result":
                    result_text = event.get("result", "")

            except json.JSONDecodeError as e:
                _LOG.warning("Failed to parse stream-json line: %s", e)
                continue

        # Save debug log to file
        self._save_debug_log(events, channel_id)

        return result_text

    def _save_debug_log(self, events: List[Dict], channel_id: int | None) -> None:
        """Save CLI events to debug log file."""
        try:
            debug_dir = Path("/data/wendy/debug_logs")
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time() * 1000)
            channel_str = str(channel_id) if channel_id else "unknown"
            log_path = debug_dir / f"{channel_str}_{timestamp}.json"

            # Extract key info for easier debugging
            debug_data = {
                "timestamp": timestamp,
                "channel_id": channel_id,
                "events": events,
                "summary": self._summarize_events(events),
            }

            log_path.write_text(json.dumps(debug_data, indent=2))
            _LOG.info("Saved CLI debug log to %s", log_path)

            # Keep only last 20 debug logs
            logs = sorted(debug_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            for old_log in logs[:-20]:
                old_log.unlink()

        except Exception as e:
            _LOG.error("Failed to save debug log: %s", e)

    def _summarize_events(self, events: List[Dict]) -> Dict:
        """Extract summary info from events for quick debugging."""
        summary = {
            "tool_uses": [],
            "assistant_messages": [],
            "total_cost_usd": None,
            "num_turns": None,
        }

        for event in events:
            event_type = event.get("type")

            if event_type == "assistant":
                msg = event.get("message", {})
                content = msg.get("content", [])
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            summary["tool_uses"].append({
                                "tool": block.get("name"),
                                "input_preview": str(block.get("input", ""))[:200],
                            })
                        elif block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                summary["assistant_messages"].append(text[:500])

            elif event_type == "result":
                summary["total_cost_usd"] = event.get("total_cost_usd")
                summary["num_turns"] = event.get("num_turns")

        return summary

    def _append_to_stream_log(self, event: Dict, channel_id: int | None) -> None:
        """Append a single event to the rolling stream log file."""
        try:
            STREAM_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Add metadata to event
            enriched_event = {
                "ts": int(time.time() * 1000),  # millisecond timestamp
                "channel_id": channel_id,
                "event": event,
            }

            # Append to file
            with open(STREAM_LOG_FILE, "a") as f:
                f.write(json.dumps(enriched_event) + "\n")

        except Exception as e:
            _LOG.error("Failed to append to stream log: %s", e)

    def _trim_stream_log_if_needed(self) -> None:
        """Trim stream log to MAX_STREAM_LOG_LINES if it gets too large."""
        try:
            if not STREAM_LOG_FILE.exists():
                return

            # Count lines
            with open(STREAM_LOG_FILE, "r") as f:
                lines = f.readlines()

            if len(lines) > MAX_STREAM_LOG_LINES:
                # Keep the last MAX_STREAM_LOG_LINES
                with open(STREAM_LOG_FILE, "w") as f:
                    f.writelines(lines[-MAX_STREAM_LOG_LINES:])
                _LOG.info("Trimmed stream log from %d to %d lines", len(lines), MAX_STREAM_LOG_LINES)
        except Exception as e:
            _LOG.error("Failed to trim stream log: %s", e)

    async def generate(
        self,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        temperature: float = 1.0,
        channel_id: int | None = None,
        **kwargs,
    ) -> str:
        """Generate response using Claude CLI with persistent sessions.

        Uses --resume for per-channel sessions. Instead of passing conversation
        history, sends a simple nudge for Wendy to check messages via her tools.

        Args:
            prompt: Ignored in resume mode - we send a simple nudge instead
            temperature: Ignored (CLI doesn't support temperature)
            channel_id: Discord channel ID (required for session management)

        Returns:
            Generated text response
        """
        if not channel_id:
            raise ValueError("channel_id is required for Claude CLI sessions")

        # Get or create session for this channel
        # _force_new_session is set when retrying after a stale session error
        force_new = kwargs.get("_force_new_session", False)
        session_info = self._get_channel_session(channel_id)
        is_new_session = session_info is None or force_new

        if is_new_session:
            session_id = self._create_channel_session(channel_id)
        else:
            session_id = session_info["session_id"]

        # Nudge prompt - must explicitly require check_messages API or Wendy may assume based on context
        nudge_prompt = f"<new messages - you MUST call curl -s http://wendy_proxy:8080/api/check_messages/{channel_id} before any other action. Do not assume what the messages contain.>"

        # Build system prompt (only needed for new sessions, but --append-system-prompt
        # is idempotent so we include it always for safety)
        system_prompt = self._get_base_system_prompt()
        system_prompt += self._get_wendys_notes()
        system_prompt += self._get_tool_instructions(channel_id)

        # Build CLI command
        cmd = [
            self.cli_path,
            "-p",  # Print mode (non-interactive)
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
        ]

        # Resume existing session or start fresh
        if not is_new_session:
            cmd.extend(["--resume", session_id])
            _LOG.info("ClaudeCLI: resuming session %s for channel %d", session_id, channel_id)
        else:
            # For new sessions, use the session ID we created
            cmd.extend(["--session-id", session_id])
            _LOG.info("ClaudeCLI: starting new session %s for channel %d", session_id, channel_id)

        # Append system prompt
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        # Enable tools with sandboxed permissions
        cmd.extend([
            "--allowedTools",
            "Read,WebSearch,WebFetch,Bash,Edit(/data/wendy/wendys_folder/**),Write(/data/wendy/wendys_folder/**),Write(/data/wendy/uploads/**)",
            "--disallowedTools",
            "Edit(/data/wendy/*.sh),Edit(/data/wendy/*.py),Edit(/app/**),Write(/app/**)",
        ])

        _LOG.info(
            "ClaudeCLI: model=%s, session=%s, is_new=%s",
            self.model,
            session_id[:8],
            is_new_session,
        )

        # Wendy's working directory
        wendy_dir = Path("/data/wendy")
        wendy_dir.mkdir(parents=True, exist_ok=True)
        self._setup_wendy_scripts(wendy_dir)

        proc = None
        try:
            # Filter out sensitive environment variables from Claude CLI
            sensitive_vars = {
                "DISCORD_TOKEN",
                "WEBHOOK_URL",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "REPLICATE_API_TOKEN",
                "GITHUB_TOKEN",
                "GH_TOKEN",
                "WENDY_DEPLOY_TOKEN",
            }
            cli_env = {k: v for k, v in os.environ.items() if k not in sensitive_vars}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=10 * 1024 * 1024,  # 10MB line buffer for large CLI responses
                cwd=wendy_dir,
                env=cli_env,
            )

            # Write input to stdin and close it
            proc.stdin.write(nudge_prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
            await proc.stdin.wait_closed()

            # Read stdout line-by-line in real-time, streaming to log
            events = []
            result_text = ""
            usage = {}

            async def read_stream_with_timeout():
                nonlocal result_text, usage
                async for line in proc.stdout:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        events.append(event)

                        # Stream to rolling log file in real-time
                        self._append_to_stream_log(event, channel_id)

                        # Extract result from final event
                        if event.get("type") == "result":
                            result_text = event.get("result", "")
                            usage = event.get("usage", {})

                    except json.JSONDecodeError as e:
                        _LOG.warning("Failed to parse stream-json line: %s", e)
                        continue

            try:
                await asyncio.wait_for(read_stream_with_timeout(), timeout=self.timeout)
            except asyncio.TimeoutError:
                _LOG.error("Claude CLI stdout read timed out after %ds", self.timeout)
                raise

            # Wait for process to complete
            await proc.wait()

            # Read any stderr
            stderr_data = await proc.stderr.read()
            stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""

            if proc.returncode != 0:
                _LOG.error("Claude CLI failed: %s", stderr_text)
                # If resume failed, try creating a fresh session and retry
                # Check for various session-related error messages
                session_error = (
                    "--resume" in cmd and (
                        "session" in stderr_text.lower() or
                        "no conversation found" in stderr_text.lower() or
                        not stderr_text.strip()  # Empty stderr often means session not found
                    )
                )
                if session_error and not kwargs.get("_force_new_session"):
                    _LOG.warning("Session resume failed, retrying with fresh session for channel %d", channel_id)
                    # Retry with fresh session (flag prevents infinite loop and forces --session-id)
                    return await self.generate(prompt, temperature, channel_id, _force_new_session=True, **kwargs)

                # If stderr is empty, check Claude's debug files for the real error
                error_detail = stderr_text
                if not stderr_text.strip():
                    error_detail = self._get_recent_cli_error() or "unknown error"

                raise ClaudeCliError(
                    f"CLI failed (code {proc.returncode}): {error_detail}"
                )

            # Save debug log (legacy format) and trim stream log
            self._save_debug_log(events, channel_id)
            self._trim_stream_log_if_needed()

            # Update session stats
            if usage:
                self._update_session_stats(channel_id, usage)

            _LOG.info("ClaudeCLI: response_len=%d, events_streamed=%d", len(result_text), len(events))
            # Don't return the text output - Wendy's responses go through send_message API
            return ""

        except asyncio.TimeoutError as e:
            _LOG.error("Claude CLI timed out after %ds", self.timeout)
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise ClaudeCliError(f"Timed out after {self.timeout}s") from e
        finally:
            self._cleanup_temp_files()

    def _get_base_system_prompt(self) -> str:
        """Get the base system prompt for Wendy."""
        # Load from file if configured
        system_prompt_file = os.getenv("SYSTEM_PROMPT_FILE", "/app/config/system_prompt.txt")
        if Path(system_prompt_file).exists():
            try:
                return Path(system_prompt_file).read_text().strip()
            except Exception as e:
                _LOG.warning("Failed to read system prompt file: %s", e)
        return ""

    def _parse_stream_json_with_usage(self, output: str, channel_id: int | None = None) -> tuple[str, Dict[str, Any]]:
        """Parse stream-json output and return (result_text, usage_dict)."""
        events = []
        result_text = ""
        usage = {}

        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                events.append(event)

                if event.get("type") == "result":
                    result_text = event.get("result", "")
                    usage = event.get("usage", {})

            except json.JSONDecodeError as e:
                _LOG.warning("Failed to parse stream-json line: %s", e)
                continue

        self._save_debug_log(events, channel_id)
        return result_text, usage
