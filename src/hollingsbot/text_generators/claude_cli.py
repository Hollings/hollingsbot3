"""Text-generation backend using Claude CLI for subscription-based usage.

This generator invokes the `claude` CLI command instead of the Anthropic API,
allowing use of subscription usage instead of API credits for cost savings.
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
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from .base import TextGeneratorAPI

_LOG = logging.getLogger(__name__)


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
        """Get instructions for Wendy's shell tools."""
        return f"""

---
REAL-TIME CHANNEL TOOLS (Channel ID: {channel_id})

CRITICAL: You are running in HEADLESS MODE. Your final output is NOT sent to Discord.
You MUST use ./send_message.sh to respond - this is the ONLY way users will see your messages!

RESPONSE EXPECTATIONS:
- You should ALMOST ALWAYS respond. Users expect you to participate in conversation.
- If you don't call ./send_message.sh, users see NOTHING - it looks like you ignored them.
- Only skip responding if users EXPLICITLY say they don't want your input (e.g., "wendy stop", "shut up", "go away").
- In ambiguous situations, neutral chats, or when unsure: RESPOND. Err on the side of engaging.
- Even a brief acknowledgment ("gotcha!", "nice", "haha") is better than silence.

1. SEND A MESSAGE (REQUIRED to respond):
   ./send_message.sh {channel_id} "your message here"
   ./send_message.sh {channel_id} "your message here" "/data/wendy/uploads/file.png"  # with attachment
   This is the ONLY way to send messages to users. Your final output goes nowhere.
   IMPORTANT: File attachments MUST be saved to /data/wendy/uploads/ (NOT /tmp/) to work!

2. CHECK FOR NEW MESSAGES (optional, use before responding):
   ./check_messages.sh {channel_id}
   Shows the last 10 messages to see if anyone sent new messages while you were thinking.

WORKFLOW:
1. Read/process the user's request
2. Do any work needed (read files, search, etc.)
3. ALWAYS call ./send_message.sh {channel_id} "your response" to reply (unless explicitly told not to)
4. You can send multiple messages if needed

IMAGES:
When users upload images, they are saved to /data/wendy/images/ and referenced as [Image: /path/to/file.jpg].
- You CANNOT see images without using the Read tool on the file path. The [Image: ...] tag is just a reference.
- If someone asks about an image or you see an [Image: ...] tag, you MUST call Read on that path first.
- Do NOT describe or comment on images you haven't actually Read - you will hallucinate.
- Each image has a unique filename with timestamp

PERSONAL FOLDER:
You have a personal folder at /data/wendy/wendys_folder/ where you can save notes or files. This persists between conversations.

SELF-CUSTOMIZATION:
You can edit /data/wendy/wendys_folder/CLAUDE.md to customize your own behavior. Anything you write there becomes part of your system instructions on the next message. Use this to remember things, set personal preferences, or adjust how you behave. Changes take effect immediately - no restart needed.

MESSAGE HISTORY DATABASE:
You have full read access to the message history at /data/hollingsbot.db. Use query_db.py to search messages, check past conversations, or find old content. Don't wait to be asked - if someone asks about history or past messages, just query it!

Usage:
  python3 ./query_db.py "SELECT * FROM message_history WHERE content LIKE '%keyword%' LIMIT 20"
  python3 ./query_db.py --schema    # Show all tables

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

    async def generate(
        self,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        temperature: float = 1.0,
        channel_id: int | None = None,
        **kwargs,
    ) -> str:
        """Generate response using Claude CLI.

        Args:
            prompt: Either a string or a list of message dicts
            temperature: Ignored (CLI doesn't support temperature)
            channel_id: Discord channel ID for message checking tools

        Returns:
            Generated text response
        """
        # Normalize input
        if isinstance(prompt, str):
            conversation = [{"role": "user", "text": prompt, "images": []}]
        elif isinstance(prompt, Sequence):
            conversation = list(prompt)
        else:
            raise TypeError("prompt must be a string or sequence of message dicts")

        # Format conversation
        system_prompt, user_prompt = self._format_conversation(conversation)

        # Append Wendy's self-editable notes to system prompt
        system_prompt += self._get_wendys_notes()

        # Add tool instructions if channel_id is provided
        if channel_id:
            user_prompt += self._get_tool_instructions(channel_id)

        # Build CLI command
        cmd = [
            self.cli_path,
            "-p",  # Print mode (non-interactive)
            "--output-format",
            "stream-json",
            "--verbose",  # Required for stream-json in print mode
            "--model",
            self.model,
        ]

        # Append system prompt to keep Claude Code's default (which is cached)
        # Using --append-system-prompt preserves the 22k token cached system prompt
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        # Enable useful Claude Code tools with sandboxed permissions
        # - Read: for reading files (unrestricted)
        # - Write/Edit: ONLY to her personal folder and uploads
        # - Bash: for running check_messages.sh, send_message.sh, and curl
        # - WebSearch/WebFetch: for web access
        cmd.extend([
            "--allowedTools",
            "Read,WebSearch,WebFetch,Bash,Edit(//data/wendy/wendys_folder/**),Write(//data/wendy/wendys_folder/**),Write(//data/wendy/uploads/**)",
            "--disallowedTools",
            "Edit(//data/wendy/*.sh),Edit(//data/wendy/*.py),Edit(//app/**),Write(//app/**)",
        ])

        # Prompt will be passed via stdin to avoid shell escaping issues
        _LOG.info(
            "ClaudeCLI: model=%s, prompt_len=%d, system_len=%d",
            self.model,
            len(user_prompt),
            len(system_prompt),
        )

        # Wendy's working directory
        wendy_dir = Path("/data/wendy")
        wendy_dir.mkdir(parents=True, exist_ok=True)

        # Ensure scripts are available in her directory
        self._setup_wendy_scripts(wendy_dir)

        proc = None
        try:
            # Create env without ANTHROPIC_API_KEY so CLI uses subscription billing
            # instead of API credits
            cli_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=wendy_dir,
                env=cli_env,
            )

            # Pass prompt via stdin
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=user_prompt.encode("utf-8")),
                timeout=self.timeout,
            )

            if proc.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                _LOG.error("Claude CLI failed: %s", error_msg)
                raise ClaudeCliError(
                    f"CLI failed (code {proc.returncode}): {error_msg}"
                )

            output = stdout.decode("utf-8")
            result = self._parse_stream_json(output, channel_id)
            _LOG.info("ClaudeCLI: response_len=%d", len(result))
            return result

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
