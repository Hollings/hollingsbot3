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
            for script in scripts_src.glob("*.sh"):
                dest = wendy_dir / script.name
                if not dest.exists() or dest.stat().st_mtime < script.stat().st_mtime:
                    import shutil
                    shutil.copy2(script, dest)
                    dest.chmod(0o755)  # Make executable

        # Also copy check_messages.py
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

    def _get_tool_instructions(self, channel_id: int) -> str:
        """Get instructions for Wendy's shell tools."""
        return f"""

---
REAL-TIME CHANNEL TOOLS (Channel ID: {channel_id})

NOTE: The conversation history above already contains the full chat log. These tools are for real-time interaction only.

1. CHECK FOR NEW MESSAGES (use before responding):
   ./check_messages.sh {channel_id}
   Shows the last 10 messages. Use this to see if anyone sent new messages while you were thinking.

2. SEND A MESSAGE:
   ./send_message.sh {channel_id} "your message here"
   Sends a message to Discord immediately. Users will NOT see your final output - only messages sent via this script.

WORKFLOW:
- Check for new messages before responding (someone may have added context while you were thinking)
- Use send_message.sh to respond - your final output is NOT sent to Discord
- You can send multiple messages if needed

IMAGES:
When users upload images, they are saved to /data/wendy/images/ and referenced in the conversation as [Image: /path/to/file.jpg].
- For RECENT images (in the current message or last few messages), use the Read tool to view them
- Older images in the conversation history can be viewed on request if needed
- Each image has a unique filename with timestamp, e.g. img_0_1767638760763.jpg

PERSONAL FOLDER:
You have a personal folder at /data/wendy/wendys_folder/ where you can save notes, remember things, or store any files you want. This is your private workspace that persists between conversations.
"""

    def _parse_cli_output(self, output: str) -> str:
        """Parse JSON output from Claude CLI."""
        try:
            data = json.loads(output)

            # Handle different output structures
            if "result" in data:
                return str(data["result"])

            # Stream output format
            if "output" in data and isinstance(data["output"], list):
                texts = []
                for item in data["output"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                return "\n".join(texts)

            # Message format
            if "message" in data:
                msg = data["message"]
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                texts.append(block.get("text", ""))
                        return "\n".join(texts)
                    return str(content)

            # Fallback
            _LOG.warning("Unexpected CLI output format: %s", list(data.keys()))
            return str(data)

        except json.JSONDecodeError:
            # If not JSON, return raw output
            return output.strip()

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

        # Add tool instructions if channel_id is provided
        if channel_id:
            user_prompt += self._get_tool_instructions(channel_id)

        # Build CLI command
        cmd = [
            self.cli_path,
            "-p",  # Print mode (non-interactive)
            "--output-format",
            "json",
            "--model",
            self.model,
        ]

        # Append system prompt to keep Claude Code's default (which is cached)
        # Using --append-system-prompt preserves the 22k token cached system prompt
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        # Enable useful Claude Code tools with sandboxed permissions
        # - Read: for reading files (unrestricted)
        # - Write/Edit: ONLY to her personal folder
        # - Bash: for running check_messages.sh and send_message.sh
        # - WebSearch/WebFetch: for web access
        cmd.extend([
            "--allowedTools",
            "Read,WebSearch,WebFetch,Bash,Edit(//data/wendy/wendys_folder/**),Write(//data/wendy/wendys_folder/**)",
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
            result = self._parse_cli_output(output)
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
