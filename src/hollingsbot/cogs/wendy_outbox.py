"""Outbox watcher for Wendy's async message sending.

Watches /data/wendy/outbox/ for JSON files and sends messages to Discord.
"""
import json
import logging
import os
import re
from pathlib import Path

import discord
from discord.ext import commands, tasks

_LOG = logging.getLogger(__name__)

OUTBOX_DIR = Path(os.getenv("WENDY_OUTBOX_DIR", "/data/wendy/outbox"))
MESSAGE_LOG_FILE = Path("/data/wendy/message_log.jsonl")
MAX_MESSAGE_LOG_LINES = 1000  # Rolling log limit


class WendyOutbox(commands.Cog):
    """Watches Wendy's outbox and sends messages to Discord."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._ensure_outbox_dir()
        self.watch_outbox.start()
        _LOG.info("WendyOutbox initialized, watching %s", OUTBOX_DIR)

    def _ensure_outbox_dir(self) -> None:
        """Create outbox directory if it doesn't exist."""
        OUTBOX_DIR.mkdir(parents=True, exist_ok=True)

    def _log_sent_message(
        self,
        discord_msg_id: int,
        outbox_ts: int,
        channel_id: int,
        content: str
    ) -> None:
        """Log a sent message to message_log.jsonl for debug correlation."""
        try:
            MESSAGE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

            log_entry = {
                "discord_msg_id": discord_msg_id,
                "outbox_ts": outbox_ts,
                "channel_id": channel_id,
                "content_preview": content[:100] if content else "",
            }

            with open(MESSAGE_LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            _LOG.debug("Logged message correlation: discord=%d, outbox_ts=%d", discord_msg_id, outbox_ts)

            # Trim log if needed
            self._trim_message_log_if_needed()

        except Exception as e:
            _LOG.error("Failed to log sent message: %s", e)

    def _trim_message_log_if_needed(self) -> None:
        """Trim message log to MAX_MESSAGE_LOG_LINES if it gets too large."""
        try:
            if not MESSAGE_LOG_FILE.exists():
                return

            with open(MESSAGE_LOG_FILE) as f:
                lines = f.readlines()

            if len(lines) > MAX_MESSAGE_LOG_LINES:
                with open(MESSAGE_LOG_FILE, "w") as f:
                    f.writelines(lines[-MAX_MESSAGE_LOG_LINES:])
                _LOG.info("Trimmed message log from %d to %d lines", len(lines), MAX_MESSAGE_LOG_LINES)
        except Exception as e:
            _LOG.error("Failed to trim message log: %s", e)

    def _extract_outbox_timestamp(self, filename: str) -> int | None:
        """Extract timestamp from outbox filename like '123456_1234567890123.json'."""
        # Format: {channel_id}_{timestamp_ns}.json
        match = re.match(r"\d+_(\d+)\.json$", filename)
        if match:
            return int(match.group(1))
        return None

    def cog_unload(self):
        self.watch_outbox.cancel()

    @tasks.loop(seconds=0.5)
    async def watch_outbox(self):
        """Check for new messages in the outbox."""
        try:
            for file_path in OUTBOX_DIR.glob("*.json"):
                await self._process_outbox_file(file_path)
        except Exception as e:
            _LOG.error("Error watching outbox: %s", e)

    @watch_outbox.before_loop
    async def before_watch(self):
        await self.bot.wait_until_ready()

    async def _process_outbox_file(self, outbox_file: Path) -> None:
        """Process a single outbox file and send the message."""
        try:
            data = json.loads(outbox_file.read_text())
            channel_id = int(data["channel_id"])
            message_text = data["message"]

            channel = self.bot.get_channel(channel_id)
            if not channel:
                _LOG.warning("Channel %s not found, skipping message", channel_id)
                outbox_file.unlink()
                return

            # Check for file attachment
            file_path_str = data.get("file_path")
            attachment = None
            if file_path_str:
                attachment_path = Path(file_path_str)
                if attachment_path.exists():
                    attachment = discord.File(attachment_path)
                    _LOG.info("Attaching file: %s", attachment_path)
                else:
                    _LOG.warning("Attachment file not found: %s", file_path_str)

            # Send the message (with or without attachment) and capture message ID
            if attachment:
                sent_msg = await channel.send(message_text, file=attachment)
            else:
                sent_msg = await channel.send(message_text)

            _LOG.info("Sent Wendy outbox message to channel %s (msg_id=%s): %s...",
                     channel_id, sent_msg.id, message_text[:50])

            # Log the message correlation for debug lookups
            outbox_ts = self._extract_outbox_timestamp(outbox_file.name)
            if outbox_ts:
                self._log_sent_message(
                    discord_msg_id=sent_msg.id,
                    outbox_ts=outbox_ts,
                    channel_id=channel_id,
                    content=message_text,
                )

            # Delete the file after successful send
            outbox_file.unlink()

        except json.JSONDecodeError as e:
            _LOG.error("Invalid JSON in outbox file %s: %s", outbox_file, e)
            outbox_file.unlink()  # Remove corrupt file
        except Exception as e:
            _LOG.error("Error processing outbox file %s: %s", outbox_file, e)
            # Don't delete - might be temporary error


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(WendyOutbox(bot))
