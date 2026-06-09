"""Outbox watcher for Wendy's async message sending.

Watches /data/wendy/outbox/ for JSON files and sends messages to Discord.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import discord
from discord.ext import commands, tasks

_LOG = logging.getLogger(__name__)

OUTBOX_DIR = Path(os.getenv("WENDY_OUTBOX_DIR", "/data/wendy/outbox"))
MESSAGE_LOG_FILE = Path("/data/wendy/message_log.jsonl")
MAX_MESSAGE_LOG_LINES = 1000  # Rolling log limit
MIN_FILE_AGE_SECONDS = 2.0  # Skip files this young - the producer may still be writing
MAX_SEND_ATTEMPTS = 5  # After this many failures, quarantine the file as .failed
DISCORD_MESSAGE_LIMIT = 2000


class WendyOutbox(commands.Cog):
    """Watches Wendy's outbox and sends messages to Discord."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._send_attempts: dict[str, int] = {}
        self._ensure_outbox_dir()
        self.watch_outbox.start()
        _LOG.info("WendyOutbox initialized, watching %s", OUTBOX_DIR)

    def _ensure_outbox_dir(self) -> None:
        """Create outbox directory if it doesn't exist."""
        OUTBOX_DIR.mkdir(parents=True, exist_ok=True)

    def _log_sent_message(self, discord_msg_id: int, outbox_ts: int, channel_id: int, content: str) -> None:
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

    def _outbox_sort_key(self, file_path: Path) -> tuple[int, str]:
        """Sort outbox files chronologically by embedded timestamp."""
        ts = self._extract_outbox_timestamp(file_path.name)
        return (ts if ts is not None else 0, file_path.name)

    @tasks.loop(seconds=0.5)
    async def watch_outbox(self):
        """Check for new messages in the outbox."""
        try:
            for file_path in sorted(OUTBOX_DIR.glob("*.json"), key=self._outbox_sort_key):
                await self._process_outbox_file(file_path)
        except Exception as e:
            _LOG.error("Error watching outbox: %s", e)

    @watch_outbox.before_loop
    async def before_watch(self):
        await self.bot.wait_until_ready()

    @staticmethod
    def _chunk_message(text: str) -> list[str]:
        """Split a message into Discord-sized chunks, preferring newline boundaries."""
        if len(text) <= DISCORD_MESSAGE_LIMIT:
            return [text]
        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= DISCORD_MESSAGE_LIMIT:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, DISCORD_MESSAGE_LIMIT)
            if split_at <= 0:
                split_at = DISCORD_MESSAGE_LIMIT
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")
        return chunks

    def _discard_file(self, outbox_file: Path) -> None:
        """Remove an outbox file and forget its retry count."""
        self._send_attempts.pop(outbox_file.name, None)
        try:
            outbox_file.unlink()
        except FileNotFoundError:
            pass

    def _quarantine_file(self, outbox_file: Path) -> None:
        """Move a repeatedly-failing file out of the watch glob for inspection."""
        self._send_attempts.pop(outbox_file.name, None)
        try:
            outbox_file.rename(outbox_file.with_suffix(".json.failed"))
            _LOG.error("Quarantined outbox file after %d failed attempts: %s", MAX_SEND_ATTEMPTS, outbox_file)
        except OSError:
            _LOG.exception("Failed to quarantine outbox file %s", outbox_file)

    async def _process_outbox_file(self, outbox_file: Path) -> None:
        """Process a single outbox file and send the message."""
        try:
            data = json.loads(outbox_file.read_text())
            channel_id = int(data["channel_id"])
            message_text = data["message"]
        except FileNotFoundError:
            return  # Already handled (e.g. picked up by a previous iteration)
        except json.JSONDecodeError as e:
            # The producer writes non-atomically, so a fresh file may simply be
            # mid-write. Only treat it as corrupt once it has had time to finish.
            try:
                age = time.time() - outbox_file.stat().st_mtime
            except FileNotFoundError:
                return
            if age < MIN_FILE_AGE_SECONDS:
                return
            _LOG.error("Invalid JSON in outbox file %s: %s", outbox_file, e)
            self._discard_file(outbox_file)
            return
        except (KeyError, TypeError, ValueError) as e:
            # Valid JSON but missing/malformed fields - it will never send.
            _LOG.error("Malformed outbox file %s: %s", outbox_file, e)
            self._discard_file(outbox_file)
            return

        try:
            channel = self.bot.get_channel(channel_id)
            if not channel:
                _LOG.warning("Channel %s not found, skipping message", channel_id)
                self._discard_file(outbox_file)
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

            # Send the message (chunked if needed, attachment on the first chunk)
            chunks = self._chunk_message(message_text) if message_text else [message_text]
            sent_msg = await channel.send(chunks[0], file=attachment) if attachment else await channel.send(chunks[0])
            for extra_chunk in chunks[1:]:
                await channel.send(extra_chunk)

            _LOG.info(
                "Sent Wendy outbox message to channel %s (msg_id=%s): %s...", channel_id, sent_msg.id, message_text[:50]
            )

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
            self._discard_file(outbox_file)

        except Exception as e:
            attempts = self._send_attempts.get(outbox_file.name, 0) + 1
            self._send_attempts[outbox_file.name] = attempts
            _LOG.error(
                "Error processing outbox file %s (attempt %d/%d): %s", outbox_file, attempts, MAX_SEND_ATTEMPTS, e
            )
            if attempts >= MAX_SEND_ATTEMPTS:
                self._quarantine_file(outbox_file)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(WendyOutbox(bot))
