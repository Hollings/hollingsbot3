"""Outbox watcher for Wendy's async message sending.

Watches /data/wendy/outbox/ for JSON files and sends messages to Discord.
"""
import asyncio
import json
import logging
import os
from pathlib import Path

import discord
from discord.ext import commands, tasks

_LOG = logging.getLogger(__name__)

OUTBOX_DIR = Path(os.getenv("WENDY_OUTBOX_DIR", "/data/wendy/outbox"))


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

    async def _process_outbox_file(self, file_path: Path) -> None:
        """Process a single outbox file and send the message."""
        try:
            data = json.loads(file_path.read_text())
            channel_id = int(data["channel_id"])
            message_text = data["message"]

            channel = self.bot.get_channel(channel_id)
            if not channel:
                _LOG.warning("Channel %s not found, skipping message", channel_id)
                file_path.unlink()
                return

            # Send the message
            await channel.send(message_text)
            _LOG.info("Sent Wendy outbox message to channel %s: %s...",
                     channel_id, message_text[:50])

            # Delete the file after successful send
            file_path.unlink()

        except json.JSONDecodeError as e:
            _LOG.error("Invalid JSON in outbox file %s: %s", file_path, e)
            file_path.unlink()  # Remove corrupt file
        except Exception as e:
            _LOG.error("Error processing outbox file %s: %s", file_path, e)
            # Don't delete - might be temporary error


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(WendyOutbox(bot))
