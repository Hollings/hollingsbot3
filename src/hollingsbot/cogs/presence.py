"""Shows the running version and start time in the bot's Discord presence."""

from __future__ import annotations

import asyncio
import logging

import discord
from discord.ext import commands

from hollingsbot.version import presence_text

__all__ = ["PresenceCog"]

_log = logging.getLogger(__name__)


class PresenceCog(commands.Cog):
    """Sets a custom status like ``v2e0c5d0 · up Sep 17 14:02`` so the live commit is visible at a glance."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        # The SHA can't change while this process lives (a deploy or hot-reload
        # starts a new one), so resolve it once, off the event loop.
        self._text: str | None = None

    async def _update_presence(self) -> None:
        try:
            if self._text is None:
                self._text = await asyncio.to_thread(presence_text)
            await self.bot.change_presence(activity=discord.CustomActivity(name=self._text))
            _log.info("Presence set to %r", self._text)
        except Exception:
            _log.exception("Failed to update presence")

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        # Fires on every fresh gateway session, so the presence survives reconnects.
        await self._update_presence()

    async def cog_load(self) -> None:
        # Extension (re)loaded while already connected: on_ready won't fire again.
        if self.bot.is_ready():
            await self._update_presence()


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PresenceCog(bot))
