"""Auto-pin cog for pinning messages that get 2+ 📌 reactions."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

if TYPE_CHECKING:
    from discord.abc import GuildChannel

__all__ = ["AutoPin"]

_LOG = logging.getLogger(__name__)

PIN_EMOJI = "📌"
PIN_THRESHOLD = 2


class AutoPin(commands.Cog):
    """Automatically pin messages that receive enough 📌 reactions.

    This cog listens for 📌 reactions and pins messages when they reach
    the threshold (default: 2 reactions).
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the AutoPin cog.

        Args:
            bot: The Discord bot instance
        """
        self.bot = bot

    async def _resolve_channel(self, channel_id: int) -> GuildChannel | None:
        """Resolve a channel by ID, attempting cache first then fetch.

        Args:
            channel_id: The Discord channel ID to resolve

        Returns:
            The resolved channel or None if not found/accessible
        """
        channel = self.bot.get_channel(channel_id)
        if channel is not None:
            return channel
        with suppress(discord.NotFound, discord.Forbidden, discord.HTTPException):
            return await self.bot.fetch_channel(channel_id)
        return None

    async def _fetch_message_safely(
        self, channel: GuildChannel, message_id: int
    ) -> discord.Message | None:
        """Safely fetch a message from a channel.

        Args:
            channel: The channel containing the message
            message_id: The message ID to fetch

        Returns:
            The fetched message or None if not accessible
        """
        try:
            return await channel.fetch_message(message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
            _LOG.debug("Failed to fetch message %s: %s", message_id, e)
            return None

    async def _count_pin_reactions(self, message: discord.Message) -> int:
        """Count the number of 📌 reactions on a message from non-bot users.

        Args:
            message: The message to check

        Returns:
            Number of 📌 reactions from non-bot users
        """
        for reaction in message.reactions:
            if str(reaction.emoji) == PIN_EMOJI:
                # Get users who reacted and filter out bots
                users = [user async for user in reaction.users() if not user.bot]
                return len(users)
        return 0

    async def _pin_message(self, message: discord.Message) -> bool:
        """Attempt to pin a message.

        Args:
            message: The message to pin

        Returns:
            True if successfully pinned, False otherwise
        """
        try:
            await message.pin()
            _LOG.info("Pinned message %s in channel %s", message.id, message.channel.id)
            return True
        except discord.Forbidden:
            _LOG.warning("Missing permissions to pin message %s", message.id)
            return False
        except discord.HTTPException as e:
            _LOG.warning("Failed to pin message %s: %s", message.id, e)
            return False

    @commands.Cog.listener("on_raw_reaction_add")
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        """Handle reaction additions and pin messages with enough 📌 reactions.

        Args:
            payload: The raw reaction event data
        """
        # Only process 📌 reactions
        if str(payload.emoji) != PIN_EMOJI:
            return

        # Ignore DMs
        if payload.guild_id is None:
            return

        # Resolve channel
        channel = await self._resolve_channel(payload.channel_id)
        if channel is None:
            return

        # Fetch message
        message = await self._fetch_message_safely(channel, payload.message_id)
        if message is None:
            return

        # Skip if already pinned
        if message.pinned:
            return

        # Check if threshold is reached
        pin_count = await self._count_pin_reactions(message)
        if pin_count >= PIN_THRESHOLD:
            await self._pin_message(message)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AutoPin(bot))
