from __future__ import annotations

import os
from typing import Set

import discord
from discord.ext import commands


class Starboard(commands.Cog):
    """Repost bot messages that receive reactions."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        channel_id = os.getenv("STARBOARD_CHANNEL_ID")
        self.channel_id = int(channel_id) if channel_id else None
        ignore_channels = os.getenv("STARBOARD_IGNORE_CHANNELS", "")
        self.ignore_channels: Set[int] = {
            int(cid) for cid in ignore_channels.split(",") if cid.strip()
        }
        self._posted: Set[int] = set()

    async def _get_channel(self) -> discord.abc.Messageable | None:
        if self.channel_id is None:
            return None
        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(self.channel_id)
            except discord.HTTPException:
                return None
        return channel

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        if user.bot:
            return
        message = reaction.message
        if getattr(message.channel, "id", None) in self.ignore_channels:
            return
        if not message.author.bot:
            return
        if message.id in self._posted:
            return
        channel = await self._get_channel()
        if channel is None:
            return

        content = message.content or ""
        link = message.jump_url
        text = f"{content}\n{link}" if content else link
        files = []
        for attachment in message.attachments:
            try:
                files.append(await attachment.to_file())
            except Exception:
                continue
        await channel.send(text, files=files)
        self._posted.add(message.id)


async def setup(bot: commands.Bot):
    await bot.add_cog(Starboard(bot))
