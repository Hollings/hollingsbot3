from __future__ import annotations

import asyncio
import os

import discord
from discord.ext import commands
from tasks import generate_text
from typing import Callable, Awaitable


class GPT2Chat(commands.Cog):
    """Chat with GPT-2 in a designated channel."""

    def __init__(
        self,
        bot: commands.Bot,
        *,
        channel_id: int | None = None,
        model: str = "gpt2-large",
        task_func: Callable[[str, str], Awaitable[str]] | None = None,
    ) -> None:
        self.bot = bot
        if channel_id is None:
            cid = os.getenv("GPT2_CHANNEL_ID")
            channel_id = int(cid) if cid else None
        self.channel_id = channel_id
        self.model = model
        self.task_func = task_func or self._celery_task

    def _should_respond(self, message: discord.Message) -> bool:
        if message.author.bot:
            return False
        if self.channel_id is None:
            return True
        return getattr(message.channel, "id", None) == self.channel_id

    async def _celery_task(self, model: str, prompt: str) -> str:
        task = generate_text.delay(model, prompt)
        return await asyncio.to_thread(task.get)

    async def _generate(self, prompt: str) -> str:
        return await self.task_func(self.model, prompt)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not self._should_respond(message):
            return
        prompt = message.content.strip()
        if not prompt:
            return
        reply = await self._generate(prompt)
        if reply:
            await message.channel.send(reply)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(GPT2Chat(bot))
