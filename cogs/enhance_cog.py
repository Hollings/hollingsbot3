from __future__ import annotations

import os

import discord
from discord.ext import commands

import anthropic
from typing import Callable, Awaitable


class EnhanceCog(commands.Cog):
    """Reply to quoted messages with an enhanced version via Claude."""

    def __init__(
        self,
        bot: commands.Bot,
        *,
        prompt: str | None = None,
        model: str | None = None,
        enhance_func: Callable[[str, str], Awaitable[str]] | None = None,
    ) -> None:
        self.bot = bot
        self.prompt = prompt or os.getenv("ENHANCE_PROMPT", "")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        self.enhance_func = enhance_func or self._api_call
        self.client = anthropic.Client(auth_token=os.getenv("ANTHROPIC_API_KEY", ""))

    async def _api_call(self, prompt: str, text: str) -> str:
        full_prompt = f"{prompt}\n=====\n{text}"
        return str(self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        ).content)


    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if message.content.strip().lower() != "enhance":
            return
        reference = message.reference
        if not reference or not reference.message_id:
            return
        original: discord.Message | None = getattr(reference, "resolved", None)
        if original is None:
            try:
                original = await message.channel.fetch_message(reference.message_id)
            except discord.HTTPException:
                return
        text = (original.content or "")[:2000]
        if not text:
            return
        reply = await self.enhance_func(self.prompt, text)
        if reply:
            await message.reply(reply[:2000])


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(EnhanceCog(bot))
