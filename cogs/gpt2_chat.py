# cogs/gpt2_chat.py
from __future__ import annotations

import asyncio
import os
from typing import Callable, Awaitable, Dict

import discord
from celery.exceptions import TimeoutError as CeleryTimeoutError
from discord.ext import commands

from tasks import generate_text


class GPT2Chat(commands.Cog):
    """Respond to messages in a designated channel using GPT-2 (via Celery).

    If multiple messages arrive faster than the model can reply, only the most
    recent prompt per channel will be answered. Older, still-running generation
    tasks are allowed to finish but their results are discarded.
    """

    _MAX_DISCORD_LEN: int = 2_000

    def __init__(
        self,
        bot: commands.Bot,
        *,
        channel_id: int | None = None,
        api: str = "huggingface",
        model: str = "gpt2-medium",
        task_func: Callable[[str, str, str], Awaitable[str]] | None = None,
        timeout: int | None = None,
    ) -> None:
        self.bot = bot

        # Channel configuration ------------------------------------------------
        if channel_id is None:
            cid = os.getenv("GPT2_CHANNEL_ID")
            channel_id = int(cid) if cid else None
        self.channel_id = channel_id

        # Generation configuration --------------------------------------------
        self.api = api
        self.model = model
        self.task_func = task_func or self._celery_task
        self.timeout = timeout or int(os.getenv("GPT2_RESPONSE_TIMEOUT", "120"))

        # Keep track of the newest message per channel so we can ignore stale jobs
        self._latest: Dict[int, int] = {}

    # ---------------------------------------------------------------- helpers

    def _should_respond(self, message: discord.Message) -> bool:
        if message.content.lower() == "enhance":
            return False
        if message.author.bot:
            return False
        if self.channel_id is None:
            return True
        return getattr(message.channel, "id", None) == self.channel_id

    async def _celery_task(self, api: str, model: str, prompt: str) -> str:
        task = generate_text.apply_async((api, model, prompt), queue="text")
        try:
            # Run the potentially blocking ``task.get`` call in a thread.
            return await asyncio.to_thread(task.get, timeout=self.timeout)
        except CeleryTimeoutError:
            raise RuntimeError(
                f"Model did not respond within {self.timeout} s. Please try again."
            )

    async def _generate(self, prompt: str) -> str:
        """Wrapper around *task_func* that converts exceptions to friendly text."""
        try:
            return await self.task_func(self.api, self.model, prompt)
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ Error generating response: {exc}"

    # ---------------------------------------------------------------- listeners

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not self._should_respond(message):
            return

        prompt = message.content.strip()
        if not prompt:
            return

        channel_id = getattr(message.channel, "id", None)
        if channel_id is None:
            return  # should never happen

        # Mark this as the newest prompt we care about for *channel_id*
        self._latest[channel_id] = message.id
        reply = await self._generate(prompt)

        # If a newer prompt arrived while we were waiting, drop this response.
        if self._latest.get(channel_id) != message.id:
            return

        if reply:
            await message.channel.send(reply[: self._MAX_DISCORD_LEN])


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(GPT2Chat(bot))
