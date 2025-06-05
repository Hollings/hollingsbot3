from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Callable, Mapping, Awaitable

from caption import add_caption
from prompt_db import add_prompt, init_db

import discord
from discord.ext import commands

from tasks import generate_image
import base64
import asyncio


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "image_gen_config.json"


class ImageGenCog(commands.Cog):
    """Generate images using configurable providers."""

    def __init__(
        self,
        bot: commands.Bot,
        *,
        config: Mapping[str, Mapping[str, str]] | None = None,
        task_func: Callable[[int, str, str, str], Awaitable[str]] | None = None,
    ) -> None:
        self.bot = bot
        init_db()
        if config is None:
            with open(DEFAULT_CONFIG_PATH) as f:
                config = json.load(f)
        self.config = config
        self.task_func = task_func or self._celery_task

    async def _celery_task(self, prompt_id: int, api: str, model: str, prompt: str) -> str:
        task = generate_image.delay(prompt_id, api, model, prompt)
        return await asyncio.to_thread(task.get)

    def _parse_prompt(
        self, message: discord.Message
    ) -> tuple[Mapping[str, str], str] | None:
        """Return the generator spec and prompt if the message has a known prefix."""
        for prefix, spec in self.config.items():
            if message.content.startswith(prefix):
                prompt = message.content[len(prefix) :].strip()
                if prompt:
                    return spec, prompt
        return None

    async def _generate_and_send(
        self, message: discord.Message, spec: Mapping[str, str], prompt: str
    ) -> None:
        """Generate an image via Celery and respond with the appropriate reactions."""
        thinking = "\N{THINKING FACE}"
        checkmark = "\N{WHITE HEAVY CHECK MARK}"

        try:
            await message.add_reaction(thinking)
        except discord.HTTPException:
            pass

        prompt_id = add_prompt(prompt, str(message.author.id), spec.get("api"), spec.get("model"))
        try:
            b64 = await self.task_func(prompt_id, spec.get("api"), spec.get("model"), prompt)
            image_bytes = base64.b64decode(b64)
            image_bytes = add_caption(image_bytes, prompt)
            file = discord.File(BytesIO(image_bytes), filename="output.png")
            await message.channel.send(file=file)
            await message.clear_reaction(thinking)
            await message.add_reaction(checkmark)
        except Exception as e:  # noqa: BLE001
            await message.clear_reaction(thinking)
            await message.add_reaction("\N{CROSS MARK}")
            await message.channel.send(f"Error generating image: {e}")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        parsed = self._parse_prompt(message)
        if not parsed:
            return

        spec, prompt = parsed
        await self._generate_and_send(message, spec, prompt)


async def setup(bot: commands.Bot):
    await bot.add_cog(ImageGenCog(bot))
