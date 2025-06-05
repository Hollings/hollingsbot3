from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Callable, Mapping

import discord
from discord.ext import commands

from image_generators import ImageGeneratorAPI, ReplicateImageGenerator


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "image_gen_config.json"


class ImageGenCog(commands.Cog):
    """Generate images using configurable providers."""

    def __init__(
        self,
        bot: commands.Bot,
        *,
        config: Mapping[str, Mapping[str, str]] | None = None,
        factories: Mapping[str, Callable[[str | None], ImageGeneratorAPI]] | None = None,
    ) -> None:
        self.bot = bot
        if config is None:
            with open(DEFAULT_CONFIG_PATH) as f:
                config = json.load(f)
        self.generators: dict[str, ImageGeneratorAPI] = {}
        factories = factories or {"replicate": ReplicateImageGenerator}
        for prefix, spec in config.items():
            api = spec.get("api")
            model = spec.get("model")
            factory = factories.get(api)
            if factory is None:
                raise ValueError(f"Unknown API: {api}")
            self.generators[prefix] = factory(model)

    def _parse_prompt(
        self, message: discord.Message
    ) -> tuple[ImageGeneratorAPI, str] | None:
        """Return the generator and prompt if the message has a known prefix."""
        for prefix, generator in self.generators.items():
            if message.content.startswith(prefix):
                prompt = message.content[len(prefix) :].strip()
                if prompt:
                    return generator, prompt
        return None

    async def _generate_and_send(
        self, message: discord.Message, generator: ImageGeneratorAPI, prompt: str
    ) -> None:
        """Generate an image and respond with the appropriate reactions."""
        thinking = "\N{THINKING FACE}"
        checkmark = "\N{WHITE HEAVY CHECK MARK}"

        try:
            await message.add_reaction(thinking)
        except discord.HTTPException:
            pass

        try:
            image_bytes = await generator.generate(prompt)
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

        generator, prompt = parsed
        await self._generate_and_send(message, generator, prompt)


async def setup(bot: commands.Bot):
    await bot.add_cog(ImageGenCog(bot))
