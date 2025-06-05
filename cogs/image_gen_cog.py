from io import BytesIO

import discord
from discord.ext import commands

from image_generators import ImageGeneratorAPI, ReplicateImageGenerator


class ImageGenCog(commands.Cog):
    """Generate images using a configurable provider."""

    def __init__(self, bot: commands.Bot, generator: ImageGeneratorAPI | None = None):
        self.bot = bot
        self.generator = generator or ReplicateImageGenerator()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if not message.content.startswith("!"):
            return

        prompt = message.content[1:].strip()
        if not prompt:
            return

        thinking = "\N{THINKING FACE}"
        checkmark = "\N{WHITE HEAVY CHECK MARK}"
        try:
            await message.add_reaction(thinking)
        except discord.HTTPException:
            pass

        try:
            image_bytes = await self.generator.generate(prompt)
            file = discord.File(BytesIO(image_bytes), filename="output.png")
            await message.channel.send(file=file)
            if thinking:
                await message.clear_reaction(thinking)
                await message.add_reaction(checkmark)
        except Exception as e:  # noqa: BLE001
            if thinking:
                await message.clear_reaction(thinking)
            await message.add_reaction("\N{CROSS MARK}")
            await message.channel.send(f"Error generating image: {e}")


async def setup(bot: commands.Bot):
    await bot.add_cog(ImageGenCog(bot))
