import asyncio
import io
import os
import re
import textwrap
from typing import Optional

import discord
from discord.ext import commands
from PIL import Image
import openai
from hollingsbot.image_generators.upscalers import RealESRGANUpscaler


class ImageEditCog(commands.Cog):
    """
    Listens for a user reply to a message that contains an image.
    The reply’s content is forwarded to ChatGPT, which responds with
    a Python function called `edit_image(img)` taking a PIL.Image and
    returning a new PIL.Image. That function is executed on the image,
    and the bot sends the edited result as a reply.
    """

    CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)

    def __init__(self, bot: commands.Bot, *, model: str = "gpt-4o-mini"):
        self.bot = bot
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        # Enable AI upscaling by default (if Replicate token is configured)
        self._upscaler = RealESRGANUpscaler()

    async def cog_unload(self) -> None:
        try:
            await self._upscaler.aclose()
        except Exception:
            pass

    # -------- helpers -------- #

    async def _download_image(self, attachment: discord.Attachment) -> Image.Image:
        """Download the attachment and return a PIL Image."""
        data = await attachment.read()
        return Image.open(io.BytesIO(data)).convert("RGBA")

    def _extract_code(self, response: str) -> Optional[str]:
        """Grab the first Python code block from ChatGPT’s reply."""
        match = self.CODE_BLOCK_RE.search(response)
        return match.group(1) if match else None

    async def _run_edit_function(self, code: str, img: Image.Image) -> Image.Image:
        """
        Execute the provided code to obtain `edit_image`
        and run it in a thread to avoid blocking.
        """
        local_ns = {}
        exec(textwrap.dedent(code), {}, local_ns)
        func = local_ns.get("edit_image")
        if not callable(func):
            raise RuntimeError("No function named `edit_image` was found.")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, img)

    # -------- main listener -------- #

    @commands.Cog.listener("on_message")
    async def handle_edit_request(self, message: discord.Message) -> None:
        # Ignore bot messages and DMs
        if message.author.bot or not message.guild:
            return

        # Must be a reply to a message with an image attachment
        ref = message.reference
        if not ref or not ref.resolved:
            return

        original: discord.Message = ref.resolved  # type: ignore
        img_attachments = [
            a for a in original.attachments if (a.content_type or "").startswith("image/")
        ]
        if not img_attachments:
            return

        # Special case: 'upscale' command on a reply to an image
        cmd = (message.content or "").strip().lower()
        if cmd == "upscale":
            try:
                first_att = img_attachments[0]
                raw = await first_att.read()
                target_bytes = getattr(first_att, "size", None) or len(raw)
                upscaled = await self._upscaler.upscale(raw, target_bytes=target_bytes)
                buf = io.BytesIO(upscaled)
                fname = f"upscaled_{first_att.filename.rsplit('.',1)[0]}.jpg"
                await message.reply(file=discord.File(buf, filename=fname), mention_author=False)
            except Exception as exc:
                await message.reply(f"Upscale failed: {exc}", mention_author=False)
            return

        # Fetch the image (first one only) for general edit instructions
        try:
            first_att = img_attachments[0]
            image = await self._download_image(first_att)
        except Exception as exc:
            await message.reply(f"Failed to download image: {exc}")
            return

        # Ask ChatGPT for an editing function
        prompt = (
            "You are an image‑editing assistant. "
            "Given the user’s instruction below, return only one Python 3 function "
            "named `edit_image` that takes a `PIL.Image` and returns a new `PIL.Image`. "
            "Respond with **just** the code inside a ```python``` block—no prose.\n\n"
            f"Instruction: {message.content.strip()}"
        )

        try:
            resp = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            gpt_reply = resp.choices[0].message.content
            code = self._extract_code(gpt_reply)
            if not code:
                raise ValueError("No code block returned by ChatGPT.")
        except Exception as exc:
            await message.reply(f"ChatGPT error: {exc}")
            return

        # Execute the code on the image
        try:
            edited = await self._run_edit_function(code, image)
            buf = io.BytesIO()
            edited.save(buf, format="PNG")
            buf.seek(0)
            filename = f"edited_{first_att.filename.rsplit('.',1)[0]}.png"
        except Exception as exc:
            await message.reply(f"Image processing error: {exc}")
            return

        # Send the edited image as a reply
        await message.reply(
            file=discord.File(buf, filename=filename),
            mention_author=False,
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(ImageEditCog(bot))
