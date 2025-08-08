from __future__ import annotations

import os
import asyncio
from io import BytesIO
from typing import Callable, Awaitable

import discord
from discord.ext import commands
import anthropic

from image_generators import get_image_generator


class EnhanceCog(commands.Cog):
    """Reply to quoted messages with an enhanced version *and* a matching image.

    The original behaviour (text enhancement via Claude/Anthropic) is preserved.
    After generating the improved text we also call a Replicate‑backed image
    generator and attach the result to the reply.
    """

    _MAX_TEXT_LEN: int = 2_000
    _MAX_PROMPT_LEN: int = 300  # feed at most this many chars to the image model

    def __init__(
        self,
        bot: commands.Bot,
        *,
        prompt: str | None = None,
        model: str | None = None,
        image_model: str | None = None,
        enhance_func: Callable[[str, str], Awaitable[str]] | None = None,
    ) -> None:
        self.bot = bot

        # Anthropic / Claude settings --------------------------------------
        self.prompt = prompt or os.getenv("ENHANCE_PROMPT", "")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        self.enhance_func = enhance_func or self._api_call
        self.client = anthropic.Client(auth_token=os.getenv("ANTHROPIC_API_KEY", ""))

        # Image‑generation settings ----------------------------------------
        self.image_api: str = "replicate"  # for future extensibility
        self.image_model: str = (
            image_model
            or os.getenv("ENHANCE_IMAGE_MODEL", "black-forest-labs/flux-schnell")
        )
        self._image_gen = get_image_generator(self.image_api, self.image_model)

    # ----------------------------------------------------------------- Anthropic

    async def _api_call(self, prompt: str, text: str) -> str:
        """Call the Anthropic API **without blocking** the event‑loop."""
        full_prompt = f"{prompt} ```{text}```"

        def _sync_call():
            return self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": full_prompt}],
            )

        response = await asyncio.to_thread(_sync_call)
        content = response.content
        if isinstance(content, str):  # older models
            return content.strip()

        parts: list[str] = []
        for block in content:  # type: ignore[arg-type]
            if hasattr(block, "text"):
                parts.append(str(block.text))
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()

    # -------------------------------------------------------------- Image helper

    async def _generate_image(self, prompt: str) -> bytes:
        """Generate an image for *prompt* using the configured provider."""
        # Truncate: extremely long prompts can blow up token limits/costs.
        return await self._image_gen.generate(prompt[: self._MAX_PROMPT_LEN])

    async def cog_unload(self) -> None:
        """Ensure the image generator is cleaned up when the cog is unloaded."""
        await self._image_gen.aclose()

    # ---------------------------------------------------------------- listeners

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
        text = (original.content or "")[: self._MAX_TEXT_LEN]
        if not text:
            return

        # Step 1 ─ enhanced text
        reply_text = await self.enhance_func(self.prompt, text)
        if not reply_text:
            return

        # Step 2 ─ generate image (best‑effort)
        file: discord.File | None = None
        try:
            image_bytes = await self._generate_image(reply_text)
            file = discord.File(BytesIO(image_bytes), filename="enhanced.png")
        except Exception as exc:  # noqa: BLE001
            # Don’t fail the whole command – just log and continue.
            print(f"[EnhanceCog] Image generation failed: {exc}")

        # Step 3 ─ reply with text (+ image if available)
        await message.reply(reply_text[: self._MAX_TEXT_LEN], file=file)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(EnhanceCog(bot))
