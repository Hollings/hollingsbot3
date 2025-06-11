from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from inspect import signature
from io import BytesIO
from pathlib import Path
from typing import Awaitable, Callable, Mapping

import discord
from discord.ext import commands

from caption import add_caption
from prompt_db import add_prompt, init_db
from tasks import generate_image

__all__ = ["ImageGenCog"]

_log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- constants

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "image_gen_config.json"

THINKING = "\N{THINKING FACE}"
SUCCESS  = "\N{WHITE HEAVY CHECK MARK}"
FAILURE  = "\N{CROSS MARK}"

_MAX_DISCORD_FILESIZE = 25 * 2**20  # 25 MiB


# --------------------------------------------------------------------------- data model


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    api: str
    model: str


# --------------------------------------------------------------------------- cog


class ImageGenCog(commands.Cog):
    """Turn specially‑prefixed messages into AI‑generated images.

    A user may optionally embed a seed at the very start of the prompt::

        !{1234}A photorealistic otter in a suit

    Curly‑braces *must* immediately follow the prefix.  The seed is forwarded
    to the Replicate backend for deterministic output.
    """

    _SEED_RE = re.compile(r"^\{\s*(\d+)\s*}", re.ASCII)

    def __init__(
        self,
        bot: commands.Bot,
        *,
        config: Mapping[str, Mapping[str, str]] | None = None,
        task_runner: Callable[..., Awaitable[str]] | None = None,
    ) -> None:
        self.bot = bot
        init_db()

        # ------------ load prefix → GeneratorSpec mapping ------------
        raw_cfg: Mapping[str, Mapping[str, str]]
        if config is None:
            try:
                raw_cfg = json.loads(_DEFAULT_CONFIG_PATH.read_text("utf8"))
            except FileNotFoundError:
                _log.warning("No image_gen_config.json found – cog disabled.")
                raw_cfg = {}
        else:
            raw_cfg = config

        self._prefix_map: dict[str, GeneratorSpec] = {
            p.strip(): GeneratorSpec(**spec) for p, spec in raw_cfg.items()
        }

        # ------------ task execution helper --------------------------
        self._run_task = task_runner or self._default_celery_runner
        self._pending: set[asyncio.Task[None]] = set()

    # ---------------------------------------------------------------- life‑cycle

    async def cog_unload(self) -> None:
        for task in self._pending:
            task.cancel()
        await asyncio.gather(*self._pending, return_exceptions=True)

    # ---------------------------------------------------------------- celery helper

    async def _default_celery_runner(
        self,
        prompt_id: int,
        api: str,
        model: str,
        prompt: str,
        seed: int | None,
    ) -> str:
        """Submit the Celery task and wait for the Base‑64 image."""

        task = generate_image.apply_async(
            (prompt_id, api, model, prompt, seed), queue="image"
        )
        return await asyncio.to_thread(task.get)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    async def _react(msg: discord.Message, emoji: str, *, remove: bool = False) -> None:
        try:
            if remove:
                await msg.clear_reaction(emoji)
            else:
                await msg.add_reaction(emoji)
        except discord.HTTPException:
            _log.debug("Could not %s reaction %s on %s", "remove" if remove else "add", emoji, msg.id)

    def _split_prompt(self, content: str) -> tuple[str, GeneratorSpec] | None:
        """If *content* starts with a known prefix return stripped‑prompt & spec."""
        for prefix, spec in self._prefix_map.items():
            if content.startswith(prefix):
                return content[len(prefix):].lstrip(), spec
        return None

    # ---------------------------------------------------------------- main pipeline

    async def _handle_generation(
        self,
        message: discord.Message,
        raw_prompt: str,
        spec: GeneratorSpec,
    ) -> None:
        """Extract seed, call Celery, upload image, manage UX reactions."""

        await self._react(message, THINKING)

        # 1. Extract `{seed}` if present
        seed: int | None = None
        m = self._SEED_RE.match(raw_prompt)
        if m:
            seed = int(m.group(1))
            raw_prompt = raw_prompt[m.end():].lstrip()

        if not raw_prompt:
            await self._react(message, THINKING, remove=True)
            await self._react(message, FAILURE)
            await message.channel.send("⚠️ Prompt may not be empty.")
            return

        # 2. Persist prompt in DB
        prompt_id = add_prompt(raw_prompt, str(message.author.id), spec.api, spec.model)

        # 3. Kick off generation worker
        try:
            b64_img = await self._run_task(prompt_id, spec.api, spec.model, raw_prompt, seed)
            image_bytes = base64.b64decode(b64_img, validate=True)

            if len(image_bytes) > _MAX_DISCORD_FILESIZE:
                raise RuntimeError(
                    f"Image {len(image_bytes)/2**20:.1f} MiB exceeds Discord’s 25 MiB limit."
                )

            captioned = add_caption(image_bytes, raw_prompt)
            await message.channel.send(file=discord.File(BytesIO(captioned), filename="generated.png"))

            await self._react(message, SUCCESS)
        except Exception as exc:  # noqa: BLE001
            _log.exception("Generation failed: %s", exc)
            await self._react(message, FAILURE)
            await message.channel.send(f"⚠️ Image generation failed: **{exc}**")
        finally:
            await self._react(message, THINKING, remove=True)

    # ---------------------------------------------------------------- listener

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return

        split = self._split_prompt(message.content)
        if not split:
            return

        prompt, spec = split
        task = asyncio.create_task(self._handle_generation(message, prompt, spec))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)


# ---------------------------------------------------------------- setup hook


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ImageGenCog(bot))
