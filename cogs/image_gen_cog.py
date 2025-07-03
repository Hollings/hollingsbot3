from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
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

_MAX_DISCORD_FILESIZE = 25 * 2**20  # 25 MiB


# --------------------------------------------------------------------------- data model


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    api: str
    model: str


# --------------------------------------------------------------------------- cog


class ImageGenCog(commands.Cog):
    """Turn specially-prefixed messages into AI-generated images.

    A user may optionally embed a seed at the very start of the prompt::

        !{1234}A photorealistic otter in a suit

    Curly-braces *must* immediately follow the prefix.  The seed is forwarded
    to the Replicate backend for deterministic output.

    The generator-prefix mapping is loaded from *image_gen_config.json*.
    The file is watched for changes on every incoming message so edits take
    effect immediately – no bot restart required.
    """

    _SEED_RE = re.compile(r"^\{\s*(\d+)\s*}", re.ASCII)

    # --------------------------------------------------------------------- init

    def __init__(
        self,
        bot: commands.Bot,
        *,
        config: Mapping[str, Mapping[str, str]] | None = None,
        config_path: Path | None = None,
        task_runner: Callable[..., Awaitable[str]] | None = None,
    ) -> None:
        self.bot = bot
        init_db()

        # ------------ dynamic configuration ------------------------------
        #
        # If an explicit mapping is supplied we treat it as static.
        # Otherwise the mapping is loaded from *config_path* (defaults to
        # _DEFAULT_CONFIG_PATH) and transparently reloaded when the file
        # changes on disk.
        #
        if config is not None:
            self._prefix_map: dict[str, GeneratorSpec] = {
                p.strip(): GeneratorSpec(**spec) for p, spec in config.items()
            }
            self._cfg_path: Path | None = None
            self._cfg_mtime: float = 0.0
        else:
            self._cfg_path = config_path or _DEFAULT_CONFIG_PATH
            self._cfg_mtime = 0.0
            self._prefix_map = {}
            self._reload_config()  # initial load

        # ------------ task execution helper ------------------------------
        self._run_task = task_runner or self._default_celery_runner
        self._pending: set[asyncio.Task[None]] = set()

    # --------------------------------------------------------------------- config helpers

    def _reload_config(self) -> None:
        """(Re)load prefix → GeneratorSpec mapping from disk if file changed."""
        if self._cfg_path is None:
            return

        try:
            mtime = self._cfg_path.stat().st_mtime
        except FileNotFoundError:
            if self._prefix_map:
                _log.warning("Config file vanished – keeping last-known map.")
            return

        # Update only when timestamp changed
        if mtime == self._cfg_mtime:
            return

        try:
            raw_cfg: Mapping[str, Mapping[str, str]] = json.loads(
                self._cfg_path.read_text("utf8")
            )
        except Exception as exc:  # noqa: BLE001
            _log.exception("Failed to parse %s: %s – keeping old config", self._cfg_path, exc)
            return

        self._prefix_map = {
            p.strip(): GeneratorSpec(**spec) for p, spec in raw_cfg.items()
        }
        self._cfg_mtime = mtime
        _log.info("Reloaded image-generator config (%d prefixes).", len(self._prefix_map))

    # ---------------------------------------------------------------- life-cycle

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
        *,
        poll_interval: float = 0.5,
    ) -> str:
        """
        Launch an image‑generation task on the “image” queue and poll Redis
        until the result is ready.

        Polling sidesteps Celery’s thread‑unsafe Pub/Sub consumer, so many
        image requests can be awaited concurrently inside the Discord bot.
        The text queue logic remains untouched, keeping GPU usage serialized.
        """
        async_result = generate_image.apply_async(
            (prompt_id, api, model, prompt, seed),
            queue="image",
        )

        # Simple polling loop – cheap GET per iteration, no shared socket
        while not async_result.ready():
            await asyncio.sleep(poll_interval)

        # Task completed: fetch (and propagate) the result or remote exception
        return await asyncio.to_thread(async_result.get)

    # ---------------------------------------------------------------- helpers

    async def _react(self, msg: discord.Message, emoji: str, *, remove: bool = False) -> None:
        try:
            if remove:
                await msg.clear_reaction(emoji)
            else:
                await msg.add_reaction(emoji)
        except discord.HTTPException:
            _log.debug(
                "Could not %s reaction %s on %s",
                "remove" if remove else "add",
                emoji,
                msg.id,
            )

    def _split_prompt(self, content: str) -> tuple[str, GeneratorSpec] | None:
        """Return (*prompt_without_prefix*, *generator_spec*) if *content*
        begins with a configured prefix.

        Multiple-character prefixes are supported and disambiguated by always
        preferring the *longest* matching prefix (e.g. `'$$'` has priority over
        `'$'` when both exist).  Prefix comparison is case-sensitive.
        """
        # Always start with the freshest prefix-to-spec mapping
        self._reload_config()

        # Try the longest prefixes first so that more-specific matches win
        for prefix in sorted(self._prefix_map, key=len, reverse=True):
            if content.startswith(prefix):
                spec = self._prefix_map[prefix]
                return content[len(prefix):].lstrip(), spec

        return None


    # ---------------------------------------------------------------- main pipeline

    def _build_filename(
            self,
            prompt: str,
            spec: GeneratorSpec,
            seed: int | None,
            *,
            max_len: int = 32,
    ) -> str:
        """Return a Discord-friendly filename based on *prompt*, *spec*, and *seed*.

        Format:
            <prompt_snippet>_<api>-<model>_<seed|rand>.png
        """
        snippet = re.sub(r"[^A-Za-z0-9]+", "_", prompt).strip("_")
        if not snippet:
            snippet = "image"
        if len(snippet) > max_len:
            snippet = snippet[:max_len].rstrip("_")

        api_model = f"{spec.api}-{spec.model}".replace("/", "-")
        seed_part = str(seed) if seed is not None else "rand"
        return f"{snippet}_{api_model}_seed_{seed_part}.png".lower()

    # ---------------------------------------------------------------- main pipeline

    async def _handle_generation(
            self,
            message: discord.Message,
            raw_prompt: str,
            spec: GeneratorSpec,
    ) -> None:
        """Extract seed, call Celery, upload image, manage UX reactions."""
        await self._react(message, THINKING)

        # 1. Extract `{seed}` if present
        seed: int = random.randint(1,1000)
        m = self._SEED_RE.match(raw_prompt)
        if m:
            seed = int(m.group(1))
            raw_prompt = raw_prompt[m.end():].lstrip()

        if not raw_prompt:
            await self._react(message, THINKING, remove=True)
            await self._react(message, FAILURE)
            await message.channel.send("⚠️ Prompt may not be empty.")
            return

        # 2. Persist prompt in DB
        prompt_id = add_prompt(raw_prompt, str(message.author.id), spec.api, spec.model)

        # 3. Kick off generation worker
        try:
            b64_img = await self._run_task(
                prompt_id, spec.api, spec.model, raw_prompt, seed
            )
            image_bytes = base64.b64decode(b64_img, validate=True)

            if len(image_bytes) > _MAX_DISCORD_FILESIZE:
                raise RuntimeError(
                    f"Image {len(image_bytes) / 2 ** 20:.1f} MiB exceeds Discord’s 25 MiB limit."
                )

            captioned = add_caption(image_bytes, raw_prompt)
            filename = self._build_filename(raw_prompt, spec, seed)
            await message.channel.send(
                file=discord.File(BytesIO(captioned), filename=filename)
            )

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
