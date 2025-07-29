from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
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

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "image_gen_config.json"

THINKING = "\N{THINKING FACE}"
SUCCESS = "\N{WHITE HEAVY CHECK MARK}"
FAILURE = "\N{CROSS MARK}"

_MAX_DISCORD_FILESIZE = 25 * 2**20  # 25 MiB


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    api: str
    model: str


class ImageGenCog(commands.Cog):
    """Generate images **only** in channels listed in the
    ``STABLE_DIFFUSION_CHANNEL_IDS`` environment variable (comma‑separated).
    Responds to ``!models`` with a listing of configured generators.
    """

    _SEED_RE = re.compile(r"^\{\s*(\d+)\s*}", re.ASCII)
    _LIST_RE = re.compile(r"<([^<>]+)>", re.ASCII)  # first (outer‑most) <> list

    # ------------------------------------------------------------------ init

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

        # channel allow‑list
        env_ids = os.getenv("STABLE_DIFFUSION_CHANNEL_IDS", "")
        self._allowed_channel_ids: set[int] = {
            int(cid.strip()) for cid in env_ids.split(",") if cid.strip().isdigit()
        }
        if not self._allowed_channel_ids:
            _log.warning(
                "STABLE_DIFFUSION_CHANNEL_IDS is empty – image generation disabled."
            )

        # dynamic configuration
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

        # task execution helper
        self._run_task = task_runner or self._default_celery_runner
        self._pending: set[asyncio.Task[None]] = set()

    # ---------------------------------------------------------------- config helpers

    def _reload_config(self) -> None:
        """(Re)load prefix → GeneratorSpec mapping from disk if file changed."""
        if self._cfg_path is None:
            return

        try:
            mtime = self._cfg_path.stat().st_mtime
        except FileNotFoundError:
            if self._prefix_map:
                _log.warning("Config file vanished – keeping last‑known map.")
            return

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
        _log.info("Reloaded image‑generator config (%d prefixes).", len(self._prefix_map))

    # ---------------------------------------------------------------- new helper

    def _format_model_listing(self) -> str:
        """Return a human‑readable description of all configured generators."""
        self._reload_config()
        if not self._prefix_map:
            return "⚠️ No image generators are configured."

        lines: list[str] = ["**Available image generators:**"]
        for prefix, spec in sorted(self._prefix_map.items()):
            prefix_display = f"`{prefix}`" if prefix else "(default)"
            lines.append(f"- {prefix_display}: **{spec.api}** / **{spec.model}**")
        return "\n".join(lines)

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
        *,
        poll_interval: float = 0.5,
    ) -> str:
        """
        Launch an image‑generation task on the “image” queue and poll Redis
        until the result is ready.
        """
        async_result = generate_image.apply_async(
            (prompt_id, api, model, prompt, seed),
            queue="image",
        )

        while not async_result.ready():
            await asyncio.sleep(poll_interval)

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
        """Return (prompt_without_prefix, generator_spec) if *content*
        begins with a configured prefix."""
        self._reload_config()

        for prefix in sorted(self._prefix_map, key=len, reverse=True):
            if content.startswith(prefix):
                spec = self._prefix_map[prefix]
                return content[len(prefix):].lstrip(), spec

        return None

    def _build_filename(
        self,
        prompt: str,
        spec: GeneratorSpec,
        seed: int | None,
        *,
        max_len: int = 32,
    ) -> str:
        """Return a Discord‑friendly filename based on *prompt*, *spec*, and *seed*."""
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
        """Extract seed, optionally expand <item, …>, kick off workers and upload images."""
        await self._react(message, THINKING)

        # 1 – Seed handling
        seed = random.randint(1, 1000)
        m_seed = self._SEED_RE.match(raw_prompt)
        if m_seed:
            seed = int(m_seed.group(1))
            raw_prompt = raw_prompt[m_seed.end():].lstrip()

        if not raw_prompt:
            await self._react(message, THINKING, remove=True)
            await self._react(message, FAILURE)
            await message.channel.send("⚠️ Prompt may not be empty.")
            return

        # 2 – Expand optional “<a, b, c>” list
        m_list = self._LIST_RE.search(raw_prompt)
        if m_list:
            items = [s.strip() for s in m_list.group(1).split(",") if s.strip()]
            if not items:
                items = [m_list.group(0)]  # treat “<>” as literal if empty
            prefix = raw_prompt[:m_list.start()]
            suffix = raw_prompt[m_list.end():]
            prompts = [f"{prefix}{item}{suffix}".strip() for item in items]
        else:
            prompts = [raw_prompt]

        # 3 – Kick off generation workers concurrently

        async def _launch_prompt(p: str) -> tuple[str, bytes] | Exception:
            """
            Launch a single Celery task and return (prompt_text, image_bytes) on success.
            The worker now gives us a **file‑path**, so we read the file and delete it.
            """
            prompt_id = add_prompt(p, str(message.author.id), spec.api, spec.model)
            try:
                file_path_str = await self._run_task(
                    prompt_id, spec.api, spec.model, p, seed
                )
                img_path = Path(file_path_str)
                img_bytes = img_path.read_bytes()
                try:
                    img_path.unlink(missing_ok=True)
                except Exception:
                    _log.debug("Temp image %s could not be deleted", img_path)
                return p, img_bytes
            except Exception as exc:  # noqa: BLE001
                return exc

        coros = [_launch_prompt(p) for p in prompts]
        results = await asyncio.gather(*coros, return_exceptions=True)

        overall_success = True
        for prompt_variant, result in zip(prompts, results):
            if isinstance(result, Exception):
                overall_success = False
                _log.exception("Generation failed for '%s': %s", prompt_variant, result)
                await message.channel.send(
                    f"⚠️ Image generation failed for **{prompt_variant}**:\n> {result}"
                )
                continue

            prompt_text, img_bytes = result
            try:
                if len(img_bytes) > _MAX_DISCORD_FILESIZE:
                    raise RuntimeError(
                        f"Image {len(img_bytes) / 2 ** 20:.1f} MiB exceeds Discord’s 25 MiB limit."
                    )

                captioned = add_caption(img_bytes, prompt_text)
                filename = self._build_filename(prompt_text, spec, seed)
                await message.channel.send(
                    file=discord.File(BytesIO(captioned), filename=filename)
                )
            except Exception as exc:  # noqa: BLE001
                overall_success = False
                _log.exception("Post‑processing failed: %s", exc)
                await message.channel.send(
                    f"⚠️ Image post‑processing failed for **{prompt_variant}**:\n> {exc}"
                )

        await self._react(message, THINKING, remove=True)
        await self._react(message, SUCCESS if overall_success else FAILURE)

    # ---------------------------------------------------------------- listener

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        # Only respond in whitelisted channels
        if message.channel.id not in self._allowed_channel_ids:
            return
        if message.author.bot or message.guild is None:
            return

        cleaned = message.content.strip()

        # ---- new command: !models ---------------------------------------
        if cleaned.lower() == "!models":
            await message.channel.send(self._format_model_listing())
            return

        split = self._split_prompt(cleaned)
        if not split:
            return

        prompt, spec = split
        task = asyncio.create_task(self._handle_generation(message, prompt, spec))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ImageGenCog(bot))
