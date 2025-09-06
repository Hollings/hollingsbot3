from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Awaitable, Callable, Mapping

import discord
from discord.ext import commands

from hollingsbot.caption import add_caption
from hollingsbot.prompt_db import add_prompt, init_db
from hollingsbot.tasks import generate_image  # celery task

__all__ = ["ImageGenCog"]

_log = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "image_gen_config.json"

THINKING = "\N{THINKING FACE}"
SUCCESS = "\N{WHITE HEAVY CHECK MARK}"
FAILURE = "\N{CROSS MARK}"

_MAX_DISCORD_FILESIZE = 25 * 2**20  # 25 MiB


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    api: str
    model: str
    # Optional: enable editing route by config
    # Example: {"nb:": {"api":"replicate","model":"google/nano-banana","mode":"edit"}}
    mode: str = "generate"


class ImageGenCog(commands.Cog):
    _SEED_RE = re.compile(r"^\{\s*(\d+)\s*}", re.ASCII)
    _LIST_RE = re.compile(r"<([^<>]+)>", re.ASCII)

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

        env_ids = os.getenv("STABLE_DIFFUSION_CHANNEL_IDS", "")
        self._allowed_channel_ids: set[int] = {
            int(cid.strip()) for cid in env_ids.split(",") if cid.strip().isdigit()
        }

        # Optional separate allowlist for the "edit:" command specifically.
        # If set, messages starting with "edit:" are also allowed in these channels
        # in addition to any general image channels above.
        edit_ids = os.getenv("EDIT_CHANNEL_IDS", "")
        self._edit_channel_ids: set[int] = {
            int(cid.strip()) for cid in edit_ids.split(",") if cid.strip().isdigit()
        }

        allow_str = os.getenv("STABLE_DIFFUSION_ALLOW_DMS", "1").strip().lower()
        self._allow_dms: bool = allow_str in {"1", "true", "yes", "on"}

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
            self._reload_config()

        self._run_task = task_runner or self._default_celery_runner
        self._pending: set[asyncio.Task[None]] = set()

    def _reload_config(self) -> None:
        if self._cfg_path is None:
            return
        try:
            mtime = self._cfg_path.stat().st_mtime
        except FileNotFoundError:
            if self._prefix_map:
                _log.warning("Config file vanished; keeping last-known map.")
            return
        if mtime == self._cfg_mtime:
            return
        try:
            raw_cfg: Mapping[str, Mapping[str, str]] = json.loads(
                self._cfg_path.read_text("utf8")
            )
        except Exception as exc:
            _log.exception("Failed to parse %s: %s; keeping old config", self._cfg_path, exc)
            return
        self._prefix_map = {p.strip(): GeneratorSpec(**spec) for p, spec in raw_cfg.items()}
        self._cfg_mtime = mtime
        _log.info("Reloaded image-generator config (%d prefixes).", len(self._prefix_map))

    def _format_model_listing(self) -> str:
        self._reload_config()
        if not self._prefix_map:
            return "No image generators are configured."
        lines: list[str] = ["Available image generators:"]
        for prefix, spec in sorted(self._prefix_map.items()):
            prefix_display = f"`{prefix}`" if prefix else "(default)"
            mode_display = f" / mode={spec.mode}" if getattr(spec, "mode", "generate") != "generate" else ""
            lines.append(f"- {prefix_display}: {spec.api} / {spec.model}{mode_display}")
        lines.append(f"\nDM support: {'enabled' if self._allow_dms else 'disabled'}")
        allowlist_desc = (
            "all guild channels"
            if not self._allowed_channel_ids
            else f"{len(self._allowed_channel_ids)} whitelisted channel(s)"
        )
        lines.append(f"Guild scope: {allowlist_desc}")
        if self._edit_channel_ids:
            lines.append(f"Edit scope: {len(self._edit_channel_ids)} whitelisted channel(s)")
        return "\n".join(lines)

    async def cog_unload(self) -> None:
        for task in self._pending:
            task.cancel()
        await asyncio.gather(*self._pending, return_exceptions=True)

    async def _default_celery_runner(
            self,
            prompt_id: int,
            api: str,
            model: str,
            prompt: str,
            seed: int | None,
            *,
            image_input: list[bytes] | None = None,
            output_format: str | None = None,
            poll_interval: float = 0.5,
    ) -> str:
        """
        Launch an image-generation task and poll. Keep routing through Celery.
        The task must accept image_input and output_format as keyword-only.
        """
        # Celery's default JSON serializer cannot carry raw bytes; encode as data URLs.
        def _as_data_urls(images: list[bytes]) -> list[str]:
            def _mime(b: bytes) -> str:
                try:
                    if b.startswith(b"\x89PNG"):
                        return "image/png"
                    if b.startswith(b"\xff\xd8"):
                        return "image/jpeg"
                    if b.startswith(b"RIFF") and b[8:12] == b"WEBP":
                        return "image/webp"
                    if b.startswith(b"BM"):
                        return "image/bmp"
                except Exception:
                    pass
                return "application/octet-stream"

            out: list[str] = []
            for img in images:
                b64 = base64.b64encode(img).decode("ascii")
                out.append(f"data:{_mime(img)};base64,{b64}")
            return out

        payload_images: list[str] | None
        if image_input:
            payload_images = _as_data_urls(image_input)
        else:
            payload_images = None

        async_result = generate_image.apply_async(
            (prompt_id, api, model, prompt, seed),  # positional args only
            kwargs={
                "image_input": payload_images,  # keyword-only
                "output_format": output_format,  # keyword-only
                # "timeout": 45.0,  # example if you want to override
            },
            queue="image",
        )
        while not async_result.ready():
            await asyncio.sleep(poll_interval)
        return await asyncio.to_thread(async_result.get)

    async def _react(self, msg: discord.Message, emoji: str, *, remove: bool = False) -> None:
        try:
            if remove:
                # Remove only the bot's own reaction to avoid requiring
                # Manage Messages permission (clear_reaction clears all).
                if self.bot.user is not None:
                    await msg.remove_reaction(emoji, self.bot.user)
                else:
                    await msg.clear_reaction(emoji)
            else:
                await msg.add_reaction(emoji)
        except discord.HTTPException:
            _log.debug("Could not %s reaction %s on %s", "remove" if remove else "add", emoji, msg.id)

    def _split_prompt(self, content: str) -> tuple[str, GeneratorSpec] | None:
        """Return (prompt_without_prefix, spec) if content starts with a known prefix.
        Matching is case-insensitive for human-friendly prefixes like "edit:".
        """
        self._reload_config()
        content_l = content.lower()
        for prefix in sorted(self._prefix_map, key=len, reverse=True):
            if content_l.startswith(prefix.lower()):
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
        snippet = re.sub(r"[^A-Za-z0-9]+", "_", prompt).strip("_")
        if not snippet:
            snippet = "image"
        if len(snippet) > max_len:
            snippet = snippet[:max_len].rstrip("_")
        api_model = f"{spec.api}-{spec.model}".replace("/", "-")
        seed_part = str(seed) if seed is not None else "rand"
        return f"{snippet}_{api_model}_seed_{seed_part}.png".lower()

    async def _handle_generation(
        self,
        message: discord.Message,
        raw_prompt: str,
        spec: GeneratorSpec,
    ) -> None:
        await self._react(message, THINKING)

        seed = random.randint(1, 1000)
        m_seed = self._SEED_RE.match(raw_prompt)
        if m_seed:
            seed = int(m_seed.group(1))
            raw_prompt = raw_prompt[m_seed.end():].lstrip()

        if not raw_prompt:
            await self._react(message, THINKING, remove=True)
            await self._react(message, FAILURE)
            await message.channel.send("Prompt may not be empty.")
            return

        # Expand optional "<a, b, c>"
        m_list = self._LIST_RE.search(raw_prompt)
        if m_list:
            items = [s.strip() for s in m_list.group(1).split(",") if s.strip()]
            if not items:
                items = [m_list.group(0)]
            prefix = raw_prompt[:m_list.start()]
            suffix = raw_prompt[m_list.end():]
            prompts = [f"{prefix}{item}{suffix}".strip() for item in items]
        else:
            prompts = [raw_prompt]

        # Collect image attachments for possible editing path (from this message and, if replying, from the replied message)
        images: list[bytes] = []
        for att in message.attachments:
            ct = (att.content_type or "").lower()
            if ct.startswith("image/") or att.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                try:
                    images.append(await att.read())
                except discord.HTTPException:
                    _log.debug("Could not download attachment %s", att.id)

        # If the user is replying to a message with images, include those images as edit inputs too
        reply_images: list[bytes] = []
        if message.reference is not None:
            replied_msg: discord.Message | None = None
            resolved = getattr(message.reference, "resolved", None)
            if isinstance(resolved, discord.Message):
                replied_msg = resolved
            else:
                ref_id = getattr(message.reference, "message_id", None)
                if ref_id:
                    try:
                        replied_msg = await message.channel.fetch_message(ref_id)
                    except Exception:
                        replied_msg = None
            if replied_msg is not None:
                for att in replied_msg.attachments:
                    ct = (att.content_type or "").lower()
                    if ct.startswith("image/") or att.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                        try:
                            reply_images.append(await att.read())
                        except discord.HTTPException:
                            _log.debug("Could not download replied attachment %s", att.id)

        all_edit_images = images + reply_images

        # Editing is triggered if there are images from either source AND either model indicates nano-banana or mode=="edit"
        do_edit_mode = ("nano-banana" in spec.model) or (getattr(spec, "mode", "") == "edit")
        do_edit = bool(all_edit_images) and do_edit_mode

        # If user explicitly asked for edit mode but we found no images anywhere, fail early with guidance
        if getattr(spec, "mode", "") == "edit" and not all_edit_images:
            await self._react(message, THINKING, remove=True)
            await self._react(message, FAILURE)
            await message.channel.send("No images found to edit. Reply to a message with an image or attach an image with your `edit:` prompt.")
            return

        async def _launch_prompt(p: str) -> tuple[str, bytes] | Exception:
            prompt_id = add_prompt(p, str(message.author.id), spec.api, spec.model)
            try:
                # Route through Celery, passing image_input when editing
                file_or_b64 = await self._run_task(
                    prompt_id, spec.api, spec.model, p, seed,
                    image_input=all_edit_images if do_edit else None,
                    output_format="png" if do_edit else None,
                )
                img_path = Path(file_or_b64)
                if img_path.exists():
                    img_bytes = img_path.read_bytes()
                    try:
                        img_path.unlink(missing_ok=True)
                    except Exception:
                        _log.debug("Temp image %s could not be deleted", img_path)
                else:
                    img_bytes = base64.b64decode(file_or_b64)
                return p, img_bytes
            except Exception as exc:
                return exc

        results = await asyncio.gather(*(_launch_prompt(p) for p in prompts), return_exceptions=True)

        overall_success = True
        for prompt_variant, result in zip(prompts, results):
            if isinstance(result, Exception):
                overall_success = False
                _log.exception("Generation failed for %r: %s", prompt_variant, result)
                continue

            prompt_text, img_bytes = result
            try:
                if len(img_bytes) > _MAX_DISCORD_FILESIZE:
                    raise RuntimeError("Image exceeds Discord 25 MiB limit.")

                captioned = img_bytes if do_edit else add_caption(img_bytes, prompt_text)

                filename = self._build_filename(prompt_text, spec, seed)
                await message.channel.send(file=discord.File(BytesIO(captioned), filename=filename))
            except Exception as exc:
                overall_success = False
                _log.exception("Post-processing failed: %s", exc)
                await message.channel.send(f"Image post-processing failed for **{prompt_variant}**:\n> {exc}")

        await self._react(message, THINKING, remove=True)
        await self._react(message, SUCCESS if overall_success else FAILURE)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        is_dm = message.guild is None
        if is_dm:
            if not self._allow_dms:
                return
        else:
            # Check content early to allow a separate allowlist for edit:
            cleaned = (message.content or "").strip()
            if self._allowed_channel_ids and message.channel.id not in self._allowed_channel_ids:
                # Not in general image channels. Allow if this is an edit: prompt
                # and the channel is in EDIT_CHANNEL_IDS.
                if not (cleaned.lower().startswith("edit:") and message.channel.id in self._edit_channel_ids):
                    return
        # For DMs or allowed guild messages, continue.
        cleaned = (message.content or "").strip()

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
