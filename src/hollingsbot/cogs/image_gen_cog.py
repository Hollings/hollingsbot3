"""Image generation cog for Discord bot.

This cog handles all image generation commands, including:
- Text-to-image generation via multiple providers (Flux, Stable Diffusion, etc.)
- Image-to-image transformations (upscaling, outpainting)
- Cost tracking and rate limiting per user
- Batch generation and prompt enhancement

Commands are triggered by prefixes (e.g., !flux, !sd) which route to different
generation backends. Image generation is offloaded to Celery workers to prevent
blocking the bot.

Configuration is loaded from image_gen_config.json and environment variables.
See docs/CONFIGURATION.md for details.
"""

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
import emoji
from discord.ext import commands

from hollingsbot.caption import add_caption
from hollingsbot.cost_tracking import CostTracker
from hollingsbot.utils.image_utils import compress_image_to_fit
from hollingsbot.utils.outpaint_utils import create_outpaint_images
from hollingsbot.prompt_db import bulk_add_prompts, init_db
from hollingsbot.tasks import generate_image  # celery task
from hollingsbot.text_generators import get_text_generator

__all__ = ["ImageGenCog"]

_log = logging.getLogger(__name__)

# Configuration
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "image_gen_config.json"

# Reaction emojis
THINKING = "\N{THINKING FACE}"
SUCCESS = "\N{WHITE HEAVY CHECK MARK}"
FAILURE = "\N{CROSS MARK}"

# Discord file size limits
_MAX_DISCORD_FILESIZE = 25 * 2**20  # 25 MiB (fallback)
_DEFAULT_FILESIZE_LIMIT = 8 * 2**20  # 8 MiB default
_COMPRESSION_SAFETY_FACTOR = 0.95

# Generation defaults
_DEFAULT_OUTPAINT_PROMPT_TIMEOUT = 30.0
_DEFAULT_CELERY_POLL_INTERVAL = 0.5
_HISTORY_SEARCH_LIMIT = 10
_MAX_FILENAME_LENGTH = 32
_MAX_PROMPT_LENGTH = 10000  # Maximum allowed prompt length to prevent abuse


# ============================================================================
# Utility Functions
# ============================================================================


def detect_mime_type(data: bytes) -> str:
    """
    Detect the MIME type of image data by examining magic bytes.

    Args:
        data: Raw image bytes

    Returns:
        MIME type string (e.g., 'image/png', 'image/jpeg')
    """
    try:
        if data.startswith(b"\x89PNG"):
            return "image/png"
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if data.startswith(b"RIFF") and len(data) > 12 and data[8:12] == b"WEBP":
            return "image/webp"
        if data.startswith(b"BM"):
            return "image/bmp"
    except Exception:
        pass
    return "application/octet-stream"


def detect_image_extension(data: bytes) -> str:
    """
    Detect the file extension for image data by examining magic bytes.

    Args:
        data: Raw image bytes

    Returns:
        File extension without dot (e.g., 'png', 'jpg')
    """
    try:
        if data.startswith(b"\x89PNG"):
            return "png"
        if data.startswith(b"\xff\xd8"):
            return "jpg"
        if data.startswith(b"RIFF") and len(data) > 12 and data[8:12] == b"WEBP":
            return "webp"
        if data.startswith(b"BM"):
            return "bmp"
    except Exception:
        pass
    return "png"


def bytes_to_data_url(data: bytes) -> str:
    """
    Convert image bytes to a data URL for embedding in JSON.

    Args:
        data: Raw image bytes

    Returns:
        Data URL string (e.g., 'data:image/png;base64,...')
    """
    mime_type = detect_mime_type(data)
    b64_data = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64_data}"


def bytes_list_to_data_urls(images: list[bytes]) -> list[str]:
    """
    Convert a list of image bytes to data URLs.

    Args:
        images: List of raw image bytes

    Returns:
        List of data URL strings
    """
    return [bytes_to_data_url(img) for img in images]


@dataclass(frozen=True)
class GeneratorSpec:
    """Configuration for a specific image generator."""

    api: str
    model: str
    mode: str = "generate"  # "generate", "edit", or "outpaint"
    price_per_image: float | None = None
    quality: str = "medium"  # For OpenAI: "low", "medium", "high"
    aspect_ratio: str | None = None  # For OpenAI: "1:1", "3:2", "2:3", etc.
    default_prompt: str | None = None  # Default prompt if user provides none
    model_options: dict | None = None  # Extra model-specific options (go_fast, safety_tolerance, etc.)


# ============================================================================
# Main Cog
# ============================================================================


class ImageGenCog(commands.Cog):
    """
    Discord cog for image generation, editing, and outpainting.

    Supports multiple image generation APIs with configurable prefixes,
    rate limiting, and multiple generation modes.
    """

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
            self._daily_free_budget: float = 0.50  # Default
            self._default_price: float = 0.03  # Default
        else:
            self._cfg_path = config_path or _DEFAULT_CONFIG_PATH
            self._cfg_mtime = 0.0
            self._prefix_map = {}
            self._daily_free_budget = 0.50  # Will be overridden by config
            self._default_price = 0.03  # Will be overridden by config
            self._reload_config()

        # Initialize cost tracker
        db_path = os.getenv("PROMPT_DB_PATH", "prompts.db")
        self._cost_tracker = CostTracker(db_path, self._daily_free_budget)

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
            raw_cfg: dict = json.loads(self._cfg_path.read_text("utf8"))
        except Exception as exc:
            _log.exception("Failed to parse %s: %s; keeping old config", self._cfg_path, exc)
            return

        # Extract global config values
        self._daily_free_budget = raw_cfg.get("daily_free_budget", 0.50)
        self._default_price = raw_cfg.get("default_price_per_image", 0.03)

        # Build prefix map, excluding non-prefix keys
        self._prefix_map = {}
        known_keys = {"api", "model", "mode", "price_per_image", "quality", "aspect_ratio", "default_prompt"}
        for key, spec in raw_cfg.items():
            if key in ("daily_free_budget", "default_price_per_image"):
                continue
            if isinstance(spec, dict):
                # Separate known GeneratorSpec fields from extra model options
                spec_kwargs = {k: v for k, v in spec.items() if k in known_keys}
                extra_opts = {k: v for k, v in spec.items() if k not in known_keys}
                if extra_opts:
                    spec_kwargs["model_options"] = extra_opts
                self._prefix_map[key.strip()] = GeneratorSpec(**spec_kwargs)

        # Update cost tracker with new budget
        if hasattr(self, "_cost_tracker"):
            self._cost_tracker.daily_free_budget = self._daily_free_budget

        self._cfg_mtime = mtime
        _log.info(
            "Reloaded image-generator config (%d prefixes, budget=$%.2f, default_price=$%.3f).",
            len(self._prefix_map),
            self._daily_free_budget,
            self._default_price,
        )

    def _format_model_listing(self) -> str:
        self._reload_config()
        if not self._prefix_map:
            return "No image generators are configured."
        lines: list[str] = ["Available image generators:"]
        for prefix, spec in sorted(self._prefix_map.items()):
            prefix_display = f"`{prefix}`" if prefix else "(default)"
            mode_display = f" / mode={spec.mode}" if getattr(spec, "mode", "generate") != "generate" else ""
            price = getattr(spec, "price_per_image", self._default_price)
            price_display = f" / ${price:.3f}" if price < 0.01 else f" / ${price:.2f}"
            lines.append(f"- {prefix_display}: {spec.api} / {spec.model}{mode_display}{price_display}")
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
        mask: bytes | None = None,
        output_format: str | None = None,
        quality: str = "medium",
        aspect_ratio: str | None = None,
        model_options: dict | None = None,
        poll_interval: float = _DEFAULT_CELERY_POLL_INTERVAL,
        timeout: float = 300.0,
    ) -> str | list[str]:
        """
        Launch an image-generation task via Celery and poll for results.

        Args:
            prompt_id: Database ID for this prompt
            api: Image generation API name
            model: Model identifier
            prompt: Text prompt for generation
            seed: Random seed (optional)
            image_input: Input images for edit/outpaint modes
            mask: Mask image for inpainting/outpainting
            output_format: Desired output format
            quality: Quality level for OpenAI ('low', 'medium', 'high')
            aspect_ratio: Aspect ratio for OpenAI ('1:1', '3:2', '2:3', etc.)
            model_options: Extra model-specific options
            poll_interval: Seconds between result checks
            timeout: Maximum time to wait for result in seconds (default 5 min)

        Returns:
            File path(s) or base64-encoded image data

        Raises:
            TimeoutError: If task doesn't complete within timeout
            RuntimeError: If Celery task fails
        """
        # Convert images to data URLs for JSON serialization
        payload_images = bytes_list_to_data_urls(image_input) if image_input else None
        payload_mask = bytes_to_data_url(mask) if mask else None

        async_result = generate_image.apply_async(
            (prompt_id, api, model, prompt, seed),
            kwargs={
                "image_input": payload_images,
                "mask": payload_mask,
                "output_format": output_format,
                "quality": quality,
                "aspect_ratio": aspect_ratio,
                "model_options": model_options,
            },
            queue="image",
        )

        elapsed = 0.0
        while not async_result.ready():
            if elapsed >= timeout:
                async_result.revoke(terminate=True)
                raise TimeoutError(f"Image generation timed out after {timeout}s")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Check for task failure
        if async_result.failed():
            error = async_result.result
            raise RuntimeError(f"Image generation failed: {error}")

        return await asyncio.to_thread(async_result.get)

    async def _react(self, msg: discord.Message, emoji: str, *, remove: bool = False) -> None:
        """
        Add or remove a reaction from a message.

        Args:
            msg: Discord message to react to
            emoji: Emoji string or Unicode character
            remove: If True, remove the reaction instead of adding it
        """
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

    async def _get_thematic_emoji(self, prompt: str) -> str:
        """
        Get a thematic emoji for an image generation prompt using Claude Haiku.

        Args:
            prompt: The image generation prompt

        Returns:
            A single emoji string, or THINKING as fallback
        """
        try:
            generator = get_text_generator("anthropic", "claude-haiku-4-5")
            response = await asyncio.wait_for(
                generator.generate(
                    f"Return exactly one emoji shortcode that fits this image prompt. Use Discord format like :fire: or :art:. Just the shortcode, nothing else.\n\nPrompt: {prompt}",
                    temperature=0.7,
                ),
                timeout=3.0,
            )
            shortcode = response.strip()
            _log.info("Haiku returned shortcode: %r for prompt: %s", shortcode, prompt[:50])
            # Ensure it's in :name: format
            if not shortcode.startswith(":"):
                shortcode = f":{shortcode}"
            if not shortcode.endswith(":"):
                shortcode = f"{shortcode}:"
            _log.info("Normalized shortcode: %r", shortcode)
            # Convert shortcode to unicode emoji
            unicode_emoji = emoji.emojize(shortcode, language="alias")
            _log.info("After emojize: %r (changed=%s)", unicode_emoji, unicode_emoji != shortcode)
            # Check if conversion succeeded (emojize returns input unchanged if not found)
            if unicode_emoji != shortcode and len(unicode_emoji) <= 8:
                return unicode_emoji
            _log.info("Emoji conversion failed, falling back to THINKING")
        except Exception as exc:
            _log.warning("Failed to get thematic emoji: %s", exc)
        return THINKING

    async def _send_error_message(
        self,
        message: discord.Message,
        error_text: str,
        *,
        reply_to: discord.Message | None = None,
    ) -> None:
        """
        Send an error message, attempting to reply first, then falling back to channel send.

        Args:
            message: Original message that triggered the error
            error_text: Error message to send
            reply_to: Message to reply to (defaults to original message)
        """
        target = reply_to or message
        try:
            await target.reply(error_text, mention_author=False)
        except Exception:
            await message.channel.send(error_text)

    async def _gather_attachment_images(self, message: discord.Message) -> list[bytes]:
        """
        Collect all image attachments from a Discord message.

        Args:
            message: Discord message to extract images from

        Returns:
            List of image bytes
        """
        collected: list[bytes] = []
        for att in message.attachments:
            ct = (att.content_type or "").lower()
            is_image = ct.startswith("image/") or att.filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp", ".bmp")
            )
            if is_image:
                try:
                    collected.append(await att.read())
                except discord.HTTPException:
                    _log.debug("Could not download attachment %s", att.id)
        return collected

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

    async def _generate_outpaint_prompt(self, image_bytes: bytes) -> str:
        """
        Use an LLM to generate a descriptive prompt for outpainting.

        Args:
            image_bytes: Image to analyze

        Returns:
            Generated prompt text (or fallback "." on error)
        """
        try:
            mime_type = detect_mime_type(image_bytes)
            b64_data = base64.b64encode(image_bytes).decode("ascii")

            # Get LLM configuration from environment
            llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic").lower()
            llm_model = os.getenv("DEFAULT_LLM_MODEL", "claude-3-5-sonnet-20241022")
            generator = get_text_generator(llm_provider, llm_model)

            # Build message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image and describe what a zoomed-out version of this scene "
                                "would look like. Focus on what surroundings, environment, or context would "
                                "naturally extend beyond the current frame. Be concise and descriptive "
                                "(1-2 sentences). Don't mention that this is a zoomed out version, just "
                                "describe the fuller scene."
                            ),
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64_data,
                            },
                        },
                    ],
                }
            ]

            # Generate prompt using LLM
            prompt = await asyncio.wait_for(
                generator.generate(messages), timeout=_DEFAULT_OUTPAINT_PROMPT_TIMEOUT
            )

            _log.info("Generated outpaint prompt: %s", prompt)
            return prompt.strip()

        except Exception as exc:
            _log.warning("Failed to generate outpaint prompt: %s", exc)
            return "."  # Fallback to minimal prompt

    async def _get_aspect_ratio_for_prompt(self, prompt: str) -> str:
        """
        Use Claude to determine the best aspect ratio for an image generation prompt.

        Args:
            prompt: The image generation prompt

        Returns:
            Aspect ratio string (defaults to "3:2" on error)
        """
        valid_ratios = {"1:1", "2:3", "3:2", "16:9", "9:16"}
        default_ratio = "3:2"

        try:
            generator = get_text_generator("anthropic", "claude-haiku-4-5")

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Select the optimal aspect ratio for this AI image generation prompt.\n\n"
                        f"Available ratios:\n"
                        f"- 1:1 = square (centered compositions, profile pics, symmetrical)\n"
                        f"- 2:3 = portrait/tall (people, buildings, standing figures)\n"
                        f"- 3:2 = landscape/wide (scenery, groups, horizontal subjects)\n"
                        f"- 16:9 = ultrawide cinematic (panoramas, epic vistas)\n"
                        f"- 9:16 = vertical/phone (TikTok, tall narrow scenes)\n\n"
                        f"Prompt: {prompt}\n\n"
                        f"Reply with ONLY the ratio. Nothing else."
                    ),
                }
            ]

            response = await asyncio.wait_for(generator.generate(messages), timeout=30.0)
            response_text = response.strip()

            # Extract ratio from response (may contain reasoning text)
            for valid in valid_ratios:
                if valid in response_text:
                    _log.info("Aspect ratio %s for '%s': %s", valid, prompt[:50], response_text)
                    return valid

            _log.warning("Invalid aspect ratio response '%s', using default %s", response_text, default_ratio)
            return default_ratio

        except Exception as exc:
            _log.warning("Failed to get aspect ratio: %s, using default %s", exc, default_ratio)
            return default_ratio

    def _build_filename(
        self,
        prompt: str,
        spec: GeneratorSpec,
        seed: int | None,
        *,
        max_len: int = _MAX_FILENAME_LENGTH,
    ) -> str:
        """
        Build a descriptive filename for generated images.

        Args:
            prompt: Generation prompt
            spec: Generator specification
            seed: Random seed
            max_len: Maximum length for prompt snippet

        Returns:
            Filename string
        """
        snippet = re.sub(r"[^A-Za-z0-9]+", "_", prompt).strip("_")
        if not snippet:
            snippet = "image"
        if len(snippet) > max_len:
            snippet = snippet[:max_len].rstrip("_")
        api_model = f"{spec.api}-{spec.model}".replace("/", "-")
        seed_part = str(seed) if seed is not None else "rand"
        return f"{snippet}_{api_model}_seed_{seed_part}.png".lower()

    def _parse_seed_from_prompt(self, raw_prompt: str) -> tuple[str, int]:
        """
        Extract seed from prompt if present in {seed} format.

        Args:
            raw_prompt: Raw prompt text potentially containing seed

        Returns:
            Tuple of (prompt_without_seed, seed_value)
        """
        m_seed = self._SEED_RE.match(raw_prompt)
        if m_seed:
            seed = int(m_seed.group(1))
            prompt = raw_prompt[m_seed.end() :].lstrip()
            return prompt, seed
        return raw_prompt, random.randint(1, 1000)

    def _expand_prompt_list(self, raw_prompt: str) -> list[str]:
        """
        Expand prompt containing <a, b, c> syntax into multiple prompts.

        Args:
            raw_prompt: Prompt potentially containing list syntax

        Returns:
            List of expanded prompts
        """
        m_list = self._LIST_RE.search(raw_prompt)
        if not m_list:
            return [raw_prompt]

        items = [s.strip() for s in m_list.group(1).split(",") if s.strip()]
        if not items:
            return [raw_prompt]

        prefix = raw_prompt[: m_list.start()]
        suffix = raw_prompt[m_list.end() :]
        return [f"{prefix}{item}{suffix}".strip() for item in items]

    async def _collect_images_for_editing(
        self, message: discord.Message
    ) -> tuple[list[bytes], discord.Message | None]:
        """
        Collect images from the message, its reply reference, and recent history.

        Args:
            message: Discord message to collect images from

        Returns:
            Tuple of (all_images, history_source_message)
        """
        # Collect from current message
        images = await self._gather_attachment_images(message)

        # Collect from replied-to message
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
                reply_images = await self._gather_attachment_images(replied_msg)

        # If no images yet, search recent history
        history_images: list[bytes] = []
        history_source: discord.Message | None = None
        if not images and not reply_images:
            try:
                async for prev_msg in message.channel.history(
                    limit=_HISTORY_SEARCH_LIMIT, before=message
                ):
                    history_images = await self._gather_attachment_images(prev_msg)
                    if history_images:
                        history_source = prev_msg
                        break
            except Exception:
                _log.debug("Failed to backfill images from history", exc_info=True)

        all_images = images + reply_images + history_images
        return all_images, history_source

    async def _prepare_outpaint(
        self,
        message: discord.Message,
        all_edit_images: list[bytes],
        needs_generated_prompt: bool,
        working_emoji: str = THINKING,
    ) -> tuple[list[str], list[bytes], bytes] | None:
        """
        Prepare images and prompt for outpaint mode.

        Args:
            message: Discord message (for error reporting)
            all_edit_images: Collected images
            needs_generated_prompt: Whether to generate prompt with LLM
            working_emoji: Emoji to remove on error

        Returns:
            Tuple of (prompts, prepared_images, mask) or None on error
        """
        try:
            prompts: list[str]
            if needs_generated_prompt:
                generated_prompt = await self._generate_outpaint_prompt(all_edit_images[0])
                prompts = [generated_prompt]
                _log.info("Using LLM-generated prompt for outpaint: %s", generated_prompt)
            else:
                prompts = []  # Will be filled by caller

            # Prepare outpaint images
            scaled_image, mask = create_outpaint_images(all_edit_images[0])
            return prompts, [scaled_image], mask

        except Exception as exc:
            await self._react(message, working_emoji, remove=True)
            await self._react(message, FAILURE)
            _log.exception("Failed to prepare outpaint images: %s", exc)
            await self._send_error_message(message, f"Failed to prepare outpaint images: {exc}")
            return None


    def _load_generation_result(self, item: str) -> bytes:
        """
        Load a single generation result (from file path or base64 string).

        Args:
            item: File path or base64-encoded string

        Returns:
            Image bytes
        """
        path = Path(item)
        if path.exists():
            data = path.read_bytes()
            try:
                path.unlink(missing_ok=True)
            except Exception:
                _log.debug("Temp image %s could not be deleted", path)
            return data
        return base64.b64decode(item)

    async def _execute_generation_tasks(
        self,
        prompt_ids: list[int],
        prompts: list[str],
        spec: GeneratorSpec,
        seed: int,
        image_input: list[bytes] | None,
        mask: bytes | None,
        use_png_format: bool,
        aspect_ratio_override: str | None = None,
    ) -> list[tuple[str, list[bytes]] | Exception]:
        """
        Execute multiple generation tasks concurrently.

        Args:
            prompt_ids: Database IDs for prompts
            prompts: Text prompts
            spec: Generator specification
            seed: Random seed
            image_input: Input images for edit/outpaint
            mask: Mask for inpainting
            use_png_format: Whether to force PNG output
            aspect_ratio_override: Override aspect ratio (used for dynamic gpt-image aspect ratios)

        Returns:
            List of results (prompt, images) or exceptions
        """
        aspect_ratio = aspect_ratio_override if aspect_ratio_override else spec.aspect_ratio

        async def _launch_single(prompt_id: int, prompt: str) -> tuple[str, list[bytes]] | Exception:
            try:
                result = await self._run_task(
                    prompt_id,
                    spec.api,
                    spec.model,
                    prompt,
                    seed,
                    image_input=image_input,
                    mask=mask,
                    output_format="png" if use_png_format else None,
                    quality=spec.quality,
                    aspect_ratio=aspect_ratio,
                    model_options=spec.model_options,
                )

                # Convert result to bytes
                if isinstance(result, list):
                    images_bytes = [self._load_generation_result(x) for x in result]
                else:
                    images_bytes = [self._load_generation_result(result)]

                return prompt, images_bytes
            except Exception as exc:
                return exc

        return await asyncio.gather(
            *(_launch_single(pid, p) for pid, p in zip(prompt_ids, prompts)),
            return_exceptions=True,
        )

    def _prepare_discord_files(
        self,
        images_bytes: list[bytes],
        prompt: str,
        spec: GeneratorSpec,
        seed: int,
        limit_bytes: int,
        skip_caption: bool = False,
    ) -> list[discord.File]:
        """
        Prepare image bytes as Discord File objects with compression if needed.

        Args:
            images_bytes: List of image bytes
            prompt: Prompt text (for filename and caption)
            spec: Generator specification
            seed: Random seed
            limit_bytes: Maximum file size in bytes
            skip_caption: Skip adding caption to images

        Returns:
            List of Discord File objects
        """
        files: list[discord.File] = []
        base_name = self._build_filename(prompt, spec, seed)
        root, _orig_ext = (base_name.rsplit(".", 1) + ["png"])[:2]

        for idx, img_bytes in enumerate(images_bytes, start=1):
            # Add caption unless skipped
            processed = img_bytes if skip_caption else add_caption(img_bytes, prompt)

            # Compress if needed
            if len(processed) > limit_bytes:
                processed, new_ext = compress_image_to_fit(processed, limit_bytes)
            else:
                new_ext = detect_image_extension(processed)

            # Try harder compression if still too large
            if len(processed) > limit_bytes:
                processed, new_ext = compress_image_to_fit(
                    processed, int(limit_bytes * _COMPRESSION_SAFETY_FACTOR)
                )

            # Skip if still too large
            if len(processed) > limit_bytes:
                _log.warning(
                    "Image still exceeds limit after compression: %d > %d",
                    len(processed),
                    limit_bytes,
                )
                continue

            # Generate filename with index if multiple results
            fname = f"{root}.{new_ext}" if len(images_bytes) == 1 else f"{root}_{idx}.{new_ext}"
            files.append(discord.File(BytesIO(processed), filename=fname))

        return files

    async def _process_and_send_results(
        self,
        message: discord.Message,
        results: list[tuple[str, list[bytes]] | Exception],
        prompts: list[str],
        spec: GeneratorSpec,
        seed: int,
        skip_caption: bool,
        reply_target: discord.Message,
    ) -> bool:
        """
        Process generation results and send them to Discord.

        Args:
            message: Original Discord message
            results: Generation results or exceptions
            prompts: Original prompts
            spec: Generator specification
            seed: Random seed
            skip_caption: Whether to skip adding captions (for edit/outpaint)
            reply_target: Message to reply to for edit/outpaint modes

        Returns:
            True if all results succeeded, False otherwise
        """
        overall_success = True

        # Determine file size limit
        guild_limit = getattr(message.guild, "filesize_limit", None)
        limit_bytes = int(guild_limit) if guild_limit else _DEFAULT_FILESIZE_LIMIT
        limit_bytes = max(1, limit_bytes)

        for prompt_variant, result in zip(prompts, results):
            if isinstance(result, Exception):
                overall_success = False
                _log.exception("Generation failed for %r: %s", prompt_variant, result)
                continue

            prompt_text, images_bytes = result
            try:
                files = self._prepare_discord_files(
                    images_bytes, prompt_text, spec, seed, limit_bytes, skip_caption
                )

                if not files:
                    raise RuntimeError(
                        f"All generated images exceed Discord limit (>{limit_bytes} bytes) "
                        "even after compression."
                    )

                # Send files
                if skip_caption:  # Edit/outpaint mode
                    await reply_target.reply(files=files, mention_author=False)
                else:  # Regular generation
                    await message.channel.send(files=files)

            except Exception as exc:
                overall_success = False
                _log.exception("Post-processing failed: %s", exc)
                error_msg = f"Image post-processing failed for **{prompt_variant}**:\n> {exc}"
                await self._send_error_message(message, error_msg, reply_to=reply_target if skip_caption else None)

        return overall_success

    async def _handle_generation(
        self,
        message: discord.Message,
        raw_prompt: str,
        spec: GeneratorSpec,
    ) -> None:
        """
        Handle the complete image generation workflow.

        Args:
            message: Discord message that triggered generation
            raw_prompt: User's prompt text
            spec: Generator specification
        """
        # Get thematic emoji for the prompt (runs in parallel with parsing)
        working_emoji = await self._get_thematic_emoji(raw_prompt)
        await self._react(message, working_emoji)

        # Parse seed and clean prompt
        raw_prompt, seed = self._parse_seed_from_prompt(raw_prompt)

        # Use default prompt if user provided none and spec has one
        if not raw_prompt and spec.default_prompt:
            raw_prompt = spec.default_prompt

        # Check for outpaint mode with no prompt
        is_outpaint = spec.mode == "outpaint"
        needs_generated_prompt = not raw_prompt and is_outpaint

        # Validate prompt
        if not raw_prompt and not is_outpaint:
            await self._react(message, working_emoji, remove=True)
            await self._react(message, FAILURE)
            await message.channel.send("Prompt may not be empty.")
            return

        # Validate prompt length to prevent abuse
        if len(raw_prompt) > _MAX_PROMPT_LENGTH:
            await self._react(message, working_emoji, remove=True)
            await self._react(message, FAILURE)
            await self._send_error_message(
                message,
                f"Prompt is too long ({len(raw_prompt)} characters). Maximum allowed is {_MAX_PROMPT_LENGTH} characters.",
            )
            return

        # Set placeholder for outpaint mode if needed
        if needs_generated_prompt:
            raw_prompt = "placeholder"

        # Expand prompt list syntax
        prompts = self._expand_prompt_list(raw_prompt)

        # Collect images for edit/outpaint modes
        is_edit_mode = spec.mode == "edit"
        all_edit_images, history_source = await self._collect_images_for_editing(message)

        do_edit = bool(all_edit_images) and is_edit_mode
        do_outpaint = bool(all_edit_images) and is_outpaint
        reply_target = history_source or message

        # Validate images for edit mode
        if spec.mode == "edit" and not all_edit_images:
            await self._react(message, working_emoji, remove=True)
            await self._react(message, FAILURE)
            await self._send_error_message(
                message,
                "No images found to edit. Reply to a message with an image or attach an image with your `edit:` prompt.",
            )
            return

        # Validate images for outpaint mode
        if is_outpaint and not all_edit_images:
            await self._react(message, working_emoji, remove=True)
            await self._react(message, FAILURE)
            await self._send_error_message(
                message,
                "No images found to outpaint. Reply to a message with an image or attach an image with your `zoom out` prompt.",
            )
            return

        # Prepare outpaint if needed
        outpaint_mask: bytes | None = None
        if do_outpaint and all_edit_images:
            result = await self._prepare_outpaint(message, all_edit_images, needs_generated_prompt, working_emoji)
            if result is None:
                return  # Error already handled
            outpaint_prompts, all_edit_images, outpaint_mask = result
            if needs_generated_prompt:
                prompts = outpaint_prompts

        # Check cost affordability
        cost = spec.price_per_image if spec.price_per_image is not None else self._default_price
        can_afford, error_msg = self._cost_tracker.can_afford(message.author.id, cost)

        if not can_afford:
            await self._react(message, working_emoji, remove=True)
            await self._react(message, FAILURE)
            await self._send_error_message(message, error_msg)
            return

        # Record prompt for tracking (no rate limiting)
        prompt_ids = bulk_add_prompts(
            prompts,
            str(message.author.id),
            spec.api,
            spec.model,
        )

        # For models that support aspect ratio (non-edit), get dynamic aspect ratio from Claude
        aspect_ratio_override: str | None = None
        model_lower = spec.model.lower()
        supports_dynamic_aspect = (
            "gpt-image" in model_lower or
            "black-forest-labs/" in model_lower or
            "flux" in model_lower
        )
        if supports_dynamic_aspect and not (do_edit or do_outpaint):
            # Use the first prompt for aspect ratio detection (they're usually all similar)
            aspect_ratio_override = await self._get_aspect_ratio_for_prompt(prompts[0])

        # Execute generation tasks
        results = await self._execute_generation_tasks(
            prompt_ids,
            prompts,
            spec,
            seed,
            all_edit_images if (do_edit or do_outpaint) else None,
            outpaint_mask if do_outpaint else None,
            do_edit or do_outpaint,
            aspect_ratio_override,
        )

        # Process and send results
        overall_success = await self._process_and_send_results(
            message,
            results,
            prompts,
            spec,
            seed,
            do_edit or do_outpaint,
            reply_target,
        )

        # Deduct cost only if generation was successful
        if overall_success:
            try:
                self._cost_tracker.deduct_cost(message.author.id, cost)
            except Exception as exc:
                _log.exception("Failed to deduct cost for user %s: %s", message.author.id, exc)
                # Don't fail the generation - user already received their image

        await self._react(message, working_emoji, remove=True)
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
            # Check content early to allow a separate allowlist for edit: and zoom out
            cleaned = (message.content or "").strip()
            if self._allowed_channel_ids and message.channel.id not in self._allowed_channel_ids:
                # Not in general image channels. Allow if this is an edit prompt or zoom out
                # and the channel is in EDIT_CHANNEL_IDS.
                cleaned_lower = cleaned.lower()
                is_edit_cmd = (
                    cleaned_lower.startswith("edit:") or
                    cleaned_lower.startswith("edit high:") or
                    cleaned_lower.startswith("edit pro:") or
                    cleaned_lower.startswith("zoom out")
                )
                if not (is_edit_cmd and message.channel.id in self._edit_channel_ids):
                    return
        # For DMs or allowed guild messages, continue.
        cleaned = (message.content or "").strip()

        if cleaned.lower() == "!models":
            await message.channel.send(self._format_model_listing())
            return

        # Check if this is a bot command (starts with ! followed by a known command)
        # to prevent treating commands like !usage, !grant, !balance as image prompts
        if cleaned.startswith("!"):
            # Get the potential command word (everything between ! and first space)
            potential_command = cleaned[1:].split()[0].lower() if len(cleaned) > 1 else ""
            # List of known bot commands that should not trigger image generation
            bot_commands = {
                "usage", "balance", "grant", "set_price", "set_budget",
                "reset", "ping", "help", "model", "system", "models"
            }
            if potential_command in bot_commands:
                return  # Let the command system handle it

        split = self._split_prompt(cleaned)
        if not split:
            return

        prompt, spec = split
        task = asyncio.create_task(self._handle_generation(message, prompt, spec))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ImageGenCog(bot))
