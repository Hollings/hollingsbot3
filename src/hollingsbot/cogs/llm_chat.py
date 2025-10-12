from __future__ import annotations

import asyncio
import base64
import functools
import io
import json
import logging
import os
import re
import textwrap
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterable, Sequence

import discord
from discord.ext import commands
from PIL import Image

try:  # Optional at runtime; fallback to raw SVG attachments if unavailable.
    import cairosvg  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    cairosvg = None  # type: ignore

from celery.result import AsyncResult

from hollingsbot.settings import clear_system_prompt_cache, get_default_system_prompt
from hollingsbot.tasks import generate_llm_chat_response

_LOG = logging.getLogger(__name__)


_SVG_BLOCK_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.IGNORECASE | re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+\-]*)\n(?P<body>.*?)(```)$", re.DOTALL | re.MULTILINE)
_TEXT_ATTACHMENT_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".log",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rb",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".css",
    ".html",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
    ".xml",
    ".csv",
}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_CODE_EXTENSION_MAP = {
    "": "txt",
    "text": "txt",
    "plaintext": "txt",
    "py": "py",
    "python": "py",
    "ts": "ts",
    "tsx": "tsx",
    "js": "js",
    "javascript": "js",
    "json": "json",
    "yaml": "yml",
    "yml": "yml",
    "bash": "sh",
    "sh": "sh",
    "shell": "sh",
    "go": "go",
    "rs": "rs",
    "rust": "rs",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "xml": "xml",
    "php": "php",
    "rb": "rb",
    "ruby": "rb",
    "cs": "cs",
}
_MAX_TEXT_ATTACHMENT_BYTES = 120_000
_IMAGE_MAX_EDGE = 2048
_IMAGE_MAX_BYTES = 9_500_000
_MESSAGE_CHUNK = 1900


@dataclass(slots=True)
class ImageAttachment:
    name: str
    url: str | None
    data_url: str | None
    width: int | None = None
    height: int | None = None
    size: int | None = None

    def clone(self) -> "ImageAttachment":
        return ImageAttachment(
            name=self.name,
            url=self.url,
            data_url=self.data_url,
            width=self.width,
            height=self.height,
            size=self.size,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "data_url": self.data_url,
            "width": self.width,
            "height": self.height,
            "size": self.size,
        }


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str
    images: list[ImageAttachment] = field(default_factory=list)
    message_id: int | None = None
    author_id: int | None = None
    author_name: str | None = None


@dataclass(slots=True)
class ModelTurn:
    role: str
    text: str
    images: list[ImageAttachment] = field(default_factory=list)


@dataclass
class GenerationJob:
    task: asyncio.Task[None] | None = None
    result: AsyncResult | None = None


class LLMChatNewCog(commands.Cog):
    """LLM-powered chat cog rebuilt per the LLM_CHAT_FEATURES design."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.whitelist_channels = self._parse_channel_ids(os.getenv("LLM_WHITELIST_CHANNELS", ""))
        self.history_limit = max(1, int(os.getenv("LLM_HISTORY_LIMIT", "50")))
        self.max_turns_sent = max(1, int(os.getenv("LLM_MAX_TURNS_SENT", "8")))
        self.text_timeout = float(os.getenv("TEXT_TIMEOUT", "180"))
        self.default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai").strip().lower() or "openai"
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o").strip() or "gpt-4o"
        self.available_models = self._load_available_models()
        self._model_lookup = {
            (provider.lower(), model): (provider, model)
            for provider, model in self.available_models
        }
        if (self.default_provider, self.default_model) not in {
            (p.lower(), m) for p, m in self.available_models
        }:
            self.available_models.append((self.default_provider, self.default_model))
            self._model_lookup[(self.default_provider, self.default_model)] = (
                self.default_provider,
                self.default_model,
            )

        self.state_path = Path("generated") / "llm_chat_new_state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        stored_prompt = self._state.get("system_prompt") if isinstance(self._state, dict) else None
        self.system_prompt = (
            str(stored_prompt)
            if stored_prompt
            else get_default_system_prompt()
        )
        raw_prefs = self._state.get("model_preferences") if isinstance(self._state, dict) else {}
        if isinstance(raw_prefs, dict):
            self.model_preferences: dict[str, dict[str, dict[str, str]]] = raw_prefs
        else:
            self.model_preferences = {}

        self.channel_histories: dict[int, Deque[ConversationTurn]] = {}
        self._history_locks: dict[int, asyncio.Lock] = {}
        self._warmed_channels: set[int] = set()
        self._active_generations: dict[int, GenerationJob] = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _parse_channel_ids(raw: str) -> set[int]:
        values: set[int] = set()
        for token in raw.split(","):
            token = token.strip()
            if token.isdigit():
                values.add(int(token))
        return values

    def _load_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {}
        try:
            text = self.state_path.read_text("utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            _LOG.exception("Failed to load %s", self.state_path)
        return {}

    def _save_state(self) -> None:
        payload = {
            "system_prompt": self.system_prompt,
            "model_preferences": self.model_preferences,
        }
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        tmp_path.replace(self.state_path)

    def _load_available_models(self) -> list[tuple[str, str]]:
        raw = os.getenv("AVAILABLE_MODELS", "")
        models: list[tuple[str, str]] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if "/" in token:
                provider, model = token.split("/", 1)
            else:
                provider, model = self.default_provider, token
            provider = provider.strip().lower()
            model = model.strip()
            if provider and model:
                models.append((provider, model))
        if not models:
            models.append((self.default_provider, self.default_model))
        return models

    def _history_for_channel(self, channel_id: int) -> Deque[ConversationTurn]:
        history = self.channel_histories.get(channel_id)
        if history is None or history.maxlen != self.history_limit:
            existing = list(history) if history else []
            history = deque(existing, maxlen=self.history_limit)
            self.channel_histories[channel_id] = history
        return history

    def _lock_for_channel(self, channel_id: int) -> asyncio.Lock:
        lock = self._history_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._history_locks[channel_id] = lock
        return lock

    # ----------------------------------------------------------------- helpers
    def _channel_allowed(self, channel: discord.abc.MessageableChannel) -> bool:
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            return False
        if not self.whitelist_channels:
            # Strict interpretation: empty whitelist disables the feature entirely.
            return False
        return channel_id in self.whitelist_channels

    @staticmethod
    def _should_ignore_message(content: str | None) -> bool:
        if not content:
            return False
        stripped = content.lstrip()
        if not stripped:
            return False
        lowered = stripped.lower()
        return stripped.startswith("!") or stripped.startswith("-") or lowered.startswith("edit:")

    def _is_text_attachment(self, attachment: discord.Attachment) -> bool:
        if attachment.size == 0:
            return False
        if attachment.content_type:
            ctype = attachment.content_type.lower()
            if ctype.startswith("text/"):
                return True
            if ctype in {
                "application/json",
                "application/javascript",
                "application/xml",
                "application/x-yaml",
            }:
                return True
        _, ext = os.path.splitext(attachment.filename)
        return ext.lower() in _TEXT_ATTACHMENT_EXTENSIONS

    def _is_image_attachment(self, attachment: discord.Attachment) -> bool:
        if attachment.content_type:
            if attachment.content_type.lower().startswith("image/"):
                return True
        _, ext = os.path.splitext(attachment.filename)
        return ext.lower() in _IMAGE_EXTENSIONS

    async def _collect_text_attachments_full(
        self, message: discord.Message
    ) -> tuple[list[str], list[str]]:
        full_blocks: list[str] = []
        placeholders: list[str] = []
        for attachment in message.attachments:
            if not self._is_text_attachment(attachment):
                continue
            try:
                data = await attachment.read()
            except Exception:  # noqa: BLE001
                _LOG.exception("Failed to read text attachment %s", attachment.filename)
                continue
            truncated = False
            if len(data) > _MAX_TEXT_ATTACHMENT_BYTES:
                data = data[:_MAX_TEXT_ATTACHMENT_BYTES]
                truncated = True
            text = data.decode("utf-8", errors="replace")
            block = (
                f"[begin uploaded file: {attachment.filename}]\n{text}\n[end uploaded file]"
            )
            if truncated:
                block += "\n[truncated]"
            full_blocks.append(block)
            placeholder = f"[uploaded file {attachment.filename} removed]"
            if truncated:
                placeholder += " (truncated)"
            placeholders.append(placeholder)
        return full_blocks, placeholders

    async def _collect_text_placeholders(self, message: discord.Message) -> list[str]:
        placeholders: list[str] = []
        for attachment in message.attachments:
            if self._is_text_attachment(attachment):
                placeholders.append(f"[uploaded file {attachment.filename} removed]")
        return placeholders

    async def _prepare_image_attachment(self, attachment: discord.Attachment) -> ImageAttachment | None:
        try:
            data = await attachment.read()
        except Exception:  # noqa: BLE001
            _LOG.exception("Failed to download image attachment %s", attachment.filename)
            return None
        try:
            with Image.open(io.BytesIO(data)) as img:
                img = img.convert("RGB")
                width, height = img.size
                longest = max(width, height)
                if longest > _IMAGE_MAX_EDGE:
                    scale = _IMAGE_MAX_EDGE / float(longest)
                    resized = (
                        max(1, int(width * scale)),
                        max(1, int(height * scale)),
                    )
                    img = img.resize(resized, Image.LANCZOS)
                    width, height = img.size
                jpeg_bytes = self._encode_jpeg(img)
        except Exception:  # noqa: BLE001
            _LOG.exception("Failed to process image attachment %s", attachment.filename)
            return ImageAttachment(
                name=attachment.filename,
                url=attachment.url,
                data_url=None,
                width=None,
                height=None,
                size=attachment.size,
            )
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")
        return ImageAttachment(
            name=attachment.filename,
            url=attachment.url,
            data_url=data_url,
            width=width,
            height=height,
            size=len(jpeg_bytes),
        )

    def _encode_jpeg(self, image: Image.Image) -> bytes:
        for quality in (90, 85, 80, 75, 70, 60, 50):
            out = io.BytesIO()
            image.save(out, format="JPEG", optimize=True, quality=quality)
            if out.tell() <= _IMAGE_MAX_BYTES:
                return out.getvalue()
        return out.getvalue()

    def _image_from_bytes(self, name: str, data: bytes) -> ImageAttachment:
        try:
            with Image.open(io.BytesIO(data)) as img:
                width, height = img.size
        except Exception:  # noqa: BLE001
            width = height = None
        data_url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        return ImageAttachment(name=name, url=None, data_url=data_url, width=width, height=height, size=len(data))

    def _images_from_history(self, channel_id: int, message_id: int | None) -> list[ImageAttachment]:
        if message_id is None:
            return []
        history = self.channel_histories.get(channel_id)
        if not history:
            return []
        for turn in reversed(history):
            if turn.message_id == message_id:
                return [img.clone() for img in turn.images]
        return []

    async def _collect_image_attachments(self, message: discord.Message) -> list[ImageAttachment]:
        images: list[ImageAttachment] = []
        for attachment in message.attachments:
            if not self._is_image_attachment(attachment):
                continue
            img = await self._prepare_image_attachment(attachment)
            if img:
                images.append(img)
        return images

    async def _build_reply_hint(
        self, message: discord.Message
    ) -> tuple[str | None, list[ImageAttachment]]:
        ref = message.reference
        if not ref or not ref.message_id:
            return None, []
        resolved = ref.resolved if isinstance(ref.resolved, discord.Message) else None
        ref_message: discord.Message | None = resolved
        if ref_message is None:
            try:
                ref_message = await message.channel.fetch_message(ref.message_id)
            except Exception:  # noqa: BLE001
                return None, []
        display = ref_message.author.display_name if ref_message.author else "Unknown"
        snippet = ref_message.clean_content.strip()
        if snippet:
            snippet = textwrap.shorten(snippet.replace("\n", " "), width=140, placeholder="…")
            hint = f"(Replying to <{display}>: {snippet})"
        else:
            hint = f"(Replying to <{display}>.)"
        images = self._images_from_history(message.channel.id, ref_message.id)
        if not images:
            images = await self._collect_image_attachments(ref_message)
        return hint, [img.clone() for img in images]

    async def _prepare_user_turn(
        self, message: discord.Message
    ) -> tuple[ModelTurn, ConversationTurn]:
        display = message.author.display_name
        hint, reply_images = await self._build_reply_hint(message)
        base_text = message.clean_content.strip()
        body_parts: list[str] = []
        if hint:
            body_parts.append(hint)
        if base_text:
            body_parts.append(base_text)
        body = "\n".join(part for part in body_parts if part).strip()
        if not body:
            body = "[no content]"
        prefixed = f"<{display}>: {body}"

        text_blocks, placeholders = await self._collect_text_attachments_full(message)
        full_text = prefixed
        for block in text_blocks:
            full_text += f"\n{block}"
        history_text = prefixed
        for placeholder in placeholders:
            history_text += f"\n{placeholder}"

        current_images = await self._collect_image_attachments(message)
        merged_images = [img.clone() for img in reply_images + current_images]

        model_turn = ModelTurn(role="user", text=full_text, images=[img.clone() for img in merged_images])
        history_turn = ConversationTurn(
            role="user",
            content=history_text,
            images=merged_images,
            message_id=message.id,
            author_id=message.author.id,
            author_name=display,
        )
        return model_turn, history_turn

    async def _build_history_user_turn(self, message: discord.Message) -> ConversationTurn | None:
        if self._should_ignore_message(message.content):
            return None
        display = message.author.display_name
        hint, reply_images = await self._build_reply_hint(message)
        base_text = message.clean_content.strip()
        parts: list[str] = []
        if hint:
            parts.append(hint)
        if base_text:
            parts.append(base_text)
        body = "\n".join(parts).strip() or "[no content]"
        text = f"<{display}>: {body}"
        placeholders = await self._collect_text_placeholders(message)
        for placeholder in placeholders:
            text += f"\n{placeholder}"
        images = reply_images + await self._collect_image_attachments(message)
        return ConversationTurn(
            role="user",
            content=text,
            images=images,
            message_id=message.id,
            author_id=message.author.id,
            author_name=display,
        )

    async def _build_assistant_turn_from_message(self, message: discord.Message) -> ConversationTurn:
        text = (message.content or "").strip() or "[no content]"
        placeholders = await self._collect_text_placeholders(message)
        for placeholder in placeholders:
            text += f"\n{placeholder}"
        images = await self._collect_image_attachments(message)
        display = message.author.display_name if message.author else "Bot"
        return ConversationTurn(
            role="assistant",
            content=text,
            images=images,
            message_id=message.id,
            author_id=message.author.id if message.author else None,
            author_name=display,
        )

    def _build_conversation_payload(
        self,
        history: Sequence[ConversationTurn],
        current_turn: ModelTurn,
    ) -> list[dict[str, object]]:
        recent = list(history)[-self.max_turns_sent :]
        conversation: list[dict[str, object]] = [
            {
                "role": "system",
                "text": self.system_prompt,
                "images": [],
            }
        ]
        for turn in recent:
            conversation.append(
                {
                    "role": turn.role,
                    "text": turn.content,
                    "images": [img.to_payload() for img in turn.images],
                }
            )
        conversation.append(
            {
                "role": current_turn.role,
                "text": current_turn.text,
                "images": [img.to_payload() for img in current_turn.images],
            }
        )
        return conversation

    def _is_valid_model(self, provider: str, model: str) -> bool:
        key = (provider.lower(), model)
        return key in self._model_lookup

    def _get_model_for_user(self, guild_id: int | None, user_id: int) -> tuple[str, str]:
        gid = str(guild_id or 0)
        uid = str(user_id)
        entry = self.model_preferences.get(gid, {}).get(uid)
        if isinstance(entry, dict):
            provider = entry.get("provider")
            model = entry.get("model")
            if isinstance(provider, str) and isinstance(model, str) and self._is_valid_model(provider, model):
                return provider.lower(), model
        return self.default_provider, self.default_model

    def _set_model_for_user(self, guild_id: int | None, user_id: int, provider: str, model: str) -> None:
        gid = str(guild_id or 0)
        uid = str(user_id)
        guild_entry = self.model_preferences.setdefault(gid, {})
        guild_entry[uid] = {"provider": provider.lower(), "model": model}
        self._save_state()

    async def _ensure_channel_warm(self, channel: discord.abc.MessageableChannel) -> None:
        channel_id = getattr(channel, "id", None)
        if channel_id is None or channel_id in self._warmed_channels:
            return
        history = self._history_for_channel(channel_id)
        limit = max(self.history_limit, self.max_turns_sent) * 3
        try:
            messages: list[discord.Message] = []
            async for item in channel.history(limit=limit, oldest_first=True):
                messages.append(item)
        except Exception:  # noqa: BLE001
            _LOG.exception("Failed to preload history for channel %s", channel_id)
            self._warmed_channels.add(channel_id)
            return
        for msg in messages:
            if msg.author and msg.author.bot and msg.author != self.bot.user:
                continue
            if msg.author == self.bot.user:
                turn = await self._build_assistant_turn_from_message(msg)
            else:
                turn = await self._build_history_user_turn(msg)
                if turn is None:
                    continue
            history.append(turn)
        self._warmed_channels.add(channel_id)

    def _code_extension(self, language: str) -> str:
        return _CODE_EXTENSION_MAP.get(language.lower(), "txt")

    def _chunk_text(self, text: str) -> list[str]:
        text = text.strip()
        if len(text) <= _MESSAGE_CHUNK:
            return [text] if text else []
        chunks: list[str] = []
        buffer = ""
        for line in text.splitlines(keepends=True):
            if len(buffer) + len(line) > _MESSAGE_CHUNK:
                if buffer:
                    chunks.append(buffer.rstrip())
                    buffer = ""
                while len(line) > _MESSAGE_CHUNK:
                    chunks.append(line[:_MESSAGE_CHUNK])
                    line = line[_MESSAGE_CHUNK :]
            buffer += line
        if buffer.strip():
            chunks.append(buffer.rstrip())
        return [chunk for chunk in chunks if chunk]

    def _conversation_for_celery(
        self,
        conversation: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        # Normalize conversation entries into the compact structure expected by the Celery task.
        normalized: list[dict[str, object]] = []
        for entry in conversation:
            normalized.append(
                {
                    "role": entry.get("role", "user"),
                    "text": entry.get("text", ""),
                    "images": entry.get("images", []),
                }
            )
        return normalized

    async def _cancel_generation(self, channel_id: int) -> None:
        job = self._active_generations.pop(channel_id, None)
        if not job or not job.task:
            return
        job.task.cancel()
        with suppress(asyncio.CancelledError):
            await job.task

    # ------------------------------------------------------------- core flow
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.bot.user:
            return
        if message.author.bot:
            return
        channel = message.channel
        if not self._channel_allowed(channel):
            return
        if self._should_ignore_message(message.content):
            return
        await self._ensure_channel_warm(channel)
        provider, model = self._get_model_for_user(getattr(message.guild, "id", None), message.author.id)
        channel_id = channel.id
        lock = self._lock_for_channel(channel_id)
        async with lock:
            model_turn, history_turn = await self._prepare_user_turn(message)
            history = self._history_for_channel(channel_id)
            snapshot = list(history)
            conversation = self._build_conversation_payload(snapshot, model_turn)
            history.append(history_turn)
        await self._cancel_generation(channel_id)
        job = GenerationJob()
        task = self.bot.loop.create_task(
            self._generate_and_send_response(
                message,
                model_turn,
                self._conversation_for_celery(conversation),
                provider,
                model,
                job,
            )
        )
        job.task = task
        self._active_generations[channel_id] = job

    async def _generate_and_send_response(
        self,
        message: discord.Message,
        current_turn: ModelTurn,
        conversation: list[dict[str, object]],
        provider: str,
        model: str,
        job: GenerationJob,
    ) -> None:
        channel = message.channel
        async_result: AsyncResult | None = None
        try:
            async with channel.typing():
                async_result = generate_llm_chat_response.apply_async(
                    (provider, model, conversation),
                    kwargs={"temperature": 1.0},
                )
                job.result = async_result
                start = time.monotonic()
                while True:
                    if async_result.ready():
                        break
                    if (time.monotonic() - start) > self.text_timeout:
                        async_result.revoke(terminate=True)
                        raise TimeoutError(f"timed out after {self.text_timeout:.0f}s")
                    await asyncio.sleep(0.5)
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    functools.partial(async_result.get, timeout=0.1),
                )
            text = str(result.get("text", "")) if isinstance(result, dict) else str(result)
        except asyncio.CancelledError:
            if async_result and not async_result.ready():
                async_result.revoke(terminate=True)
            raise
        except Exception as exc:  # noqa: BLE001
            _LOG.exception(
                "Generation failed for channel %s (provider=%s model=%s): %s",
                channel.id,
                provider,
                model,
                exc,
            )
            await channel.send(f"Generation failed: {exc}")
            return
        finally:
            if self._active_generations.get(channel.id) is job:
                self._active_generations.pop(channel.id, None)
        if not text.strip():
            await channel.send("Generation failed: empty response.")
            return
        # Check for <no response> directive
        if text.strip().lower() == "<no response>":
            _LOG.info("LLM chose not to respond in channel %s", channel.id)
            return
        assistant_turn = await self._deliver_response(channel, text)
        assistant_turn.role = "assistant"
        lock = self._lock_for_channel(channel.id)
        async with lock:
            history = self._history_for_channel(channel.id)
            # Set a synthetic message id by referencing the most recent Discord message.
            if channel.last_message_id:
                assistant_turn.message_id = channel.last_message_id
            history.append(assistant_turn)

    async def _deliver_response(
        self, channel: discord.abc.MessageableChannel, raw_text: str
    ) -> ConversationTurn:
        text = raw_text.strip()
        svg_files: list[discord.File] = []
        svg_images: list[ImageAttachment] = []

        def _replace_svg(match: re.Match[str]) -> str:
            svg = match.group(0)
            idx = len(svg_files) + 1
            timestamp = int(time.time())
            png_name = f"svg_{timestamp}_{idx}.png"
            if cairosvg:
                try:
                    png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"))  # type: ignore[arg-type]
                    svg_files.append(discord.File(io.BytesIO(png_bytes), filename=png_name))
                    svg_images.append(self._image_from_bytes(png_name, png_bytes))
                    return f"[see file: {png_name}]"
                except Exception:  # noqa: BLE001
                    _LOG.exception("Failed to rasterise SVG; falling back to raw SVG file.")
            svg_name = f"svg_{timestamp}_{idx}.svg"
            svg_files.append(discord.File(io.BytesIO(svg.encode("utf-8")), filename=svg_name))
            return f"[see file: {svg_name}]"

        text = _SVG_BLOCK_RE.sub(_replace_svg, text)

        code_files: list[discord.File] = []
        if len(text) > 2000:
            def _extract_code(match: re.Match[str]) -> str:
                lang = (match.group("lang") or "").strip().lower()
                body = match.group("body") or ""
                filename = f"code_{len(code_files) + 1}.{self._code_extension(lang)}"
                code_files.append(discord.File(io.BytesIO(body.encode("utf-8")), filename=filename))
                return f"[see file: {filename}]"

            text = _CODE_BLOCK_RE.sub(_extract_code, text)

        chunks = self._chunk_text(text)
        sent_messages: list[discord.Message] = []
        if not chunks:
            chunks = ["[no content]"]
        for chunk in chunks:
            sent = await channel.send(chunk)
            sent_messages.append(sent)
        all_files = svg_files + code_files
        if all_files:
            sent = await channel.send("Generated attachments:", files=all_files)
            sent_messages.append(sent)
        history_text = "\n".join(chunk for chunk in chunks if chunk).strip()
        if not history_text:
            history_text = "[no content]"
        turn = ConversationTurn(role="assistant", content=history_text, images=svg_images)
        if sent_messages:
            turn.message_id = sent_messages[-1].id
        return turn

    # --------------------------------------------------------------- commands
    def _format_model_listing(self, current: tuple[str, str] | None = None) -> str:
        current_normalized = None
        if current:
            current_normalized = f"{current[0].lower()}/{current[1]}"
        lines = ["Available models:"]
        for provider, model in sorted(self.available_models):
            token = f"{provider}/{model}"
            if token.lower() == current_normalized:
                lines.append(f"- **{token}** (current)")
            else:
                lines.append(f"- {token}")
        return "\n".join(lines)

    @commands.command(name="models")
    async def models_command(self, ctx: commands.Context) -> None:
        if not self._channel_allowed(ctx.channel):
            return
        await ctx.send(self._format_model_listing())

    @commands.command(name="model")
    async def model_command(self, ctx: commands.Context, *, selection: str | None = None) -> None:
        if not self._channel_allowed(ctx.channel):
            return
        current = self._get_model_for_user(getattr(ctx.guild, "id", None), ctx.author.id)
        if not selection:
            await ctx.send(self._format_model_listing(current))
            return
        candidate = selection.strip()
        if "/" not in candidate:
            await ctx.send("Please provide a model in the format `provider/model`.\n" + self._format_model_listing(current))
            return
        provider, model = candidate.split("/", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if not self._is_valid_model(provider, model):
            await ctx.send(
                "Unknown model.\n" + self._format_model_listing(current)
            )
            return
        self._set_model_for_user(getattr(ctx.guild, "id", None), ctx.author.id, provider, model)
        await ctx.send(f"Model preference updated to {provider}/{model}.")

    @commands.command(name="clear")
    async def clear_command(self, ctx: commands.Context) -> None:
        if not self._channel_allowed(ctx.channel):
            return
        channel_id = ctx.channel.id
        await self._cancel_generation(channel_id)
        lock = self._lock_for_channel(channel_id)
        async with lock:
            self.channel_histories[channel_id] = deque(maxlen=self.history_limit)
        await ctx.send("Cleared saved conversation context for this channel.")

    def _reset_histories(self) -> None:
        self.channel_histories.clear()
        self._warmed_channels.clear()

    async def _cancel_all_generations(self) -> None:
        pending = list(self._active_generations.items())
        self._active_generations.clear()
        for channel_id, job in pending:
            if not job.task:
                continue
            job.task.cancel()
        await asyncio.gather(*(job.task for _, job in pending if job.task), return_exceptions=True)

    @commands.command(name="system")
    async def system_command(self, ctx: commands.Context, *, text: str | None = None) -> None:
        if not self._channel_allowed(ctx.channel):
            return
        argument = (text or "").strip()
        if not argument:
            await self._send_system_prompt(ctx)
            return
        if argument.lower() == "reset":
            clear_system_prompt_cache()
            default_prompt = get_default_system_prompt()
            if self.system_prompt == default_prompt:
                await ctx.send("System prompt already matches the default.")
                await self._send_system_prompt(ctx)
                return
            self.system_prompt = default_prompt
        else:
            if argument == self.system_prompt:
                await ctx.send("System prompt unchanged (new text matches existing value).")
                await self._send_system_prompt(ctx)
                return
            self.system_prompt = argument
        self._save_state()
        await self._cancel_all_generations()
        self._reset_histories()
        await ctx.send("System prompt updated and chat history cleared.")
        await self._send_system_prompt(ctx)

    async def _send_system_prompt(self, ctx: commands.Context) -> None:
        data = io.StringIO(self.system_prompt)
        file = discord.File(data, filename="system_prompt.txt")
        await ctx.send("Current global system prompt is attached.", file=file)

    @commands.command(name="h")
    async def help_command(self, ctx: commands.Context) -> None:
        if not self._channel_allowed(ctx.channel):
            return
        lines = [
            "LLM chat quick help:",
            "- Conversation is available only in whitelisted channels.",
            "- Speak normally; avoid leading `!`, `-`, or `edit:` unless you intend a command.",
            "- `!system`, `!system <text>`, and `!system reset` manage the shared system prompt.",
            "- `!clear` wipes the current channel's saved chat context if you need a fresh start.",
            "- `!models` lists options; `!model provider/model` sets your per-guild default.",
            "- Text file attachments are inlined with markers; history keeps lightweight placeholders.",
            "- Images are resized for OpenAI and stay attached to the conversation history.",
            "- Responses show typing, split long messages, convert SVGs to PNG, and attach code blocks when needed.",
            "- `Generation failed: ...` indicates provider or timeout issues; resend your message to retry.",
        ]
        await ctx.send("\n".join(lines))

    # -------------------------------------------------------------- lifecycle
    async def cog_unload(self) -> None:
        await self._cancel_all_generations()


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(LLMChatNewCog(bot))
