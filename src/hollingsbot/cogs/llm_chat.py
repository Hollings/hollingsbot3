from __future__ import annotations
import os
import re
import io
import asyncio
import collections
import logging
from typing import Iterable, List, Dict, Any, Optional
import base64
from PIL import Image
import aiohttp
from datetime import datetime, timezone

import discord
from discord.ext import commands

from hollingsbot.tasks import generate_text
from hollingsbot.settings import DEFAULT_SYSTEM_PROMPT, get_default_system_prompt
from hollingsbot.prompt_db import get_model_pref, set_model_pref
from hollingsbot.utils.svg_utils import extract_render_and_strip_svgs

# Env config
WHITELIST = {int(x) for x in os.getenv("LLM_WHITELIST_CHANNELS", "").split(",") if x.strip().isdigit()}
DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
# If provider is set to openai (default), prefer gpt-5; otherwise keep a sensible default per provider.
DEFAULT_MODEL = os.getenv(
    "DEFAULT_LLM_MODEL",
    "gpt-5" if DEFAULT_PROVIDER == "openai" else "claude-4o",
)
TEXT_TIMEOUT = float(os.getenv("TEXT_TIMEOUT", "180"))
HISTORY_LIMIT = int(os.getenv("LLM_HISTORY_LIMIT", "50"))
# Cap how many recent turns we actually send to the model (slice of history)
SEND_TURNS_LIMIT = int(os.getenv("LLM_MAX_TURNS_SENT", "16"))

AVAILABLE_MODELS = [m.strip() for m in os.getenv("AVAILABLE_MODELS", "").split(",") if m.strip()]

_log = logging.getLogger(__name__)


def _chunks(s: str, limit: int = 1900) -> Iterable[str]:
    for i in range(0, len(s), limit):
        yield s[i: i + limit]


def _is_image_attachment(att: discord.Attachment) -> bool:
    if att.content_type and att.content_type.startswith("image/"):
        return True
    return any(att.filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"))


def _is_text_attachment(att: discord.Attachment) -> bool:
    """
    Treat common text-like uploads as inlineable for a single turn.
    These are read and sent to the model only for the current message.
    They are not persisted in history. History stores a placeholder instead.
    """
    ct = (att.content_type or "").lower()
    if ct.startswith("text/"):
        return True
    if ct in {"application/json", "application/xml", "application/javascript"}:
        return True
    name = att.filename.lower()
    text_exts = (
        ".txt", ".md", ".markdown", ".json", ".csv", ".tsv",
        ".py", ".js", ".ts", ".html", ".css", ".xml",
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".log", ".sql", ".sh", ".bat", ".ps1",
        ".java", ".kt", ".rs", ".go", ".rb", ".php",
        ".c", ".h", ".cpp", ".hpp", ".cs",
    )
    return any(name.endswith(ext) for ext in text_exts)


def _build_internal_parts(user_text: str, images: List[discord.Attachment]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = [{"kind": "text", "text": user_text}]
    for att in images:
        media_type = att.content_type if (att.content_type and att.content_type.startswith("image/")) else None
        parts.append({
            "kind": "image",
            "url": att.url,
            "media_type": media_type or "image/png",
            "filename": att.filename,
        })
    return parts


def _fmt_ts(dt: datetime) -> str:
    try:
        ts = dt.astimezone(timezone.utc)
    except Exception:
        ts = datetime.now(timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _to_provider_content(parts_or_text: Any, provider: str) -> Any:
    if isinstance(parts_or_text, str):
        return parts_or_text
    if not isinstance(parts_or_text, list):
        return str(parts_or_text)
    out: List[Dict[str, Any]] = []
    for p in parts_or_text:
        kind = p.get("kind")
        if kind == "text":
            out.append({"type": "text", "text": p.get("text", "")})
        elif kind == "image":
            url = p.get("url")
            media_type = p.get("media_type") or "image/png"
            if provider == "openai":
                out.append({"type": "image_url", "image_url": {"url": url}})
            else:
                out.append({"type": "image", "source": {"type": "url", "url": url, "media_type": media_type}})
    return out if out else ""


class _DebouncedGeneration:
    __slots__ = ("channel_id", "message_id", "task", "async_result", "cancel_requested")

    def __init__(self, channel_id: int, message_id: int) -> None:
        self.channel_id = channel_id
        self.message_id = message_id
        self.task: Optional[asyncio.Task] = None
        self.async_result: Any | None = None
        self.cancel_requested = False

    async def cancel(self) -> None:
        if self.cancel_requested:
            return
        self.cancel_requested = True

        async_result = self.async_result
        if async_result is not None:
            try:
                ready = getattr(async_result, "ready", None)
                if callable(ready) and ready():
                    self.async_result = None
                    async_result = None
            except Exception:
                # If ready() itself fails, continue with revoke attempt.
                pass

        if async_result is not None:
            try:
                await asyncio.to_thread(async_result.revoke, terminate=True)
            except Exception as exc:  # noqa: BLE001
                _log.debug(
                    "Failed to revoke pending LLM generation for channel %s: %s",
                    self.channel_id,
                    exc,
                )

        task = self.task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                _log.debug(
                    "Debounced generation task for channel %s raised during cancel",
                    self.channel_id,
                    exc_info=True,
                )

class LLMAPIChat(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY_LIMIT))
        self.system_prompts = {}
        self._preload_once = False
        self._warming: set[int] = set()
        self._inflight_generations: Dict[int, _DebouncedGeneration] = {}

    async def _attachment_to_data_url(
        self,
        att: discord.Attachment,
        *,
        max_side: int = 2048,
        target_bytes: int = 9_500_000,
    ) -> str:
        """Download, downscale, and JPEG-compress an image attachment to a data URL.

        - Scales the longest side to at most `max_side`.
        - Tries multiple scale/quality combinations to fit under `target_bytes`.
        - Converts to RGB and flattens transparency against white.
        """
        raw = await att.read()
        img = Image.open(io.BytesIO(raw))
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "LA":
                rgb = img.convert("RGBA")
                bg.paste(rgb, mask=rgb.split()[-1])
            else:
                bg.paste(img, mask=img.split()[-1])
            base = bg
        else:
            base = img.convert("RGB")

        # Initial constrain to max_side
        w, h = base.size
        scale0 = 1.0
        if max(w, h) > max_side:
            scale0 = max_side / float(max(w, h))
        def _resized(im: Image.Image, s: float) -> Image.Image:
            if s >= 0.999:
                return im
            nw, nh = max(1, int(im.width * s)), max(1, int(im.height * s))
            return im.resize((nw, nh), Image.LANCZOS)

        candidates_scale = [scale0, scale0 * 0.85, scale0 * 0.7, scale0 * 0.5]
        qualities = [85, 75, 65, 50, 40, 30]

        for s in candidates_scale:
            im = _resized(base, s)
            for q in qualities:
                buf = io.BytesIO()
                try:
                    im.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
                except Exception:
                    buf.seek(0); buf.truncate(0)
                    im.save(buf, format="JPEG", quality=q)
                data = buf.getvalue()
                if len(data) <= target_bytes:
                    b64 = base64.b64encode(data).decode("ascii")
                    return f"data:image/jpeg;base64,{b64}"
        # Fallback: return the smallest we tried anyway
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    async def _build_provider_parts(
        self,
        provider: str,
        user_text: str,
        images: List[discord.Attachment],
    ) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"kind": "text", "text": user_text}]
        for att in images:
            media_type = att.content_type if (att.content_type and att.content_type.startswith("image/")) else None
            url = att.url
            # For OpenAI, always prefer a data URL to avoid provider-side fetch failures
            if provider == "openai":
                try:
                    url = await self._attachment_to_data_url(att)
                except Exception:
                    # On failure, fall back to original URL
                    url = att.url
            parts.append({
                "kind": "image",
                "url": url,
                "media_type": media_type or "image/png",
                "filename": att.filename,
            })
        return parts

    async def _bytes_to_jpeg_data_url(
        self,
        raw: bytes,
        *,
        max_side: int = 2048,
        target_bytes: int = 9_500_000,
    ) -> str:
        img = Image.open(io.BytesIO(raw))
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "LA":
                rgb = img.convert("RGBA")
                bg.paste(rgb, mask=rgb.split()[-1])
            else:
                bg.paste(img, mask=img.split()[-1])
            base = bg
        else:
            base = img.convert("RGB")

        w, h = base.size
        scale0 = 1.0
        if max(w, h) > max_side:
            scale0 = max_side / float(max(w, h))

        def _resized(im: Image.Image, s: float) -> Image.Image:
            if s >= 0.999:
                return im
            nw, nh = max(1, int(im.width * s)), max(1, int(im.height * s))
            return im.resize((nw, nh), Image.LANCZOS)

        candidates_scale = [scale0, scale0 * 0.85, scale0 * 0.7, scale0 * 0.5]
        qualities = [85, 75, 65, 50, 40, 30]
        last = raw
        for s in candidates_scale:
            im = _resized(base, s)
            for q in qualities:
                buf = io.BytesIO()
                try:
                    im.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
                except Exception:
                    buf.seek(0); buf.truncate(0)
                    im.save(buf, format="JPEG", quality=q)
                data = buf.getvalue()
                last = data
                if len(data) <= target_bytes:
                    b64 = base64.b64encode(data).decode("ascii")
                    return f"data:image/jpeg;base64,{b64}"
        b64 = base64.b64encode(last).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    async def _url_to_data_url(
        self,
        url: str,
        *,
        max_side: int = 2048,
        target_bytes: int = 9_500_000,
        timeout: float = 20.0,
    ) -> str:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as resp:
                    raw = await resp.read()
        except Exception:
            # If fetch fails, just return original URL; provider will decide
            return url
        return await self._bytes_to_jpeg_data_url(raw, max_side=max_side, target_bytes=target_bytes)

    async def _prepare_provider_content(self, content_or_parts: Any, provider: str) -> Any:
        """Like _to_provider_content, but compresses images for OpenAI across history.

        - If provider != openai, falls back to _to_provider_content.
        - For lists of parts, converts image URLs to compressed data URLs.
        """
        if provider != "openai":
            return _to_provider_content(content_or_parts, provider)
        if isinstance(content_or_parts, str):
            return content_or_parts
        if not isinstance(content_or_parts, list):
            return _to_provider_content(content_or_parts, provider)
        new_parts: List[Dict[str, Any]] = []
        for p in content_or_parts:
            try:
                if p.get("kind") == "image":
                    url = p.get("url")
                    # Only allow http(s) or data URLs; drop anything else to avoid provider 400s
                    if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://") or url.startswith("data:")):
                        continue
                    if isinstance(url, str) and not url.startswith("data:"):
                        try:
                            url = await self._url_to_data_url(url)
                        except Exception:
                            # Failed to fetch/convert -> drop this image from the prompt
                            continue
                        # If conversion didn't result in a data URL, drop to avoid provider-side fetch
                        if not isinstance(url, str) or not url.startswith("data:"):
                            continue
                    np = dict(p)
                    np["url"] = url
                    new_parts.append(np)
                else:
                    new_parts.append(p)
            except Exception:
                new_parts.append(p)
        return _to_provider_content(new_parts, provider)

    async def _resolve_referenced_message(self, msg: discord.Message) -> discord.Message | None:
        ref = msg.reference
        if not ref:
            return None
        resolved = getattr(ref, "resolved", None)
        if isinstance(resolved, discord.Message):
            return resolved
        ref_id = getattr(ref, "message_id", None)
        if ref_id:
            try:
                return await msg.channel.fetch_message(ref_id)
            except Exception:
                return None
        return None

    async def _preload_history_for_channel(self, channel_id: int) -> None:
        # Skip if already has content
        if self.history.get(channel_id):
            return
        channel = self.bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except Exception as exc:  # noqa: BLE001
                _log.debug("LLM preload: cannot fetch channel %s: %s", channel_id, exc)
                return
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return
        # Collect most recent messages and build a lightweight history
        try:
            # Fetch newest-first limited slice, then reverse for chronological processing
            msgs: List[discord.Message] = [m async for m in channel.history(limit=HISTORY_LIMIT, oldest_first=False)]
            msgs.reverse()
        except Exception as exc:  # noqa: BLE001
            _log.debug("LLM preload: history fetch failed for %s: %s", channel_id, exc)
            return

        dq = collections.deque(maxlen=HISTORY_LIMIT)
        for m in msgs:
            try:
                # Ignore commands and unrelated bots
                if m.content.startswith(("!", "-")):
                    continue
                if m.author.bot and (self.bot.user is None or m.author.id != self.bot.user.id):
                    continue

                # Assistant messages: store as assistant text (no timestamp)
                if self.bot.user and m.author.id == self.bot.user.id:
                    assistant_text_raw = (m.content or "").strip()
                    if assistant_text_raw:
                        dq.append({"role": "assistant", "parts": [{"kind": "text", "text": assistant_text_raw}]})
                    continue

                # User message
                images = [att for att in m.attachments if _is_image_attachment(att)]
                text_files = [att for att in m.attachments if _is_text_attachment(att) and att not in images]

                # Reply prefix + merge reply images (best-effort)
                reply_prefix = ""
                replied_msg = await self._resolve_referenced_message(m)
                if replied_msg is not None:
                    reply_author = replied_msg.author.display_name if replied_msg.author else "unknown"
                    reply_text = (replied_msg.content or "").strip()
                    reply_prefix = f"(Replying to <{reply_author}>: {reply_text})\n" if reply_text else f"(Replying to <{reply_author}>.)\n"
                    reply_images = [att for att in replied_msg.attachments if _is_image_attachment(att)]
                    if reply_images:
                        seen = {att.url for att in images}
                        for att in reply_images:
                            if att.url not in seen:
                                images.append(att)
                                seen.add(att.url)

                user_text_base = f"<{m.author.display_name}> {(m.content or '').strip()}".strip()
                history_user_text = f"{reply_prefix}{user_text_base}" if reply_prefix else user_text_base

                # Placeholders for text files, not persisted
                if text_files:
                    for att in text_files:
                        history_user_text += f"\n\n[uploaded file {att.filename} removed]"

                user_parts_for_history = _build_internal_parts(history_user_text, images)
                dq.append({"role": "user", "parts": user_parts_for_history})
            except Exception as exc:  # noqa: BLE001
                _log.debug("LLM preload: skipping message due to error: %s", exc)

        if dq:
            self.history[channel_id] = dq
            _log.info("LLM preload: initialized history for channel %s with %d items", channel_id, len(dq))

    async def _ensure_history_for_message_channel(self, message: discord.Message) -> None:
        """On-demand warmup of history for a channel if empty (e.g., startup race)."""
        cid = message.channel.id
        if cid in self._warming:
            return
        if self.history.get(cid):
            return
        # Only warm whitelisted channels
        if cid not in WHITELIST:
            return
        self._warming.add(cid)
        try:
            await self._preload_history_for_channel(cid)
        finally:
            self._warming.discard(cid)

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        # Only run once per process
        if self._preload_once:
            return
        self._preload_once = True
        # Preload for whitelisted channels only
        tasks = [self._preload_history_for_channel(cid) for cid in WHITELIST]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @commands.command(name="models")
    async def list_models(self, ctx: commands.Context) -> None:
        if ctx.channel.id not in WHITELIST:
            return
        if not AVAILABLE_MODELS:
            await ctx.reply("No models configured in AVAILABLE_MODELS.")
            return
        await ctx.reply("Available models:\n" + "\n".join(f"- `{m}`" for m in AVAILABLE_MODELS))

    @commands.command(name="model")
    async def set_or_list_model(self, ctx: commands.Context, *, spec: str = None) -> None:
        if ctx.channel.id not in WHITELIST:
            return
        if not AVAILABLE_MODELS:
            await ctx.reply("No models configured in AVAILABLE_MODELS.")
            return
        gid = ctx.guild.id if ctx.guild else 0
        pref = get_model_pref(gid, ctx.author.id)
        current_provider, current_model = pref if pref else (DEFAULT_PROVIDER, DEFAULT_MODEL)
        current_human = "chatgpt" if current_provider == "openai" else current_provider
        current_str = f"{current_human}/{current_model}"
        if not spec:
            msg_lines = []
            for m in AVAILABLE_MODELS:
                if m.lower() == current_str.lower():
                    msg_lines.append(f"- **{m}** (current)")
                else:
                    msg_lines.append(f"- {m}")
            await ctx.reply("Use `!model api/model-name` to select model.\nAvailable models:\n" + "\n".join(msg_lines))
            return
        match = None
        for m in AVAILABLE_MODELS:
            if m.lower() == spec.lower():
                match = m
                break
        if not match:
            await ctx.reply("Model not found. Use `!model` to see the list.")
            return
        provider_raw, model = match.split("/", 1)
        provider = "openai" if provider_raw.lower() in ("chatgpt", "openai") else provider_raw.lower()
        set_model_pref(gid, ctx.author.id, provider, model.strip())
        await ctx.reply(f"Model set to **{match}** for you.")

    @commands.command(name="system")
    async def set_system_prompt(self, ctx: commands.Context, *, text: str = None) -> None:
        if ctx.channel.id not in WHITELIST:
            return
        gid = ctx.guild.id if ctx.guild else 0
        key = (gid, ctx.author.id)
        current = self.system_prompts.get(key, get_default_system_prompt())
        # No args shows current prompt and does not change it
        if text is None or not text.strip():
            await ctx.reply(f"Your current system prompt is:\n```\n{current}\n```")
            return
        cmd = text.strip().lower()
        if cmd in {"clear", "reset"}:
            new_default = get_default_system_prompt()
            self.system_prompts[key] = new_default
            await ctx.reply(f"Your system prompt has been reset to the default:\n```\n{new_default}\n```")
            return
        self.system_prompts[key] = text.strip()
        await ctx.reply(f"System prompt set for you:\n```\n{text.strip()}\n```")

    async def _start_debounced_generation(self, message: discord.Message) -> None:
        channel_id = message.channel.id
        prev = self._inflight_generations.get(channel_id)
        if prev is not None:
            await prev.cancel()

        handle = _DebouncedGeneration(channel_id, message.id)
        task = asyncio.create_task(self._run_generation(message, handle))
        handle.task = task
        self._inflight_generations[channel_id] = handle

        def _log_task_result(task_obj: asyncio.Task) -> None:
            try:
                task_obj.result()
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                _log.exception(
                    "LLM chat task failed for channel %s during debounce handling",
                    channel_id,
                )

        task.add_done_callback(_log_task_result)

    async def _run_generation(self, message: discord.Message, handle: _DebouncedGeneration) -> None:
        channel_id = message.channel.id
        appended_user_history = False
        user_history_entry: Optional[Dict[str, Any]] = None

        images: List[discord.Attachment] = []
        text_files: List[discord.Attachment] = []

        try:
            gid = message.guild.id if message.guild else 0
            key = (gid, message.author.id)
            sys_prompt = self.system_prompts.get(key, get_default_system_prompt())
            pref = get_model_pref(gid, message.author.id) or (DEFAULT_PROVIDER, DEFAULT_MODEL)
            provider, model = pref

            images = [att for att in message.attachments if _is_image_attachment(att)]
            text_files = [att for att in message.attachments if _is_text_attachment(att) and att not in images]

            replied_msg: discord.Message | None = None
            if message.reference is not None:
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

            reply_prefix = ""
            if replied_msg is not None:
                reply_author = replied_msg.author.display_name if replied_msg.author else "unknown"
                reply_text = (replied_msg.content or "").strip()
                reply_prefix = (
                    f"(Replying to <{reply_author}>: {reply_text})\n"
                    if reply_text
                    else f"(Replying to <{reply_author}>.)\n"
                )
                reply_images = [att for att in replied_msg.attachments if _is_image_attachment(att)]
                if reply_images:
                    seen_urls = {att.url for att in images}
                    for att in reply_images:
                        if att.url not in seen_urls:
                            images.append(att)
                            seen_urls.add(att.url)

            user_text_base = f"<{message.author.display_name}> {message.content}".strip()
            provider_user_text = f"{reply_prefix}{user_text_base}" if reply_prefix else user_text_base
            history_user_text = f"{reply_prefix}{user_text_base}" if reply_prefix else user_text_base

            if text_files:
                for att in text_files:
                    history_user_text += f"\n\n[uploaded file {att.filename} removed]"

            user_parts_for_history = _build_internal_parts(history_user_text, images)
            user_history_entry = {"role": "user", "parts": user_parts_for_history}

            recent_turns = list(self.history[channel_id])[-max(SEND_TURNS_LIMIT, 0):]

            convo = [{"role": "system", "content": sys_prompt}]
            for m in recent_turns:
                content_or_parts = m.get("parts") if "parts" in m else m.get("content")
                if content_or_parts is None:
                    continue
                content_prepared = await self._prepare_provider_content(content_or_parts, provider)
                convo.append({"role": m.get("role", "user"), "content": content_prepared})

            if text_files:
                for att in text_files:
                    try:
                        raw = await att.read()
                        decoded = raw.decode("utf-8", errors="replace")
                    except Exception:
                        decoded = "[error reading file]"
                    provider_user_text += (
                        f"\n\n[begin uploaded file: {att.filename}]\n{decoded}\n[end uploaded file]"
                    )

            user_parts_for_provider = await self._build_provider_parts(provider, provider_user_text, images)
            convo.append({"role": "user", "content": _to_provider_content(user_parts_for_provider, provider)})

            self.history[channel_id].append(user_history_entry)
            appended_user_history = True

            async with message.channel.typing():
                async_result = await asyncio.to_thread(generate_text.delay, provider, model, convo)
                handle.async_result = async_result
                try:
                    reply_text: str = await asyncio.to_thread(async_result.get, timeout=TEXT_TIMEOUT)
                except Exception as exc:
                    if handle.cancel_requested:
                        raise asyncio.CancelledError
                    await message.channel.send(f"Generation failed: {exc}")
                    return
                finally:
                    handle.async_result = None

            if handle.cancel_requested:
                raise asyncio.CancelledError

            cleaned_text, svg_files = extract_render_and_strip_svgs(reply_text)

            attachments = []
            if len(cleaned_text) > 2000:
                def repl(match: re.Match) -> str:
                    lang = match.group(1) or "txt"
                    code = match.group(2)
                    filename = f"code.{lang.strip() if lang.strip() else 'txt'}"
                    buf = io.BytesIO(code.encode("utf-8"))
                    attachments.append((filename, buf))
                    return f"[see file: {filename}]"

                cleaned_text = re.sub(r"```([^\n]*)\n([\s\S]*?)```", repl, cleaned_text)

            if handle.cancel_requested:
                raise asyncio.CancelledError

            self.history[channel_id].append({"role": "assistant", "parts": [{"kind": "text", "text": cleaned_text}]})

            for chunk in _chunks(cleaned_text):
                if handle.cancel_requested:
                    raise asyncio.CancelledError
                await message.channel.send(chunk)

            for filename, buf in attachments:
                if handle.cancel_requested:
                    raise asyncio.CancelledError
                buf.seek(0)
                await message.channel.send(file=discord.File(buf, filename))

            for filename, buf in svg_files:
                if handle.cancel_requested:
                    raise asyncio.CancelledError
                buf.seek(0)
                await message.channel.send(file=discord.File(buf, filename))

        except asyncio.CancelledError:
            if not appended_user_history and user_history_entry is not None:
                self.history[channel_id].append(user_history_entry)
                appended_user_history = True
            elif not appended_user_history:
                fallback_text = f"<{message.author.display_name}> {(message.content or '').strip()}".strip()
                fallback_entry = {"role": "user", "parts": _build_internal_parts(fallback_text, images)}
                self.history[channel_id].append(fallback_entry)
                appended_user_history = True

            async_result = handle.async_result
            if async_result is not None:
                try:
                    ready = getattr(async_result, "ready", None)
                    if not (callable(ready) and ready()):
                        await asyncio.to_thread(async_result.revoke, terminate=True)
                except Exception:
                    _log.debug(
                        "Failed to revoke cancelled LLM generation for channel %s",
                        channel_id,
                        exc_info=True,
                    )
            handle.async_result = None
            raise

        except Exception as exc:  # noqa: BLE001
            if not appended_user_history and user_history_entry is not None:
                self.history[channel_id].append(user_history_entry)
                appended_user_history = True
            elif not appended_user_history:
                fallback_text = f"<{message.author.display_name}> {(message.content or '').strip()}".strip()
                fallback_entry = {"role": "user", "parts": _build_internal_parts(fallback_text, images)}
                self.history[channel_id].append(fallback_entry)
                appended_user_history = True

            if not handle.cancel_requested:
                _log.exception("LLM generation failed for channel %s", channel_id)
                try:
                    await message.channel.send(f"Generation failed: {exc}")
                except Exception:  # noqa: BLE001
                    _log.debug("Unable to notify channel %s about failure", channel_id, exc_info=True)

        finally:
            handle.async_result = None
            current = self._inflight_generations.get(channel_id)
            if current is handle:
                self._inflight_generations.pop(channel_id, None)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if message.channel.id not in WHITELIST:
            return
        content_clean = (message.content or "").strip()
        if content_clean.startswith(("!", "-")):
            return
        if content_clean.lower().startswith("edit:"):
            return

        await self._ensure_history_for_message_channel(message)
        await self._start_debounced_generation(message)

    @commands.command(name="h")
    async def help_command(self, ctx: commands.Context) -> None:
        if ctx.channel.id not in WHITELIST:
            return
        help_text = (
            "**LLM Bot Commands & Features**\n\n"
            "__Model Selection__\n"
            "`!model` - List available models and show your current one.\n"
            "`!model <name>` - Set your preferred model from the list.\n"
            "\n"
            "__System Prompt__\n"
            "`!system` - Show your current system prompt.\n"
            "`!system <text>` - Set a custom system prompt for your replies.\n"
            "`!system clear` or `!system reset` - Reset your system prompt to the default.\n"
            "\n"
            "__Conversation Behavior__\n"
            "- The bot responds automatically to all non-command messages in allowed channels.\n"
            "- It remembers the last N messages in the channel for conversation context.\n"
            "- Messages starting with `!` or `-` are ignored and not added to history.\n"
            "- Images are supported. Attach images to your message and they are preserved in chat history and sent to the model when supported by the selected provider.\n"
            "- Replies: when you reply to a message, the bot includes that message's text and any of its image attachments as context for your turn.\n"
            "- Text file uploads are supported for a single turn. The model reads the file contents for that message only, and chat history records a note like `[uploaded file filename.txt removed]` instead of the full contents.\n"
            "- SVG blocks in model replies are rendered to PNG if possible and attached. Otherwise the original `.svg` is attached. The SVG code block is replaced in the text with a note.\n"
            "\n"
            "__Code Blocks__\n"
            "- If a response is over 2000 characters, code blocks are removed from the text and uploaded as files.\n"
            "- Otherwise, code blocks stay inline.\n"
            "- When extracted, the text will contain `[see file: filename]` where the code block was.\n"
            "\n"
            "__Model Info__\n"
            "`!models` - List all models from the configured environment variable if that separate command is kept.\n"
        )
        await ctx.reply(help_text)


async def setup(bot: commands.Bot):
    await bot.add_cog(LLMAPIChat(bot))
