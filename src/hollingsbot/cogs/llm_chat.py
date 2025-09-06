from __future__ import annotations
import os
import re
import io
import asyncio
import collections
import logging
from typing import Iterable, List, Dict, Any
from datetime import datetime, timezone

import discord
from discord.ext import commands

from hollingsbot.tasks import generate_text
from hollingsbot.settings import DEFAULT_SYSTEM_PROMPT
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


class LLMAPIChat(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY_LIMIT))
        self.system_prompts = {}
        self._preload_once = False
        self._warming: set[int] = set()

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
        current = self.system_prompts.get(key, DEFAULT_SYSTEM_PROMPT)
        # No args shows current prompt and does not change it
        if text is None or not text.strip():
            await ctx.reply(f"Your current system prompt is:\n```\n{current}\n```")
            return
        cmd = text.strip().lower()
        if cmd in {"clear", "reset"}:
            self.system_prompts[key] = DEFAULT_SYSTEM_PROMPT
            await ctx.reply(f"Your system prompt has been reset to the default:\n```\n{DEFAULT_SYSTEM_PROMPT}\n```")
            return
        self.system_prompts[key] = text.strip()
        await ctx.reply(f"System prompt set for you:\n```\n{text.strip()}\n```")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if message.channel.id not in WHITELIST:
            return
        # Ignore explicit commands and image edit prompts (handled by image cog)
        content_clean = (message.content or "").strip()
        if content_clean.startswith(("!", "-")):
            return
        if content_clean.lower().startswith("edit:"):
            return

        # Ensure history is warmed for this channel in case preload hasn't finished yet
        await self._ensure_history_for_message_channel(message)

        gid = message.guild.id if message.guild else 0
        key = (gid, message.author.id)
        sys_prompt = self.system_prompts.get(key, DEFAULT_SYSTEM_PROMPT)
        pref = get_model_pref(gid, message.author.id) or (DEFAULT_PROVIDER, DEFAULT_MODEL)
        provider, model = pref

        # Collect images from this message and, if replying, from the referenced message too.
        images = [att for att in message.attachments if _is_image_attachment(att)]
        text_files = [att for att in message.attachments if _is_text_attachment(att) and att not in images]

        replied_msg: discord.Message | None = None
        if message.reference is not None:
            # Try to resolve locally first; fall back to fetch by ID
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

        # Build a reply context prefix and merge in images from the replied message
        reply_prefix = ""
        if replied_msg is not None:
            reply_author = replied_msg.author.display_name if replied_msg.author else "unknown"
            reply_text = (replied_msg.content or "").strip()
            if reply_text:
                reply_prefix = f"(Replying to <{reply_author}>: {reply_text})\n"
            else:
                reply_prefix = f"(Replying to <{reply_author}>.)\n"
            # Merge in image attachments from the replied message
            reply_images = [att for att in replied_msg.attachments if _is_image_attachment(att)]
            if reply_images:
                # Deduplicate by URL to avoid repeats
                seen_urls = {att.url for att in images}
                for att in reply_images:
                    if att.url not in seen_urls:
                        images.append(att)
                        seen_urls.add(att.url)

        # No timestamps in history or provider content
        user_text_base = f"<{message.author.display_name}> {message.content}".strip()
        provider_user_text = f"{reply_prefix}{user_text_base}" if reply_prefix else user_text_base
        history_user_text = f"{reply_prefix}{user_text_base}" if reply_prefix else user_text_base

        # Inline text files for the model for this turn only. Persist a small note instead.
        if text_files:
            for att in text_files:
                try:
                    raw = await att.read()
                    decoded = raw.decode("utf-8", errors="replace")
                except Exception:
                    decoded = "[error reading file]"
                provider_user_text += f"\n\n[begin uploaded file: {att.filename}]\n{decoded}\n[end uploaded file]"
                history_user_text += f"\n\n[uploaded file {att.filename} removed]"

        # Build parts separately so history never contains the file contents
        user_parts_for_provider = _build_internal_parts(provider_user_text, images)
        user_parts_for_history = _build_internal_parts(history_user_text, images)

        convo = [{"role": "system", "content": sys_prompt}]
        for m in self.history[message.channel.id]:
            content_or_parts = m.get("parts") if "parts" in m else m.get("content")
            if content_or_parts is None:
                continue
            convo.append({"role": m.get("role", "user"), "content": _to_provider_content(content_or_parts, provider)})
        convo.append({"role": "user", "content": _to_provider_content(user_parts_for_provider, provider)})

        # Store only the sanitized version in history
        self.history[message.channel.id].append({"role": "user", "parts": user_parts_for_history})

        async with message.channel.typing():
            async_result = await asyncio.to_thread(generate_text.delay, provider, model, convo)
            try:
                reply_text: str = await asyncio.to_thread(async_result.get, timeout=TEXT_TIMEOUT)
            except Exception as exc:
                await message.channel.send(f"Generation failed: {exc}")
                return

        # SVG handling FIRST: detect, render or attach, and strip with a note.
        cleaned_text, svg_files = extract_render_and_strip_svgs(reply_text)

        # Then do large-reply code or file extraction on the already cleaned text.
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

        # Store assistant reply without timestamp in history
        self.history[message.channel.id].append({"role": "assistant", "parts": [{"kind": "text", "text": cleaned_text}]})

        for chunk in _chunks(cleaned_text):
            await message.channel.send(chunk)
        for filename, buf in attachments:
            buf.seek(0)
            await message.channel.send(file=discord.File(buf, filename))
        for filename, buf in svg_files:
            buf.seek(0)
            await message.channel.send(file=discord.File(buf, filename))

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
