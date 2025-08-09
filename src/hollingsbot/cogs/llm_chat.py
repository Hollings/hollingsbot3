from __future__ import annotations
import os
import re
import io
import asyncio
import collections
from typing import Iterable, List, Dict, Any

import discord
from discord.ext import commands

from hollingsbot.tasks import generate_text
from hollingsbot.prompt_db import get_model_pref, set_model_pref
from hollingsbot.utils.svg_utils import extract_render_and_strip_svgs

# Env config
WHITELIST = {int(x) for x in os.getenv("LLM_WHITELIST_CHANNELS", "").split(",") if x.strip().isdigit()}
DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic").lower()
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "claude-4o" if DEFAULT_PROVIDER == "anthropic" else "gpt-4o")
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful assistant.")
TEXT_TIMEOUT = float(os.getenv("TEXT_TIMEOUT", "60"))
HISTORY_LIMIT = int(os.getenv("LLM_HISTORY_LIMIT", "10"))

AVAILABLE_MODELS = [m.strip() for m in os.getenv("AVAILABLE_MODELS", "").split(",") if m.strip()]


def _chunks(s: str, limit: int = 1900) -> Iterable[str]:
    for i in range(0, len(s), limit):
        yield s[i : i + limit]


def _is_image_attachment(att: discord.Attachment) -> bool:
    if att.content_type and att.content_type.startswith("image/"):
        return True
    return any(att.filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"))


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
        if not text or text.lower().strip() in {"clear", "reset"}:
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
        if message.content.startswith(("!", "-")):
            return

        gid = message.guild.id if message.guild else 0
        key = (gid, message.author.id)
        sys_prompt = self.system_prompts.get(key, DEFAULT_SYSTEM_PROMPT)
        pref = get_model_pref(gid, message.author.id) or (DEFAULT_PROVIDER, DEFAULT_MODEL)
        provider, model = pref

        images = [att for att in message.attachments if _is_image_attachment(att)]
        user_text = f"<{message.author.display_name}> {message.content}".strip()
        user_parts = _build_internal_parts(user_text, images)

        convo = [{"role": "system", "content": sys_prompt}]
        for m in self.history[message.channel.id]:
            content_or_parts = m.get("parts") if "parts" in m else m.get("content")
            if content_or_parts is None:
                continue
            convo.append({"role": m.get("role", "user"), "content": _to_provider_content(content_or_parts, provider)})
        convo.append({"role": "user", "content": _to_provider_content(user_parts, provider)})

        self.history[message.channel.id].append({"role": "user", "parts": user_parts})

        async with message.channel.typing():
            async_result = await asyncio.to_thread(generate_text.delay, provider, model, convo)
            try:
                reply_text: str = await asyncio.to_thread(async_result.get, timeout=TEXT_TIMEOUT)
            except Exception as exc:
                await message.channel.send(f"Generation failed: {exc}")
                return

        # SVG handling FIRST: detect, render/attach (or fallback), and strip with a note.
        cleaned_text, svg_files = extract_render_and_strip_svgs(reply_text)

        # Then do large-reply code/file extraction on the already cleaned text.
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

        assistant_text = f"<sent by: {self.bot.user.display_name if self.bot.user else 'Bot'}> {cleaned_text}"
        self.history[message.channel.id].append({"role": "assistant", "parts": [{"kind": "text", "text": assistant_text}]})

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
            "`!model` — List available models and show your current one.\n"
            "`!model <name>` — Set your preferred model from the list.\n"
            "\n"
            "__System Prompt__\n"
            "`!system <text>` — Set a custom system prompt for your replies.\n"
            "`!system clear` or `!system reset` — Reset your system prompt to the default.\n"
            "\n"
            "__Conversation Behavior__\n"
            "- The bot responds automatically to all non-command messages in allowed channels.\n"
            "- It remembers the last N messages in the channel for conversation context.\n"
            "- Messages starting with `!` or `-` are ignored (and not added to history).\n"
            "- **Images are supported**: attach images to your message and they will be preserved in the chat history and sent to the model when supported by the selected provider.\n"
            "- **SVG blocks in model replies** are rendered to PNG (if possible) and attached; otherwise the original `.svg` is attached. The SVG code block is replaced in the text with a note.\n"
            "\n"
            "__Code Blocks__\n"
            "- If a response is over 2000 characters, code blocks are removed from the text and uploaded as files.\n"
            "- Otherwise, code blocks stay inline.\n"
            "- When extracted, the text will contain `[see file: filename]` where the code block was.\n"
            "\n"
            "__Model Info__\n"
            "`!models` — List all models from the configured environment variable (if separate command is kept).\n"
        )
        await ctx.reply(help_text)


async def setup(bot: commands.Bot):
    await bot.add_cog(LLMAPIChat(bot))
