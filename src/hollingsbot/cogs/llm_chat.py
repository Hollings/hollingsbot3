from __future__ import annotations
import os
import re
import io
import asyncio
import collections
from typing import Iterable

import discord
from discord.ext import commands

from hollingsbot.tasks import generate_text
from hollingsbot.prompt_db import get_model_pref, set_model_pref

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

class LLMAPIChat(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY_LIMIT))
        self.system_prompts = {}  # (guild_id, user_id) -> str

    @commands.command(name="models")
    async def list_models(self, ctx: commands.Context) -> None:
        """List all available models from the env variable."""
        if ctx.channel.id not in WHITELIST:
            return
        if not AVAILABLE_MODELS:
            await ctx.reply("No models configured in AVAILABLE_MODELS.")
            return
        await ctx.reply("Available models:\n" + "\n".join(f"- `{m}`" for m in AVAILABLE_MODELS))

    @commands.command(name="model")
    async def set_or_list_model(self, ctx: commands.Context, *, spec: str = None) -> None:
        """Set your preferred model from the allowed list, or list models if no arg given."""
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

        # No arg → list models, mark current
        if not spec:
            msg_lines = []
            for m in AVAILABLE_MODELS:
                if m.lower() == current_str.lower():
                    msg_lines.append(f"- **{m}** (current)")
                else:
                    msg_lines.append(f"- {m}")
            await ctx.reply("Use `!model api/model-name` to select model.\nAvailable models:\n" + "\n".join(msg_lines))
            return

        # Arg provided → set model
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
            # Explicitly set back to default system prompt
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

        convo = [{"role": "system", "content": sys_prompt}]
        convo.extend(self.history[message.channel.id])

        user_text = f"<sent by: {message.author.display_name}> {message.content}"
        convo.append({"role": "user", "content": user_text})
        self.history[message.channel.id].append({"role": "user", "content": user_text})

        pref = get_model_pref(gid, message.author.id) or (DEFAULT_PROVIDER, DEFAULT_MODEL)
        provider, model = pref

        async with message.channel.typing():
            async_result = await asyncio.to_thread(
                generate_text.delay,
                provider,
                model,
                convo
            )
            try:
                reply_text: str = await asyncio.to_thread(async_result.get, timeout=TEXT_TIMEOUT)
            except Exception as exc:
                await message.channel.send(f"Generation failed: {exc}")
                return

        # Extract code blocks
        attachments = []
        def repl(match: re.Match) -> str:
            lang = match.group(1) or "txt"
            code = match.group(2)
            filename = f"code.{lang.strip() if lang.strip() else 'txt'}"
            buf = io.BytesIO(code.encode("utf-8"))
            attachments.append((filename, buf))
            return f"[see file: {filename}]"

        cleaned_text = re.sub(r"```([^\n]*)\n([\s\S]*?)```", repl, reply_text)

        assistant_text = f"<sent by: {self.bot.user.display_name if self.bot.user else 'Bot'}> {cleaned_text}"
        self.history[message.channel.id].append({"role": "assistant", "content": assistant_text})

        for chunk in _chunks(cleaned_text):
            await message.channel.send(chunk)
        for filename, buf in attachments:
            buf.seek(0)
            await message.channel.send(file=discord.File(buf, filename))

    @commands.command(name="h")
    async def help_command(self, ctx: commands.Context) -> None:
        """Show all available commands and features."""
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
            "- Each message is labeled with `<sent by: Nickname>` for context.\n"
            "\n"
            "__Code Blocks__\n"
            "- Any triple-backtick code blocks in responses are removed from the text and uploaded as files.\n"
            "- The text will contain `[see file: filename]` where the code block was.\n"
            "\n"
            "__Model Info__\n"
            "`!models` — List all models from the configured environment variable (if separate command is kept).\n"
        )
        await ctx.reply(help_text)


async def setup(bot: commands.Bot):
    await bot.add_cog(LLMAPIChat(bot))
