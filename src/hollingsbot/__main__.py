"""Hollingsbot3 main entry point.

This module initializes and runs the Discord bot, loading all cogs (extensions)
and handling the connection lifecycle.

Usage:
    python -m hollingsbot

The bot connects to Discord using the token from DISCORD_TOKEN environment variable.
All cogs are loaded on startup, and the bot supports automatic reconnection with
retry logic for transient connection errors.

Configuration is done via environment variables. See docs/CONFIGURATION.md for
the complete reference.
"""

import asyncio
import logging
import os
import time

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("DISCORD_TOKEN")
logger = logging.getLogger("hollingsbot")
logging.basicConfig(level=logging.INFO)

intents = discord.Intents.default()
intents.message_content = True


def _ids_from_env(name: str) -> set[int]:
    raw = os.getenv(name, "") or ""
    return {int(x.strip()) for x in raw.split(",") if x.strip().isdigit()}


_IMG_CHANNEL_IDS = _ids_from_env("STABLE_DIFFUSION_CHANNEL_IDS")


def _dynamic_prefix(bot: commands.Bot, message: discord.Message):
    """
    Allow mention prefix everywhere.
    Allow "!" prefix everywhere now that the image cog properly filters out
    known bot commands. This lets users use !usage, !balance, etc. in image channels.
    """
    return commands.when_mentioned_or("!")(bot, message)


bot = commands.Bot(command_prefix=_dynamic_prefix, intents=intents, case_insensitive=True, help_command=None)


@bot.event
async def on_ready():
    print(f"Bot is ready. Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"Loaded cogs: {list(bot.cogs.keys())}")
    print(f"Commands: {[c.name for c in bot.commands]}")


RESTART_INTERVAL = int(os.getenv("BOT_RESTART_INTERVAL", 6 * 60 * 60))


@tasks.loop(seconds=RESTART_INTERVAL)
async def restart_task():
    print("Restart interval reached; exiting for restart")
    await bot.close()
    os._exit(0)


async def main():
    async with bot:

        async def _ensure_loaded(name: str) -> None:
            # Avoid double-loading across crash/retry loops
            if name in bot.extensions:
                return
            await bot.load_extension(name)

        await _ensure_loaded("hollingsbot.cogs.message_logger")
        await _ensure_loaded("hollingsbot.cogs.general")
        await _ensure_loaded("hollingsbot.cogs.image_gen_cog")
        # await _ensure_loaded("hollingsbot.cogs.gpt2_chat")  # Disabled
        await _ensure_loaded("hollingsbot.cogs.admin")
        await _ensure_loaded("hollingsbot.cogs.credits_cog")
        await _ensure_loaded("hollingsbot.cogs.gif_chain")
        enable_starboard = os.getenv("ENABLE_STARBOARD", "0")
        if enable_starboard not in {"0", "false", "False"}:
            await _ensure_loaded("hollingsbot.cogs.starboard")
        await _ensure_loaded("hollingsbot.cogs.auto_pin")
        await _ensure_loaded("hollingsbot.cogs.chat_coordinator")
        # await _ensure_loaded("hollingsbot.cogs.wendy_outbox")  # Moved to wendy-bot service
        await _ensure_loaded("hollingsbot.cogs.temp_bot_commands")
        await _ensure_loaded("hollingsbot.cogs.debug_commands")
        await _ensure_loaded("hollingsbot.cogs.feature_requests")
        await _ensure_loaded("hollingsbot.cogs.best_bot_posts")
        await _ensure_loaded("hollingsbot.cogs.yeah_streak")
        logger.info("starting bot")
        await bot.start(token)


@bot.event
async def on_message(message):
    if message.author.bot:
        return
    privacy = os.getenv("STABLE_DIFFUSION_PRIVACY", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not privacy:
        logger.info("on_message: %s: %s", message.author, message.content[:120])
    # Important: keep this so other command cogs still work
    await bot.process_commands(message)


if __name__ == "__main__":
    # Robust launcher: retry on transient connect errors (e.g., gateway timeouts)
    while True:
        try:
            asyncio.run(main())
            break  # Normal exit
        except Exception as e:
            # Log and retry with backoff; discord.py sometimes raises during initial connect
            logger.exception("Bot crashed during startup/connect; retrying in 5s: %s", e)
            time.sleep(5)
