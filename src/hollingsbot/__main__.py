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

from hollingsbot.settings import parse_id_set

load_dotenv()

token = os.getenv("DISCORD_TOKEN")
logger = logging.getLogger("hollingsbot")
logging.basicConfig(level=logging.INFO)

intents = discord.Intents.default()
intents.message_content = True


def _ids_from_env(name: str) -> set[int]:
    return parse_id_set(os.getenv(name))


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
    if not restart_task.is_running():
        restart_task.start()


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
    """Surface command errors to the user instead of swallowing them.

    Without this handler discord.py only logs, so a typo'd `!spawn 10` or
    `!grant @user lots` failed in total silence.
    """
    # "!" doubles as an image-gen prefix, so unknown commands are normal noise.
    if isinstance(error, commands.CommandNotFound):
        return
    # Commands with their own error handlers deal with it themselves.
    if ctx.command and ctx.command.has_error_handler():
        return
    if isinstance(error, commands.MissingRequiredArgument | commands.BadArgument | commands.BadUnionArgument):
        sig = f"{ctx.prefix}{ctx.command.qualified_name} {ctx.command.signature}".strip()
        doc = ctx.command.short_doc or ""
        await ctx.reply(f"Usage: `{sig}`\n{doc}".strip(), mention_author=False)
        return
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.reply(f"Slow down - try again in {error.retry_after:.0f}s.", mention_author=False)
        return
    if isinstance(error, commands.CheckFailure):
        await ctx.reply("You don't have permission to use that command.", mention_author=False)
        return
    logger.error("Command %s failed", ctx.command, exc_info=error)
    await ctx.reply("Something went wrong running that command.", mention_author=False)


RESTART_INTERVAL = int(os.getenv("BOT_RESTART_INTERVAL", 6 * 60 * 60))


@tasks.loop(seconds=RESTART_INTERVAL)
async def restart_task():
    # tasks.loop runs its first iteration immediately on start; the restart
    # should only happen once a full interval has elapsed.
    if restart_task.current_loop == 0:
        return
    logger.warning("Restart interval (%ss) reached; exiting for supervisor restart", RESTART_INTERVAL)
    await bot.close()
    # Hard-exit so Docker's restart policy (unless-stopped) brings up a fresh
    # process even if some background task would keep the loop alive.
    os._exit(0)


@restart_task.before_loop
async def _before_restart_task():
    await bot.wait_until_ready()


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
        await _ensure_loaded("hollingsbot.cogs.admin")
        await _ensure_loaded("hollingsbot.cogs.credits_cog")
        await _ensure_loaded("hollingsbot.cogs.gif_chain")
        enable_starboard = os.getenv("ENABLE_STARBOARD", "0")
        if enable_starboard not in {"0", "false", "False"}:
            await _ensure_loaded("hollingsbot.cogs.starboard")
        await _ensure_loaded("hollingsbot.cogs.chat_coordinator")
        await _ensure_loaded("hollingsbot.cogs.temp_bot_commands")
        await _ensure_loaded("hollingsbot.cogs.yeah_streak")
        logger.info("starting bot")
        await bot.start(token)


@bot.event
async def on_message(message):
    if message.author.bot:
        # Allow bots to use commands but skip logging
        await bot.process_commands(message)
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
