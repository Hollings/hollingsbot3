# bot.py
import os
import time
from dotenv import load_dotenv
import discord
from discord.ext import commands, tasks
import asyncio
import logging

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
    Allow "!" only outside image-gen channels and outside DMs.
    This prevents the command layer from consuming bang-prefixed image prompts
    where the image cog should handle them.
    """
    extras = []
    if message.guild is not None:
        if getattr(message.channel, "id", None) not in _IMG_CHANNEL_IDS:
            extras.append("!")
    return commands.when_mentioned_or(*extras)(bot, message)

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

        await _ensure_loaded("hollingsbot.cogs.general")
        await _ensure_loaded("hollingsbot.cogs.image_gen_cog")
        await _ensure_loaded("hollingsbot.cogs.gpt2_chat")
        await _ensure_loaded("hollingsbot.cogs.admin")
        await _ensure_loaded("hollingsbot.cogs.gif_chain")
        enable_starboard = os.getenv("ENABLE_STARBOARD", "0")
        if enable_starboard not in {"0", "false", "False"}:
            await _ensure_loaded("hollingsbot.cogs.starboard")
        await _ensure_loaded("hollingsbot.cogs.llm_chat_new")
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
        except Exception as e:  # noqa: BLE001
            # Log and retry with backoff; discord.py sometimes raises during initial connect
            logger.exception("Bot crashed during startup/connect; retrying in 5s: %s", e)
            time.sleep(5)
