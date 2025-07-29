import os
from dotenv import load_dotenv
import discord
from discord.ext import commands, tasks
import asyncio

load_dotenv()

token = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    if not restart_task.is_running():
        restart_task.start()


RESTART_INTERVAL = int(os.getenv('BOT_RESTART_INTERVAL', 6 * 60 * 60))


@tasks.loop(seconds=RESTART_INTERVAL)
async def restart_task():
    print('Restart interval reached; exiting for restart')
    await bot.close()
    os._exit(0)

async def main():
    async with bot:
        await bot.load_extension('cogs.general')
        await bot.load_extension('cogs.image_gen_cog')
        await bot.load_extension('cogs.gpt2_chat')
        await bot.load_extension('cogs.enhance_cog')
        # await bot.load_extension('cogs.image_edit')
        enable_starboard = os.getenv('ENABLE_STARBOARD', '0')
        if enable_starboard not in {'0', 'false', 'False'}:
            await bot.load_extension('cogs.starboard')
        # await bot.load_extension('cogs.pr_manager')
        await bot.start(token)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
