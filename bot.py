import os
from dotenv import load_dotenv
import discord
from discord.ext import commands

load_dotenv()

token = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

async def main():
    async with bot:
        await bot.load_extension('cogs.general')
        await bot.load_extension('cogs.image_gen_cog')
        await bot.load_extension('cogs.gpt2_chat')
        enable_starboard = os.getenv('ENABLE_STARBOARD', '0')
        if enable_starboard not in {'0', 'false', 'False'}:
            await bot.load_extension('cogs.starboard')
        await bot.load_extension('cogs.pr_manager')
        await bot.start(token)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
