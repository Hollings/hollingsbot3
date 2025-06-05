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
        await bot.start(token)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
