import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
import discord
from discord.ext import commands

from cogs.general import General


class DummyCtx:
    def __init__(self):
        self.sent = []

    async def send(self, content):
        self.sent.append(content)


@pytest.mark.asyncio
async def test_ping():
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    cog = General(bot)
    ctx = DummyCtx()
    await cog.ping.callback(cog, ctx)
    assert ctx.sent == ["Pong!"]
