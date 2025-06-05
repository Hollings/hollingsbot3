import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import discord
from discord.ext import commands

from cogs.starboard import Starboard


class FakeChannel:
    def __init__(self):
        self.sent = []

    async def send(self, *args, **kwargs):
        self.sent.append({'args': args, 'kwargs': kwargs})


class FakeAuthor:
    def __init__(self, *, bot: bool = False):
        self.bot = bot


class FakeMessage:
    def __init__(self, content: str, author: FakeAuthor):
        self.content = content
        self.author = author
        self.channel = FakeChannel()
        self.attachments = []
        self.jump_url = "http://jump"
        self.id = 1


class FakeReaction:
    def __init__(self, message: FakeMessage):
        self.message = message


class FakeUser:
    def __init__(self, *, bot: bool = False):
        self.bot = bot


@pytest.mark.asyncio
async def test_first_reaction_reposts(monkeypatch):
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix='!', intents=intents)
    os.environ['STARBOARD_CHANNEL_ID'] = '99'
    starboard_channel = FakeChannel()
    monkeypatch.setattr(bot, 'get_channel', lambda cid: starboard_channel if cid == 99 else None)

    cog = Starboard(bot)
    message = FakeMessage('hello', FakeAuthor(bot=True))
    reaction = FakeReaction(message)
    user = FakeUser()

    await cog.on_reaction_add(reaction, user)

    assert starboard_channel.sent
    sent = starboard_channel.sent[0]
    assert 'hello' in sent['args'][0]
    assert message.jump_url in sent['args'][0]


@pytest.mark.asyncio
async def test_second_reaction_does_not_repost(monkeypatch):
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix='!', intents=intents)
    os.environ['STARBOARD_CHANNEL_ID'] = '99'
    starboard_channel = FakeChannel()
    monkeypatch.setattr(bot, 'get_channel', lambda cid: starboard_channel if cid == 99 else None)

    cog = Starboard(bot)
    message = FakeMessage('hello', FakeAuthor(bot=True))
    reaction = FakeReaction(message)
    user = FakeUser()

    await cog.on_reaction_add(reaction, user)
    await cog.on_reaction_add(reaction, user)

    assert len(starboard_channel.sent) == 1
