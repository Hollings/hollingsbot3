import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import types
import pytest
import discord
from discord.ext import commands

from cogs.enhance_cog import EnhanceCog


class FakeChannel:
    def __init__(self):
        self.sent = []
        self.messages = {}
        self.id = 1

    async def send(self, content: str):
        self.sent.append(content)

    async def fetch_message(self, mid: int):
        return self.messages[mid]


class FakeAuthor:
    def __init__(self, *, bot: bool = False):
        self.bot = bot


class FakeMessage:
    def __init__(self, content: str, author: FakeAuthor, channel: FakeChannel, *, message_id: int = 1):
        self.content = content
        self.author = author
        self.channel = channel
        self.id = message_id
        self.reference = None

    async def reply(self, content: str):
        await self.channel.send(content)


class FakeReference:
    def __init__(self, message: FakeMessage):
        self.resolved = message
        self.message_id = message.id


@pytest.mark.asyncio
async def test_enhance_reply(monkeypatch):
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix='!', intents=intents)
    called = []

    async def enhance(prompt: str, text: str) -> str:
        called.append((prompt, text))
        return 'better ' + text

    cog = EnhanceCog(bot, prompt='Improve:', enhance_func=enhance)
    channel = FakeChannel()
    original = FakeMessage('hello', FakeAuthor(), channel, message_id=42)
    channel.messages[42] = original
    msg = FakeMessage('enhance', FakeAuthor(), channel, message_id=43)
    msg.reference = FakeReference(original)

    await cog.on_message(msg)

    assert called == [('Improve:', 'hello')]
    assert channel.sent == ['better hello']


@pytest.mark.asyncio
async def test_ignore_without_reference():
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix='!', intents=intents)

    cog = EnhanceCog(bot, prompt='p', enhance_func=lambda p, t: 'x')
    channel = FakeChannel()
    msg = FakeMessage('enhance', FakeAuthor(), channel)

    await cog.on_message(msg)

    assert not channel.sent


@pytest.mark.asyncio
async def test_trim_input_and_output():
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix='!', intents=intents)
    received = []

    async def enhance(prompt: str, text: str) -> str:
        received.append(text)
        return 'y' * 2100

    cog = EnhanceCog(bot, prompt='p', enhance_func=enhance)
    channel = FakeChannel()
    long_text = 'x' * 2500
    original = FakeMessage(long_text, FakeAuthor(), channel, message_id=5)
    channel.messages[5] = original
    msg = FakeMessage('enhance', FakeAuthor(), channel)
    msg.reference = FakeReference(original)

    await cog.on_message(msg)

    assert len(received[0]) == 2000
    assert channel.sent and len(channel.sent[0]) == 2000
