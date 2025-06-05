import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import types
mock_replicate = types.SimpleNamespace(Client=object, helpers=types.SimpleNamespace(FileOutput=object))
sys.modules.setdefault("replicate", mock_replicate)

import pytest
import discord
from discord.ext import commands

from cogs.image_gen_cog import ImageGenCog


class MockGenerator:
    def __init__(self, *, should_fail: bool = False):
        self.should_fail = should_fail
        self.prompts = []

    async def generate(self, prompt: str) -> bytes:
        self.prompts.append(prompt)
        if self.should_fail:
            raise RuntimeError("boom")
        return b"image-bytes"


class FakeChannel:
    def __init__(self):
        self.sent = []

    async def send(self, *args, **kwargs):
        self.sent.append({"args": args, "kwargs": kwargs})


class FakeAuthor:
    def __init__(self, *, bot: bool = False):
        self.bot = bot


class FakeMessage:
    def __init__(self, content: str, author: FakeAuthor):
        self.content = content
        self.author = author
        self.channel = FakeChannel()
        self.added = []
        self.cleared = []

    async def add_reaction(self, emoji: str):
        self.added.append(emoji)

    async def clear_reaction(self, emoji: str):
        self.cleared.append(emoji)


@pytest.mark.asyncio
async def test_image_generation_success():
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    gen = MockGenerator()
    config = {"!": {"api": "mock", "model": "m"}}
    factories = {"mock": lambda model: gen}
    cog = ImageGenCog(bot, config=config, factories=factories)
    author = FakeAuthor()
    msg = FakeMessage("!cat", author)
    await cog.on_message(msg)

    assert gen.prompts == ["cat"]
    assert any(isinstance(entry["kwargs"].get("file"), discord.File) for entry in msg.channel.sent)
    assert "\N{WHITE HEAVY CHECK MARK}" in msg.added


@pytest.mark.asyncio
async def test_image_generation_failure():
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    gen = MockGenerator(should_fail=True)
    config = {"!": {"api": "mock", "model": "m"}}
    factories = {"mock": lambda model: gen}
    cog = ImageGenCog(bot, config=config, factories=factories)
    author = FakeAuthor()
    msg = FakeMessage("!cat", author)
    await cog.on_message(msg)

    assert gen.prompts == ["cat"]
    assert any("Error generating image" in entry["args"][0] for entry in msg.channel.sent)
    assert "\N{CROSS MARK}" in msg.added


@pytest.mark.asyncio
async def test_multiple_prefixes():
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    gen1 = MockGenerator()
    gen2 = MockGenerator()
    config = {
        "!": {"api": "mock1", "model": "m1"},
        "$": {"api": "mock2", "model": "m2"},
    }
    factories = {
        "mock1": lambda model: gen1,
        "mock2": lambda model: gen2,
    }
    cog = ImageGenCog(bot, config=config, factories=factories)
    author = FakeAuthor()
    msg = FakeMessage("$dog", author)
    await cog.on_message(msg)

    assert gen1.prompts == []
    assert gen2.prompts == ["dog"]
