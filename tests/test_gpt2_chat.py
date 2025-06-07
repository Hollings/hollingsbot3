import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import discord
from discord.ext import commands
from cogs.gpt2_chat import GPT2Chat


class DummyGenerator:
    def __init__(self):
        self.prompts = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return prompt + " response"


class FakeChannel:
    def __init__(self):
        self.sent = []
        self.id = 1

    async def send(self, content: str):
        self.sent.append(content)


class FakeAuthor:
    def __init__(self, *, bot: bool = False):
        self.bot = bot


class FakeMessage:
    def __init__(self, content: str, channel: FakeChannel, author: FakeAuthor):
        self.content = content
        self.channel = channel
        self.author = author


@pytest.mark.asyncio
async def test_gpt2_chat_response(monkeypatch):
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    gen = DummyGenerator()

    async def task(model, prompt):
        assert model == "gpt2-large"
        return await gen.generate(prompt)

    cog = GPT2Chat(bot, channel_id=1, task_func=task)
    channel = FakeChannel()
    msg = FakeMessage("hello", channel, FakeAuthor())
    await cog.on_message(msg)

    assert gen.prompts == ["hello"]
    assert channel.sent and "response" in channel.sent[0]


@pytest.mark.asyncio
async def test_ignore_other_channels(monkeypatch):
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)
    gen = DummyGenerator()

    async def task(model, prompt):
        return await gen.generate(prompt)

    cog = GPT2Chat(bot, channel_id=2, task_func=task)
    channel = FakeChannel()
    msg = FakeMessage("hello", channel, FakeAuthor())
    await cog.on_message(msg)

    assert not channel.sent
