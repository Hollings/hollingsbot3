"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx2
import pytest


class FakeMessagesAPI:
    """Scripted stand-in for the Anthropic Messages endpoint.

    Tests talk to it through a real ``AsyncAnthropic`` client, so the installed
    SDK's method signatures and request serialisation are exercised: a keyword
    the SDK no longer accepts fails here exactly as it does in production,
    which an ``AsyncMock`` on ``messages.create`` (accepting anything) never can.
    """

    def __init__(self) -> None:
        self.reply = "ok"
        self.status = 200
        self.requests: list[dict[str, Any]] = []
        self._client: anthropic.AsyncAnthropic | None = None

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx2.Response(
                self.status, json={"type": "error", "error": {"type": "api_error", "message": "scripted failure"}}
            )
        return httpx2.Response(
            200,
            json={
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": self.reply}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key="test-key",
                max_retries=0,
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle)),
            )
        return self._client


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Route every ``AnthropicTextGenerator`` through a :class:`FakeMessagesAPI`."""
    from hollingsbot.text_generators import anthropic as anthropic_generator

    fake = FakeMessagesAPI()
    monkeypatch.setattr(anthropic_generator, "get_client", lambda _name, _factory: fake.client())
    return fake


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary database file."""
    db_file = temp_dir / "test.db"
    yield str(db_file)


@pytest.fixture
def mock_discord_channel():
    """Create a mock Discord channel."""
    channel = MagicMock()
    channel.id = 123456789
    channel.name = "test-channel"
    channel.send = AsyncMock()
    return channel


@pytest.fixture
def mock_discord_message():
    """Create a mock Discord message."""
    message = MagicMock()
    message.id = 987654321
    message.content = "Test message content"
    message.author = MagicMock()
    message.author.id = 111222333
    message.author.name = "TestUser"
    message.channel = MagicMock()
    message.channel.id = 123456789
    message.reply = AsyncMock()
    return message


@pytest.fixture
def mock_discord_ctx():
    """Create a mock Discord command context."""
    ctx = MagicMock()
    ctx.author = MagicMock()
    ctx.author.id = 111222333
    ctx.author.name = "TestUser"
    ctx.channel = MagicMock()
    ctx.channel.id = 123456789
    ctx.send = AsyncMock()
    ctx.reply = AsyncMock()
    return ctx


@pytest.fixture
def mock_bot():
    """Create a mock Discord bot."""
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = 999888777
    bot.user.name = "TestBot"
    bot.loop = MagicMock()
    return bot
