"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest


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
