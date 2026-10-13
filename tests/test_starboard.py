"""Starboard suppresses duplicates but allows retries after failed delivery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hollingsbot.cogs.starboard import Starboard


@pytest.fixture
def board(monkeypatch):
    monkeypatch.setenv("STARBOARD_CHANNEL_ID", "2")
    cog = Starboard(Mock())
    cog._is_valid_reactor = Mock(return_value=True)
    cog._is_channel_eligible = Mock(return_value=True)
    cog._resolve_channel = AsyncMock(return_value=SimpleNamespace(id=2))
    cog._validate_starboard_channel = Mock(return_value=True)
    cog._fetch_message_safely = AsyncMock(return_value=SimpleNamespace(id=3, author=SimpleNamespace(bot=True)))
    cog._log_starboard_entry = Mock()
    return cog


def reaction():
    return SimpleNamespace(guild_id=1, channel_id=4, message_id=3)


async def test_failed_delivery_can_be_retried(board):
    board._send_to_starboard = AsyncMock(side_effect=[None, SimpleNamespace(id=5)])
    await board.on_raw_reaction_add(reaction())
    assert 3 not in board._seen_message_ids
    await board.on_raw_reaction_add(reaction())
    assert 3 in board._seen_message_ids
    board._log_starboard_entry.assert_called_once()


async def test_concurrent_reactions_post_once(board):
    entered, resume = asyncio.Event(), asyncio.Event()

    async def send(*args):
        entered.set()
        await resume.wait()
        return SimpleNamespace(id=5)

    board._send_to_starboard = AsyncMock(side_effect=send)
    first = asyncio.create_task(board.on_raw_reaction_add(reaction()))
    await entered.wait()
    await board.on_raw_reaction_add(reaction())
    resume.set()
    await first
    board._send_to_starboard.assert_awaited_once()
