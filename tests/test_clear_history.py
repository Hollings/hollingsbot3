"""`!clear`: the bots (Jev included) forget the chat up to a message, even across a restart."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from hollingsbot import prompt_db
from hollingsbot.cogs.chat_coordinator import ChatCoordinator
from hollingsbot.cogs.conversation import ConversationTurn
from hollingsbot.cogs.temp_bot_commands import TempBotCommands

CHANNEL = 1553193310439608371


@pytest.fixture
def db(monkeypatch, temp_db):
    monkeypatch.setattr(prompt_db, "DB_PATH", Path(temp_db))


def said(mid):
    return ConversationTurn(role="user", content=f"<A>: message {mid}", message_id=mid, author_name="A")


def ids(coordinator):
    return [t.message_id for t in coordinator._history_for_channel(CHANNEL)]


async def test_stalled_historical_image_keeps_text_and_allows_backfill_to_finish(db, mock_bot, monkeypatch):
    from hollingsbot.cogs import chat_coordinator as coordinator_mod

    monkeypatch.setattr(coordinator_mod, "_HISTORY_IMAGE_TIMEOUT", 0.01)
    cancelled = asyncio.Event()

    async def stalled(message):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(coordinator_mod.chat_utils, "collect_image_attachments", stalled)
    coordinator = ChatCoordinator(mock_bot)
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = CHANNEL
    msg = MagicMock(spec=discord.Message)
    msg.id, msg.content, msg.webhook_id, msg.attachments = 9, "where is the cat?", None, []
    msg.author = MagicMock(bot=False, nick=None, global_name="Human")
    msg.mentions = []
    msg.guild = None

    async def history(limit):
        yield msg

    channel.history = history
    await asyncio.wait_for(coordinator._ensure_channel_warm(channel), 1)
    assert cancelled.is_set()
    assert CHANNEL in coordinator._warmed_channels
    assert ids(coordinator) == [9]
    assert "where is the cat?" in coordinator.channel_histories[CHANNEL][0].content
    assert coordinator.channel_histories[CHANNEL][0].images == []


async def test_clear_forgets_up_to_the_message_and_stops_old_replies(db, mock_bot):
    coordinator = ChatCoordinator(mock_bot)
    jev = MagicMock(spec=["receive_message", "_cancel_generation"])
    jev._cancel_generation = AsyncMock()
    coordinator.bots = [jev]
    coordinator._history_for_channel(CHANNEL).extend(said(m) for m in (1, 2, 3, 4, 5))

    await coordinator.clear_history(CHANNEL, 3)

    assert ids(coordinator) == [4, 5]  # sent after the clear point: kept
    jev._cancel_generation.assert_awaited_once_with(CHANNEL)  # no reply from the old chat
    assert prompt_db.get_channel_clear_point(CHANNEL) == 3
    assert prompt_db.get_channel_clear_time(CHANNEL) is not None
    assert prompt_db.get_channel_clear_point(1) is None


async def test_a_restart_backfill_skips_what_was_cleared(db, mock_bot):
    prompt_db.set_channel_clear_point(CHANNEL, 3)
    coordinator = ChatCoordinator(mock_bot)
    coordinator._build_history_turn = AsyncMock(side_effect=lambda msg: said(msg.id))
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = CHANNEL

    def message(mid):
        msg = MagicMock(spec=discord.Message)
        msg.id, msg.content, msg.webhook_id = mid, f"message {mid}", None
        msg.author = MagicMock(bot=False)
        return msg

    async def newest_first(limit):
        for mid in (5, 4, 3, 2, 1):
            yield message(mid)

    channel.history = newest_first

    await coordinator._warm_channel(channel)

    assert ids(coordinator) == [4, 5]


async def test_clear_during_slow_backfill_does_not_restore_old_messages(db, mock_bot):
    coordinator = ChatCoordinator(mock_bot)
    coordinator._build_history_turn = AsyncMock(side_effect=lambda msg: said(msg.id))
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = CHANNEL
    started, resume = asyncio.Event(), asyncio.Event()

    async def newest_first(limit):
        started.set()
        await resume.wait()
        for mid in (5, 4, 3, 2, 1):
            msg = MagicMock(id=mid, content=f"message {mid}", webhook_id=None)
            msg.author.bot = False
            yield msg

    channel.history = newest_first
    warming = asyncio.create_task(coordinator._warm_channel(channel))
    await started.wait()
    await coordinator.clear_history(CHANNEL, 3)
    resume.set()
    await warming
    assert ids(coordinator) == [4, 5]


async def test_concurrent_messages_cancel_the_previous_registered_response(db, mock_bot):
    coordinator = ChatCoordinator(mock_bot)
    coordinator._ensure_channel_warm = AsyncMock()
    preparing, resume = asyncio.Event(), asyncio.Event()
    responding, cancelled = asyncio.Event(), asyncio.Event()

    async def prepare(msg):
        if msg.id == 1:
            preparing.set()
            await resume.wait()
        return said(msg.id)

    async def respond(channel_id, msg, history):
        if msg.id == 1:
            responding.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    coordinator._prepare_full_turn = prepare
    coordinator._try_bot_responses = respond

    def message(mid):
        msg = MagicMock(id=mid, content="hello", attachments=[], webhook_id=None)
        msg.channel.id = CHANNEL
        msg.author.bot = False
        return msg

    first = asyncio.create_task(coordinator.on_message(message(1)))
    await preparing.wait()
    second = asyncio.create_task(coordinator.on_message(message(2)))
    resume.set()
    await asyncio.wait_for(responding.wait(), 1)
    await asyncio.wait_for(asyncio.gather(first, second), 1)
    assert cancelled.is_set()
    assert ids(coordinator) == [1, 2]
    assert not coordinator._active_message_tasks


async def test_delayed_event_before_clear_point_is_ignored(db, mock_bot):
    coordinator = ChatCoordinator(mock_bot)
    await coordinator.clear_history(CHANNEL, 3)
    message = MagicMock(id=2, content="hello", attachments=[], webhook_id=None)
    message.channel.id = CHANNEL
    coordinator._prepare_full_turn = AsyncMock()
    await coordinator.on_message(message)
    coordinator._prepare_full_turn.assert_not_awaited()
    assert ids(coordinator) == []


@pytest.mark.parametrize(("reply_to", "expected"), [(None, 77), (40, 40)])
async def test_clear_command_clears_up_to_itself_or_the_message_it_replies_to(reply_to, expected):
    coordinator = MagicMock()
    coordinator.clear_history = AsyncMock()
    bot = MagicMock()
    bot.get_cog.return_value = coordinator
    ctx = MagicMock()
    ctx.channel.id = CHANNEL
    ctx.message.id = 77
    ctx.message.reference = MagicMock(message_id=reply_to) if reply_to else None
    ctx.message.add_reaction = AsyncMock()

    await TempBotCommands.clear_history_command.callback(TempBotCommands(bot), ctx)

    coordinator.clear_history.assert_awaited_once_with(CHANNEL, expected)
    ctx.message.add_reaction.assert_awaited_once_with("✅")
