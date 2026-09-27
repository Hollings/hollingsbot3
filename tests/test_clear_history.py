"""`!clear`: the bots (Jev included) forget the chat up to a message, even across a restart."""

from __future__ import annotations

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
