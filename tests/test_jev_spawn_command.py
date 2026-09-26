"""`!spawn jev [N]` / `!despawn jev`: the temp bot commands hand Jev's spawns to JevBot."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hollingsbot.cogs import temp_bot_commands as cmds
from hollingsbot.cogs.chat_bots.jev_bot import SPAWN_REPLIES_DEFAULT, JevBotSettings
from hollingsbot.cogs.temp_bot_commands import TempBotCommands, jev_spawn_replies

HOME, AWAY = 1, 2
NAMES = {"jev"}


@pytest.mark.parametrize(
    ("first", "rest", "expected"),
    [
        ("jev", "10", 10),
        ("JEV", " 4 ", 4),
        ("jev", "", SPAWN_REPLIES_DEFAULT),
        ("10", "jev", 10),
        ("10", "Jev", 10),
        ("10", "a grumpy pirate", None),  # a temp bot
        ("10", "jev the pirate", None),  # a temp bot that happens to mention jev
    ],
)
def test_jev_spawn_replies(first, rest, expected):
    assert jev_spawn_replies(first, rest, NAMES, default=SPAWN_REPLIES_DEFAULT) == expected


def test_jev_spawn_needs_a_number():
    with pytest.raises(ValueError):
        jev_spawn_replies("jev", "lots", NAMES, default=10)


class JevBot:  # the commands find chat bots by class name
    def __init__(self, spawned=None):
        self.settings = JevBotSettings(channels=frozenset({HOME}))
        self.whitelist_channels = {HOME}
        self.spawned = dict(spawned or {})
        self.spawn = AsyncMock()
        self.despawn = AsyncMock(side_effect=lambda channel: self.spawned.pop(channel.id, None) is not None)


class TempBotManager:
    def __init__(self):
        self.handle_spawn_command = AsyncMock()
        self.handle_despawn_command = AsyncMock()


def make_cog(*bots):
    discord_bot = MagicMock()
    coordinator = MagicMock()
    coordinator.bots = list(bots)
    discord_bot.get_cog.return_value = coordinator
    return TempBotCommands(discord_bot)


def make_ctx(channel_id=AWAY):
    ctx = MagicMock()
    ctx.channel.id = channel_id
    ctx.message.reference = None
    ctx.send = AsyncMock()
    ctx.reply = AsyncMock()
    return ctx


async def spawn(cog, ctx, first, rest=""):
    await TempBotCommands.spawn_command.callback(cog, ctx, first, initial_prompt=rest)


async def despawn(cog, ctx, name=None):
    await TempBotCommands.despawn_command.callback(cog, ctx, name)


async def test_spawn_jev_goes_to_jevbot_and_the_rest_to_temp_bots():
    jev, manager = JevBot(), TempBotManager()
    cog = make_cog(manager, jev)

    ctx = make_ctx()
    await spawn(cog, ctx, "jev", "10")
    jev.spawn.assert_awaited_once_with(ctx, 10)
    await spawn(cog, make_ctx(), "jev")
    assert jev.spawn.await_args.args[1] == SPAWN_REPLIES_DEFAULT
    manager.handle_spawn_command.assert_not_awaited()

    ctx = make_ctx()
    await spawn(cog, ctx, "5", "a grumpy pirate")
    manager.handle_spawn_command.assert_awaited_once()
    assert manager.handle_spawn_command.await_args.args == (ctx, 5)
    assert manager.handle_spawn_command.await_args.kwargs["initial_prompt"] == "a grumpy pirate"


@pytest.mark.parametrize(("first", "rest"), [("jev", "lots"), ("lots", "a pirate")])
async def test_spawn_with_a_bad_count_shows_usage(first, rest):
    jev, manager = JevBot(), TempBotManager()
    ctx = make_ctx()
    await spawn(make_cog(manager, jev), ctx, first, rest)
    ctx.reply.assert_awaited_once_with(cmds.SPAWN_USAGE, mention_author=False)
    jev.spawn.assert_not_awaited()
    manager.handle_spawn_command.assert_not_awaited()


async def test_despawn_jev():
    jev, manager = JevBot({AWAY: 4}), TempBotManager()
    cog = make_cog(manager, jev)
    ctx = make_ctx()
    await despawn(cog, ctx, "Jev")
    ctx.send.assert_awaited_once_with("Despawned: **Jev**")
    ctx = make_ctx()
    await despawn(cog, ctx, "jev")
    assert "isn't spawned" in ctx.send.await_args.args[0]
    ctx = make_ctx(HOME)
    await despawn(cog, ctx, "jev")
    assert "lives in this channel" in ctx.send.await_args.args[0]
    manager.handle_despawn_command.assert_not_awaited()


async def test_despawn_lists_and_clears_jev_along_with_temp_bots(monkeypatch):
    jev, manager = JevBot({AWAY: 4}), TempBotManager()
    cog = make_cog(manager, jev)
    temp_bots = []
    monkeypatch.setattr(cmds, "get_temp_bots_for_channel", lambda channel_id: temp_bots)

    ctx = make_ctx()
    await despawn(cog, ctx)  # no temp bots: just Jev's line
    assert "here for 4 more replies" in ctx.send.await_args.args[0]
    manager.handle_despawn_command.assert_not_awaited()

    temp_bots.append({"name": "Pirate"})
    ctx = make_ctx()
    await despawn(cog, ctx, "all")  # Jev, then the temp bots
    ctx.send.assert_awaited_once_with("Despawned: **Jev**")
    manager.handle_despawn_command.assert_awaited_once_with(ctx, "all")
    assert jev.spawned == {}


async def test_despawn_without_jev_visiting_is_all_temp_bots():
    jev, manager = JevBot(), TempBotManager()
    ctx = make_ctx()
    await despawn(make_cog(manager, jev), ctx, "Pirate")
    manager.handle_despawn_command.assert_awaited_once_with(ctx, "Pirate")
    ctx.send.assert_not_awaited()
