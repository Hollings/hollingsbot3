"""`!spawn jev [N]` / `!spawn jev2` / `!despawn jev`: the temp bot commands hand Jev spawns to the Jevs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hollingsbot.cogs import temp_bot_commands as cmds
from hollingsbot.cogs.chat_bots.jev_bot import SPAWN_REPLIES_DEFAULT, JevBotSettings
from hollingsbot.cogs.temp_bot_commands import TempBotCommands, jev_spawn

HOME, AWAY = 1, 2


def find(word: str) -> str | None:
    return {"jev": "Jev", "jev2": "Jev2"}.get(word.strip().lower())


@pytest.mark.parametrize(
    ("first", "rest", "expected"),
    [
        ("jev", "10", ("Jev", 10)),
        ("jev", " 4 ", ("Jev", 4)),
        ("jev", "", ("Jev", SPAWN_REPLIES_DEFAULT)),
        ("jev2", "7", ("Jev2", 7)),
        ("10", "jev", ("Jev", 10)),
        ("3", "jev2", ("Jev2", 3)),
        ("10", "a grumpy pirate", None),  # a temp bot
        ("10", "jev the pirate", None),  # a temp bot that happens to mention jev
    ],
)
def test_jev_spawn(first, rest, expected):
    assert jev_spawn(first, rest, find, default=SPAWN_REPLIES_DEFAULT) == expected


def test_jev_spawn_needs_a_number():
    with pytest.raises(ValueError):
        jev_spawn("jev", "lots", find, default=10)


class JevBot:  # the commands find chat bots by class name
    def __init__(self, name="Jev", spawned=None, home=frozenset({HOME})):
        self.settings = JevBotSettings(channels=home, name=name)
        self.whitelist_channels = set(home)
        self.spawned = dict(spawned or {})
        self.spawn = AsyncMock()
        self.despawn = AsyncMock(side_effect=lambda channel: self.spawned.pop(channel.id, None) is not None)


class TempBotManager:
    def __init__(self):
        self.handle_spawn_command = AsyncMock()
        self.handle_despawn_command = AsyncMock()


def jevs(jev_spawned=None, jev2_spawned=None):
    return JevBot(spawned=jev_spawned), JevBot("Jev2", jev2_spawned, home=frozenset())


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


async def test_spawn_goes_to_the_jev_named_and_the_rest_to_temp_bots():
    (jev, jev2), manager = jevs(), TempBotManager()
    cog = make_cog(manager, jev, jev2)

    ctx = make_ctx()
    await spawn(cog, ctx, "jev", "10")
    jev.spawn.assert_awaited_once_with(ctx, 10)
    ctx = make_ctx()
    await spawn(cog, ctx, "Jev2", "4")
    jev2.spawn.assert_awaited_once_with(ctx, 4)
    await spawn(cog, make_ctx(), "jev")
    assert jev.spawn.await_args.args[1] == SPAWN_REPLIES_DEFAULT
    await spawn(cog, make_ctx(), "6", "jev2")
    assert jev2.spawn.await_args.args[1] == 6
    manager.handle_spawn_command.assert_not_awaited()

    ctx = make_ctx()
    await spawn(cog, ctx, "5", "a grumpy pirate")
    manager.handle_spawn_command.assert_awaited_once()
    assert manager.handle_spawn_command.await_args.args == (ctx, 5)
    assert manager.handle_spawn_command.await_args.kwargs["initial_prompt"] == "a grumpy pirate"


def test_plain_jev_means_the_main_jev_even_when_it_is_renamed():
    jevvy, jev2 = JevBot("Jevvy"), JevBot("Jev2", home=frozenset())
    cog = make_cog(jevvy, jev2)
    assert cog._find_jev("jev") is jevvy and cog._find_jev("JEVVY") is jevvy
    assert cog._find_jev("jev2") is jev2 and cog._find_jev("jev3") is None


@pytest.mark.parametrize(("first", "rest"), [("jev", "lots"), ("jev2", "lots"), ("lots", "a pirate")])
async def test_spawn_with_a_bad_count_shows_usage(first, rest):
    (jev, jev2), manager = jevs(), TempBotManager()
    ctx = make_ctx()
    await spawn(make_cog(manager, jev, jev2), ctx, first, rest)
    ctx.reply.assert_awaited_once_with(cmds.SPAWN_USAGE, mention_author=False)
    jev.spawn.assert_not_awaited()
    jev2.spawn.assert_not_awaited()
    manager.handle_spawn_command.assert_not_awaited()


async def test_despawn_a_jev_by_name():
    (jev, jev2), manager = jevs({AWAY: 4}, {AWAY: 2}), TempBotManager()
    cog = make_cog(manager, jev, jev2)
    ctx = make_ctx()
    await despawn(cog, ctx, "jev2")
    ctx.send.assert_awaited_once_with("Despawned: **Jev2**")
    assert (jev.spawned, jev2.spawned) == ({AWAY: 4}, {})
    ctx = make_ctx()
    await despawn(cog, ctx, "Jev2")
    assert "isn't spawned" in ctx.send.await_args.args[0]
    ctx = make_ctx(HOME)
    await despawn(cog, ctx, "jev")
    assert "lives in this channel" in ctx.send.await_args.args[0]
    manager.handle_despawn_command.assert_not_awaited()


async def test_despawn_lists_and_clears_every_jev_along_with_temp_bots(monkeypatch):
    (jev, jev2), manager = jevs({AWAY: 4}, {AWAY: 2}), TempBotManager()
    cog = make_cog(manager, jev, jev2)
    temp_bots = []
    monkeypatch.setattr(cmds, "get_temp_bots_for_channel", lambda channel_id: temp_bots)

    ctx = make_ctx()
    await despawn(cog, ctx)  # no temp bots: just the Jevs
    listed = ctx.send.await_args.args[0]
    assert "**Jev** is here for 4 more replies" in listed and "**Jev2** is here for 2 more replies" in listed
    manager.handle_despawn_command.assert_not_awaited()

    temp_bots.append({"name": "Pirate"})
    ctx = make_ctx()
    await despawn(cog, ctx, "all")  # the Jevs, then the temp bots
    ctx.send.assert_awaited_once_with("Despawned: **Jev**, **Jev2**")
    manager.handle_despawn_command.assert_awaited_once_with(ctx, "all")
    assert (jev.spawned, jev2.spawned) == ({}, {})


async def test_despawn_without_jevs_visiting_is_all_temp_bots():
    (jev, jev2), manager = jevs(), TempBotManager()
    ctx = make_ctx()
    await despawn(make_cog(manager, jev, jev2), ctx, "Pirate")
    manager.handle_despawn_command.assert_awaited_once_with(ctx, "Pirate")
    ctx.send.assert_not_awaited()
