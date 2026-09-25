"""Tests for LiveMessage: send once, then throttled edits, final text always lands."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import discord

from hollingsbot.utils.live_message import LiveMessage


class Recorder:
    def __init__(self):
        self.sent: list[str] = []
        self.edits: list[str] = []
        self.message = MagicMock(spec=discord.Message)

    async def send(self, text):
        self.sent.append(text)
        return self.message

    async def edit(self, message, text):
        assert message is self.message
        self.edits.append(text)


async def test_first_update_sends_later_ones_edit_and_finish_lands_final_text():
    r = Recorder()
    live = LiveMessage(r.send, r.edit, interval=0.05)
    await live.update("i")
    assert r.sent == ["i"]
    await live.update("i like")
    await live.update("i like pizza")
    await asyncio.sleep(0.12)
    assert r.edits and r.edits[-1] == "i like pizza"
    assert await live.finish("i like pizza.") is r.message
    assert r.edits[-1] == "i like pizza."
    assert r.sent == ["i"]


async def test_edits_are_throttled():
    r = Recorder()
    live = LiveMessage(r.send, r.edit, interval=10)
    await live.update("a")
    for text in ("a b", "a b c", "a b c d"):
        await live.update(text)
    await asyncio.sleep(0.05)
    assert r.edits == []  # still inside the interval
    await live.finish("a b c d e")
    assert r.edits == ["a b c d e"]


async def test_finish_without_updates_sends_once():
    r = Recorder()
    live = LiveMessage(r.send, r.edit)
    await live.finish("paris")
    assert r.sent == ["paris"] and r.edits == []


async def test_finish_with_nothing_to_say_sends_nothing():
    r = Recorder()
    assert await LiveMessage(r.send, r.edit).finish("") is None
    assert r.sent == []


async def test_failed_intermediate_edit_is_not_fatal():
    r = Recorder()

    async def flaky_edit(message, text):
        if text == "a b":
            raise discord.HTTPException(MagicMock(status=500), "boom")
        r.edits.append(text)

    live = LiveMessage(r.send, flaky_edit, interval=0.01)
    await live.update("a")
    await live.update("a b")
    await asyncio.sleep(0.05)
    await live.finish("a b c")
    assert r.edits[-1] == "a b c"
