"""Tests for TypingTracker.wait_until_quiet: bots hold a post while a human is mid-message."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from hollingsbot.cogs.typing_tracker import TypingTracker

CHANNEL = 5
BOT_ID = 1


async def typing(tracker: TypingTracker, user_id: int) -> None:
    await tracker.on_typing(MagicMock(id=CHANNEL), MagicMock(id=user_id), datetime.now(timezone.utc))


async def test_returns_at_once_when_nobody_is_typing():
    tracker = TypingTracker()
    started = time.monotonic()
    await tracker.wait_until_quiet(CHANNEL, BOT_ID, max_wait=5, poll=0.01)
    assert time.monotonic() - started < 0.5


async def test_the_bots_own_typing_does_not_hold_it():
    tracker = TypingTracker()
    await typing(tracker, BOT_ID)
    started = time.monotonic()
    await tracker.wait_until_quiet(CHANNEL, BOT_ID, max_wait=5, poll=0.01)
    assert time.monotonic() - started < 0.5


async def test_waits_for_a_typing_human_but_not_forever():
    tracker = TypingTracker()
    await typing(tracker, 42)
    started = time.monotonic()
    await tracker.wait_until_quiet(CHANNEL, BOT_ID, max_wait=0.2, poll=0.05)
    assert 0.15 <= time.monotonic() - started < 1.0


async def test_stops_waiting_once_the_human_stops_typing():
    tracker = TypingTracker()
    await typing(tracker, 42)
    polls = 0
    real = tracker.is_human_typing

    def stops_after_two_polls(channel_id, bot_user_id):
        nonlocal polls
        polls += 1
        return real(channel_id, bot_user_id) and polls <= 2

    tracker.is_human_typing = stops_after_two_polls
    await tracker.wait_until_quiet(CHANNEL, BOT_ID, max_wait=5, poll=0.01)
    assert polls == 3
