"""Regression tests for the temp bot cleanup race condition.

Bug: when `_cleanup_temp_bot` was awaiting slow steps (especially summary
generation), a new message could trigger `_cancel_generation`, which cancelled
the cleanup task AND called `increment_temp_bot_replies` as a refund. The
refund revived the bot (replies_remaining back to 1, is_active still 1), so
it would respond again, "leave" again, get refunded again - forever.

Fix: `_cleanup_temp_bot` now calls `delete_temp_bot(webhook_id)` BEFORE any
await. `increment_temp_bot_replies` has `WHERE is_active = 1` so refunds
become no-ops once the bot is inactive.

These tests verify both the DB-level guard and the cleanup ordering.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def temp_bot_db(tmp_path: Path, monkeypatch):
    """Point the temp bot DB module at an isolated sqlite file.

    Both `prompt_db` and `temp_bot_db` cache `DB_PATH` at import time, so we
    have to patch both bindings.
    """
    db_file = tmp_path / "test.db"

    from hollingsbot import prompt_db, temp_bot_db

    monkeypatch.setattr(prompt_db, "DB_PATH", db_file)
    monkeypatch.setattr(temp_bot_db, "DB_PATH", db_file)

    prompt_db.init_db()
    return db_file


def _make_bot(temp_bot_db, **overrides) -> int:
    """Create a temp bot row and return its webhook_id."""
    from hollingsbot.temp_bot_db import create_temp_bot

    defaults = {
        "channel_id": 111,
        "webhook_id": 999,
        "name": "Veiled Cipher",
        "avatar_url": None,
        "spawn_prompt": "test bot",
        "replies_remaining": 1,
        "spawn_message_id": 222,
        "avatar_bytes": None,
    }
    defaults.update(overrides)
    create_temp_bot(**defaults)
    return defaults["webhook_id"]


# ---------------------------------------------------------------------------
# DB-level: the actual mechanism that prevents the bug
# ---------------------------------------------------------------------------


def test_increment_after_delete_is_noop(temp_bot_db):
    """Once a bot is marked inactive, refund attempts must not revive it.

    This is the load-bearing invariant: `increment_temp_bot_replies` has
    `WHERE is_active = 1`, so a cancellation that fires after `delete_temp_bot`
    cannot bring the bot back from the dead.
    """
    from hollingsbot.temp_bot_db import (
        delete_temp_bot,
        get_temp_bot_by_webhook_id,
        increment_temp_bot_replies,
    )

    webhook_id = _make_bot(temp_bot_db, replies_remaining=0)

    delete_temp_bot(webhook_id)

    new_count = increment_temp_bot_replies(webhook_id)

    assert new_count == -1, "increment must report 'not found / inactive'"

    bot = get_temp_bot_by_webhook_id(webhook_id)
    assert bot is not None
    assert bot["replies_remaining"] == 0, "replies_remaining must not have ticked up"


def test_decrement_after_delete_is_noop(temp_bot_db):
    """Symmetric guard: decrement also requires is_active = 1."""
    from hollingsbot.temp_bot_db import (
        decrement_temp_bot_replies,
        delete_temp_bot,
    )

    webhook_id = _make_bot(temp_bot_db, replies_remaining=5)
    delete_temp_bot(webhook_id)

    remaining, should_cleanup = decrement_temp_bot_replies(webhook_id)

    assert remaining == -1
    assert should_cleanup is False


def test_inactive_bot_excluded_from_active_listings(temp_bot_db):
    """`get_temp_bots_for_channel` and `get_depleted_temp_bots` must not
    return inactive bots. After cleanup, the bot is invisible to the
    response-selection and periodic-cleanup paths.
    """
    from hollingsbot.temp_bot_db import (
        delete_temp_bot,
        get_depleted_temp_bots,
        get_temp_bots_for_channel,
    )

    webhook_id = _make_bot(temp_bot_db, channel_id=111, replies_remaining=0)

    assert len(get_temp_bots_for_channel(111)) == 1
    assert len(get_depleted_temp_bots()) == 1

    delete_temp_bot(webhook_id)

    assert get_temp_bots_for_channel(111) == []
    assert get_depleted_temp_bots() == []


# ---------------------------------------------------------------------------
# Cleanup ordering: delete_temp_bot must run before any await
# ---------------------------------------------------------------------------


def _make_manager_skeleton():
    """Build just enough of a TempBotManager to call `_cleanup_temp_bot`.

    The real `__init__` starts a discord.py tasks.loop and reads a system
    prompt file. We bypass it entirely and attach only the attributes the
    method touches.
    """
    from hollingsbot.cogs.chat_bots.temp_bot.manager import TempBotManager

    manager = TempBotManager.__new__(TempBotManager)
    manager.bot = MagicMock()
    manager.bot.fetch_webhook = AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_cleanup_marks_inactive_before_any_await(temp_bot_db):
    """Even if every await inside `_cleanup_temp_bot` is cancelled, the bot
    must already be marked inactive by the time control yields.

    We simulate a worst-case mid-cleanup cancellation by making
    `bot.fetch_webhook` raise CancelledError - this is the very first await
    in the method. If `delete_temp_bot` weren't called first, the bot would
    still be active when the cancellation propagates.
    """
    from hollingsbot.temp_bot_db import get_temp_bot_by_webhook_id

    webhook_id = _make_bot(temp_bot_db, replies_remaining=0)
    manager = _make_manager_skeleton()
    manager.bot.fetch_webhook.side_effect = asyncio.CancelledError()

    # Patch the summary helpers to network-free no-ops; we only care about
    # the inactive-marking ordering here.
    with (
        patch(
            "hollingsbot.cogs.chat_bots.temp_bot.manager.generate_conversation_summary",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "hollingsbot.cogs.chat_bots.temp_bot.manager.save_conversation_summary",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(asyncio.CancelledError):
            await manager._cleanup_temp_bot(webhook_id, "Veiled Cipher", send_depletion_message=True)

    bot = get_temp_bot_by_webhook_id(webhook_id)
    assert bot is not None
    # The DB query in get_temp_bot_by_webhook_id doesn't return is_active,
    # so check it directly.
    import sqlite3

    with sqlite3.connect(temp_bot_db) as conn:
        cur = conn.execute("SELECT is_active FROM temp_bots WHERE webhook_id = ?", (webhook_id,))
        is_active = cur.fetchone()[0]
    assert is_active == 0, "bot must be marked inactive even when cleanup is cancelled"


@pytest.mark.asyncio
async def test_cleanup_survives_refund_attempt_during_cleanup(temp_bot_db):
    """End-to-end simulation of the original bug: cleanup runs, a parallel
    `_cancel_generation` fires `increment_temp_bot_replies` mid-flight, and
    the bot must still end up dead.
    """
    from hollingsbot.temp_bot_db import (
        get_temp_bot_by_webhook_id,
        increment_temp_bot_replies,
    )

    webhook_id = _make_bot(temp_bot_db, replies_remaining=0)
    manager = _make_manager_skeleton()

    refund_results: list[int] = []

    async def slow_fetch_webhook(_wid):
        # Simulate a parallel `_cancel_generation` racing with the cleanup.
        # It tries to refund the reply - but because we already called
        # delete_temp_bot, the refund must be a no-op.
        refund_results.append(increment_temp_bot_replies(webhook_id))
        wh = MagicMock()
        wh.send = AsyncMock()
        wh.delete = AsyncMock()
        return wh

    manager.bot.fetch_webhook.side_effect = slow_fetch_webhook

    with (
        patch(
            "hollingsbot.cogs.chat_bots.temp_bot.manager.generate_conversation_summary",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "hollingsbot.cogs.chat_bots.temp_bot.manager.save_conversation_summary",
            new=AsyncMock(),
        ),
    ):
        await manager._cleanup_temp_bot(webhook_id, "Veiled Cipher", send_depletion_message=True)

    assert refund_results == [-1], "refund during cleanup must be a no-op"

    bot = get_temp_bot_by_webhook_id(webhook_id)
    assert bot is not None
    assert bot["replies_remaining"] == 0, "bot must not have been revived"

    import sqlite3

    with sqlite3.connect(temp_bot_db) as conn:
        cur = conn.execute("SELECT is_active FROM temp_bots WHERE webhook_id = ?", (webhook_id,))
        is_active = cur.fetchone()[0]
    assert is_active == 0
