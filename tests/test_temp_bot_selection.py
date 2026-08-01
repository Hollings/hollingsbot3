"""Tests for temp bot response selection.

Bug: `_select_responding_bot` force-picked a bot whenever every bot failed
its RESPONSE_PROBABILITY roll, so a temp bot responded to 100% of messages.
Combined with LLMs narrating departures ("*walks away*") without the literal
!despawn token, a bot could "leave" in-fiction and then be dragged back into
the conversation forever until manually despawned.

Fix: no force-pick (the coordinator falls through to the main bots when all
temp bots decline), and a direct reply to a temp bot's webhook message now
always gets a response from that bot.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import discord

from hollingsbot.cogs.chat_bots.temp_bot.manager import TempBotManager


def _manager() -> TempBotManager:
    """Bare instance - selection methods don't touch instance state."""
    return object.__new__(TempBotManager)


def _message(author_is_bot=False, webhook_id=None, reference=None):
    return SimpleNamespace(
        author=SimpleNamespace(bot=author_is_bot),
        webhook_id=webhook_id,
        reference=reference,
    )


def _reply_to_webhook(webhook_id: int):
    """A message.reference whose resolved message came from the given webhook."""
    resolved = MagicMock(spec=discord.Message)
    resolved.webhook_id = webhook_id
    return SimpleNamespace(resolved=resolved)


BOT_A = {"name": "Bot A", "webhook_id": 1}
BOT_B = {"name": "Bot B", "webhook_id": 2}


def test_no_response_when_all_bots_decline():
    """All bots failing the probability roll means no temp bot responds."""
    with patch("hollingsbot.cogs.chat_bots.temp_bot.manager.random.random", return_value=0.99):
        selected = _manager()._select_responding_bot(_message(), [BOT_A, BOT_B])
    assert selected is None


def test_willing_bot_is_selected():
    with patch("hollingsbot.cogs.chat_bots.temp_bot.manager.random.random", return_value=0.01):
        selected = _manager()._select_responding_bot(_message(), [BOT_A])
    assert selected == BOT_A


def test_speaking_bot_excluded():
    """A temp bot never responds to its own message."""
    msg = _message(author_is_bot=True, webhook_id=BOT_A["webhook_id"])
    with patch("hollingsbot.cogs.chat_bots.temp_bot.manager.random.random", return_value=0.01):
        selected = _manager()._select_responding_bot(msg, [BOT_A])
    assert selected is None


def test_direct_reply_always_gets_response():
    """Replying to a temp bot's message bypasses the probability roll."""
    msg = _message(reference=_reply_to_webhook(BOT_B["webhook_id"]))
    with patch("hollingsbot.cogs.chat_bots.temp_bot.manager.random.random", return_value=0.99):
        selected = _manager()._select_responding_bot(msg, [BOT_A, BOT_B])
    assert selected == BOT_B


def test_reply_to_non_temp_bot_message_falls_through_to_roll():
    """A reply to some unrelated message still uses the probability roll."""
    msg = _message(reference=_reply_to_webhook(12345))
    with patch("hollingsbot.cogs.chat_bots.temp_bot.manager.random.random", return_value=0.99):
        selected = _manager()._select_responding_bot(msg, [BOT_A, BOT_B])
    assert selected is None
