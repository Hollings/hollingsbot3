"""A reply reservation is refunded exactly once, only before delivery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hollingsbot.cogs.chat_bots.temp_bot import manager as temp
from hollingsbot.cogs.conversation import ConversationTurn


def manager():
    bot = object.__new__(temp.TempBotManager)
    bot.bot = SimpleNamespace(fetch_webhook=AsyncMock())
    bot._active_generations = {}
    bot._generating_webhooks = {99}
    bot._cancellation_counts = {}
    bot.debug_log_store = Mock()
    bot._wait_for_typing_to_clear = AsyncMock()
    return bot


@pytest.mark.parametrize("delivered", [False, True])
async def test_cancel_refunds_only_undelivered_reply(monkeypatch, delivered):
    refund = Mock()
    monkeypatch.setattr(temp, "increment_temp_bot_replies", refund)
    bot = manager()
    job = temp.GenerationJob(webhook_id=99)
    bot._active_generations[11] = job
    if delivered:
        bot._generate_response_with_job = AsyncMock(return_value=("hello", {}, []))
        bot.bot.fetch_webhook.return_value.send = AsyncMock(return_value=SimpleNamespace(id=123))
        bot._cleanup_temp_bot = AsyncMock(side_effect=asyncio.CancelledError)
    else:
        bot._generate_response_with_job = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await bot._generate_and_send(11, 99, "Test", [], job, should_cleanup=delivered)
    assert refund.call_count == (0 if delivered else 1)
    assert not bot._generating_webhooks
    assert not bot._active_generations


async def test_cancel_during_preparation_refunds_and_propagates(monkeypatch):
    selected = {"webhook_id": 99, "name": "Test", "spawn_prompt": "test"}
    monkeypatch.setattr(temp, "get_temp_bots_for_channel", Mock(return_value=[selected]))
    monkeypatch.setattr(temp, "decrement_temp_bot_replies", Mock(return_value=(2, False)))
    refund = Mock()
    monkeypatch.setattr(temp, "increment_temp_bot_replies", refund)
    bot = manager()
    bot._select_responding_bot = Mock(return_value=selected)
    bot._build_temp_bot_system_prompt = Mock(return_value="system")
    bot._build_conversation_payload = Mock(return_value=[])
    bot._cancel_generation = AsyncMock(side_effect=asyncio.CancelledError)
    message = SimpleNamespace(id=123, author="user", content="hello", channel=SimpleNamespace(id=11))
    with pytest.raises(asyncio.CancelledError):
        await bot.receive_message(message, [ConversationTurn(role="user", content="hello", message_id=123)])
    refund.assert_called_once_with(99)
    assert not bot._generating_webhooks


async def test_cancel_before_generation_starts_refunds_and_removes_job(monkeypatch):
    selected = {"webhook_id": 99, "name": "Test", "spawn_prompt": "test"}
    monkeypatch.setattr(temp, "get_temp_bots_for_channel", Mock(return_value=[selected]))
    monkeypatch.setattr(temp, "decrement_temp_bot_replies", Mock(return_value=(2, False)))
    refund = Mock()
    monkeypatch.setattr(temp, "increment_temp_bot_replies", refund)
    bot = manager()
    bot._select_responding_bot = Mock(return_value=selected)
    bot._build_temp_bot_system_prompt = Mock(return_value="system")
    bot._build_conversation_payload = Mock(return_value=[])
    bot._cancel_generation = AsyncMock(return_value=False)

    def cancelled_task(coroutine):
        task = asyncio.create_task(coroutine)
        task.cancel()
        return task

    bot.bot.loop = SimpleNamespace(create_task=cancelled_task)
    message = SimpleNamespace(id=123, author="user", content="hello", channel=SimpleNamespace(id=11))
    with pytest.raises(asyncio.CancelledError):
        await bot.receive_message(message, [ConversationTurn(role="user", content="hello", message_id=123)])
    refund.assert_called_once_with(99)
    assert not bot._generating_webhooks
    assert not bot._active_generations
