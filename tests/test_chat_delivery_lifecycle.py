import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hollingsbot.cogs import chat_utils
from hollingsbot.cogs.chat_bots.gemini_bot import GeminiBot
from hollingsbot.cogs.chat_bots.grok_bot import GrokBot
from hollingsbot.cogs.chat_bots.llama_bot import LlamaBot
from hollingsbot.cogs.chat_coordinator import ChatCoordinator


def bot_shell(bot_type):
    instance = object.__new__(bot_type)
    instance.channel_webhooks = {}
    instance.channel_names = {}
    instance._store_response_log = Mock()
    instance._wait_for_typing_to_clear = AsyncMock()
    return instance


@pytest.mark.parametrize("bot_type", [GeminiBot, GrokBot, LlamaBot])
async def test_later_send_failure_preserves_confirmed_partial_response(bot_type):
    instance = bot_shell(bot_type)
    first = SimpleNamespace(id=123)
    channel = SimpleNamespace(id=1, send=AsyncMock(side_effect=[first, RuntimeError("send failed")]))
    text = "a" * 1900 + "\n" + "b" * 1900
    args = (channel, text) if bot_type is LlamaBot else (channel, text, [])
    with pytest.raises(chat_utils.PartialDeliveryError) as error:
        await instance._send_to_discord(*args)
    assert error.value.messages == [first]
    assert error.value.text == chat_utils.chunk_message(text)[0]


@pytest.mark.parametrize("bot_type", [GeminiBot, GrokBot])
async def test_partial_delivery_records_visible_text_and_stops_fallback(bot_type):
    instance = bot_shell(bot_type)
    first = SimpleNamespace(id=123, webhook_id=None)
    channel = SimpleNamespace(id=1, send=AsyncMock(side_effect=[first, RuntimeError("send failed")]))
    message = SimpleNamespace(id=50, channel=channel)
    text = "a" * 1900 + "\n" + "b" * 1900
    instance._run_generation = AsyncMock(return_value=(text, {}, []))

    async def receive(_message, history):
        return await instance._generate_and_send(_message, history, "fake", "fake", None)

    instance.receive_message = receive
    fallback = SimpleNamespace(receive_message=AsyncMock())
    coordinator = SimpleNamespace(
        _bots_in_turn_order=lambda channel_id: [instance, fallback],
        _add_response_to_history=AsyncMock(),
    )
    await ChatCoordinator._try_bot_responses(coordinator, 1, message, [])
    fallback.receive_message.assert_not_awaited()
    recorded = coordinator._add_response_to_history.await_args.args
    assert recorded[1:3] == (123, chat_utils.chunk_message(text)[0])


@pytest.mark.parametrize("bot_type", [GeminiBot, GrokBot])
async def test_cancel_after_provider_finishes_still_prevents_delivery(bot_type):
    instance = bot_shell(bot_type)
    instance.bot = SimpleNamespace(loop=asyncio.get_running_loop())
    instance._active_generations = {}
    instance._should_respond = AsyncMock(return_value=True)
    instance._get_model_for_user = Mock(return_value=("fake", "fake"))
    instance._extract_current_turn = Mock(return_value=SimpleNamespace(text="hello"))
    instance._translate_history = Mock(return_value=[])
    instance._get_user_system_prompt = Mock(return_value=None)
    instance._build_full_system_prompt = Mock(return_value="system")
    instance._build_conversation_payload = Mock(return_value=[])
    instance._run_generation = AsyncMock(return_value=("reply", {}, []))
    waiting = asyncio.Event()

    async def typing_wait(channel_id):
        waiting.set()
        await asyncio.Event().wait()

    instance._wait_for_typing_to_clear = typing_wait
    channel = SimpleNamespace(id=1, send=AsyncMock())
    message = SimpleNamespace(id=50, channel=channel, author=SimpleNamespace(id=2), guild=None, content="hello")
    task = asyncio.create_task(instance.receive_message(message, []))
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        assert 1 in instance._active_generations
        await instance._cancel_generation(1)
        with pytest.raises(asyncio.CancelledError):
            await task
        channel.send.assert_not_awaited()
        assert instance._active_generations == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
