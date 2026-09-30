"""The spawn path should start promptly and retain its generated avatar."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import replicate

from hollingsbot import prompt_db, temp_bot_db
from hollingsbot.cogs.chat_bots.temp_bot import identity, manager
from hollingsbot.image_generators import generate_avatar
from hollingsbot.image_generators.replicate_api import ReplicateImageGenerator


async def test_identity_uses_fast_direct_model_and_keeps_avatar_on_fallback(monkeypatch):
    generator = MagicMock()
    generator.generate = AsyncMock(return_value="NAME: Captain Fern\nAVATAR: a green portrait")
    factory = MagicMock(return_value=generator)
    monkeypatch.setattr(identity, "get_text_generator", factory)

    assert await identity.generate_bot_identity("a cheerful captain") == ("Captain Fern", "a green portrait")
    factory.assert_called_once_with("anthropic", "claude-haiku-4-5")

    generator.generate.side_effect = RuntimeError("service unavailable")
    name, avatar_prompt = await identity.generate_bot_identity("a cheerful captain")
    assert name
    assert "a cheerful captain" in avatar_prompt


async def test_avatar_requests_png_and_handles_missing_credentials(monkeypatch):
    instances = []

    class Generator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.generate = AsyncMock(return_value=b"png")
            self.aclose = AsyncMock()
            instances.append(self)

    monkeypatch.setattr("hollingsbot.image_generators.ReplicateImageGenerator", Generator)
    assert await generate_avatar("a portrait") == b"png"
    assert instances[0].kwargs["model_options"] == {"output_format": "png"}
    instances[0].aclose.assert_awaited_once()

    def missing_token(**kwargs):
        raise RuntimeError("REPLICATE_API_TOKEN is required")

    monkeypatch.setattr("hollingsbot.image_generators.ReplicateImageGenerator", missing_token)
    assert await generate_avatar("a portrait") is None


async def test_replicate_file_output_reads_data_uri_without_http():
    output = replicate.helpers.FileOutput("data:image/png;base64,YWJj", MagicMock())
    generator = ReplicateImageGenerator(api_token="test-token")
    assert await generator._normalise_output([output]) == b"abc"
    assert await generator._collect_all([output]) == [b"abc"]


async def test_avatar_model_sends_png_option_to_replicate():
    generator = ReplicateImageGenerator(api_token="test-token", model_options={"output_format": "png"})
    generator._client.async_run = AsyncMock(return_value=[b"image"])
    assert await generator.generate("a portrait") == b"image"
    assert generator._client.async_run.await_args.kwargs["input"]["output_format"] == "png"


async def test_avatar_is_saved_after_webhook_edit(monkeypatch):
    bot_manager = object.__new__(manager.TempBotManager)
    webhook = MagicMock(id=123)
    webhook.edit = AsyncMock()
    monkeypatch.setattr(manager, "generate_avatar", AsyncMock(return_value=b"image"))
    save_avatar = MagicMock()
    monkeypatch.setattr(manager, "set_temp_bot_avatar", save_avatar)

    await bot_manager._apply_spawn_avatar(webhook, "a portrait", "Captain Fern")
    webhook.edit.assert_awaited_once_with(avatar=b"image")
    save_avatar.assert_called_once_with(123, b"image")


def test_avatar_is_available_when_bot_is_recalled(tmp_path, monkeypatch):
    db_file = tmp_path / "temp-bots.db"
    monkeypatch.setattr(prompt_db, "DB_PATH", db_file)
    monkeypatch.setattr(temp_bot_db, "DB_PATH", db_file)
    temp_bot_db.create_temp_bot(1, 123, "Captain Fern", None, "a captain", 4)

    temp_bot_db.set_temp_bot_avatar(123, b"image")

    assert temp_bot_db.get_temp_bot_by_webhook_id(123)["avatar_bytes"] == b"image"


async def test_first_reply_waits_for_avatar_while_text_generates(monkeypatch):
    bot_manager = object.__new__(manager.TempBotManager)
    bot_manager.bot = MagicMock()
    webhook = MagicMock()
    webhook.send = AsyncMock()
    bot_manager.bot.fetch_webhook = AsyncMock(return_value=webhook)
    bot_manager._translate_history = MagicMock(return_value=[])
    bot_manager._build_temp_bot_system_prompt = MagicMock(return_value="system")
    bot_manager._build_conversation_payload = MagicMock(return_value=[])

    text_started = asyncio.Event()
    allow_text = asyncio.Event()
    avatar_done = asyncio.Event()

    async def make_reply(_conversation):
        text_started.set()
        await allow_text.wait()
        return "Hello!"

    async def make_avatar():
        await avatar_done.wait()

    bot_manager._generate_response = make_reply
    monkeypatch.setattr(manager, "decrement_temp_bot_replies", lambda _id: (2, False))
    avatar_task = asyncio.create_task(make_avatar())
    response_task = asyncio.create_task(
        bot_manager._send_initial_response(MagicMock(), 123, "Captain Fern", "a captain", 456, avatar_task=avatar_task)
    )

    await asyncio.wait_for(text_started.wait(), 1)
    allow_text.set()
    await asyncio.sleep(0)
    webhook.send.assert_not_awaited()
    avatar_done.set()
    await response_task
    webhook.send.assert_awaited_once_with("Hello!", username="Captain Fern", wait=True)
