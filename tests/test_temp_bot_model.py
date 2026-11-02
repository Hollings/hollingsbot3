"""Temp bot replies go to the configured chat model, Anthropic by default."""

from unittest.mock import AsyncMock

import pytest

from hollingsbot.cogs.chat_bots.temp_bot import manager as temp


def manager():
    return object.__new__(temp.TempBotManager)


@pytest.fixture
def run_task(monkeypatch):
    mock = AsyncMock(return_value={"text": "hi", "debug": {}, "conversation": []})
    monkeypatch.setattr(temp, "run_task", mock)
    monkeypatch.setattr(temp, "generate_llm_chat_response", "task")
    return mock


async def test_default_model_is_anthropic(monkeypatch, run_task):
    monkeypatch.delenv("TEMP_BOT_PROVIDER", raising=False)
    monkeypatch.delenv("TEMP_BOT_MODEL", raising=False)
    bot = manager()
    bot.text_timeout = 5
    conversation = [{"role": "user", "text": "yo", "images": []}]
    await bot._generate_response_with_job(conversation, temp.GenerationJob())
    task, args = run_task.call_args.args
    assert task == "task"
    assert args == (temp.DEFAULT_TEMP_BOT_PROVIDER, temp.DEFAULT_TEMP_BOT_MODEL, conversation)
    assert temp.DEFAULT_TEMP_BOT_PROVIDER == "anthropic"


async def test_env_overrides_model(monkeypatch, run_task):
    monkeypatch.setenv("TEMP_BOT_PROVIDER", "openrouter")
    monkeypatch.setenv("TEMP_BOT_MODEL", "some/model")
    bot = manager()
    bot.text_timeout = 5
    await bot._generate_response_with_job([{"role": "user", "text": "yo", "images": []}], temp.GenerationJob())
    assert run_task.call_args.args[1][:2] == ("openrouter", "some/model")
