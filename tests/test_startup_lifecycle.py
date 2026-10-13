"""Startup retries and scheduled restarts release loop-owned resources."""

import asyncio
import importlib
from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.fixture
def entrypoint(monkeypatch):
    with patch("dotenv.load_dotenv"):
        module = importlib.import_module("hollingsbot.__main__")
    monkeypatch.setattr(module, "_restart_requested", False)
    monkeypatch.setattr(module, "bot", Mock())
    monkeypatch.setattr(module, "restart_task", Mock())
    module.restart_task.get_task.return_value = None
    monkeypatch.setattr(module, "close_clients", AsyncMock())
    return module


async def test_retry_reopens_bot_and_stays_on_one_loop(entrypoint, monkeypatch):
    loops = []

    async def startup():
        loops.append(asyncio.get_running_loop())
        if len(loops) == 1:
            raise OSError("connection reset")

    monkeypatch.setattr(entrypoint, "main", startup)
    with patch.object(entrypoint.asyncio, "sleep", new_callable=AsyncMock) as sleep:
        await entrypoint.run_bot()
    assert loops[0] is loops[1]
    entrypoint.bot.clear.assert_called_once()
    sleep.assert_awaited_once_with(5)
    assert entrypoint.close_clients.await_count == 2
    assert entrypoint.restart_task.cancel.call_count == 2


async def test_cancelled_startup_cleans_up_without_retry(entrypoint, monkeypatch):
    monkeypatch.setattr(entrypoint, "main", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await entrypoint.run_bot()
    entrypoint.close_clients.assert_awaited_once()
    entrypoint.bot.clear.assert_not_called()


async def test_scheduled_restart_exits_after_cleanup(entrypoint, monkeypatch):
    events = []

    async def close():
        events.append("close")

    monkeypatch.setattr(entrypoint, "main", AsyncMock())
    monkeypatch.setattr(entrypoint, "_restart_requested", True)
    monkeypatch.setattr(entrypoint, "close_clients", close)
    monkeypatch.setattr(entrypoint, "_exit_for_restart", lambda: events.append("exit"))
    await entrypoint.run_bot()
    assert events == ["close", "exit"]
