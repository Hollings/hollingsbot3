"""Public reset requests serialize and cannot bypass cooldown by restarting."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from hollingsbot import prompt_db
from hollingsbot.cogs import admin


async def test_reset_in_flight_and_cooldown_survive_a_new_cog(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_db, "DB_PATH", tmp_path / "bot.db")
    first = admin.Admin(Mock())
    entered, finish = asyncio.Event(), asyncio.Event()

    async def restart(_ctx):
        entered.set()
        await finish.wait()

    first._restart_containers = AsyncMock(side_effect=restart)
    ctx1, ctx2 = SimpleNamespace(send=AsyncMock()), SimpleNamespace(send=AsyncMock())
    pending = asyncio.create_task(first.reset.callback(first, ctx1))
    await asyncio.wait_for(entered.wait(), timeout=5)
    try:
        await first.reset.callback(first, ctx2)
        ctx2.send.assert_awaited_once_with("A reset is already in progress.")
    finally:
        finish.set()
        await pending
    first._restart_containers.assert_awaited_once()

    reloaded = admin.Admin(Mock())
    reloaded._restart_containers = AsyncMock()
    ctx3 = SimpleNamespace(send=AsyncMock())
    await reloaded.reset.callback(reloaded, ctx3)
    assert ctx3.send.await_args.args[0].startswith("Reset is on cooldown.")
    reloaded._restart_containers.assert_not_called()


async def test_reset_discovers_peers_in_own_project_and_requests_supervisor_restart(monkeypatch):
    monkeypatch.setattr(admin.shutil, "which", lambda _tool: "docker")
    monkeypatch.setattr(admin.socket, "gethostname", lambda: "bot-container")
    fake = object.__new__(admin.Admin)
    fake.bot = SimpleNamespace(request_restart=AsyncMock())
    fake._capture_subprocess = AsyncMock(
        side_effect=[
            json.dumps({"com.docker.compose.project": "hollingsbot3", "com.docker.compose.service": "bot"}),
            "redis-container",
            "text-container",
            "image-container",
        ]
    )
    fake._run_subprocess = AsyncMock(return_value=True)
    assert await fake._restart_docker_peers()
    calls = [call.args[0] for call in fake._capture_subprocess.await_args_list]
    assert calls[0][-1] == "bot-container"
    assert all("label=com.docker.compose.project=hollingsbot3" in call for call in calls[1:])
    assert [call.args[0][-1] for call in fake._run_subprocess.await_args_list] == [
        "redis-container",
        "text-container",
        "image-container",
    ]
    await fake._request_bot_restart(SimpleNamespace(send=AsyncMock()), peers_restarted=True)
    fake.bot.request_restart.assert_awaited_once()
