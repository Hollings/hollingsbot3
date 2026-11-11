import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from hollingsbot import image_prompt_rewriter as rewrite, prompt_db
from hollingsbot.cogs import image_gen_cog as images
from test_image_gen_workflow import message


@pytest.mark.parametrize("response", ['["directed cat", "directed dog"]', "[]", '[""]', "not json", "[42, 42]"])
async def test_validates_complete_batch_without_losing_originals(monkeypatch, response):
    generator = Mock(generate=AsyncMock(return_value=response))
    monkeypatch.setattr(rewrite, "get_text_generator", Mock(return_value=generator))
    originals = ["cat", "dog"]
    actual = await rewrite.rewrite_image_prompts(originals)
    assert actual == (["directed cat", "directed dog"] if response.startswith('["directed') else originals)
    messages = generator.generate.call_args.args[0]
    assert json.loads(messages[1]["content"]) == originals
    assert messages[0]["role"] == "system"


async def test_timeout_uses_original_prompt(monkeypatch):
    async def hang(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(rewrite, "get_text_generator", Mock(return_value=Mock(generate=hang)))
    monkeypatch.setattr(rewrite, "REWRITE_TIMEOUT", 0.01)
    assert await rewrite.rewrite_image_prompts(["cat"]) == ["cat"]


async def test_cancellation_during_rewrite_releases_hold_and_replay_does_not_rewrite(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_db, "DB_PATH", tmp_path / "bot.db")
    cog = images.ImageGenCog(
        Mock(), config={"$$$$": {"api": "replicate", "model": "fake", "rewrite_prompt": True, "price_per_image": 0.2}}
    )
    cog._react = AsyncMock()
    cog._execute_generation_tasks = AsyncMock()
    started = asyncio.Event()

    async def hang(*args):
        started.set()
        await asyncio.Event().wait()

    generator = Mock(generate=AsyncMock(side_effect=hang))
    monkeypatch.setattr(rewrite, "get_text_generator", Mock(return_value=generator))
    msg = message()
    task = asyncio.create_task(cog._handle_generation(msg, "cat", cog._prefix_map["$$$$"]))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert cog._cost_tracker.get_user_status(1)["reserved_for_images"] == pytest.approx(0.2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert cog._cost_tracker.get_user_status(1)["reserved_for_images"] == 0
    assert cog._cost_tracker.get_history(1) == []
    await cog._handle_generation(msg, "cat", cog._prefix_map["$$$$"])
    generator.generate.assert_awaited_once()
    cog._execute_generation_tasks.assert_not_awaited()


def test_rewrite_configuration_is_not_sent_to_image_provider():
    spec = images.spec_from_dict({"api": "replicate", "model": "fake", "rewrite_prompt": True})
    assert spec.rewrite_prompt
    assert not spec.model_options
    assert not images.spec_from_dict({"api": "replicate", "model": "fake"}).rewrite_prompt
    with pytest.raises(ValueError):
        images.spec_from_dict({"api": "replicate", "model": "fake", "rewrite_prompt": "false"})
