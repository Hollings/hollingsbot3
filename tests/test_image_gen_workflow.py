"""Exercise configuration, delivery and accounting without external services."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hollingsbot import prompt_db
from hollingsbot.cogs import image_gen_cog as images
from hollingsbot.cogs.admin import Admin
from hollingsbot.cogs.credits_cog import CreditsCog


@pytest.fixture
def cog(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_db, "DB_PATH", tmp_path / "bot.db")
    return images.ImageGenCog(Mock(), config={"!": {"api": "replicate", "model": "test"}})


def test_listing_uses_default_price(cog):
    assert "$0.03" in cog._format_model_listing()


@pytest.mark.parametrize(
    "bad", [[], {"daily_free_budget": -1}, {"!": {"api": "replicate"}}, {"default_price_per_image": "free"}]
)
def test_invalid_reload_preserves_all_previous_config(cog, tmp_path, bad):
    original = cog._prefix_map
    cog._cfg_path = tmp_path / "config.json"
    cog._cfg_path.write_text(json.dumps(bad))
    cog._reload_config()
    assert cog._prefix_map is original
    assert cog._default_price == 0.03
    assert cog._daily_free_budget == 0.5
    assert cog._cost_tracker.daily_free_budget == 0.5
    assert cog._cfg_mtime == 0


def test_accounting_cogs_share_database_and_budget(cog):
    bot = Mock()
    bot.get_cog.return_value = cog
    admin, credits = Admin(bot), CreditsCog(bot)
    assert admin._cost_tracker is cog._cost_tracker
    assert credits._cost_tracker is cog._cost_tracker
    assert cog._cost_tracker.db_path == str(prompt_db.DB_PATH)
    cog._cost_tracker.daily_free_budget = 1.25
    assert credits._cost_tracker.daily_free_budget == 1.25


def message(user=1):
    return SimpleNamespace(
        id=123,
        author=SimpleNamespace(id=user),
        guild=None,
        channel=SimpleNamespace(send=AsyncMock()),
        reply=AsyncMock(),
    )


async def test_partial_delivery_counts_only_successes(cog):
    cog._prepare_discord_files = Mock(return_value=[Mock()])
    cog._send_error_message = AsyncMock()
    msg = message()
    delivered = await cog._process_and_send_results(
        msg,
        [("cat", [b"image"]), RuntimeError("failed")],
        ["cat", "dog"],
        cog._prefix_map["!"],
        42,
        False,
        msg,
    )
    assert delivered == 1
    msg.channel.send.assert_awaited_once()


async def test_batch_charges_delivered_variants_without_downloading_edit_images(cog, monkeypatch):
    cog._react = AsyncMock()
    cog._get_thematic_emoji = AsyncMock(return_value=images.THINKING)
    cog._collect_images_for_editing = AsyncMock()
    cog._execute_generation_tasks = AsyncMock(return_value=[("cat", [b"img"]), RuntimeError("failed")])
    cog._process_and_send_results = AsyncMock(return_value=1)
    cog._cost_tracker = Mock()
    cog._cost_tracker.can_afford.return_value = (True, "")
    monkeypatch.setattr(images, "bulk_add_prompts", Mock(return_value=[1, 2]))
    await cog._handle_generation(message(), "<cat,dog>", cog._prefix_map["!"])
    cog._cost_tracker.can_afford.assert_called_once_with(1, 0.06)
    cog._cost_tracker.deduct_cost.assert_called_once_with(1, 0.03)
    cog._collect_images_for_editing.assert_not_awaited()


async def test_same_user_is_serialized_but_other_users_can_generate(cog):
    first_started, release = asyncio.Event(), asyncio.Event()
    entered = []

    async def generate(msg, prompt, spec):
        entered.append(prompt)
        if prompt == "first":
            first_started.set()
            await release.wait()

    cog._generate_for_user = generate
    spec = cog._prefix_map["!"]
    first = asyncio.create_task(cog._handle_generation(message(), "first", spec))
    await first_started.wait()
    second = asyncio.create_task(cog._handle_generation(message(), "second", spec))
    await cog._handle_generation(message(user=2), "other", spec)
    await asyncio.sleep(0)
    assert entered == ["first", "other"]
    release.set()
    await asyncio.gather(first, second)
    assert entered == ["first", "other", "second"]


async def test_cancellation_during_emoji_lookup_stops_image_job(cog, monkeypatch):
    started, stopped = asyncio.Event(), asyncio.Event()

    async def generate(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    cog._react = AsyncMock()

    async def emoji(prompt):
        await asyncio.Event().wait()

    cog._get_thematic_emoji = emoji
    cog._execute_generation_tasks = generate
    cog._cost_tracker = Mock()
    cog._cost_tracker.can_afford.return_value = (True, "")
    monkeypatch.setattr(images, "bulk_add_prompts", Mock(return_value=[1]))
    request = asyncio.create_task(cog._handle_generation(message(), "cat", cog._prefix_map["!"]))
    await started.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert stopped.is_set()
    cog._cost_tracker.deduct_cost.assert_not_called()
