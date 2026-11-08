"""Exercise configuration, delivery and accounting without external services."""

import asyncio
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from hollingsbot import prompt_db
from hollingsbot.cogs import image_gen_cog as images
from hollingsbot.cogs.admin import Admin
from hollingsbot.cogs.credits_cog import CreditsCog
from hollingsbot.image_generators.base import ImageGenerationError


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


async def test_batch_charges_delivered_variants_without_downloading_edit_images(cog):
    cog._react = AsyncMock()
    cog._get_thematic_emoji = AsyncMock(return_value=images.THINKING)
    cog._collect_images_for_editing = AsyncMock()
    cog._execute_generation_tasks = AsyncMock(return_value=[("cat", [b"img"]), ImageGenerationError("incomplete SVG")])
    cog._prepare_discord_files = Mock(return_value=[Mock()])
    cog._send_error_message = AsyncMock()
    msg = message()
    await cog._handle_generation(msg, "<cat,dog>", cog._prefix_map["!"])
    assert cog._cost_tracker.get_history(1)[0]["total_cost"] == pytest.approx(0.03)
    msg.channel.send.assert_awaited_once()
    cog._send_error_message.assert_awaited_once()
    cog._collect_images_for_editing.assert_not_awaited()


@pytest.mark.parametrize("reference", [None, b"attached still"])
async def test_gif_command_preserves_frames_and_charges_ten_cents(cog, tmp_path, reference):
    cog._cfg_path = images._DEFAULT_CONFIG_PATH
    cog._reload_config()
    split = cog._split_prompt("gif: a lantern swings")
    assert split is not None
    prompt, spec = split
    assert cog._split_prompt("gif") is None
    msg = message()
    msg.reference = None
    msg.channel.history = Mock(side_effect=AssertionError("GIF prompts must not search unrelated history"))
    cog._gather_attachment_images = AsyncMock(return_value=[reference] if reference else [])
    cog._get_aspect_ratio_for_prompt = AsyncMock(side_effect=AssertionError("GIF uses the configured square ratio"))
    cog._react = AsyncMock()
    cog._get_thematic_emoji = AsyncMock(return_value=images.THINKING)

    data = BytesIO()
    Image.new("RGB", (8, 8), "red").save(
        data, format="GIF", save_all=True, append_images=[Image.new("RGB", (8, 8), "blue")], duration=200, loop=0
    )
    result = tmp_path / "animation.gif"
    result.write_bytes(data.getvalue())
    cog._run_task = AsyncMock(return_value=[str(result)])
    sent = []

    async def send(*, files):
        sent.extend((file.filename, file.fp.read()) for file in files)

    msg.channel.send.side_effect = send
    await cog._generate_for_user(msg, prompt, spec)

    call = cog._run_task.await_args
    assert call.args[2] == "openai/gpt-image-2.6-flare"
    assert call.kwargs["output_format"] == "gif"
    assert call.kwargs["image_input"] == ([reference] if reference else None)
    msg.channel.history.assert_not_called()
    msg.channel.send.assert_awaited_once()
    assert len(sent) == 1 and sent[0][0].endswith(".gif")
    assert sent[0][1] == data.getvalue()
    assert not result.exists()
    assert cog._cost_tracker.get_history(1)[0]["total_cost"] == pytest.approx(0.10)


async def test_slow_request_does_not_block_same_user_or_other_users(cog):
    cog._react = AsyncMock()
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
    assert entered == ["first", "other", "second"]
    release.set()
    await asyncio.gather(first, second)
    assert entered == ["first", "other", "second"]


async def test_cancellation_during_emoji_lookup_stops_image_job(cog):
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
    cog._cost_tracker.reserve_images.return_value = (True, "")
    request = asyncio.create_task(cog._handle_generation(message(), "cat", cog._prefix_map["!"]))
    await started.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert stopped.is_set()
    cog._cost_tracker.deduct_cost.assert_not_called()
    assert cog._react.await_args.kwargs == {"remove": True}


async def test_acknowledged_variant_is_settled_before_next_send_and_replay_is_ignored(cog):
    cog._react = AsyncMock()
    cog._get_thematic_emoji = AsyncMock(return_value=images.THINKING)
    cog._execute_generation_tasks = AsyncMock(return_value=[("cat", [b"one"]), ("dog", [b"two"])])
    cog._prepare_discord_files = Mock(return_value=[Mock()])
    msg = message()
    msg.channel.send.side_effect = [None, asyncio.CancelledError()]
    spec = cog._prefix_map["!"]
    with pytest.raises(asyncio.CancelledError):
        await cog._handle_generation(msg, "<cat,dog>", spec)

    with prompt_db._connect() as conn:
        rows = conn.execute("SELECT id, status FROM prompts ORDER BY id").fetchall()
    assert [status for _, status in rows] == ["delivered", "outcome_unknown"]
    # A duplicate settlement and replay of the same Discord event do no work.
    cog._cost_tracker.deduct_cost(1, 0.03, prompt_id=rows[0][0])
    await cog._handle_generation(msg, "<cat,dog>", spec)
    assert cog._cost_tracker.get_history(1)[0]["total_cost"] == pytest.approx(0.03)
    assert cog._cost_tracker.get_history(1)[0]["generation_count"] == 1
    cog._execute_generation_tasks.assert_awaited_once()
    assert msg.channel.send.await_count == 2
