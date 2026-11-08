import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier
from unittest.mock import AsyncMock, Mock

import pytest

from hollingsbot import prompt_db
from hollingsbot.cogs import image_gen_cog as images
from hollingsbot.cost_tracking import CostTracker
from test_image_gen_workflow import message


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_db, "DB_PATH", tmp_path / "bot.db")
    prompt_db.init_db()
    tracker = CostTracker(str(prompt_db.DB_PATH), 0.5)
    minute = tracker._get_current_minute()
    monkeypatch.setattr(tracker, "_get_current_minute", lambda: minute)
    return tracker


def claim(message_id, count=1):
    return prompt_db.bulk_add_prompts(["cat"] * count, "1", "replicate", "fake", request_message_id=message_id)


def deliver(tracker, pid, cost):
    assert prompt_db.transition_image_prompt(pid, "generating")
    assert prompt_db.transition_image_prompt(pid, "generated")
    assert prompt_db.transition_image_prompt(pid, "delivering")
    tracker.deduct_cost(1, cost, prompt_id=pid)


def test_parallel_admissions_cannot_reserve_the_same_money(tracker):
    ids = [claim(100), claim(101)]
    barrier = Barrier(2)

    def reserve(pids):
        independent = CostTracker(tracker.db_path, 0.5)
        barrier.wait(timeout=5)
        return independent.reserve_images(1, pids, 0.4)[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve, ids)) == [False, True]
    status = tracker.get_user_status(1)
    assert status["current_budget"] == pytest.approx(0.1)
    assert status["reserved_for_images"] == pytest.approx(0.4)
    assert tracker.get_history(1) == []  # Holds are not charges.


def test_out_of_order_settlement_release_and_replay_preserve_budget(tracker):
    first, second = claim(100), claim(101)
    assert tracker.reserve_images(1, first, 0.3)[0]
    assert tracker.reserve_images(1, second, 0.15)[0]
    assert not tracker.can_afford(1, 0.1)[0]
    deliver(tracker, second[0], 0.15)
    tracker.deduct_cost(1, 0.15, prompt_id=second[0])
    prompt_db.abandon_image_prompts(first)
    status = tracker.get_user_status(1)
    assert status["current_budget"] == pytest.approx(0.35)
    assert status["reserved_for_images"] == 0
    assert status["generation_count"] == 1
    assert tracker.get_history(1)[0]["total_cost"] == pytest.approx(0.15)


def test_unaffordable_batch_reserves_nothing(tracker):
    assert not tracker.reserve_images(1, claim(100, count=3), 0.2)[0]
    assert tracker.get_user_status(1)["reserved_for_images"] == 0
    assert tracker.reserve_images(1, claim(101), 0.5)[0]


def test_redeemed_credits_are_spendable_while_another_image_runs(tracker):
    assert tracker.reserve_images(1, claim(100), 0.5)[0]
    for _ in range(47):
        prompt_db.give_user_token(1)
    assert prompt_db.redeem_user_tokens(1, 47, 0.37)[0]
    second = claim(101)
    assert tracker.reserve_images(1, second, 0.3)[0]
    deliver(tracker, second[0], 0.3)
    status = tracker.get_user_status(1)
    assert status["reserved_for_images"] == pytest.approx(0.5)
    assert status["current_budget"] + status["credit_balance"] == pytest.approx(0.07)


def test_expired_owner_releases_budget_and_cannot_send_late_result(tracker):
    pids = claim(100)
    assert tracker.reserve_images(1, pids, 0.5)[0]
    assert prompt_db.transition_image_prompt(pids[0], "generating")
    assert prompt_db.transition_image_prompt(pids[0], "generated")
    with prompt_db._connect() as conn:
        conn.execute(
            "UPDATE image_reservations SET expires_at = ?", ((datetime.utcnow() - timedelta(seconds=1)).isoformat(),)
        )
    restarted = CostTracker(tracker.db_path, 0.5)
    assert restarted.can_afford(1, 0.5)[0]
    assert not prompt_db.transition_image_prompt(pids[0], "delivering")
    with prompt_db._connect() as conn:
        assert conn.execute("SELECT status FROM prompts WHERE id = ?", pids).fetchone()[0] == "outcome_unknown"
    assert tracker.reserve_images(1, claim(101), 0.5)[0]


async def test_same_user_generates_concurrently_and_cancellation_releases_only_its_hold(tracker):
    cog = images.ImageGenCog(Mock(), config={"!": {"api": "replicate", "model": "fake", "price_per_image": 0.2}})
    cog._react = AsyncMock()
    cog._get_thematic_emoji = AsyncMock(return_value=images.THINKING)
    first_started, second_started = asyncio.Event(), asyncio.Event()

    async def generate(ids, prompts, *args):
        (first_started if prompts == ["first"] else second_started).set()
        await asyncio.Event().wait()

    cog._execute_generation_tasks = generate
    first, second = message(), message()
    second.id = 124
    task1 = asyncio.create_task(cog._handle_generation(first, "first", cog._prefix_map["!"]))
    task2 = None
    try:
        await asyncio.wait_for(first_started.wait(), 1)
        task2 = asyncio.create_task(cog._handle_generation(second, "second", cog._prefix_map["!"]))
        await asyncio.wait_for(second_started.wait(), 1)
        assert tracker.get_user_status(1)["reserved_for_images"] == pytest.approx(0.4)
        task1.cancel()
        await asyncio.gather(task1, return_exceptions=True)
        assert tracker.get_user_status(1)["reserved_for_images"] == pytest.approx(0.2)
        assert not task2.done()
    finally:
        task1.cancel()
        if task2:
            task2.cancel()
        await asyncio.gather(task1, *([task2] if task2 else []), return_exceptions=True)
    assert tracker.get_user_status(1)["reserved_for_images"] == 0
    assert tracker.get_history(1) == []


def test_failed_settlement_rolls_back_hold_and_debit(tracker):
    pids = claim(100)
    assert tracker.reserve_images(1, pids, 0.2)[0]
    with prompt_db._connect() as conn:
        conn.execute(
            "CREATE TRIGGER reject_charge BEFORE INSERT ON user_daily_costs BEGIN SELECT RAISE(ABORT, 'failed write'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        deliver(tracker, pids[0], 0.2)
    assert tracker.get_user_status(1)["reserved_for_images"] == pytest.approx(0.2)
    assert tracker.get_history(1) == []
    with prompt_db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM image_settlements").fetchone()[0] == 0
