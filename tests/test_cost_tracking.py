"""Tests for cost tracking and credit management."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from hollingsbot.cost_tracking import CostTracker


@pytest.fixture
def db_path():
    """Create a temporary database file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test.db"
        yield str(db_file)


@pytest.fixture
def setup_db(db_path):
    """Initialize the database tables needed for cost tracking."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_hourly_budget (
                user_id INTEGER PRIMARY KEY,
                current_budget REAL DEFAULT 0,
                last_tick_minute TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_daily_costs (
                user_id INTEGER,
                date TEXT,
                total_cost REAL DEFAULT 0,
                free_budget_used REAL DEFAULT 0,
                credits_used REAL DEFAULT 0,
                generation_count INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_credits (
                user_id INTEGER PRIMARY KEY,
                balance REAL DEFAULT 0,
                lifetime_spent REAL DEFAULT 0,
                last_updated TEXT
            )
        """)
        conn.commit()
    return db_path


@pytest.fixture
def tracker(setup_db):
    """Create a CostTracker instance with test database."""
    return CostTracker(setup_db, daily_free_budget=1.00)


class TestCostTrackerInit:
    """Tests for CostTracker initialization."""

    def test_init_with_valid_params(self, setup_db):
        """Test initialization with valid parameters."""
        tracker = CostTracker(setup_db, daily_free_budget=2.50)
        assert tracker.db_path == setup_db
        assert tracker.daily_free_budget == 2.50

    def test_init_with_zero_budget(self, setup_db):
        """Test initialization with zero daily budget."""
        tracker = CostTracker(setup_db, daily_free_budget=0)
        assert tracker.daily_free_budget == 0


class TestCanAfford:
    """Tests for the can_afford method."""

    def test_new_user_can_afford_within_budget(self, tracker):
        """Test that a new user can afford a cost within daily budget."""
        user_id = 12345
        can_afford, error = tracker.can_afford(user_id, 0.50)
        assert can_afford is True
        assert error == ""

    def test_new_user_cannot_afford_over_budget(self, tracker):
        """Test that a new user cannot afford a cost over daily budget."""
        user_id = 12345
        can_afford, error = tracker.can_afford(user_id, 2.00)
        assert can_afford is False
        assert "Insufficient funds" in error

    def test_can_afford_with_credits(self, tracker):
        """Test that user can afford using credits."""
        user_id = 12345
        # Grant credits
        tracker.grant_credits(user_id, 5.00)
        # Try to afford more than daily budget
        can_afford, error = tracker.can_afford(user_id, 3.00)
        assert can_afford is True
        assert error == ""

    def test_can_afford_zero_cost(self, tracker):
        """Test that zero cost is always affordable."""
        user_id = 12345
        can_afford, error = tracker.can_afford(user_id, 0)
        assert can_afford is True
        assert error == ""

    def test_can_afford_negative_cost_raises(self, tracker):
        """Test that negative cost raises ValueError."""
        user_id = 12345
        with pytest.raises(ValueError, match="non-negative"):
            tracker.can_afford(user_id, -1.00)

    def test_can_afford_invalid_type_raises(self, tracker):
        """Test that invalid cost type raises ValueError."""
        user_id = 12345
        with pytest.raises(ValueError):
            tracker.can_afford(user_id, "invalid")


class TestDeductCost:
    """Tests for the deduct_cost method."""

    def test_deduct_from_free_budget(self, tracker):
        """Test deducting cost from free budget."""
        user_id = 12345
        # First check we can afford it (initializes budget)
        tracker.can_afford(user_id, 0.50)
        # Deduct the cost
        tracker.deduct_cost(user_id, 0.50)
        # Check remaining budget
        status = tracker.get_user_status(user_id)
        assert status["current_budget"] == pytest.approx(0.50, abs=0.01)

    def test_deduct_uses_credits_after_free_budget(self, tracker):
        """Test that credits are used after free budget is exhausted."""
        user_id = 12345
        # Grant credits
        tracker.grant_credits(user_id, 1.00)
        # Check we can afford (initializes budget)
        tracker.can_afford(user_id, 1.50)
        # Deduct more than free budget
        tracker.deduct_cost(user_id, 1.50)
        # Check that credits were used
        status = tracker.get_user_status(user_id)
        assert status["current_budget"] == pytest.approx(0.0, abs=0.01)
        assert status["credit_balance"] == pytest.approx(0.50, abs=0.01)

    def test_deduct_negative_cost_raises(self, tracker):
        """Test that deducting negative cost raises ValueError."""
        user_id = 12345
        with pytest.raises(ValueError, match="non-negative"):
            tracker.deduct_cost(user_id, -1.00)

    def test_deduct_zero_cost(self, tracker):
        """Test that deducting zero cost works."""
        user_id = 12345
        tracker.can_afford(user_id, 0)  # Initialize
        tracker.deduct_cost(user_id, 0)
        status = tracker.get_user_status(user_id)
        assert status["generation_count"] == 1


class TestGetUserStatus:
    """Tests for the get_user_status method."""

    def test_new_user_status(self, tracker):
        """Test status for a new user."""
        user_id = 12345
        status = tracker.get_user_status(user_id)
        assert status["current_budget"] == pytest.approx(1.00, abs=0.01)
        assert status["free_budget_total"] == 1.00
        assert status["hourly_rate"] == pytest.approx(1.00 / 24, abs=0.001)
        assert status["credits_used_today"] == 0
        assert status["generation_count"] == 0
        assert status["credit_balance"] == 0

    def test_status_after_deduction(self, tracker):
        """Test status after a cost deduction."""
        user_id = 12345
        tracker.can_afford(user_id, 0.25)  # Initialize
        tracker.deduct_cost(user_id, 0.25)
        status = tracker.get_user_status(user_id)
        assert status["current_budget"] == pytest.approx(0.75, abs=0.01)
        assert status["generation_count"] == 1


class TestGrantCredits:
    """Tests for the grant_credits method."""

    def test_grant_credits_to_new_user(self, tracker):
        """Test granting credits to a new user."""
        user_id = 12345
        tracker.grant_credits(user_id, 10.00)
        status = tracker.get_user_status(user_id)
        assert status["credit_balance"] == 10.00

    def test_grant_credits_cumulative(self, tracker):
        """Test that credits are cumulative."""
        user_id = 12345
        tracker.grant_credits(user_id, 5.00)
        tracker.grant_credits(user_id, 3.00)
        status = tracker.get_user_status(user_id)
        assert status["credit_balance"] == 8.00

    def test_grant_negative_credits(self, tracker):
        """Test granting negative credits (deduction)."""
        user_id = 12345
        tracker.grant_credits(user_id, 10.00)
        tracker.grant_credits(user_id, -3.00)
        status = tracker.get_user_status(user_id)
        assert status["credit_balance"] == 7.00


class TestGetHistory:
    """Tests for the get_history method."""

    def test_history_empty_for_new_user(self, tracker):
        """Test that history is empty for a new user."""
        user_id = 12345
        history = tracker.get_history(user_id)
        assert history == []

    def test_history_after_usage(self, tracker):
        """Test history after some usage."""
        user_id = 12345
        tracker.can_afford(user_id, 0.10)  # Initialize
        tracker.deduct_cost(user_id, 0.10)
        tracker.deduct_cost(user_id, 0.20)
        history = tracker.get_history(user_id)
        assert len(history) == 1
        assert history[0]["total_cost"] == pytest.approx(0.30, abs=0.01)
        assert history[0]["generation_count"] == 2


class TestBudgetRefresh:
    """Tests for budget refresh over time."""

    def test_budget_accrues_over_time(self, tracker):
        """Test that budget accrues as time passes."""
        user_id = 12345
        # Exhaust budget
        tracker.can_afford(user_id, 1.00)  # Initialize
        tracker.deduct_cost(user_id, 1.00)
        status = tracker.get_user_status(user_id)
        assert status["current_budget"] == pytest.approx(0.0, abs=0.01)

        # Mock time passing (60 minutes)
        minute_rate = 1.00 / 1440.0
        expected_accrual = minute_rate * 60

        # Manually update last_tick_minute to simulate time passing
        with sqlite3.connect(tracker.db_path) as conn:
            past_time = datetime.now(timezone.utc) - timedelta(minutes=60)
            conn.execute(
                "UPDATE user_hourly_budget SET last_tick_minute = ? WHERE user_id = ?",
                (past_time.isoformat(), user_id),
            )
            conn.commit()

        status = tracker.get_user_status(user_id)
        assert status["current_budget"] == pytest.approx(expected_accrual, abs=0.001)

    def test_budget_caps_at_daily_limit(self, tracker):
        """Test that budget doesn't exceed daily limit."""
        user_id = 12345
        # Initialize with full budget
        tracker.can_afford(user_id, 0)

        # Simulate 48 hours passing (should still cap at daily budget)
        with sqlite3.connect(tracker.db_path) as conn:
            past_time = datetime.now(timezone.utc) - timedelta(hours=48)
            conn.execute(
                "UPDATE user_hourly_budget SET last_tick_minute = ? WHERE user_id = ?",
                (past_time.isoformat(), user_id),
            )
            conn.commit()

        status = tracker.get_user_status(user_id)
        # Should be capped at 1.00 (daily budget)
        assert status["current_budget"] == pytest.approx(1.00, abs=0.01)
