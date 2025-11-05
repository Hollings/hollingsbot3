"""Cost tracking and credit management for image generation.

This module provides the CostTracker class which manages:
- Daily free budget tracking per user
- Credit balance management
- Cost deduction for image generations
- Usage history and reporting
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

_log = logging.getLogger(__name__)


class CostTracker:
    """Manages cost tracking and credit balances for users."""

    def __init__(self, db_path: str, daily_free_budget: float) -> None:
        """Initialize the cost tracker.

        Args:
            db_path: Path to the SQLite database file
            daily_free_budget: Global daily free budget amount in dollars
        """
        self.db_path = db_path
        self.daily_free_budget = daily_free_budget
        self._ensure_tables_exist()
        _log.info(
            f"CostTracker initialized with db_path={db_path}, "
            f"daily_free_budget=${daily_free_budget:.2f}"
        )

    def _ensure_tables_exist(self) -> None:
        """Create the cost tracking tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_daily_costs (
                    user_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    total_cost REAL DEFAULT 0.0,
                    free_budget_used REAL DEFAULT 0.0,
                    credits_used REAL DEFAULT 0.0,
                    generation_count INTEGER DEFAULT 0,
                    PRIMARY KEY (user_id, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_credits (
                    user_id INTEGER PRIMARY KEY,
                    balance REAL DEFAULT 0.0,
                    lifetime_spent REAL DEFAULT 0.0,
                    last_updated TIMESTAMP
                )
            """)
            conn.commit()
        _log.debug("Database tables ensured to exist")

    def _get_today_string(self) -> str:
        """Get today's date as a string in YYYY-MM-DD format (UTC).

        Returns:
            Today's date string
        """
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _get_seconds_until_reset(self) -> int:
        """Calculate seconds until the daily reset (midnight UTC).

        Returns:
            Seconds until midnight UTC
        """
        now = datetime.now(timezone.utc)
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = tomorrow.replace(day=tomorrow.day + 1)
        return int((tomorrow - now).total_seconds())

    def _format_reset_time(self, seconds: int) -> str:
        """Format seconds until reset as a human-readable string.

        Args:
            seconds: Seconds until reset

        Returns:
            Formatted string like "4h 23m"
        """
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

    def can_afford(self, user_id: int, cost: float) -> tuple[bool, str]:
        """Check if a user can afford a generation.

        Args:
            user_id: Discord user ID
            cost: Cost of the generation in dollars

        Returns:
            Tuple of (can_afford: bool, error_message: str or empty)
        """
        today = self._get_today_string()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get today's usage
            daily_row = conn.execute(
                "SELECT free_budget_used FROM user_daily_costs WHERE user_id = ? AND date = ?",
                (user_id, today),
            ).fetchone()

            free_budget_used = daily_row["free_budget_used"] if daily_row else 0.0
            remaining_free_budget = max(0.0, self.daily_free_budget - free_budget_used)

            # Get credit balance
            credit_row = conn.execute(
                "SELECT balance FROM user_credits WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            credit_balance = credit_row["balance"] if credit_row else 0.0

        # Check if affordable
        if remaining_free_budget >= cost:
            # Can afford with free budget
            _log.debug(
                f"User {user_id} can afford ${cost:.3f} using free budget "
                f"(${remaining_free_budget:.3f} remaining)"
            )
            return (True, "")

        shortfall = cost - remaining_free_budget
        if credit_balance >= shortfall:
            # Can afford with free budget + credits
            _log.debug(
                f"User {user_id} can afford ${cost:.3f} using "
                f"${remaining_free_budget:.3f} free + ${shortfall:.3f} credits"
            )
            return (True, "")

        # Cannot afford
        reset_seconds = self._get_seconds_until_reset()
        reset_time = self._format_reset_time(reset_seconds)

        error_msg = (
            f"❌ Insufficient funds for this generation.\n\n"
            f"This costs ${cost:.2f} but you have:\n"
            f"  • Free budget remaining today: ${remaining_free_budget:.2f} / ${self.daily_free_budget:.2f}\n"
            f"  • Credit balance: ${credit_balance:.2f}\n\n"
            f"Your daily free budget resets in {reset_time}.\n"
            f"Ask an admin about purchasing credits!"
        )

        _log.info(
            f"User {user_id} cannot afford ${cost:.3f} "
            f"(free: ${remaining_free_budget:.3f}, credits: ${credit_balance:.2f})"
        )
        return (False, error_msg)

    def deduct_cost(self, user_id: int, cost: float) -> None:
        """Deduct cost from user's free budget and/or credits.

        This should only be called after a successful generation.

        Args:
            user_id: Discord user ID
            cost: Cost to deduct in dollars
        """
        today = self._get_today_string()

        with sqlite3.connect(self.db_path) as conn:
            # Use immediate transaction to prevent race conditions
            conn.execute("BEGIN IMMEDIATE")

            try:
                # Get current daily usage
                daily_row = conn.execute(
                    "SELECT free_budget_used FROM user_daily_costs WHERE user_id = ? AND date = ?",
                    (user_id, today),
                ).fetchone()

                free_budget_used = daily_row[0] if daily_row else 0.0
                remaining_free_budget = max(0.0, self.daily_free_budget - free_budget_used)

                # Calculate deductions
                free_deduction = min(cost, remaining_free_budget)
                credit_deduction = cost - free_deduction

                # Update daily costs
                conn.execute(
                    """
                    INSERT INTO user_daily_costs (user_id, date, total_cost, free_budget_used, credits_used, generation_count)
                    VALUES (?, ?, ?, ?, ?, 1)
                    ON CONFLICT(user_id, date) DO UPDATE SET
                        total_cost = total_cost + ?,
                        free_budget_used = free_budget_used + ?,
                        credits_used = credits_used + ?,
                        generation_count = generation_count + 1
                    """,
                    (user_id, today, cost, free_deduction, credit_deduction, cost, free_deduction, credit_deduction),
                )

                # Update credits if needed
                if credit_deduction > 0:
                    conn.execute(
                        """
                        INSERT INTO user_credits (user_id, balance, lifetime_spent, last_updated)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET
                            balance = balance - ?,
                            lifetime_spent = lifetime_spent + ?,
                            last_updated = ?
                        """,
                        (
                            user_id,
                            -credit_deduction,
                            credit_deduction,
                            datetime.now(timezone.utc),
                            credit_deduction,
                            credit_deduction,
                            datetime.now(timezone.utc),
                        ),
                    )

                conn.commit()
                _log.info(
                    f"Deducted ${cost:.3f} from user {user_id} "
                    f"(free: ${free_deduction:.3f}, credits: ${credit_deduction:.3f})"
                )

            except Exception:
                conn.rollback()
                _log.exception(f"Failed to deduct cost for user {user_id}")
                raise

    def get_user_status(self, user_id: int) -> dict:
        """Get a user's current usage status and balance.

        Args:
            user_id: Discord user ID

        Returns:
            Dictionary with keys: free_budget_used, free_budget_total, credits_used_today,
            generation_count, credit_balance, reset_time
        """
        today = self._get_today_string()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get daily usage
            daily_row = conn.execute(
                """
                SELECT free_budget_used, credits_used, generation_count
                FROM user_daily_costs
                WHERE user_id = ? AND date = ?
                """,
                (user_id, today),
            ).fetchone()

            # Get credit balance
            credit_row = conn.execute(
                "SELECT balance FROM user_credits WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        free_budget_used = daily_row["free_budget_used"] if daily_row else 0.0
        credits_used_today = daily_row["credits_used"] if daily_row else 0.0
        generation_count = daily_row["generation_count"] if daily_row else 0
        credit_balance = credit_row["balance"] if credit_row else 0.0

        reset_seconds = self._get_seconds_until_reset()

        return {
            "free_budget_used": free_budget_used,
            "free_budget_total": self.daily_free_budget,
            "credits_used_today": credits_used_today,
            "generation_count": generation_count,
            "credit_balance": credit_balance,
            "reset_time": self._format_reset_time(reset_seconds),
        }

    def grant_credits(self, user_id: int, amount: float) -> None:
        """Grant credits to a user.

        Args:
            user_id: Discord user ID
            amount: Amount of credits to grant (can be negative to deduct)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_credits (user_id, balance, lifetime_spent, last_updated)
                VALUES (?, ?, 0, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    balance = balance + ?,
                    last_updated = ?
                """,
                (user_id, amount, datetime.now(timezone.utc), amount, datetime.now(timezone.utc)),
            )
            conn.commit()

        _log.info(f"Granted ${amount:.2f} credits to user {user_id}")

    def get_history(self, user_id: int, days: int = 7) -> list[dict]:
        """Get a user's cost history for the last N days.

        Args:
            user_id: Discord user ID
            days: Number of days to retrieve (default 7)

        Returns:
            List of dicts with keys: date, total_cost, free_budget_used, credits_used, generation_count
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT date, total_cost, free_budget_used, credits_used, generation_count
                FROM user_daily_costs
                WHERE user_id = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (user_id, days),
            ).fetchall()

        return [dict(row) for row in rows]
