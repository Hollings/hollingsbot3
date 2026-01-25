"""Cost tracking and credit management for image generation.

This module provides the CostTracker class which manages:
- Daily free budget tracking per user
- Credit balance management
- Cost deduction for image generations
- Usage history and reporting

Tables are created by prompt_db.init_db() - this module just uses them.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
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
        # Tables are created by prompt_db.init_db() which runs at bot startup
        _log.info(
            f"CostTracker initialized with db_path={db_path}, "
            f"daily_free_budget=${daily_free_budget:.2f}"
        )

    def _get_today_string(self) -> str:
        """Get today's date as a string in YYYY-MM-DD format (UTC).

        Returns:
            Today's date string
        """
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _get_current_minute(self) -> datetime:
        """Get the current minute rounded down (UTC).

        Returns:
            Current datetime with seconds and microseconds set to 0
        """
        now = datetime.now(timezone.utc)
        return now.replace(second=0, microsecond=0)


    def _update_hourly_budget(self, user_id: int, conn: sqlite3.Connection) -> float:
        """Update a user's budget based on elapsed time (tracked per minute).

        This method calculates how many minutes have passed since the last tick
        and adds the appropriate amount to the user's current budget, capped
        at the daily free budget. Budget accrues at a rate of daily_budget / 1440
        (minutes per day).

        Args:
            user_id: Discord user ID
            conn: Database connection to use

        Returns:
            The user's current budget after update
        """
        current_minute = self._get_current_minute()
        minute_rate = self.daily_free_budget / 1440.0  # 24 hours * 60 minutes

        # Get or create user's budget record
        row = conn.execute(
            "SELECT current_budget, last_tick_minute FROM user_hourly_budget WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if row is None:
            # New user - check if they have usage today to migrate their budget
            today = self._get_today_string()
            daily_row = conn.execute(
                "SELECT free_budget_used FROM user_daily_costs WHERE user_id = ? AND date = ?",
                (user_id, today),
            ).fetchone()

            if daily_row:
                # Existing user with usage today - give them remaining daily budget
                free_budget_used = daily_row[0]
                current_budget = max(0.0, self.daily_free_budget - free_budget_used)
            else:
                # Brand new user - start with full budget
                current_budget = self.daily_free_budget

            # Insert new record
            conn.execute(
                "INSERT INTO user_hourly_budget (user_id, current_budget, last_tick_minute) VALUES (?, ?, ?)",
                (user_id, current_budget, current_minute),
            )
            conn.commit()
            _log.info(f"Initialized budget for user {user_id} at ${current_budget:.5f}")
            return current_budget

        current_budget, last_tick_minute = row
        last_tick_minute = datetime.fromisoformat(last_tick_minute)

        # Calculate minutes elapsed since last tick
        minutes_elapsed = int((current_minute - last_tick_minute).total_seconds() / 60)

        if minutes_elapsed > 0:
            # Add budget for elapsed minutes, capped at daily budget
            budget_to_add = minute_rate * minutes_elapsed
            new_budget = min(current_budget + budget_to_add, self.daily_free_budget)

            conn.execute(
                "UPDATE user_hourly_budget SET current_budget = ?, last_tick_minute = ? WHERE user_id = ?",
                (new_budget, current_minute, user_id),
            )
            conn.commit()

            _log.debug(
                f"Updated budget for user {user_id}: "
                f"${current_budget:.5f} + ${budget_to_add:.5f} ({minutes_elapsed}m) = ${new_budget:.5f}"
            )
            return new_budget

        # No time elapsed, return current budget
        return current_budget

    def can_afford(self, user_id: int, cost: float) -> tuple[bool, str]:
        """Check if a user can afford a generation.

        Args:
            user_id: Discord user ID
            cost: Cost of the generation in dollars

        Returns:
            Tuple of (can_afford: bool, error_message: str or empty)

        Raises:
            ValueError: If cost is negative or not a valid number
        """
        # Validate cost input
        if not isinstance(cost, (int, float)) or cost < 0:
            raise ValueError(f"Cost must be a non-negative number, got: {cost}")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Update and get current hourly budget
            current_budget = self._update_hourly_budget(user_id, conn)

            # Get credit balance
            credit_row = conn.execute(
                "SELECT balance FROM user_credits WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            credit_balance = credit_row["balance"] if credit_row else 0.0

        # Check if affordable
        if current_budget >= cost:
            # Can afford with free budget
            _log.debug(
                f"User {user_id} can afford ${cost:.5f} using free budget "
                f"(${current_budget:.5f} remaining)"
            )
            return (True, "")

        shortfall = cost - current_budget
        if credit_balance >= shortfall:
            # Can afford with free budget + credits
            _log.debug(
                f"User {user_id} can afford ${cost:.5f} using "
                f"${current_budget:.5f} free + ${shortfall:.5f} credits"
            )
            return (True, "")

        # Cannot afford
        hourly_rate = self.daily_free_budget / 24.0

        error_msg = (
            f"Insufficient funds for this generation.\n\n"
            f"This costs ${cost:.2f} but you have:\n"
            f"  - Free budget: ${current_budget:.2f} / ${self.daily_free_budget:.2f}\n"
            f"  - Credit balance: ${credit_balance:.2f}\n\n"
            f"Budget increases by ${hourly_rate:.2f}/hour (max ${self.daily_free_budget:.2f}/day).\n"
            f"Ask an admin about purchasing credits!"
        )

        _log.info(
            f"User {user_id} cannot afford ${cost:.5f} "
            f"(free: ${current_budget:.5f}, credits: ${credit_balance:.2f})"
        )
        return (False, error_msg)

    def deduct_cost(self, user_id: int, cost: float) -> None:
        """Deduct cost from user's free budget and/or credits.

        This should only be called after a successful generation.

        Args:
            user_id: Discord user ID
            cost: Cost to deduct in dollars

        Raises:
            ValueError: If cost is negative or not a valid number
        """
        # Validate cost input
        if not isinstance(cost, (int, float)) or cost < 0:
            raise ValueError(f"Cost must be a non-negative number, got: {cost}")

        today = self._get_today_string()

        with sqlite3.connect(self.db_path) as conn:
            # Use immediate transaction to prevent race conditions
            conn.execute("BEGIN IMMEDIATE")

            try:
                # Update hourly budget and get current balance
                current_budget = self._update_hourly_budget(user_id, conn)

                # Calculate deductions
                free_deduction = min(cost, current_budget)
                credit_deduction = cost - free_deduction

                # Update hourly budget
                new_budget = current_budget - free_deduction
                conn.execute(
                    "UPDATE user_hourly_budget SET current_budget = ? WHERE user_id = ?",
                    (new_budget, user_id),
                )

                # Update daily costs for reporting
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
                    f"Deducted ${cost:.5f} from user {user_id} "
                    f"(free: ${free_deduction:.5f}, credits: ${credit_deduction:.5f})"
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
            Dictionary with keys: current_budget, free_budget_total, hourly_rate,
            credits_used_today, generation_count, credit_balance
        """
        today = self._get_today_string()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Update and get current hourly budget
            current_budget = self._update_hourly_budget(user_id, conn)

            # Get daily usage
            daily_row = conn.execute(
                """
                SELECT credits_used, generation_count
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

        credits_used_today = daily_row["credits_used"] if daily_row else 0.0
        generation_count = daily_row["generation_count"] if daily_row else 0
        credit_balance = credit_row["balance"] if credit_row else 0.0
        hourly_rate = self.daily_free_budget / 24.0

        return {
            "current_budget": current_budget,
            "free_budget_total": self.daily_free_budget,
            "hourly_rate": hourly_rate,
            "credits_used_today": credits_used_today,
            "generation_count": generation_count,
            "credit_balance": credit_balance,
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
