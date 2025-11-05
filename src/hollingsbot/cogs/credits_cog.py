"""User-facing commands for checking image generation usage and credits.

This cog provides commands for users to view their current balance, daily usage,
and cost history.
"""

from __future__ import annotations

import logging
import os

import discord
from discord.ext import commands

from hollingsbot.cost_tracking import CostTracker

__all__ = ["CreditsCog"]

_log = logging.getLogger(__name__)


class CreditsCog(commands.Cog):
    """Commands for users to check their image generation usage and credits."""

    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the credits cog.

        Args:
            bot: Discord bot instance
        """
        self.bot = bot

        # Get cost tracker configuration
        db_path = os.getenv("PROMPT_DB_PATH", "prompts.db")
        daily_free_budget = float(os.getenv("DAILY_FREE_BUDGET", "0.50"))

        self._cost_tracker = CostTracker(db_path, daily_free_budget)
        _log.info("CreditsCog initialized")

    @commands.command(name="usage")
    async def usage_command(self, ctx: commands.Context) -> None:
        """Show the user's current usage and credit balance.

        Usage: !usage
        """
        try:
            status = self._cost_tracker.get_user_status(ctx.author.id)

            free_used = status["free_budget_used"]
            free_total = status["free_budget_total"]
            credits_used_today = status["credits_used_today"]
            generation_count = status["generation_count"]
            credit_balance = status["credit_balance"]
            reset_time = status["reset_time"]

            # Build the response message
            lines = [
                f"**Your usage for today** (resets in {reset_time}):",
                f"  • Free budget: ${free_used:.2f} / ${free_total:.2f} used",
            ]

            if credits_used_today > 0:
                lines.append(f"  • Credits spent today: ${credits_used_today:.2f}")

            lines.append(f"  • Total generations: {generation_count}")
            lines.append(f"\n**Credit balance:** ${credit_balance:.2f}")

            message = "\n".join(lines)
            await ctx.send(message)

            _log.debug(f"User {ctx.author.id} checked usage")

        except Exception as exc:
            _log.exception(f"Failed to get usage for user {ctx.author.id}: {exc}")
            await ctx.send("❌ Failed to retrieve usage information. Please try again later.")


async def setup(bot: commands.Bot) -> None:
    """Load the CreditsCog."""
    await bot.add_cog(CreditsCog(bot))
