"""User-facing commands for checking image generation usage and credits.

This cog provides commands for users to view their current balance, daily usage,
and cost history, and redeem tokens for credits.
"""

from __future__ import annotations

import logging
import os

from discord.ext import commands

from hollingsbot.cost_tracking import CostTracker
from hollingsbot.prompt_db import get_user_token_balance, deduct_user_tokens

__all__ = ["CreditsCog"]

_log = logging.getLogger(__name__)

# Token to credit exchange rate: 47 tokens = $0.37
TOKENS_PER_EXCHANGE = 47
CENTS_PER_EXCHANGE = 37
DOLLARS_PER_TOKEN = CENTS_PER_EXCHANGE / 100 / TOKENS_PER_EXCHANGE  # ~$0.00787 per token


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

            current_budget = status["current_budget"]
            free_total = status["free_budget_total"]
            hourly_rate = status["hourly_rate"]
            credits_used_today = status["credits_used_today"]
            generation_count = status["generation_count"]
            credit_balance = status["credit_balance"]

            # Build the response message
            lines = [
                "**Your current status:**",
                f"  - Free budget: ${current_budget:.2f} / ${free_total:.2f}",
                f"  - Budget increases by ${hourly_rate:.2f}/hour (max ${free_total:.2f}/day)",
            ]

            if credits_used_today > 0:
                lines.append(f"  - Credits spent today: ${credits_used_today:.2f}")

            lines.append(f"  - Total generations today: {generation_count}")
            lines.append(f"\n**Credit balance:** ${credit_balance:.2f}")

            message = "\n".join(lines)
            await ctx.send(message)

            _log.debug(f"User {ctx.author.id} checked usage")

        except Exception as exc:
            _log.exception(f"Failed to get usage for user {ctx.author.id}: {exc}")
            await ctx.send("Failed to retrieve usage information. Please try again later.")

    @commands.command(name="redeem")
    async def redeem_command(self, ctx: commands.Context, amount: int = None) -> None:
        """Redeem Wendy tokens for image generation credits.

        Usage: !redeem <amount>
        Example: !redeem 47
        """
        # Show usage info if no amount provided
        if amount is None:
            token_balance = get_user_token_balance(ctx.author.id)
            max_redeemable = token_balance
            max_credits = max_redeemable * DOLLARS_PER_TOKEN

            message = (
                f"**Token Redemption**\n"
                f"Exchange your Wendy tokens for image generation credits!\n\n"
                f"**Exchange rate:** {TOKENS_PER_EXCHANGE} tokens = ${CENTS_PER_EXCHANGE / 100:.2f}\n"
                f"**Your tokens:** {token_balance}\n"
                f"**Max credits:** ${max_credits:.2f}\n\n"
                f"**Usage:** `!redeem <amount>`\n"
                f"**Example:** `!redeem 47` to redeem 47 tokens for ${CENTS_PER_EXCHANGE / 100:.2f}"
            )
            await ctx.send(message)
            return

        # Validate amount
        if amount <= 0:
            await ctx.send("Amount must be a positive number.")
            return

        # Check balance and deduct tokens
        success, new_balance = deduct_user_tokens(ctx.author.id, amount)

        if not success:
            await ctx.send(
                f"You don't have enough tokens. You have {new_balance} tokens, "
                f"but tried to redeem {amount}."
            )
            return

        # Calculate credits and grant them
        credits_earned = amount * DOLLARS_PER_TOKEN
        self._cost_tracker.grant_credits(ctx.author.id, credits_earned)

        await ctx.send(
            f"Redeemed {amount} tokens for ${credits_earned:.2f} in credits!\n"
            f"Remaining tokens: {new_balance}"
        )
        _log.info(
            f"User {ctx.author.id} redeemed {amount} tokens for ${credits_earned:.2f} credits"
        )


async def setup(bot: commands.Bot) -> None:
    """Load the CreditsCog."""
    await bot.add_cog(CreditsCog(bot))
