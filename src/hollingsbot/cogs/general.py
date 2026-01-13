"""General utility commands for the Discord bot.

This cog provides basic commands like ping and help that are available across
all channels and provide core bot functionality.
"""

from __future__ import annotations

import logging

from discord.ext import commands

__all__ = ["General"]

_log = logging.getLogger(__name__)

# Discord's message limit is 2000 characters; we keep a safety margin
MAX_HELP_MESSAGE_LENGTH = 1900


class General(commands.Cog):
    """Basic commands available to all users.

    Provides fundamental bot interactions including connection testing (ping)
    and comprehensive help documentation.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the General cog.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        _log.info("General cog initialized")

    @commands.command()
    async def ping(self, ctx: commands.Context) -> None:
        """Test bot responsiveness.

        Responds with 'Pong!' to verify the bot is online and responsive.
        """
        _log.debug("Ping command invoked by %s", ctx.author)
        await ctx.send("Pong!")

    @commands.command()
    async def model(self, ctx: commands.Context, provider: str | None = None, model_name: str | None = None) -> None:
        """Set or show preferred LLM model.

        Usage:
            !model - Show current model
            !model <provider> <model> - Set model (e.g., !model claude-cli sonnet)
        """
        coordinator = self.bot.get_cog("ChatCoordinator")
        if not coordinator:
            await ctx.send("ChatCoordinator not loaded")
            return

        # Find WendyBot
        wendy = None
        for bot_instance in coordinator.bots:
            if bot_instance.__class__.__name__ == "WendyBot":
                wendy = bot_instance
                break

        if not wendy:
            await ctx.send("WendyBot not found")
            return

        await wendy.handle_model_command(ctx, provider, model_name)

    @commands.command()
    async def context(self, ctx: commands.Context) -> None:
        """Show Wendy's session stats for this channel.

        Displays token usage, message count, and session info.
        """
        coordinator = self.bot.get_cog("ChatCoordinator")
        if not coordinator:
            await ctx.send("ChatCoordinator not loaded")
            return

        wendy = None
        for bot_instance in coordinator.bots:
            if bot_instance.__class__.__name__ == "WendyBot":
                wendy = bot_instance
                break

        if not wendy:
            await ctx.send("WendyBot not found")
            return

        await wendy.handle_context_command(ctx)

    @commands.command()
    async def wreset(self, ctx: commands.Context) -> None:
        """Reset Wendy's session for this channel.

        Starts a fresh conversation session, clearing accumulated context.
        """
        coordinator = self.bot.get_cog("ChatCoordinator")
        if not coordinator:
            await ctx.send("ChatCoordinator not loaded")
            return

        wendy = None
        for bot_instance in coordinator.bots:
            if bot_instance.__class__.__name__ == "WendyBot":
                wendy = bot_instance
                break

        if not wendy:
            await ctx.send("WendyBot not found")
            return

        await wendy.handle_reset_session_command(ctx)

    @commands.command()
    async def tokens(self, ctx: commands.Context) -> None:
        """Show token leaderboard."""
        from hollingsbot.prompt_db import get_token_leaderboard, get_user_token_balance

        leaderboard = get_token_leaderboard(10)

        if not leaderboard:
            await ctx.send("No tokens have been given yet!")
            return

        lines = ["**Token Leaderboard**"]
        for rank, (user_id, tokens) in enumerate(leaderboard, 1):
            lines.append(f"**{rank}.** <@{user_id}> - {tokens} token(s)")

        # Show caller's rank if not in top 10
        caller_id = ctx.author.id
        caller_in_top = any(uid == caller_id for uid, _ in leaderboard)
        if not caller_in_top:
            balance = get_user_token_balance(caller_id)
            lines.append(f"\nYou have **{balance}** token(s).")

        import discord
        await ctx.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())

    @commands.command()
    async def repair(self, ctx: commands.Context) -> None:
        """Emergency repair for when Wendy stops responding.

        Runs Claude Code to investigate and fix critical issues.
        Only for crashes/failures - not for minor bugs.
        """
        from hollingsbot.tasks import repair_wendy

        await ctx.send("Starting emergency repair... (this may take a few minutes)")
        _log.info("Repair command invoked by %s for channel %d", ctx.author, ctx.channel.id)

        try:
            # Run the repair task
            result = repair_wendy.delay(ctx.channel.id)
            # Wait for result with timeout
            task_result = result.get(timeout=360)  # 6 min timeout

            if task_result.get("success"):
                await ctx.send("Repair completed. Check Wendy's message for details.")
            else:
                error = task_result.get("error", "Unknown error")
                await ctx.send(f"Repair failed: {error}")

        except Exception as e:
            _log.exception("Repair command failed")
            await ctx.send(f"Repair failed: {e}")

    @commands.command(name="help")
    async def help_cmd(self, ctx: commands.Context) -> None:
        """Display comprehensive bot help documentation.

        Shows available features, commands, and usage examples across all cogs.
        The message is automatically truncated if it exceeds Discord's length limit.
        """
        _log.debug("Help command invoked by %s in channel %s", ctx.author, ctx.channel)
        help_text = self._build_help_text()
        truncated_text = self._truncate_for_discord(help_text)
        await ctx.send(truncated_text)

    def _build_help_text(self) -> str:
        """Build the complete help message text.

        Constructs a comprehensive help message documenting all bot features,
        organized by category (image generation, LLM chat, admin, etc.).

        Returns:
            The complete help message as a formatted string.
        """
        return (
            "**Hollingsbot Help**\n"
            "Mention the bot to run commands anywhere (e.g., `@Bot help`).\n\n"
            "Image generation\n"
            "- `! prompt` quick image.\n"
            "- `$ prompt` higher quality; `$$ prompt` premium.\n"
            "- `^ prompt` SVG generator.\n"
            "- `edit: ...` reply to a message with an image (or attach one) to edit; the bot replies to your prompt message.\n"
            "- Tips: `{123}` sets seed; `<a, b, c>` expands to multiple prompts.\n"
            "- `!models` list available image generators.\n\n"
            "GIF from reply chain\n"
            "- Reply `gif` to any message to build a GIF from all images across the whole reply chain (root → leaf). Shows a thinking emoji while working.\n\n"
            "Chat with LLMs\n"
            "- Type normally; the bot replies with context and supports images.\n"
            "- `!models` list available chat models.\n"
            "- `!model <provider/model>` set your preferred model.\n"
            "- `!system` show; `!system <text>` set; `!system reset` clear your system prompt.\n"
            "- Long replies auto-split; large code blocks may be attached; SVG blocks are rendered.\n\n"
            "GPT‑2 channel\n"
            "- In the GPT‑2 channel, the bot replies to any message with a lightweight model.\n\n"
            "Admin\n"
            "- `!reset` restart the project containers (the bot may go offline briefly).\n\n"
            "Other\n"
            "- `ping` returns `Pong!`.\n"
            "- `!tokens` show token leaderboard.\n"
            "- If a starboard is enabled, reacting to a bot message can repost it there.\n"
        )

    def _truncate_for_discord(self, text: str) -> str:
        """Truncate text to fit within Discord's message length limit.

        Args:
            text: The text to truncate.

        Returns:
            The text truncated to MAX_HELP_MESSAGE_LENGTH if necessary.
        """
        if len(text) <= MAX_HELP_MESSAGE_LENGTH:
            return text

        _log.warning(
            "Help text (%d chars) exceeds limit (%d chars), truncating",
            len(text),
            MAX_HELP_MESSAGE_LENGTH,
        )
        return text[:MAX_HELP_MESSAGE_LENGTH]


async def setup(bot: commands.Bot) -> None:
    """Load the General cog into the bot.

    Args:
        bot: The Discord bot instance to add the cog to.
    """
    await bot.add_cog(General(bot))
