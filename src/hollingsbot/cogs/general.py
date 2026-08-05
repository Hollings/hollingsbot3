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

    @commands.command(name="help")
    async def help_cmd(self, ctx: commands.Context, *, command_name: str | None = None) -> None:
        """Display bot help documentation.

        `!help` shows the overview; `!help <command>` shows that command's
        usage and full description.
        """
        _log.debug("Help command invoked by %s in channel %s", ctx.author, ctx.channel)

        if command_name:
            cmd = self.bot.get_command(command_name.lstrip("!").strip())
            if cmd is None:
                await ctx.send(f"No command named `{command_name}`. Try `!help` for the overview.")
                return
            sig = f"!{cmd.qualified_name} {cmd.signature}".strip()
            await ctx.send(f"**{sig}**\n{cmd.help or 'No description available.'}")
            return

        from hollingsbot.cogs.chat_utils import chunk_message

        for chunk in chunk_message(self._build_help_text(), MAX_HELP_MESSAGE_LENGTH):
            await ctx.send(chunk)

    def _build_help_text(self) -> str:
        """Build the complete help message text.

        Constructs a comprehensive help message documenting all bot features,
        organized by category (image generation, LLM chat, admin, etc.).

        Returns:
            The complete help message as a formatted string.
        """
        return (
            "**Hollingsbot Help**\n"
            "Mention the bot to run commands anywhere (e.g., `@Bot help`). "
            "Use `!help <command>` for details on any command.\n\n"
            "Image generation\n"
            "- `! prompt` quick image.\n"
            "- `$ prompt` higher quality; `$$ prompt` premium.\n"
            "- `^ prompt` SVG generator.\n"
            "- `edit: ...` reply to a message with an image (or attach one) to edit; the bot replies to your prompt message.\n"
            "- Tips: `{123}` sets seed; `<a, b, c>` expands to multiple prompts.\n"
            "- `!models` list available image generators (image channels only).\n"
            "- `!usage` your budget and credits; `!redeem` trade tokens for credits; `!balance` full status.\n\n"
            "GIF from reply chain\n"
            "- Reply `gif` to any message to build a GIF from all images across the whole reply chain (root → leaf). Shows a thinking emoji while working.\n\n"
            "Chat with LLMs\n"
            "- Type normally; the bot replies with context and supports images.\n"
            "- Long replies auto-split; SVG blocks are rendered as images.\n\n"
            "Temp bots\n"
            "- `!spawn <replies> <prompt>` spawn a temporary bot with a personality (e.g. `!spawn 10 a grumpy pirate`).\n"
            "- `!despawn [name|all]` list or remove temp bots.\n"
            "- `!recall <replies> <name>` bring back a previous temp bot.\n"
            "- `!history [query]` list or search past temp bots; `!clear` clear chat history.\n\n"
            "Admin\n"
            "- `!reset` restart the project containers; `!grant`, `!set_price`, `!set_budget`.\n\n"
            "Other\n"
            "- `!ping` returns `Pong!`.\n"
            "- `!tokens` show token leaderboard; `!yeahscore` / `!yeahleaders` yeah-streak stats.\n"
            "- If a starboard is enabled, reacting to a bot message can repost it there.\n"
        )


async def setup(bot: commands.Bot) -> None:
    """Load the General cog into the bot.

    Args:
        bot: The Discord bot instance to add the cog to.
    """
    await bot.add_cog(General(bot))
