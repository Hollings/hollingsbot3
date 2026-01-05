"""Discord commands for temp bot management."""

import logging

import discord
from discord.ext import commands

from hollingsbot.prompt_db import (
    get_historical_temp_bots,
    get_temp_bot_by_name,
    search_temp_bots,
)

_LOG = logging.getLogger(__name__)


class TempBotCommands(commands.Cog):
    """Commands for spawning and managing temporary LLM bots."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        _LOG.info("TempBotCommands cog initialized")

    def _get_temp_bot_manager(self):
        """Get the TempBotManager instance from the coordinator."""
        coordinator = self.bot.get_cog("ChatCoordinator")
        if not coordinator:
            return None

        # Find TempBotManager in registered bots
        for bot_instance in coordinator.bots:
            if bot_instance.__class__.__name__ == "TempBotManager":
                return bot_instance
        return None

    @commands.command(name="spawn")
    async def spawn_command(
        self,
        ctx: commands.Context,
        reply_count: int,
        *,
        initial_prompt: str,
    ) -> None:
        """Spawn a temporary LLM bot that responds with a limited number of messages.

        The bot will immediately respond to the initial prompt, then can respond to
        subsequent messages (including from other bots) until it runs out of replies.

        Usage: !spawn <reply_count> <initial_prompt>
        Example: !spawn 10 convince wendy to cheer up

        Options:
        - Reply to a message with !spawn to include that message as initial context
        - Use -context flag to include previous 5 messages: !spawn 10 -context <prompt>
        """
        manager = self._get_temp_bot_manager()
        if not manager:
            await ctx.send("Temp bot system not available")
            return

        # Parse -context flag (comes after reply_count, before prompt)
        include_context = False
        prompt = initial_prompt
        if prompt.startswith("-context "):
            include_context = True
            prompt = prompt[9:]  # Strip "-context "
        elif prompt.startswith("-context"):
            include_context = True
            prompt = prompt[8:]  # Strip "-context" (no space)

        # Get replied-to message if this is a reply
        reply_message = None
        if ctx.message.reference and ctx.message.reference.message_id:
            try:
                reply_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            except Exception:
                _LOG.warning("Failed to fetch replied-to message for spawn")

        await manager.handle_spawn_command(
            ctx,
            reply_count,
            initial_prompt=prompt.strip(),
            reply_message=reply_message,
            include_context=include_context,
        )

    @commands.command(name="despawn")
    async def despawn_command(self, ctx: commands.Context, name: str | None = None) -> None:
        """Manually remove temporary bots from this channel.

        Usage: !despawn [name]
        If no name is provided, lists active temp bots.
        If name is provided, removes only that specific bot.
        """
        manager = self._get_temp_bot_manager()
        if not manager:
            await ctx.send("Temp bot system not available")
            return

        await manager.handle_despawn_command(ctx, name)

    @commands.command(name="clear")
    async def clear_history_command(self, ctx: commands.Context) -> None:
        """Clear conversation history for this channel.

        This removes all previous messages from the bots' memory,
        allowing you to start fresh. The bots will only see messages
        sent after this command.

        Usage: !clear
        """
        coordinator = self.bot.get_cog("ChatCoordinator")
        if not coordinator:
            await ctx.message.add_reaction("\u274c")  # X mark
            return

        channel_id = ctx.channel.id

        # Clear in-memory history for this channel
        if channel_id in coordinator.channel_histories:
            coordinator.channel_histories[channel_id].clear()

        # Set clear point in database (soft delete for summarization)
        # All summaries and cached messages before this message will be ignored
        if coordinator.summary_cache:
            coordinator.summary_cache.set_clear_point(channel_id, ctx.message.id)
            _LOG.info(
                f"Set clear point for channel {channel_id} at message {ctx.message.id}"
            )

        _LOG.info(f"Cleared conversation history for channel {channel_id}")
        await ctx.message.add_reaction("\u2705")  # Checkmark

    @commands.command(name="recall")
    async def recall_command(
        self,
        ctx: commands.Context,
        reply_count: int,
        *,
        bot_name: str,
    ) -> None:
        """Recall a previously spawned temp bot back into the chat.

        The bot will return with its original name, avatar (if available),
        and personality/purpose.

        Usage: !recall <reply_count> <bot_name>
        Example: !recall 10 Veiled Cipher

        Use !history to see a list of previous temp bots you can recall.
        """
        manager = self._get_temp_bot_manager()
        if not manager:
            await ctx.send("Temp bot system not available")
            return

        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send("This command only works in text channels.")
            return

        # Find the bot by name
        bot_data = get_temp_bot_by_name(bot_name.strip(), channel_id=ctx.channel.id)

        if not bot_data:
            # Try searching
            matches = search_temp_bots(bot_name.strip(), limit=5)
            if matches:
                names = ", ".join(f"**{m['name']}**" for m in matches)
                await ctx.send(f"Bot '{bot_name}' not found. Did you mean: {names}?")
            else:
                await ctx.send(f"No temp bot named '{bot_name}' found. Use `!history` to see available bots.")
            return

        if bot_data.get("is_active"):
            await ctx.send(f"**{bot_data['name']}** is already active in this channel!")
            return

        # Recall the bot using the manager's spawn handler
        await manager.handle_recall_command(
            ctx,
            reply_count,
            bot_data=bot_data,
        )

    @commands.command(name="history")
    async def history_command(self, ctx: commands.Context, query: str | None = None) -> None:
        """Show previously spawned temp bots that can be recalled.

        Usage: !history [search_query]

        Examples:
            !history           - Show recent temp bots from this channel
            !history cipher    - Search for bots with 'cipher' in name/prompt
        """
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send("This command only works in text channels.")
            return

        if query:
            # Search across all channels
            bots = search_temp_bots(query.strip(), limit=10)
            title = f"Temp bots matching '{query}'"
        else:
            # Show recent from this channel
            bots = get_historical_temp_bots(channel_id=ctx.channel.id, limit=10)
            title = "Recent temp bots in this channel"

        if not bots:
            if query:
                await ctx.send(f"No temp bots found matching '{query}'.")
            else:
                await ctx.send("No temp bot history found for this channel.")
            return

        # Build response
        lines = [f"**{title}**\n"]
        for bot in bots:
            name = bot["name"]
            prompt = bot["spawn_prompt"][:60] + "..." if len(bot["spawn_prompt"]) > 60 else bot["spawn_prompt"]
            created = bot.get("created_at", "unknown")[:10] if bot.get("created_at") else "unknown"
            status = "(active)" if bot.get("is_active") else ""
            lines.append(f"- **{name}** {status} - _{prompt}_ ({created})")

        lines.append(f"\nUse `!recall <count> <name>` to bring one back.")
        await ctx.send("\n".join(lines))


async def setup(bot: commands.Bot) -> None:
    """Load the temp bot commands cog."""
    await bot.add_cog(TempBotCommands(bot))
