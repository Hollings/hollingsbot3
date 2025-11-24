"""Discord commands for temp bot management."""

import logging

import discord
from discord.ext import commands

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
        """
        manager = self._get_temp_bot_manager()
        if not manager:
            await ctx.send("Temp bot system not available")
            return

        await manager.handle_spawn_command(ctx, reply_count, initial_prompt=initial_prompt)

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
            await ctx.message.add_reaction("❌")
            return

        channel_id = ctx.channel.id

        # Clear the history for this channel
        if channel_id in coordinator.channel_histories:
            coordinator.channel_histories[channel_id].clear()
            _LOG.info(f"Cleared conversation history for channel {channel_id}")
            await ctx.message.add_reaction("✅")
        else:
            await ctx.message.add_reaction("❌")


async def setup(bot: commands.Bot) -> None:
    """Load the temp bot commands cog."""
    await bot.add_cog(TempBotCommands(bot))
