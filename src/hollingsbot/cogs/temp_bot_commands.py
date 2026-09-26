"""Discord commands for temp bot management."""

import logging
from collections.abc import Callable
from typing import Any

import discord
from discord.ext import commands

from hollingsbot.cogs.chat_bots.jev_bot import SPAWN_REPLIES_DEFAULT
from hollingsbot.prompt_db import (
    get_historical_temp_bots,
    get_temp_bot_by_name,
    get_temp_bots_for_channel,
    search_temp_bots,
)

_LOG = logging.getLogger(__name__)

SPAWN_USAGE = "Usage: `!spawn <replies> <prompt>` for a temp bot, or `!spawn jev [replies]` (or `jev2`) for a Jev."


def jev_spawn(first: str, rest: str, find: Callable[[str], Any], default: int) -> tuple[Any, int] | None:
    """`!spawn jev [N]` or `!spawn N jev` (any Jev's name) -> (that Jev, N), N defaulting to ``default``.

    ``find`` looks a Jev up by a word of the command (None if no Jev is called
    that). Returns None if the spawn isn't a Jev's; raises ValueError when N
    isn't a number.
    """
    rest = rest.strip()
    if (jev := find(first)) is not None:
        return jev, int(rest) if rest else default
    if (jev := find(rest)) is not None:
        return jev, int(first)
    return None


class TempBotCommands(commands.Cog):
    """Commands for spawning and managing temporary LLM bots (and the Jevs, which `!spawn jev` brings over)."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        _LOG.info("TempBotCommands cog initialized")

    def _get_chat_bots(self, class_name: str) -> list:
        """The bots registered with the chat coordinator with this class name, in registration order."""
        coordinator = self.bot.get_cog("ChatCoordinator")
        if not coordinator:
            return []
        return [b for b in coordinator.bots if b.__class__.__name__ == class_name]

    def _get_temp_bot_manager(self):
        """Get the TempBotManager instance from the coordinator."""
        managers = self._get_chat_bots("TempBotManager")
        return managers[0] if managers else None

    def _find_jev(self, word: str):
        """The Jev called ``word`` (its name, any case; plain "jev" is always the first Jev)."""
        jevs = self._get_chat_bots("JevBot")
        word = word.strip().lower()
        for jev in jevs:
            if jev.settings.name.lower() == word:
                return jev
        return jevs[0] if jevs and word == "jev" else None

    @commands.command(name="spawn")
    async def spawn_command(
        self,
        ctx: commands.Context,
        reply_count: str,
        *,
        initial_prompt: str = "",
    ) -> None:
        """Spawn a temporary LLM bot that responds with a limited number of messages.

        The bot will immediately respond to the initial prompt, then can respond to
        subsequent messages (including from other bots) until it runs out of replies.

        Usage: !spawn <reply_count> <initial_prompt>
        Example: !spawn 10 convince everyone to cheer up

        Options:
        - Reply to a message with !spawn to include that message as initial context
        - Use -context flag to include previous 5 messages: !spawn 10 -context <prompt>

        `!spawn jev [reply_count]` brings Jev (the word-by-word decision model) instead,
        and `!spawn jev2` a second one: each answers every message here, other bots'
        included, until it has posted that many replies.
        """
        try:
            jev_and_replies = jev_spawn(reply_count, initial_prompt, self._find_jev, default=SPAWN_REPLIES_DEFAULT)
        except ValueError:
            await ctx.reply(SPAWN_USAGE, mention_author=False)
            return
        if jev_and_replies is not None:
            jev, replies = jev_and_replies
            await jev.spawn(ctx, replies)
            return

        try:
            count = int(reply_count)
        except ValueError:
            await ctx.reply(SPAWN_USAGE, mention_author=False)
            return

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
            count,
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
        `!despawn jev` (or `jev2`) sends away a spawned Jev; `!despawn all` includes them.
        """
        if await self._despawn_jevs(ctx, name):
            return

        manager = self._get_temp_bot_manager()
        if not manager:
            await ctx.send("Temp bot system not available")
            return

        await manager.handle_despawn_command(ctx, name)

    async def _despawn_jevs(self, ctx: commands.Context, name: str | None) -> bool:
        """The Jevs' part of `!despawn`; True when that was all there was to do."""
        channel_id = ctx.channel.id
        if name is not None and (jev := self._find_jev(name)) is not None:
            jev_name = jev.settings.name
            if await jev.despawn(ctx.channel):
                await ctx.send(f"Despawned: **{jev_name}**")
            elif channel_id in jev.whitelist_channels:
                await ctx.send(f"**{jev_name}** lives in this channel, so it can't be despawned.")
            else:
                await ctx.send(f"**{jev_name}** isn't spawned in this channel.")
            return True
        visiting = [jev for jev in self._get_chat_bots("JevBot") if channel_id in jev.spawned]
        if not visiting:
            return False
        if name is None:
            await ctx.send(
                "\n".join(
                    f"**{jev.settings.name}** is here for {jev.spawned[channel_id]} more replies "
                    f"(`!despawn {jev.settings.name.lower()}` sends it away)."
                    for jev in visiting
                )
            )
        elif name.lower() == "all":
            for jev in visiting:
                await jev.despawn(ctx.channel)
            await ctx.send("Despawned: " + ", ".join(f"**{jev.settings.name}**" for jev in visiting))
        else:
            return False
        return not get_temp_bots_for_channel(channel_id)

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
            _LOG.info(f"Set clear point for channel {channel_id} at message {ctx.message.id}")

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

        lines.append("\nUse `!recall <count> <name>` to bring one back.")
        await ctx.send("\n".join(lines))


async def setup(bot: commands.Bot) -> None:
    """Load the temp bot commands cog."""
    await bot.add_cog(TempBotCommands(bot))
