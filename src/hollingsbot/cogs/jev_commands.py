"""`!jev`: what Jev knows - the words it was born with and the ones it has learned from you."""

from __future__ import annotations

import asyncio

from discord.ext import commands

from hollingsbot.cogs.chat_bots.jev_bot import JevBotSettings
from hollingsbot.jev.lexicon import Lexicon, LexiconSummary
from hollingsbot.jev.vocab import load_vocab


def describe(summary: LexiconSummary, *, name: str, born: int, reach: int) -> str:
    """The `!jev` message (pure, for tests). ``reach``: learned words usable at once."""
    if not summary.total:
        return (
            f"**{name}** was born knowing {born} words and hasn't learned any yet. "
            f"Talk to {name}: every word you say, {name} remembers."
        )
    lines = [f"**{name}** was born knowing {born} words and has learned **{summary.total}** more from you."]
    if reach < summary.total:
        lines.append(f"{name} can reach {reach} learned words at a time: the ones heard most, most recently.")
    lines += [
        "Newest: " + ", ".join(f"*{word}* ({who})" for word, who in summary.newest),
        "Most heard: " + ", ".join(f"*{word}* x{uses}" for word, uses in summary.favorites),
    ]
    return "\n".join(lines)


class JevCommands(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.command(name="jev")
    async def jev(self, ctx: commands.Context) -> None:
        """Show what Jev knows: how many words it has learned, the newest, and who taught them."""
        settings = JevBotSettings.from_env()
        born_words = load_vocab()[: settings.writer.vocab_size]
        summary = await asyncio.to_thread(Lexicon().summary, exclude=born_words)
        reach = settings.reachable_learned(summary.total)
        await ctx.send(describe(summary, name=settings.name, born=len(born_words), reach=reach))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(JevCommands(bot))
