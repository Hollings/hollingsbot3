"""Yeah streak tracker - tracks consecutive 'yeah' messages per user."""

from __future__ import annotations

import contextlib
import logging
import os
import random
import sqlite3
from pathlib import Path

import discord
from discord.ext import commands

_LOG = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("/data/hollingsbot.db")
ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Streak announcement messages
STREAK_MESSAGES = [
    "🚨 YEAH STREAK ALERT 🚨 {mention} just hit {streak}x !!! 🔥🔥🔥",
    "╔══════════════════╗\n   {streak}x YEAH STREAK\n      {mention}\n╚══════════════════╝",
    "⚡ {streak}x YEAH ⚡ {mention} IS IN THE ZONE ⚡",
    "👏 YEAH 👏 STREAK 👏 {streak} 👏 {mention} 👏",
    "|| {streak}x YEAH || {mention} CANNOT BE STOPPED ||",
    "w h o a . . . {mention} just said yeah {streak} times in a row",
    "✨ {mention} ✨ {streak}x yeah streak ✨ incredible ✨",
    "🏆 CERTIFIED YEAH MOMENT 🏆 {mention} x{streak}",
    ">>> {mention} YEAH STREAK LEVEL {streak} UNLOCKED <<<",
    "DING DING DING 🛎️ {streak}x YEAH STREAK 🛎️ {mention} WINS",
]


class YeahStreakCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.db_path = Path(os.getenv("PROMPT_DB_PATH", str(DEFAULT_DB_PATH)))
        self._conn: sqlite3.Connection | None = None
        # In-memory tracking of current streaks: {(channel_id, user_id): current_streak}
        self.current_streaks: dict[tuple[int, int], int] = {}
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS yeah_streaks (
                    user_id INTEGER PRIMARY KEY,
                    high_score INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def cog_unload(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    def _get_high_score(self, user_id: int) -> int:
        conn = self._get_connection()
        row = conn.execute("SELECT high_score FROM yeah_streaks WHERE user_id = ?", (user_id,)).fetchone()
        return row[0] if row else 0

    def _update_high_score(self, user_id: int, score: int) -> bool:
        """Update high score if new score is higher. Returns True if new record."""
        conn = self._get_connection()
        current = self._get_high_score(user_id)
        if score > current:
            conn.execute(
                "INSERT OR REPLACE INTO yeah_streaks (user_id, high_score) VALUES (?, ?)",
                (user_id, score),
            )
            conn.commit()
            return True
        return False

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        # Skip DMs
        if not message.guild:
            return

        # Ignore own messages
        if message.author.id == self.bot.user.id:
            return

        key = (message.channel.id, message.author.id)
        has_yeah = "yeah" in message.content.lower()

        if has_yeah:
            # Increment streak
            self.current_streaks[key] = self.current_streaks.get(key, 0) + 1
        else:
            # Check if they had a streak worth announcing (4+)
            streak = self.current_streaks.get(key, 0)
            if streak >= 4:
                await self._announce_streak_end(message.channel, message.author, streak)
            # Reset streak
            self.current_streaks[key] = 0

    async def _announce_streak_end(self, channel: discord.abc.Messageable, user: discord.User, streak: int) -> None:
        """Announce streak end with obnoxious message and image."""
        is_new_record = self._update_high_score(user.id, streak)

        msg = random.choice(STREAK_MESSAGES).format(streak=streak, mention=user.mention)
        if is_new_record:
            msg += " NEW PERSONAL BEST!!!"

        image_choices = [ASSETS_DIR / "theyeahzone.png", ASSETS_DIR / "yeahzone2.png"]
        image_path = random.choice(image_choices)
        try:
            if image_path.exists():
                await channel.send(content=msg, file=discord.File(image_path))
            else:
                await channel.send(content=msg)
        except Exception:
            _LOG.exception("Failed to send yeah streak message")

    @commands.command(name="yeahscore")
    async def yeahscore_cmd(self, ctx: commands.Context) -> None:
        """Show your yeah streak high score."""
        score = self._get_high_score(ctx.author.id)
        await ctx.send(f"{ctx.author.mention} your yeah streak high score is {score}!")

    @commands.command(name="yeahleaders")
    async def yeahleaders_cmd(self, ctx: commands.Context) -> None:
        """Show yeah streak leaderboard."""
        conn = self._get_connection()
        rows = conn.execute("SELECT user_id, high_score FROM yeah_streaks ORDER BY high_score DESC LIMIT 10").fetchall()
        if not rows:
            await ctx.send("No yeah streaks recorded yet!")
            return
        lines = ["**Yeah Streak Leaderboard**"]
        for i, (uid, score) in enumerate(rows, 1):
            lines.append(f"{i}. <@{uid}> - {score}x streak")
        await ctx.send("\n".join(lines))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(YeahStreakCog(bot))
