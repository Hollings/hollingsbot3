from __future__ import annotations

import os
from typing import Dict, Set

import aiohttp
import discord
from discord.ext import commands, tasks


class PRManager(commands.Cog):
    """Manage pull requests via Discord reactions."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        channel_id = os.getenv("PR_CHANNEL_ID")
        self.channel_id = int(channel_id) if channel_id else None
        self.repo = os.getenv("GITHUB_REPOSITORY")
        self.token = os.getenv("GITHUB_TOKEN")
        self.seen_prs: Set[int] = set()
        self.message_pr: Dict[int, int] = {}
        if self.channel_id and self.repo and self.token:
            self.poll_prs.start()

    async def cog_unload(self) -> None:
        if self.poll_prs.is_running():
            self.poll_prs.cancel()

    async def _get_channel(self) -> discord.abc.Messageable | None:
        if self.channel_id is None:
            return None
        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(self.channel_id)
            except discord.HTTPException:
                return None
        return channel

    @tasks.loop(minutes=1)
    async def poll_prs(self) -> None:
        channel = await self._get_channel()
        if channel is None or not self.repo or not self.token:
            return
        url = f"https://api.github.com/repos/{self.repo}/pulls?state=open"
        headers = {"Authorization": f"token {self.token}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()
        except Exception:
            return

        for pr in data:
            number = pr.get("number")
            if number in self.seen_prs:
                continue
            self.seen_prs.add(number)
            title = pr.get("title")
            link = pr.get("html_url")
            msg = await channel.send(f"PR #{number}: {title}\n{link}")
            self.message_pr[msg.id] = number
            await msg.add_reaction("\N{WHITE HEAVY CHECK MARK}")
            await msg.add_reaction("\N{CROSS MARK}")

    async def _api_request(self, method: str, url: str, **kwargs) -> int:
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"token {self.token}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, **kwargs) as resp:
                return resp.status

    async def merge_pr(self, number: int) -> None:
        url = f"https://api.github.com/repos/{self.repo}/pulls/{number}/merge"
        await self._api_request("PUT", url)

    async def close_pr(self, number: int) -> None:
        url = f"https://api.github.com/repos/{self.repo}/pulls/{number}"
        await self._api_request("PATCH", url, json={"state": "closed"})

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        if user.bot:
            return
        pr_number = self.message_pr.get(reaction.message.id)
        if pr_number is None:
            return
        if reaction.emoji == "\N{WHITE HEAVY CHECK MARK}":
            await self.merge_pr(pr_number)
            await reaction.message.channel.send(f"PR #{pr_number} merged.")
            self.message_pr.pop(reaction.message.id, None)
        elif reaction.emoji == "\N{CROSS MARK}":
            await self.close_pr(pr_number)
            await reaction.message.channel.send(f"PR #{pr_number} closed.")
            self.message_pr.pop(reaction.message.id, None)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PRManager(bot))
