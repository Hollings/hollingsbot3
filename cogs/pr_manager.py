from __future__ import annotations

import os
from typing import Dict, Set

DEFAULT_WEBHOOK = (
    "https://discord.com/api/webhooks/1380252805494738974/"
    "wif1p4iK8_hZq41JwNCmHRcFj6E9cxyet1fNlTiLt-nN9dc2qyxclJcYZhoUnfRwG5p5"
)

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
        self.webhook_url = os.getenv("WEBHOOK_URL", DEFAULT_WEBHOOK)
        self.seen_prs: Set[int] = set()
        self.message_pr: Dict[int, int] = {}
        self.pr_info: Dict[int, Dict[str, str]] = {}
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
            body = pr.get("body", "")
            author = pr.get("user", {}).get("login", "")
            link = pr.get("html_url")
            msg = await channel.send(f"PR #{number}: {title}\n{link}")
            self.message_pr[msg.id] = number
            self.pr_info[number] = {"title": title or "", "body": body or ""}
            await msg.add_reaction("\N{WHITE HEAVY CHECK MARK}")
            await msg.add_reaction("\N{CROSS MARK}")
            await self._notify_open(number, title or "", body or "", author)

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

    async def _notify_open(self, number: int, title: str, body: str, author: str) -> None:
        if not self.webhook_url:
            return
        msg = (
            f"New PR #{number} opened by {author}\n"
            f"{title}\n"
            f"{body}\n"
            f"Original query: {title}"
        )
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json={"content": msg})

    async def _notify_merge(self, number: int, merged_by: discord.User) -> None:
        if not self.webhook_url:
            return
        info = self.pr_info.get(number)
        if info is None:
            url = f"https://api.github.com/repos/{self.repo}/pulls/{number}"
            headers = {"Authorization": f"token {self.token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        info = await resp.json()
                    else:
                        info = {}
        title = info.get("title", "")
        body = info.get("body", "")
        msg = (
            f"PR #{number} merged by {merged_by.name}\n"
            f"{title}\n"
            f"{body}\n"
            f"Original query: {title}"
        )
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json={"content": msg})

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
            await self._notify_merge(pr_number, user)
            self.message_pr.pop(reaction.message.id, None)
            self.pr_info.pop(pr_number, None)
        elif reaction.emoji == "\N{CROSS MARK}":
            await self.close_pr(pr_number)
            await reaction.message.channel.send(f"PR #{pr_number} closed.")
            self.message_pr.pop(reaction.message.id, None)
            self.pr_info.pop(pr_number, None)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PRManager(bot))
