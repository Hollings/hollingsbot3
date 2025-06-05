from __future__ import annotations

import os
import logging
from typing import Dict, Set

import aiohttp
import discord
from discord.ext import commands, tasks


class PRManager(commands.Cog):
    """Manage pull requests via Discord reactions."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.log = logging.getLogger(__name__)
        channel_id = os.getenv("PR_CHANNEL_ID")
        self.channel_id = int(channel_id) if channel_id else None
        self.repo = os.getenv("GITHUB_REPOSITORY")
        self.token = os.getenv("GITHUB_TOKEN")
        self.seen_prs: Set[int] = set()
        self.message_pr: Dict[int, int] = {}
        self.pr_info: Dict[int, Dict[str, str]] = {}
        self.log.info(
            "PRManager initialized (channel=%s, repo=%s)", self.channel_id, self.repo
        )
        if self.channel_id and self.repo and self.token:
            self.log.info("Starting PR polling loop")
            self.poll_prs.start()

    async def cog_unload(self) -> None:
        if self.poll_prs.is_running():
            self.poll_prs.cancel()
            self.log.info("Stopped PR polling loop")

    async def _get_channel(self) -> discord.abc.Messageable | None:
        if self.channel_id is None:
            return None
        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(self.channel_id)
            except discord.HTTPException:
                self.log.warning("Failed to fetch PR channel %s", self.channel_id)
                return None
        return channel

    @tasks.loop(minutes=1)
    async def poll_prs(self) -> None:
        self.log.debug("Polling for open PRs")
        channel = await self._get_channel()
        if channel is None or not self.repo or not self.token:
            self.log.debug("Missing channel, repo, or token; skipping poll")
            return
        url = f"https://api.github.com/repos/{self.repo}/pulls?state=open"
        headers = {"Authorization": f"token {self.token}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        self.log.error("GitHub API returned status %s", resp.status)
                        return
                    data = await resp.json()
        except Exception as exc:  # noqa: BLE001
            self.log.error("Failed to poll PRs: %s", exc)
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
            self.log.info("New PR #%s detected: %s", number, title)
            msg = await channel.send(
                f"PR #{number} by {author}: {title}\n{body}\n{link}"
            )
            self.message_pr[msg.id] = number
            self.pr_info[number] = {"title": title or "", "body": body or ""}
            await msg.add_reaction("\N{WHITE HEAVY CHECK MARK}")
            await msg.add_reaction("\N{CROSS MARK}")

    async def _api_request(self, method: str, url: str, **kwargs) -> int:
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"token {self.token}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, **kwargs) as resp:
                self.log.debug("%s %s -> %s", method, url, resp.status)
                return resp.status

    async def merge_pr(self, number: int) -> bool:
        url = f"https://api.github.com/repos/{self.repo}/pulls/{number}/merge"
        self.log.info("Merging PR #%s", number)
        status = await self._api_request(
            "PUT", url, json={"merge_method": "squash"}
        )
        if status not in (200, 201):
            self.log.error("Failed to merge PR #%s (status %s)", number, status)
            return False
        return True

    async def close_pr(self, number: int) -> None:
        url = f"https://api.github.com/repos/{self.repo}/pulls/{number}"
        self.log.info("Closing PR #%s", number)
        await self._api_request("PATCH", url, json={"state": "closed"})



    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        if user.bot:
            return
        pr_number = self.message_pr.get(reaction.message.id)
        if pr_number is None:
            return
        if reaction.emoji == "\N{WHITE HEAVY CHECK MARK}":
            self.log.info("Merge reaction received for PR #%s by %s", pr_number, user.name)
            merged = await self.merge_pr(pr_number)
            if merged:
                await reaction.message.channel.send(f"PR #{pr_number} merged.")
            else:
                await reaction.message.channel.send(
                    f"Failed to merge PR #{pr_number}."
                )
            self.message_pr.pop(reaction.message.id, None)
            self.pr_info.pop(pr_number, None)
        elif reaction.emoji == "\N{CROSS MARK}":
            self.log.info("Close reaction received for PR #%s by %s", pr_number, user.name)
            await self.close_pr(pr_number)
            await reaction.message.channel.send(f"PR #{pr_number} closed.")
            self.message_pr.pop(reaction.message.id, None)
            self.pr_info.pop(pr_number, None)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PRManager(bot))
