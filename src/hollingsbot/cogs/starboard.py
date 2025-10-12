from __future__ import annotations

import json
import logging
import os
from collections import deque
from contextlib import suppress

import discord
from discord.abc import Messageable
from discord.ext import commands

from hollingsbot.prompt_db import log_starboard_post

__all__ = ["Starboard"]

_LOG = logging.getLogger(__name__)


def _parse_channel_ids(raw: str | None) -> set[int]:
    if not raw:
        return set()
    ids: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            ids.add(int(token))
    return ids


def _parse_channel_id(raw: str | None) -> int | None:
    raw = (raw or "").strip()
    if raw.isdigit():
        return int(raw)
    return None


class Starboard(commands.Cog):
    """Mirror reacted bot messages into a starboard channel."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.starboard_channel_id = _parse_channel_id(os.getenv("STARBOARD_CHANNEL_ID"))
        self.ignore_channel_ids = _parse_channel_ids(os.getenv("STARBOARD_IGNORE_CHANNELS"))
        self.whitelist_channel_ids = _parse_channel_ids(
            os.getenv("STARBOARD_WHITELIST_CHANNEL_IDS")
        )
        self._seen_message_ids: deque[int] = deque(maxlen=2048)

        if self.starboard_channel_id is None:
            _LOG.warning("Starboard cog loaded without STARBOARD_CHANNEL_ID; cog idle.")

    async def _resolve_channel(self, channel_id: int) -> discord.abc.GuildChannel | None:
        channel = self.bot.get_channel(channel_id)
        if channel is not None:
            return channel
        with suppress(discord.NotFound, discord.Forbidden, discord.HTTPException):
            return await self.bot.fetch_channel(channel_id)
        return None

    @commands.Cog.listener("on_raw_reaction_add")
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if self.starboard_channel_id is None:
            return
        if payload.guild_id is None:  # Ignore DMs
            return
        if payload.user_id == getattr(self.bot.user, "id", None):
            return

        member = payload.member
        if member is None:
            guild = self.bot.get_guild(payload.guild_id)
            if guild is not None:
                member = guild.get_member(payload.user_id)
        if member is None or member.bot:
            return

        origin_channel = await self._resolve_channel(payload.channel_id)
        if origin_channel is None:
            return
        if origin_channel.id in self.ignore_channel_ids:
            return
        if self.whitelist_channel_ids and origin_channel.id not in self.whitelist_channel_ids:
            return

        try:
            message = await origin_channel.fetch_message(payload.message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return

        if not message.author.bot:
            return
        if message.id in self._seen_message_ids:
            return

        starboard_channel = await self._resolve_channel(self.starboard_channel_id)
        if starboard_channel is None:
            return
        if not isinstance(starboard_channel, Messageable):
            _LOG.warning(
                "Starboard channel %s is not messageable; ignoring repost.",
                self.starboard_channel_id,
            )
            return
        if getattr(starboard_channel.guild, "id", None) != payload.guild_id:
            return

        content = self._format_forward_content(message)
        try:
            sent_message = await starboard_channel.send(content)
        except (discord.Forbidden, discord.HTTPException):
            _LOG.exception("Failed to repost message %s to starboard.", message.id)
            return

        self._seen_message_ids.append(message.id)

        attachments_serialized = self._serialize_attachments(message)
        reaction = None
        with suppress(Exception):
            reaction = str(payload.emoji)

        author_name = getattr(message.author, "display_name", message.author.name)
        try:
            log_starboard_post(
                guild_id=payload.guild_id,
                source_channel_id=message.channel.id,
                starboard_channel_id=starboard_channel.id,
                original_message_id=message.id,
                starboard_message_id=sent_message.id,
                reactor_user_id=payload.user_id,
                reaction_emoji=reaction,
                original_author_id=message.author.id,
                original_author_name=author_name,
                jump_url=message.jump_url,
                content=message.content,
                attachment_urls=attachments_serialized,
            )
        except Exception:  # noqa: BLE001
            _LOG.exception("Failed to record starboard entry for message %s", message.id)

    def _format_forward_content(self, message: discord.Message) -> str:
        parts: list[str] = [message.jump_url]
        author_name = getattr(message.author, "display_name", message.author.name)
        if message.content:
            preview = message.content.strip()
            if len(preview) > 800:
                preview = f"{preview[:797]}…"
            parts.append(f"> **{author_name}**: {preview}")
        for attachment in message.attachments:
            parts.append(attachment.url)
        body = "\n".join(parts)
        if len(body) > 1900:
            body = f"{body[:1897]}…"
        return body

    def _serialize_attachments(self, message: discord.Message) -> str:
        payload: list[dict[str, object]] = []

        def _add_entry(*, filename: str | None, url: str | None, proxy_url: str | None, content_type: str | None, width: int | None = None, height: int | None = None) -> None:
            if not url and not proxy_url:
                return
            payload.append(
                {
                    "filename": filename,
                    "url": url,
                    "proxy_url": proxy_url,
                    "content_type": content_type,
                    "width": width,
                    "height": height,
                }
            )

        for attachment in message.attachments:
            proxy_url = getattr(attachment, "proxy_url", None)
            _add_entry(
                filename=attachment.filename,
                url=attachment.url,
                proxy_url=proxy_url,
                content_type=attachment.content_type,
                width=attachment.width,
                height=attachment.height,
            )

        # Some archived messages only retain embed thumbnails/images.
        for embed in message.embeds:
            img = embed.image or embed.thumbnail
            if img and getattr(img, "url", None):
                _add_entry(
                    filename=(embed.title or "embed"),
                    url=getattr(embed, "url", None) or img.url,
                    proxy_url=img.url,
                    content_type="image/embed",
                    width=getattr(img, "width", None),
                    height=getattr(img, "height", None),
                )

        return json.dumps(payload)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Starboard(bot))
