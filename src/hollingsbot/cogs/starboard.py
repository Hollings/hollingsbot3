"""Starboard cog for mirroring reacted bot messages to a dedicated channel."""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from contextlib import suppress
from typing import TYPE_CHECKING

import discord
from discord.abc import Messageable
from discord.ext import commands

from hollingsbot.prompt_db import log_starboard_post

if TYPE_CHECKING:
    from discord.abc import GuildChannel

__all__ = ["Starboard"]

_LOG = logging.getLogger(__name__)

# Constants
MAX_CONTENT_PREVIEW_LENGTH = 800
MAX_FORWARD_MESSAGE_LENGTH = 1900
SEEN_MESSAGE_CACHE_SIZE = 2048


def _parse_channel_ids(raw: str | None) -> set[int]:
    """Parse comma-separated channel IDs from environment variable.

    Args:
        raw: Comma-separated string of channel IDs

    Returns:
        Set of parsed channel IDs (empty set if raw is None or invalid)
    """
    if not raw:
        return set()
    ids: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            ids.add(int(token))
    return ids


def _parse_channel_id(raw: str | None) -> int | None:
    """Parse a single channel ID from environment variable.

    Args:
        raw: String representation of a channel ID

    Returns:
        Parsed channel ID or None if invalid
    """
    raw = (raw or "").strip()
    if raw.isdigit():
        return int(raw)
    return None


class Starboard(commands.Cog):
    """Mirror reacted bot messages into a starboard channel.

    This cog listens for reactions on bot messages and reposts them to a designated
    starboard channel. It supports channel whitelisting/blacklisting and maintains
    a cache of seen messages to prevent duplicate posts.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the Starboard cog.

        Args:
            bot: The Discord bot instance
        """
        self.bot = bot
        self.starboard_channel_id = _parse_channel_id(os.getenv("STARBOARD_CHANNEL_ID"))
        self.ignore_channel_ids = _parse_channel_ids(os.getenv("STARBOARD_IGNORE_CHANNELS"))
        self.whitelist_channel_ids = _parse_channel_ids(os.getenv("STARBOARD_WHITELIST_CHANNEL_IDS"))
        self._seen_message_ids: deque[int] = deque(maxlen=SEEN_MESSAGE_CACHE_SIZE)

        if self.starboard_channel_id is None:
            _LOG.warning("Starboard cog loaded without STARBOARD_CHANNEL_ID; cog idle.")

    async def _resolve_channel(self, channel_id: int) -> GuildChannel | None:
        """Resolve a channel by ID, attempting cache first then fetch.

        Args:
            channel_id: The Discord channel ID to resolve

        Returns:
            The resolved channel or None if not found/accessible
        """
        channel = self.bot.get_channel(channel_id)
        if channel is not None:
            return channel
        with suppress(discord.NotFound, discord.Forbidden, discord.HTTPException):
            return await self.bot.fetch_channel(channel_id)
        return None

    def _is_valid_reactor(self, payload: discord.RawReactionActionEvent) -> bool:
        """Check if the reactor is a valid non-bot user.

        Args:
            payload: The reaction event payload

        Returns:
            True if reactor is a valid human user, False otherwise
        """
        if payload.user_id == getattr(self.bot.user, "id", None):
            return False

        member = payload.member
        if member is None:
            guild = self.bot.get_guild(payload.guild_id)
            if guild is not None:
                member = guild.get_member(payload.user_id)

        return member is not None and not member.bot

    def _is_channel_eligible(self, channel: GuildChannel) -> bool:
        """Check if a channel is eligible for starboard posts.

        Args:
            channel: The channel to check

        Returns:
            True if channel is eligible (not ignored and whitelisted if applicable)
        """
        if channel.id in self.ignore_channel_ids:
            return False
        return not (self.whitelist_channel_ids and channel.id not in self.whitelist_channel_ids)

    async def _fetch_message_safely(self, channel: GuildChannel, message_id: int) -> discord.Message | None:
        """Safely fetch a message from a channel.

        Args:
            channel: The channel containing the message
            message_id: The message ID to fetch

        Returns:
            The fetched message or None if not accessible
        """
        try:
            return await channel.fetch_message(message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
            _LOG.debug("Failed to fetch message %s: %s", message_id, e)
            return None

    def _is_message_eligible(self, message: discord.Message) -> bool:
        """Check if a message is eligible for starboard posting.

        Args:
            message: The message to check

        Returns:
            True if message is from a bot and hasn't been seen before
        """
        if not message.author.bot:
            return False
        return message.id not in self._seen_message_ids

    def _validate_starboard_channel(self, channel: GuildChannel | None, guild_id: int) -> bool:
        """Validate that the starboard channel is usable.

        Args:
            channel: The resolved starboard channel
            guild_id: Expected guild ID

        Returns:
            True if channel is valid and messageable in the same guild
        """
        if channel is None:
            return False
        if not isinstance(channel, Messageable):
            _LOG.warning(
                "Starboard channel %s is not messageable; ignoring repost.",
                self.starboard_channel_id,
            )
            return False
        return getattr(channel.guild, "id", None) == guild_id

    async def _send_to_starboard(
        self, starboard_channel: Messageable, message: discord.Message
    ) -> discord.Message | None:
        """Send a formatted message to the starboard channel.

        Args:
            starboard_channel: The starboard channel to send to
            message: The original message to repost

        Returns:
            The sent message or None if sending failed
        """
        content = self._format_forward_content(message)
        try:
            return await starboard_channel.send(content)
        except (discord.Forbidden, discord.HTTPException):
            _LOG.exception("Failed to repost message %s to starboard.", message.id)
            return None

    def _log_starboard_entry(
        self,
        payload: discord.RawReactionActionEvent,
        message: discord.Message,
        starboard_channel_id: int,
        sent_message_id: int,
    ) -> None:
        """Log starboard post metadata to the database.

        Args:
            payload: The original reaction event
            message: The original message
            starboard_channel_id: ID of the starboard channel
            sent_message_id: ID of the message posted to starboard
        """
        attachments_serialized = self._serialize_attachments(message)
        reaction = None
        with suppress(Exception):
            reaction = str(payload.emoji)

        author_name = self._get_author_display_name(message.author)
        try:
            log_starboard_post(
                guild_id=payload.guild_id,
                source_channel_id=message.channel.id,
                starboard_channel_id=starboard_channel_id,
                original_message_id=message.id,
                starboard_message_id=sent_message_id,
                reactor_user_id=payload.user_id,
                reaction_emoji=reaction,
                original_author_id=message.author.id,
                original_author_name=author_name,
                jump_url=message.jump_url,
                content=message.content,
                attachment_urls=attachments_serialized,
            )
        except Exception:
            _LOG.exception("Failed to record starboard entry for message %s", message.id)

    @commands.Cog.listener("on_raw_reaction_add")
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        """Handle reaction additions and repost eligible messages to starboard.

        Args:
            payload: The raw reaction event data
        """
        # Early validation: starboard configured and in a guild
        if self.starboard_channel_id is None:
            return
        if payload.guild_id is None:  # Ignore DMs
            return

        # Validate reactor
        if not self._is_valid_reactor(payload):
            return

        # Resolve and validate origin channel
        origin_channel = await self._resolve_channel(payload.channel_id)
        if origin_channel is None:
            return
        if not self._is_channel_eligible(origin_channel):
            return

        # Fetch and validate message
        message = await self._fetch_message_safely(origin_channel, payload.message_id)
        if message is None:
            return
        if not self._is_message_eligible(message):
            return

        # Resolve and validate starboard channel
        starboard_channel = await self._resolve_channel(self.starboard_channel_id)
        if not self._validate_starboard_channel(starboard_channel, payload.guild_id):
            return

        # Send to starboard
        sent_message = await self._send_to_starboard(starboard_channel, message)
        if sent_message is None:
            return

        # Mark as seen and log
        self._seen_message_ids.append(message.id)
        self._log_starboard_entry(payload, message, starboard_channel.id, sent_message.id)

    @staticmethod
    def _get_author_display_name(author: discord.User | discord.Member) -> str:
        """Extract the display name from a Discord user or member.

        Args:
            author: The Discord user or member

        Returns:
            Display name (preferred) or username fallback
        """
        return getattr(author, "display_name", author.name)

    @staticmethod
    def _truncate_text(text: str, max_length: int) -> str:
        """Truncate text to a maximum length with ellipsis.

        Args:
            text: Text to truncate
            max_length: Maximum length (including ellipsis)

        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        return f"{text[: max_length - 1]}…"

    def _format_forward_content(self, message: discord.Message) -> str:
        """Format a message for forwarding to the starboard channel.

        Args:
            message: The original message to format

        Returns:
            Formatted message content with jump URL, preview, and attachments
        """
        parts: list[str] = [message.jump_url]
        author_name = self._get_author_display_name(message.author)

        if message.content:
            preview = self._truncate_text(message.content.strip(), MAX_CONTENT_PREVIEW_LENGTH)
            parts.append(f"> **{author_name}**: {preview}")

        for attachment in message.attachments:
            parts.append(attachment.url)

        body = "\n".join(parts)
        return self._truncate_text(body, MAX_FORWARD_MESSAGE_LENGTH)

    @staticmethod
    def _create_attachment_entry(
        *,
        filename: str | None,
        url: str | None,
        proxy_url: str | None,
        content_type: str | None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, object] | None:
        """Create a serializable attachment data entry.

        Args:
            filename: Name of the file
            url: Direct URL to the attachment
            proxy_url: Proxied/CDN URL for the attachment
            content_type: MIME type of the attachment
            width: Image width in pixels (if applicable)
            height: Image height in pixels (if applicable)

        Returns:
            Dictionary containing attachment metadata, or None if no valid URLs
        """
        if not url and not proxy_url:
            return None

        return {
            "filename": filename,
            "url": url,
            "proxy_url": proxy_url,
            "content_type": content_type,
            "width": width,
            "height": height,
        }

    def _extract_message_attachments(self, message: discord.Message) -> list[dict[str, object]]:
        """Extract all attachment metadata from a message.

        Args:
            message: The Discord message to extract attachments from

        Returns:
            List of attachment metadata dictionaries
        """
        attachments: list[dict[str, object]] = []

        # Process direct file attachments
        for attachment in message.attachments:
            entry = self._create_attachment_entry(
                filename=attachment.filename,
                url=attachment.url,
                proxy_url=getattr(attachment, "proxy_url", None),
                content_type=attachment.content_type,
                width=attachment.width,
                height=attachment.height,
            )
            if entry is not None:
                attachments.append(entry)

        # Process embed images (for archived messages that only retain embeds)
        for embed in message.embeds:
            img = embed.image or embed.thumbnail
            if img and getattr(img, "url", None):
                entry = self._create_attachment_entry(
                    filename=(embed.title or "embed"),
                    url=getattr(embed, "url", None) or img.url,
                    proxy_url=img.url,
                    content_type="image/embed",
                    width=getattr(img, "width", None),
                    height=getattr(img, "height", None),
                )
                if entry is not None:
                    attachments.append(entry)

        return attachments

    def _serialize_attachments(self, message: discord.Message) -> str:
        """Serialize message attachments to JSON format for database storage.

        Args:
            message: The Discord message containing attachments

        Returns:
            JSON-encoded string of attachment metadata
        """
        attachments = self._extract_message_attachments(message)
        return json.dumps(attachments)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Starboard(bot))
