"""Shared utility functions for chat system."""

import asyncio
import base64
import io
import logging
import os
import re
import textwrap
from collections import deque
from typing import Deque

import discord
from PIL import Image

from hollingsbot.cogs.conversation import ConversationTurn, ImageAttachment
from hollingsbot.url_metadata import (
    download_images_from_metadata,
    extract_url_metadata,
    format_metadata_for_llm,
)
from hollingsbot.utils.discord_utils import get_display_name

_LOG = logging.getLogger(__name__)

# Constants
_TEXT_ATTACHMENT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".xml", ".yaml", ".yml",
    ".html", ".css", ".sql", ".sh", ".bash", ".java", ".c", ".cpp", ".h", ".go", ".rs",
    ".rb", ".php", ".swift", ".kt", ".cs", ".log", ".env", ".toml", ".ini", ".conf"
}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_MAX_TEXT_ATTACHMENT_BYTES = 120_000  # 120KB
_IMAGE_MAX_EDGE = 2048
_IMAGE_MAX_BYTES = 10_000_000  # ~9.5MB after base64


# ==================== Message Cleaning ====================

def clean_mentions(message: discord.Message, bot: discord.Client) -> str:
    """Replace user mentions with display names instead of IDs."""
    from hollingsbot.utils.discord_utils import get_display_name

    content = message.content
    if not content:
        return ""

    # Build a mapping of user IDs to display names from message.mentions
    user_map: dict[int, str] = {}
    for user in message.mentions:
        user_map[user.id] = get_display_name(user)

    # Replace user mentions <@123456> or <@!123456> with @displayname
    def replace_user_mention(match: re.Match[str]) -> str:
        user_id_str = match.group(1)
        user_id = int(user_id_str)

        # Use the pre-built map from message.mentions
        if user_id in user_map:
            return f"@{user_map[user_id]}"

        # Fallback: try guild member lookup
        if message.guild:
            member = message.guild.get_member(user_id)
            if member:
                return f"@{get_display_name(member)}"

        # Fallback: try bot's user cache
        user = bot.get_user(user_id)
        if user:
            return f"@{get_display_name(user)}"

        # Last resort - return @Unknown or keep the mention
        return f"@User{user_id_str}"

    # Pattern matches <@123456> or <@!123456>
    content = re.sub(r'<@!?(\d+)>', replace_user_mention, content)

    return content


def should_ignore_message(content: str | None) -> bool:
    """Check if message should be ignored (bot commands or image generation)."""
    if not content:
        return False
    stripped = content.lstrip()
    if not stripped:
        return False
    lowered = stripped.lower()
    return stripped.startswith("!") or stripped.startswith("-") or lowered.startswith("edit:")


# ==================== Attachment Detection ====================

def is_text_attachment(attachment: discord.Attachment) -> bool:
    """Check if attachment is a text file."""
    if attachment.size == 0:
        return False
    if attachment.content_type:
        ctype = attachment.content_type.lower()
        if ctype.startswith("text/"):
            return True
        if ctype in {
            "application/json",
            "application/javascript",
            "application/xml",
            "application/x-yaml",
        }:
            return True
    _, ext = os.path.splitext(attachment.filename)
    return ext.lower() in _TEXT_ATTACHMENT_EXTENSIONS


def is_image_attachment(attachment: discord.Attachment) -> bool:
    """Check if attachment is an image file."""
    if attachment.content_type:
        if attachment.content_type.lower().startswith("image/"):
            return True
    _, ext = os.path.splitext(attachment.filename)
    return ext.lower() in _IMAGE_EXTENSIONS


# ==================== Text Attachments ====================

async def read_text_attachment(attachment: discord.Attachment) -> tuple[str, bool] | None:
    """Read text attachment and return (content, was_truncated) or None on error."""
    try:
        data = await attachment.read()
    except Exception:
        _LOG.exception("Failed to read text attachment %s", attachment.filename)
        return None

    truncated = len(data) > _MAX_TEXT_ATTACHMENT_BYTES
    if truncated:
        data = data[:_MAX_TEXT_ATTACHMENT_BYTES]

    text = data.decode("utf-8", errors="replace")
    return text, truncated


async def collect_text_attachments_full(message: discord.Message) -> tuple[list[str], list[str]]:
    """Collect text attachments returning (full_blocks, placeholders)."""
    full_blocks: list[str] = []
    placeholders: list[str] = []

    for attachment in message.attachments:
        if not is_text_attachment(attachment):
            continue

        result = await read_text_attachment(attachment)
        if result is None:
            continue

        text, truncated = result
        block = f"[begin uploaded file: {attachment.filename}]\n{text}\n[end uploaded file]"
        if truncated:
            block += "\n[truncated]"
        full_blocks.append(block)

        placeholder = f"[uploaded file {attachment.filename} removed]"
        if truncated:
            placeholder += " (truncated)"
        placeholders.append(placeholder)

    return full_blocks, placeholders


# ==================== Image Processing ====================

def encode_jpeg(image: Image.Image) -> bytes:
    """Encode image as JPEG, trying progressively lower quality to meet size limit."""
    for quality in (90, 85, 80, 75, 70, 60, 50):
        out = io.BytesIO()
        image.save(out, format="JPEG", optimize=True, quality=quality)
        if out.tell() <= _IMAGE_MAX_BYTES:
            return out.getvalue()
    return out.getvalue()


def resize_image_if_needed(img: Image.Image) -> Image.Image:
    """Resize image if it exceeds maximum edge length."""
    width, height = img.size
    longest = max(width, height)
    if longest <= _IMAGE_MAX_EDGE:
        return img

    scale = _IMAGE_MAX_EDGE / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, Image.LANCZOS)


async def prepare_image_attachment(attachment: discord.Attachment) -> ImageAttachment | None:
    """Download and process image attachment into ImageAttachment object."""
    try:
        data = await attachment.read()
    except Exception:
        _LOG.exception("Failed to download image attachment %s", attachment.filename)
        return None

    try:
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            img = resize_image_if_needed(img)
            width, height = img.size
            jpeg_bytes = encode_jpeg(img)
    except Exception:
        _LOG.exception("Failed to process image attachment %s", attachment.filename)
        return ImageAttachment(
            name=attachment.filename,
            url=attachment.url,
            data_url="",
            width=None,
            height=None,
            size=attachment.size,
        )

    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")
    return ImageAttachment(
        name=attachment.filename,
        url=attachment.url,
        data_url=data_url,
        width=width,
        height=height,
        size=len(jpeg_bytes),
    )


def image_from_bytes(name: str, data: bytes) -> ImageAttachment:
    """Create ImageAttachment from raw bytes (e.g., SVG conversion)."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception:
        width = height = None
    data_url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    return ImageAttachment(
        name=name, url="", data_url=data_url, width=width, height=height, size=len(data)
    )


def images_from_history(
    channel_histories: dict[int, Deque[ConversationTurn]],
    channel_id: int,
    message_id: int | None,
) -> list[ImageAttachment]:
    """Retrieve images from a previous message in channel history."""
    if message_id is None:
        return []
    history = channel_histories.get(channel_id)
    if not history:
        return []
    for turn in reversed(history):
        if turn.message_id == message_id:
            return list(turn.images)  # Return copies
    return []


async def collect_image_attachments(message: discord.Message) -> list[ImageAttachment]:
    """Collect all image attachments from a message."""
    images: list[ImageAttachment] = []
    for attachment in message.attachments:
        if not is_image_attachment(attachment):
            continue
        img = await prepare_image_attachment(attachment)
        if img:
            images.append(img)
    return images


# ==================== Reply Context ====================

async def fetch_referenced_message(message: discord.Message) -> discord.Message | None:
    """Fetch the message being replied to, if any."""
    ref = message.reference
    if not ref or not ref.message_id:
        return None

    resolved = ref.resolved if isinstance(ref.resolved, discord.Message) else None
    if resolved:
        return resolved

    try:
        return await message.channel.fetch_message(ref.message_id)
    except Exception:
        _LOG.exception("Failed to fetch referenced message %s", ref.message_id)
        return None


async def build_reply_hint(
    message: discord.Message,
    bot: discord.Client,
    channel_histories: dict[int, Deque[ConversationTurn]],
) -> tuple[str | None, list[ImageAttachment]]:
    """Build reply hint text and collect images from referenced message."""
    ref_message = await fetch_referenced_message(message)
    if ref_message is None:
        return None, []

    # Get display name with proper fallback: server nick > global display > username
    if ref_message.author:
        display = get_display_name(ref_message.author)
    else:
        display = "Unknown"

    snippet = clean_mentions(ref_message, bot).strip()

    if snippet:
        snippet = textwrap.shorten(snippet.replace("\n", " "), width=140, placeholder="…")
        hint = f"(Replying to <{display}>: {snippet})"
    else:
        hint = f"(Replying to <{display}>.)"

    images = images_from_history(channel_histories, message.channel.id, ref_message.id)
    if not images:
        images = await collect_image_attachments(ref_message)

    return hint, list(images)


# ==================== URL Metadata ====================

async def extract_url_images(base_text: str) -> tuple[list[ImageAttachment], str, str]:
    """
    Extract URL metadata and images from text.
    Returns (images, full_metadata_text, history_metadata_text).
    """
    url_images: list[ImageAttachment] = []
    full_metadata_text = ""
    history_metadata_text = ""

    if not base_text:
        return url_images, full_metadata_text, history_metadata_text

    url_metadata_list = await extract_url_metadata(base_text)
    if not url_metadata_list:
        return url_images, full_metadata_text, history_metadata_text

    # For current turn to LLM: include images
    full_metadata_parts = [
        format_metadata_for_llm(m, include_images=True) for m in url_metadata_list
    ]
    full_metadata_text = "\n\n".join(full_metadata_parts)

    # For history: exclude images
    history_metadata_parts = [
        format_metadata_for_llm(m, include_images=False) for m in url_metadata_list
    ]
    history_metadata_text = "\n\n".join(history_metadata_parts)

    # Download and process images from URL metadata
    for metadata in url_metadata_list:
        downloaded_images = await download_images_from_metadata(metadata)
        # Convert url_metadata.ImageAttachment to conversation.ImageAttachment
        for url_img in downloaded_images:
            url_images.append(
                ImageAttachment(
                    name=url_img.name,
                    url=url_img.url,
                    data_url=url_img.data_url,
                    width=url_img.width,
                    height=url_img.height,
                    size=url_img.size,
                )
            )

    return url_images, full_metadata_text, history_metadata_text


# ==================== Turn Building ====================

def build_user_message_text(display_name: str, reply_hint: str | None, base_text: str) -> str:
    """Build formatted user message text with display name and optional reply hint."""
    body_parts: list[str] = []
    if reply_hint:
        body_parts.append(reply_hint)
    if base_text:
        body_parts.append(base_text)

    body = "\n".join(part for part in body_parts if part).strip()
    if not body:
        body = "[no content]"

    return f"<{display_name}>: {body}"
