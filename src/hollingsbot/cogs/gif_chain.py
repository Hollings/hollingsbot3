"""Discord cog for creating GIFs from reply chains.

This cog monitors messages for the "gif" command, which when used as a reply
to a message in a thread, collects all images from the reply chain and combines
them into an animated GIF.
"""
from __future__ import annotations

import io
import logging
import os
from typing import List, Set

import discord
from discord.ext import commands
from PIL import Image

__all__ = ["GifReplyChainCog"]

_log = logging.getLogger(__name__)

# Constants
THINKING_EMOJI = "\N{THINKING FACE}"
DEFAULT_MAX_FRAMES = 30
DEFAULT_MAX_SIDE_PX = 768
DEFAULT_FRAME_MS = 700
DEFAULT_MAX_BYTES = 8 * 1024 * 1024  # 8 MiB Discord limit
DEFAULT_SCAN_LIMIT = 400

# GIF compression strategies
SCALE_FACTORS = [1.0, 0.85, 0.72, 0.6, 0.5, 0.4, 0.33]
FRAME_DROPS = [1, 2, 3]
PALETTE_SIZES = [128, 64, 32]

# Image processing constants
CANVAS_BACKGROUND_COLOR = (0, 0, 0, 255)  # Black background
SCALE_THRESHOLD = 0.999  # Treat scale factors above this as 1.0


def _is_image_attachment(attachment: discord.Attachment) -> bool:
    """Check if a Discord attachment is an image file.

    Args:
        attachment: Discord attachment to check

    Returns:
        True if the attachment appears to be an image based on content type
        or file extension
    """
    content_type = (attachment.content_type or "").lower()
    if content_type.startswith("image/"):
        return True

    filename = attachment.filename.lower()
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff")
    return filename.endswith(image_extensions)


class GifReplyChainCog(commands.Cog):
    """Cog that creates animated GIFs from Discord reply chains.

    When a user replies to a message with just "gif", this cog traverses
    the reply chain (both ancestors and descendants), collects all image
    attachments, and combines them into an animated GIF.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the GIF reply chain cog.

        Args:
            bot: The Discord bot instance
        """
        self.bot = bot

        # Load configuration from environment
        env_ids = os.getenv("STABLE_DIFFUSION_CHANNEL_IDS", "")
        self._allowed_channel_ids: Set[int] = {
            int(x.strip())
            for x in env_ids.split(",")
            if x.strip().isdigit()
        }

        self._max_frames = int(os.getenv("GIF_CHAIN_MAX_FRAMES", str(DEFAULT_MAX_FRAMES)))
        self._max_side_px = int(os.getenv("GIF_CHAIN_MAX_SIDE", str(DEFAULT_MAX_SIDE_PX)))
        self._frame_ms = int(os.getenv("GIF_CHAIN_FRAME_MS", str(DEFAULT_FRAME_MS)))
        self._max_bytes = int(os.getenv("GIF_CHAIN_MAX_BYTES", str(DEFAULT_MAX_BYTES)))
        self._scan_limit = int(os.getenv("GIF_CHAIN_SCAN_LIMIT", str(DEFAULT_SCAN_LIMIT)))

    def _in_allowed_channel(self, message: discord.Message) -> bool:
        """Check if a message is in an allowed channel for GIF generation.

        Args:
            message: Message to check

        Returns:
            True if the message is in an allowed channel or if no channel
            restrictions are configured
        """
        if message.guild is None:
            return False
        if not self._allowed_channel_ids:
            return True
        return message.channel.id in self._allowed_channel_ids

    async def _add_reaction(self, message: discord.Message, emoji: str) -> None:
        """Add a reaction to a message, ignoring errors.

        Args:
            message: Message to react to
            emoji: Emoji to add as reaction
        """
        try:
            await message.add_reaction(emoji)
        except Exception:
            pass

    async def _remove_reaction(self, message: discord.Message, emoji: str) -> None:
        """Remove a reaction from a message, ignoring errors.

        Args:
            message: Message to remove reaction from
            emoji: Emoji to remove
        """
        try:
            if self.bot.user is not None:
                await message.remove_reaction(emoji, self.bot.user)
            else:
                await message.clear_reaction(emoji)
        except Exception:
            pass

    async def _resolve_message_reference(
        self,
        reference: discord.MessageReference,
        channel: discord.abc.Messageable
    ) -> discord.Message | None:
        """Resolve a message reference to an actual message.

        Args:
            reference: Message reference to resolve
            channel: Channel to fetch message from if needed

        Returns:
            Resolved message, or None if resolution failed
        """
        # Try using the resolved attribute first (cached)
        resolved = getattr(reference, "resolved", None)
        if isinstance(resolved, discord.Message):
            return resolved

        # Fall back to fetching by ID
        message_id = getattr(reference, "message_id", None)
        if message_id:
            try:
                return await channel.fetch_message(message_id)
            except Exception:
                pass

        return None

    async def _find_direct_reply(self, base: discord.Message) -> discord.Message | None:
        """Find the earliest message that directly replies to the given message.

        Scans messages after the base message up to the configured scan limit
        and returns the first message whose reference points to the base message.

        Args:
            base: Message to find reply to

        Returns:
            First message that replies to base, or None if not found
        """
        try:
            async for msg in base.channel.history(
                limit=self._scan_limit,
                after=base.created_at,
                oldest_first=True
            ):  # type: ignore[attr-defined]
                ref = getattr(msg, "reference", None)
                if ref and getattr(ref, "message_id", None) == base.id:
                    return msg
        except Exception:
            pass
        return None

    async def _collect_ancestor_messages(
        self,
        start: discord.Message,
        seen: Set[int]
    ) -> List[discord.Message]:
        """Collect all ancestor messages by traversing reply references backward.

        Args:
            start: Message to start from
            seen: Set to track visited message IDs (updated in-place)

        Returns:
            List of ancestor messages in order from root to start
        """
        ancestors = []
        current = start

        while current and current.id not in seen:
            seen.add(current.id)
            ancestors.append(current)

            ref = getattr(current, "reference", None)
            if not ref:
                break

            current = await self._resolve_message_reference(ref, start.channel)
            if current is None:
                break

        ancestors.reverse()  # Return in order: root -> start
        return ancestors

    async def _collect_descendant_messages(
        self,
        start: discord.Message,
        seen: Set[int]
    ) -> List[discord.Message]:
        """Collect all descendant messages by traversing replies forward.

        Args:
            start: Message to start from
            seen: Set to track visited message IDs (updated in-place)

        Returns:
            List of descendant messages in order from start to leaf
        """
        descendants: List[discord.Message] = []
        current = start

        while True:
            next_message = await self._find_direct_reply(current)
            if not next_message or next_message.id in seen:
                break

            descendants.append(next_message)
            seen.add(next_message.id)
            current = next_message

        return descendants

    async def _collect_chain_images(self, start: discord.Message) -> List[bytes]:
        """Collect all image attachments from a reply chain.

        Traverses both backward (ancestors) and forward (descendants) from the
        start message, collecting all image attachments up to the configured
        maximum frame count.

        Args:
            start: Message in the chain to start collection from

        Returns:
            List of raw image bytes from attachments in chain order
        """
        seen: Set[int] = set()

        # Collect backward to root
        ancestors = await self._collect_ancestor_messages(start, seen)

        # Collect forward to leaf
        descendants = await self._collect_descendant_messages(start, seen)

        # Combine into full chain
        chain = ancestors + descendants

        # Extract images from chain
        images: List[bytes] = []
        for message in chain:
            if len(images) >= self._max_frames:
                break

            for attachment in message.attachments:
                if len(images) >= self._max_frames:
                    break

                if _is_image_attachment(attachment):
                    try:
                        images.append(await attachment.read())
                    except Exception:
                        _log.debug(
                            "Could not download attachment %s on message %s",
                            attachment.id,
                            message.id
                        )

        return images

    def _decode_images(self, image_blobs: List[bytes]) -> List[Image.Image]:
        """Decode raw image bytes into PIL Image objects.

        Args:
            image_blobs: List of raw image bytes

        Returns:
            List of decoded PIL Images in RGBA format
        """
        decoded = []
        for blob in image_blobs:
            try:
                image = Image.open(io.BytesIO(blob)).convert("RGBA")
                decoded.append(image)
            except Exception:
                _log.debug("Failed to decode image blob")

        return decoded

    def _scale_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """Scale images to fit within maximum dimension constraints.

        Args:
            images: List of images to scale

        Returns:
            List of scaled images
        """
        if not images:
            return []

        scaled = []
        for image in images:
            width, height = image.size
            max_dimension = max(width, height)

            if max_dimension > self._max_side_px:
                scale = self._max_side_px / float(max_dimension)
            else:
                scale = 1.0

            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))

            scaled_image = image.resize((new_width, new_height), Image.LANCZOS)
            scaled.append(scaled_image)

        return scaled

    def _create_uniform_canvas(self, images: List[Image.Image]) -> List[Image.Image]:
        """Center all images on uniform-sized canvases.

        Creates canvases sized to the maximum width/height across all images,
        centers each image on its canvas with a black background.

        Args:
            images: List of images to place on canvases

        Returns:
            List of images on uniform-sized canvases
        """
        if not images:
            return []

        # Find maximum dimensions
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        # Center each image on a uniform canvas
        canvases = []
        for image in images:
            canvas = Image.new("RGBA", (max_width, max_height), CANVAS_BACKGROUND_COLOR)
            x_offset = (max_width - image.width) // 2
            y_offset = (max_height - image.height) // 2
            canvas.paste(image, (x_offset, y_offset), image)

            # Convert to RGB for GIF encoding
            canvases.append(canvas.convert("RGB"))

        return canvases

    def _prepare_frames(self, image_blobs: List[bytes]) -> List[Image.Image]:
        """Decode and prepare images as uniform GIF frames.

        Decodes raw image bytes, scales them to fit constraints, and places
        them on uniform-sized canvases ready for GIF encoding.

        Args:
            image_blobs: List of raw image bytes

        Returns:
            List of prepared frames ready for GIF encoding
        """
        decoded = self._decode_images(image_blobs)
        if not decoded:
            return []

        scaled = self._scale_images(decoded)
        frames = self._create_uniform_canvas(scaled)

        return frames

    def _resize_frames(
        self,
        frames: List[Image.Image],
        scale: float
    ) -> List[Image.Image]:
        """Resize all frames by a scale factor.

        Args:
            frames: Frames to resize
            scale: Scale factor (1.0 = no change)

        Returns:
            Resized frames, or original frames if scale is approximately 1.0
        """
        if scale >= SCALE_THRESHOLD:
            return frames

        width, height = frames[0].size
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        return [
            frame.resize((new_width, new_height), Image.LANCZOS)
            for frame in frames
        ]

    def _encode_gif(self, frames: List[Image.Image], palette_size: int) -> bytes:
        """Encode frames as an animated GIF with palette quantization.

        Args:
            frames: Frames to encode
            palette_size: Number of colors in palette (lower = smaller file)

        Returns:
            Encoded GIF as bytes
        """
        # Quantize each frame to reduce file size
        quantized = [
            frame.convert("RGB").quantize(colors=palette_size, method=Image.MEDIANCUT)
            for frame in frames
        ]

        buffer = io.BytesIO()
        quantized[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=quantized[1:],
            duration=self._frame_ms,
            loop=0,
            optimize=False,  # Disable optimize for stability
            disposal=2,
        )
        buffer.seek(0)
        return buffer.getvalue()

    async def _build_gif_bytes(self, frames: List[Image.Image]) -> bytes:
        """Build a GIF from frames, trying compression strategies to fit size limit.

        Attempts multiple combinations of scaling, frame dropping, and palette
        reduction to produce a GIF under the configured size limit.

        Args:
            frames: Prepared frames to encode

        Returns:
            Encoded GIF bytes, or empty bytes if encoding failed
        """
        if not frames:
            return b""

        # Try progressively aggressive compression strategies
        for scale in SCALE_FACTORS:
            resized_frames = self._resize_frames(frames, scale)

            for drop_rate in FRAME_DROPS:
                # Drop frames: keep every Nth frame
                sampled_frames = resized_frames[::drop_rate] if drop_rate > 1 else resized_frames

                # Ensure at least one frame
                if not sampled_frames:
                    sampled_frames = [resized_frames[-1]]

                for palette_size in PALETTE_SIZES:
                    try:
                        gif_data = self._encode_gif(sampled_frames, palette_size)

                        if len(gif_data) <= self._max_bytes:
                            return gif_data
                    except Exception:
                        _log.debug(
                            "Failed to encode GIF with scale=%.2f, drop=%d, palette=%d",
                            scale, drop_rate, palette_size
                        )
                        continue

        # Last resort: single frame with most aggressive compression
        _log.warning("All compression strategies failed, using fallback")
        try:
            smallest_frames = self._resize_frames(frames, SCALE_FACTORS[-1])
            single_frame = [smallest_frames[-1]]
            return self._encode_gif(single_frame, PALETTE_SIZES[-1])
        except Exception:
            _log.error("Failed to encode fallback GIF")
            return b""

    def _validate_gif_request(self, message: discord.Message) -> bool:
        """Validate that a message is a valid GIF generation request.

        Args:
            message: Message to validate

        Returns:
            True if the message is a valid GIF request
        """
        if message.author.bot:
            return False

        if not message.guild:
            return False

        if not self._in_allowed_channel(message):
            return False

        if (message.content or "").strip().lower() != "gif":
            return False

        if not message.reference:
            return False

        return True

    async def _get_target_message(
        self,
        message: discord.Message
    ) -> discord.Message | None:
        """Get the target message from a reply reference.

        Args:
            message: Message containing the reference

        Returns:
            Target message, or None if resolution failed
        """
        reference = message.reference
        if not reference:
            return None

        return await self._resolve_message_reference(reference, message.channel)

    async def _create_and_send_gif(
        self,
        message: discord.Message,
        target: discord.Message
    ) -> None:
        """Create a GIF from a reply chain and send it.

        Args:
            message: Command message that triggered the request
            target: Target message in the reply chain
        """
        await self._add_reaction(message, THINKING_EMOJI)

        try:
            # Collect images from the chain
            image_blobs = await self._collect_chain_images(target)
            if not image_blobs:
                await message.reply(
                    "No images found in the reply chain.",
                    mention_author=False
                )
                return

            # Prepare frames
            frames = self._prepare_frames(image_blobs)
            if not frames:
                await message.reply(
                    "Could not decode images from the reply chain.",
                    mention_author=False
                )
                return

            # Build GIF
            gif_data = await self._build_gif_bytes(frames)
            if not gif_data:
                await message.reply(
                    "Failed to build GIF.",
                    mention_author=False
                )
                return

            # Send result
            await message.reply(
                file=discord.File(io.BytesIO(gif_data), filename="reply_chain.gif"),
                mention_author=False
            )

        except Exception as exc:
            _log.exception("GIF chain creation failed: %s", exc)
            try:
                await message.reply(
                    "Error creating GIF from reply chain.",
                    mention_author=False
                )
            except Exception:
                pass
        finally:
            await self._remove_reaction(message, THINKING_EMOJI)

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        """Listen for "gif" commands in replies and generate GIFs from chains.

        Args:
            message: Message received from Discord
        """
        if not self._validate_gif_request(message):
            return

        target = await self._get_target_message(message)
        if target is None:
            return

        await self._create_and_send_gif(message, target)


async def setup(bot: commands.Bot) -> None:
    """Set up the GIF reply chain cog.

    Args:
        bot: Discord bot instance
    """
    await bot.add_cog(GifReplyChainCog(bot))
