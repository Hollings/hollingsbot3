"""Image generation tool for LLM chat bot."""

from __future__ import annotations

import logging

from hollingsbot.image_generators import get_image_generator
from hollingsbot.tools.parser import get_current_context

_LOG = logging.getLogger(__name__)

# Store generated images to be posted after tool execution
_pending_images: dict[int, list[tuple[bytes, str]]] = {}  # channel_id -> [(image_bytes, prompt)]


def get_pending_images(channel_id: int) -> list[tuple[bytes, str]]:
    """Get and clear pending images for a channel."""
    return _pending_images.pop(channel_id, [])


async def generate_image_async(prompt: str) -> str:
    """
    Generate an image using gpt-image-1.5 low quality.

    The image will be posted to Discord after tool execution completes.

    Args:
        prompt: Text description of the image to generate

    Returns:
        Status message for the LLM
    """
    if not prompt or not prompt.strip():
        return "Error: prompt cannot be empty"

    prompt = prompt.strip()

    # Get channel context
    ctx = get_current_context()
    channel_id = ctx.get("channel_id")
    if not channel_id:
        return "Error: no channel context available"

    _LOG.info("Generating image for channel %s: %s", channel_id, prompt[:50])

    try:
        # Use the Replicate API with gpt-image-1.5 low quality
        generator = get_image_generator(api="replicate", model="openai/gpt-image-1.5", quality="low")

        # Generate the image
        image_bytes = await generator.generate(prompt)

        # Store for posting after tool execution
        if channel_id not in _pending_images:
            _pending_images[channel_id] = []
        _pending_images[channel_id].append((image_bytes, prompt))

        _LOG.info("Image generated successfully for channel %s (%d bytes)", channel_id, len(image_bytes))

        return f"Image generated successfully for prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'. The image will be posted to the chat."

    except Exception as exc:
        _LOG.exception("Image generation failed: %s", exc)
        return f"Image generation failed: {exc}"
