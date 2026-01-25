import logging

from .base import ImageGeneratorAPI
from .replicate_api import ReplicateImageGenerator
from .svg_gpt import SvgGPTImageGenerator

__all__ = [
    "ImageGeneratorAPI",
    "ReplicateImageGenerator",
    "SvgGPTImageGenerator",
    "generate_avatar",
    "get_image_generator",
]

_LOG = logging.getLogger(__name__)

# Default model for avatar generation
AVATAR_MODEL = "black-forest-labs/flux-schnell"


def get_image_generator(
    api: str,
    model: str,
    *,
    quality: str = "auto",
    aspect_ratio: str | None = None,
    model_options: dict | None = None,
) -> ImageGeneratorAPI:
    """Return an appropriate image generator instance for the given API.

    Args:
        api: API provider name ('replicate', 'svg', 'openai-svg')
        model: Model identifier
        quality: Quality level for gpt-image models ('low', 'medium', 'high', 'auto')
        aspect_ratio: Aspect ratio for gpt-image models ('1:1', '3:2', '2:3', etc.)
        model_options: Extra model-specific options (go_fast, safety_tolerance, etc.)

    Returns:
        ImageGeneratorAPI instance
    """
    if api == "replicate":
        return ReplicateImageGenerator(model=model, quality=quality, aspect_ratio=aspect_ratio, model_options=model_options)
    if api in ("svg", "openai-svg"):
        return SvgGPTImageGenerator(model)
    raise ValueError(f"Unknown API: {api}")


async def generate_avatar(prompt: str, model: str = AVATAR_MODEL) -> bytes | None:
    """Generate an avatar image from a prompt using Replicate.

    This is a convenience function for LLM cogs to generate profile pictures.
    Uses flux-schnell by default for fast generation.

    Args:
        prompt: Description of the avatar to generate
        model: Replicate model to use (default: flux-schnell)

    Returns:
        Image bytes on success, None on failure
    """
    generator = ReplicateImageGenerator(model=model)
    try:
        _LOG.info("Generating avatar with prompt: %s", prompt[:100])
        image_bytes = await generator.generate(prompt)
        _LOG.info("Avatar generated successfully (%d bytes)", len(image_bytes))
        return image_bytes
    except Exception:
        _LOG.exception("Failed to generate avatar")
        return None
    finally:
        await generator.aclose()
