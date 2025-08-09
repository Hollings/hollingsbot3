from .base import ImageGeneratorAPI
from .replicate_api import ReplicateImageGenerator
from .svg_gpt import SvgGPTImageGenerator

__all__ = [
    "ImageGeneratorAPI",
    "ReplicateImageGenerator",
    "SvgGPTImageGenerator",
    "get_image_generator",
]


def get_image_generator(api: str, model: str) -> ImageGeneratorAPI:
    """Return an appropriate image generator instance for the given API."""
    if api == "replicate":
        return ReplicateImageGenerator(model)
    if api in ("svg", "openai-svg"):
        return SvgGPTImageGenerator(model)
    raise ValueError(f"Unknown API: {api}")
