from .base import ImageGeneratorAPI
from .replicate_api import ReplicateImageGenerator

__all__ = [
    "ImageGeneratorAPI",
    "ReplicateImageGenerator",
    "get_image_generator",
]


def get_image_generator(api: str, model: str) -> ImageGeneratorAPI:
    """Return an appropriate image generator instance for the given API."""
    if api == "replicate":
        return ReplicateImageGenerator(model)
    raise ValueError(f"Unknown API: {api}")
