from .base import TextGeneratorAPI
from .huggingface import HuggingFaceTextGenerator

__all__ = ["TextGeneratorAPI", "HuggingFaceTextGenerator"]


def get_text_generator(api: str, model: str) -> TextGeneratorAPI:
    """Return an appropriate text generator instance for the given API."""
    if api == "huggingface":
        return HuggingFaceTextGenerator(model)
    raise ValueError(f"Unknown API: {api}")
