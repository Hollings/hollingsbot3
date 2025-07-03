from .base import TextGeneratorAPI
from .huggingface import HuggingFaceTextGenerator
from .anthropic import AnthropicTextGenerator

__all__ = [
    "TextGeneratorAPI",
    "HuggingFaceTextGenerator",
    "AnthropicTextGenerator",
]


def get_text_generator(api: str, model: str) -> TextGeneratorAPI:
    """Return an appropriate text-generator instance for the given API."""
    if api == "huggingface":
        return HuggingFaceTextGenerator(model)
    if api == "anthropic":
        return AnthropicTextGenerator(model)
    raise ValueError(f"Unknown API: {api}")
