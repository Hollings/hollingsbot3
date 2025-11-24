# text_generators/__init__.py
from .base import TextGeneratorAPI
from .huggingface import HuggingFaceTextGenerator
from .anthropic import AnthropicTextGenerator
from .openai_chatgpt import OpenAIChatTextGenerator
from .grok import GrokTextGenerator

__all__ = [
    "TextGeneratorAPI",
    "HuggingFaceTextGenerator",
    "AnthropicTextGenerator",
    "OpenAIChatTextGenerator",
    "GrokTextGenerator",
]


def get_text_generator(api: str, model: str) -> TextGeneratorAPI:
    """Return an appropriate text-generator instance for the given API."""
    if api == "huggingface":
        return HuggingFaceTextGenerator(model)
    if api == "anthropic":
        return AnthropicTextGenerator(model)
    if api in ("openai", "chatgpt"):
        return OpenAIChatTextGenerator(model)
    if api == "grok":
        return GrokTextGenerator(model)
    raise ValueError(f"Unknown API: {api}")
