"""Text generation providers for LLM interactions.

This package provides a unified interface for multiple LLM providers:
- Anthropic (Claude models)
- OpenAI (GPT models)
- xAI (Grok)
- Google (Gemini)
- OpenRouter (multi-provider routing)
- HuggingFace (local/HF models)

All generators implement the TextGeneratorAPI abstract base class, providing
a consistent `generate(prompt: str) -> str` method.

Usage:
    from hollingsbot.text_generators import get_text_generator

    # Get a generator for a specific API
    generator = get_text_generator("anthropic", "claude-sonnet-4")

    # Generate text
    response = await generator.generate("Hello, how are you?")

See docs/INTEGRATIONS.md for details on each provider.
"""

from .base import TextGeneratorAPI
from .huggingface import HuggingFaceTextGenerator
from .anthropic import AnthropicTextGenerator
from .openai_chatgpt import OpenAIChatTextGenerator
from .grok import GrokTextGenerator
from .openrouter import OpenRouterTextGenerator, OpenRouterCompletionsGenerator, OpenRouterLoomGenerator
from .gemini import GeminiTextGenerator
from .claude_cli import ClaudeCliTextGenerator

__all__ = [
    "TextGeneratorAPI",
    "HuggingFaceTextGenerator",
    "AnthropicTextGenerator",
    "OpenAIChatTextGenerator",
    "GrokTextGenerator",
    "OpenRouterTextGenerator",
    "OpenRouterCompletionsGenerator",
    "OpenRouterLoomGenerator",
    "GeminiTextGenerator",
    "ClaudeCliTextGenerator",
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
    if api == "openrouter":
        return OpenRouterTextGenerator(model)
    if api == "openrouter-completions":
        return OpenRouterCompletionsGenerator(model)
    if api == "openrouter-loom":
        return OpenRouterLoomGenerator(model)
    if api == "gemini":
        return GeminiTextGenerator(model)
    if api == "claude-cli":
        return ClaudeCliTextGenerator(model)
    raise ValueError(f"Unknown API: {api}")
