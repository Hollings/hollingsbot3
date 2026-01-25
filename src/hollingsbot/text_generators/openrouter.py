# text_generators/openrouter.py
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TypedDict, Union

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from .base import TextGeneratorAPI

_CLIENT_CACHE: dict[str, AsyncOpenAI] = {}
_LOG = logging.getLogger(__name__)


class _Message(TypedDict):
    role: str
    content: str


def _get_openrouter_client() -> AsyncOpenAI:
    """Get or create the shared OpenRouter client."""
    if "openrouter" not in _CLIENT_CACHE:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter")
        _CLIENT_CACHE["openrouter"] = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    return _CLIENT_CACHE["openrouter"]


class OpenRouterTextGenerator(TextGeneratorAPI):
    """Text-generation backend for OpenRouter models (chat completions).

    Uses OpenRouter's OpenAI-compatible API.
    Requires OPENROUTER_API_KEY in the environment.
    """

    def __init__(self, model: str = "meta-llama/llama-3.1-405b-instruct") -> None:
        self.model = model

    def _get_client(self) -> AsyncOpenAI:
        return _get_openrouter_client()

    async def generate(
        self,
        prompt: Union[str, Sequence[_Message]],
        *,
        temperature: float = 1.0,
    ) -> str:
        """Generate text using OpenRouter's OpenAI-compatible API."""
        # Normalize input into a list of messages
        if isinstance(prompt, str):
            messages: list[_Message] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, Sequence):
            if not all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
                raise TypeError("Each message must be a dict with 'role' and 'content' keys")
            messages = list(prompt)  # type: ignore[arg-type]
        else:
            raise TypeError("prompt must be a string or a sequence of message dicts")

        client = self._get_client()

        _LOG.debug("OpenRouter: generating with model=%s, messages=%d", self.model, len(messages))

        _LOG.info("OpenRouter request: model=%s, messages=%s", self.model, messages)

        try:
            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,      # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=500,
            )
        except RateLimitError as e:
            _LOG.warning("OpenRouter rate limit hit for model %s: %s", self.model, e)
            raise
        except APIConnectionError as e:
            _LOG.error("OpenRouter connection error for model %s: %s", self.model, e)
            raise
        except APIError as e:
            _LOG.error("OpenRouter API error for model %s (status %s): %s", self.model, getattr(e, 'status_code', 'unknown'), e)
            raise

        _LOG.info("OpenRouter response: %s", resp)

        choice = resp.choices[0]
        finish_reason = getattr(choice, 'finish_reason', None)
        content = choice.message.content

        _LOG.info("OpenRouter result: finish_reason=%s, content_len=%d, content=%s",
                  finish_reason, len(content) if content else 0, content[:200] if content else None)

        return (content or "").strip()


class OpenRouterCompletionsGenerator(TextGeneratorAPI):
    """Text-generation backend for OpenRouter models (raw completions).

    Uses OpenRouter's completions API for raw text continuation.
    Best for base models like Llama 405B base.
    Requires OPENROUTER_API_KEY in the environment.
    """

    def __init__(self, model: str = "meta-llama/llama-3.1-405b") -> None:
        self.model = model

    def _get_client(self) -> AsyncOpenAI:
        return _get_openrouter_client()

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 500,
    ) -> str:
        """Generate text continuation using OpenRouter's completions API."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string for completions API")

        client = self._get_client()

        _LOG.info(
            "OpenRouter completions: model=%s, prompt_len=%d, temp=%.2f",
            self.model, len(prompt), temperature
        )

        try:
            resp = await client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as e:
            _LOG.warning("OpenRouter completions rate limit hit for model %s: %s", self.model, e)
            raise
        except APIConnectionError as e:
            _LOG.error("OpenRouter completions connection error for model %s: %s", self.model, e)
            raise
        except APIError as e:
            _LOG.error("OpenRouter completions API error for model %s (status %s): %s", self.model, getattr(e, 'status_code', 'unknown'), e)
            raise

        _LOG.info("OpenRouter completions response: %s", resp)

        choice = resp.choices[0]
        text = choice.text

        _LOG.info(
            "OpenRouter completions result: finish_reason=%s, text_len=%d",
            getattr(choice, 'finish_reason', None),
            len(text) if text else 0
        )

        return (text or "").strip()


class OpenRouterLoomGenerator(TextGeneratorAPI):
    """Text-generation backend using Loom technique for instruct models.

    Uses CLI simulation mode to make instruct models behave like base models.
    Works with Gemini, Claude, and other instruct-tuned models.
    Requires OPENROUTER_API_KEY in the environment.
    """

    SYSTEM_PROMPT = "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with the output of the command."
    USER_PROMPT = "<cmd>cat untitled.txt</cmd>"

    def __init__(self, model: str = "google/gemini-3-flash-preview") -> None:
        self.model = model

    def _get_client(self) -> AsyncOpenAI:
        return _get_openrouter_client()

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 500,
    ) -> str:
        """Generate text continuation using Loom technique."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string for Loom completions")

        client = self._get_client()

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT},
            {"role": "assistant", "content": prompt},
        ]

        _LOG.info(
            "OpenRouter Loom: model=%s, prompt_len=%d, temp=%.2f",
            self.model, len(prompt), temperature
        )

        try:
            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as e:
            _LOG.warning("OpenRouter Loom rate limit hit for model %s: %s", self.model, e)
            raise
        except APIConnectionError as e:
            _LOG.error("OpenRouter Loom connection error for model %s: %s", self.model, e)
            raise
        except APIError as e:
            _LOG.error("OpenRouter Loom API error for model %s (status %s): %s", self.model, getattr(e, 'status_code', 'unknown'), e)
            raise

        _LOG.info("OpenRouter Loom response: %s", resp)

        choice = resp.choices[0]
        content = choice.message.content

        _LOG.info(
            "OpenRouter Loom result: finish_reason=%s, content_len=%d",
            getattr(choice, 'finish_reason', None),
            len(content) if content else 0
        )

        return (content or "").strip()
