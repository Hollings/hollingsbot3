# text_generators/grok.py
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


class GrokTextGenerator(TextGeneratorAPI):
    """Text-generation backend for xAI Grok models.

    Uses the OpenAI-compatible xAI API.
    Requires XAI_API_KEY in the environment.
    """

    def __init__(self, model: str = "grok-4-1-fast-non-reasoning-latest") -> None:
        self.model = model

    def _get_client(self) -> AsyncOpenAI:
        if "grok" not in _CLIENT_CACHE:
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                raise ValueError("XAI_API_KEY environment variable is required for Grok")
            _CLIENT_CACHE["grok"] = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        return _CLIENT_CACHE["grok"]

    async def generate(
        self,
        prompt: Union[str, Sequence[_Message]],
        *,
        temperature: float = 1.0,
    ) -> str:
        """Generate text using Grok via xAI's OpenAI-compatible API."""
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

        # Use Chat Completions API (xAI is OpenAI-compatible)
        _LOG.debug("Grok: generating with model=%s, messages=%d", self.model, len(messages))

        try:
            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,      # type: ignore[arg-type]
                temperature=temperature,
            )
        except RateLimitError as e:
            _LOG.warning("Grok rate limit hit for model %s: %s", self.model, e)
            raise
        except APIConnectionError as e:
            _LOG.error("Grok connection error for model %s: %s", self.model, e)
            raise
        except APIError as e:
            _LOG.error("Grok API error for model %s (status %s): %s", self.model, getattr(e, 'status_code', 'unknown'), e)
            raise

        choice = resp.choices[0]
        return (choice.message.content or "").strip()
