# text_generators/grok.py
from __future__ import annotations

from typing import Dict, Sequence, TypedDict, Union, List, Any
import os
import logging

from openai import AsyncOpenAI

from .base import TextGeneratorAPI

_CLIENT_CACHE: Dict[str, AsyncOpenAI] = {}
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
            messages: List[_Message] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, Sequence):
            if not all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
                raise TypeError("Each message must be a dict with 'role' and 'content' keys")
            messages = list(prompt)  # type: ignore[arg-type]
        else:
            raise TypeError("prompt must be a string or a sequence of message dicts")

        client = self._get_client()

        # Use Chat Completions API (xAI is OpenAI-compatible)
        _LOG.debug("Grok: generating with model=%s, messages=%d", self.model, len(messages))

        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,      # type: ignore[arg-type]
            temperature=temperature,
        )
        choice = resp.choices[0]
        return (choice.message.content or "").strip()
