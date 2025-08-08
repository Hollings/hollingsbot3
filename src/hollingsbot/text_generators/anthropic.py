"""Text-generation backend that calls Anthropic’s Claude models."""
from __future__ import annotations

import asyncio
from typing import Dict, Sequence, Union, TypedDict, Any, List

from anthropic import AsyncAnthropic

from .base import TextGeneratorAPI

# --------------------------------------------------------------------------- #
# A single shared client is plenty; reuse it across all requests              #
# --------------------------------------------------------------------------- #
_CLIENT_CACHE: Dict[str, AsyncAnthropic] = {}


class _Message(TypedDict):
    role: str
    content: str


class AnthropicTextGenerator(TextGeneratorAPI):
    """Generate text using Anthropic’s Claude models (default: **claude-4o**).

    The class relies on the ``anthropic`` package and an ``ANTHROPIC_API_KEY``
    environment variable being present.

    ``prompt`` may be either:

    • **str** – treated as a single ``{"role": "user", "content": <prompt>}``
      message.

    • **Sequence[dict]** – exactly the list you would pass to the Anthropic
      SDK’s ``messages`` parameter (each dict must contain ``role`` and
      ``content`` keys).

    The response text is returned directly, truncated by Anthropic to the
    specified ``max_tokens`` server-side (1 024 here).
    """

    def __init__(self, model: str = "claude-4o") -> None:
        self.model = model

    # ---------------------------------------------------------------- helpers

    def _get_client(self) -> AsyncAnthropic:
        """Return (and cache) a shared ``AsyncAnthropic`` client instance."""
        if "default" not in _CLIENT_CACHE:
            _CLIENT_CACHE["default"] = AsyncAnthropic()  # picks up API key
        return _CLIENT_CACHE["default"]

    # ---------------------------------------------------------------- public

    async def generate(
        self,
        prompt: Union[str, Sequence[_Message]],
    ) -> str:
        """Return Claude’s reply for *prompt* as a plain string."""
        # Normalise the prompt into a list of message dicts.
        if isinstance(prompt, str):
            messages: List[_Message] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, Sequence):
            # A very light validation to help catch obvious misuse.
            if not all(
                isinstance(m, dict) and "role" in m and "content" in m
                for m in prompt
            ):
                raise TypeError(
                    "Each message must be a dict with 'role' and 'content' keys"
                )
            messages = list(prompt)  # type: ignore[arg-type]
        else:
            raise TypeError(
                "prompt must be either a string or a sequence of message dicts"
            )

        client = self._get_client()

        async def _call_sdk(msgs: Sequence[_Message]) -> str:
            response = await client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=msgs,
            )
            # SDK returns a list of content blocks; first block is the main text.
            block = response.content[0]
            return (block.text if hasattr(block, "text") else str(block)).strip()

        return await _call_sdk(messages)
