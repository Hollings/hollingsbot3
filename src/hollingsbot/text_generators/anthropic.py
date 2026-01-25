"""Text-generation backend that calls Anthropic's Claude models."""
from __future__ import annotations

import logging
import os
from typing import Dict, Sequence, Union, TypedDict, Any, List

from anthropic import AsyncAnthropic, APIError, APIConnectionError, RateLimitError

from .base import TextGeneratorAPI

_log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# A single shared client is plenty; reuse it across all requests              #
# --------------------------------------------------------------------------- #
_CLIENT_CACHE: Dict[str, AsyncAnthropic] = {}


class _Message(TypedDict):
    role: str
    content: Union[str, List[Dict[str, Any]]]


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
    specified ``max_tokens`` server-side (16384 by default for Claude 4+,
    configurable via ANTHROPIC_MAX_TOKENS environment variable).
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
        temperature: float = 1.0,
    ) -> str:
        """Return Claude's reply for *prompt* as a plain string.

        Accepts either a single user string or a list of role/content messages.
        Content can be either a string or a list of content blocks (for images).
        Any messages with role "system" are moved to the top‑level "system"
        parameter as required by the Anthropic Messages API.
        """
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

        # Extract system messages (Anthropic expects top-level `system`, not a
        # message role). Concatenate multiple system entries with blank lines.
        def _content_to_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    try:
                        t = item.get("type")
                    except Exception:
                        t = None
                    if t == "text":
                        parts.append(item.get("text", ""))
                    else:
                        # Non-text in system is unusual; stringify conservatively
                        parts.append(str(item))
                return "\n".join(p for p in parts if p)
            return str(content)

        system_parts: List[str] = []
        cleaned: List[Dict[str, Any]] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            if role == "system":
                system_parts.append(_content_to_text(m.get("content")))
            else:
                # Pass through content as-is for user/assistant messages
                # (can be string or list of content blocks with images)
                content = m.get("content")
                is_cacheable = m.get("_cacheable", False)

                # If cacheable, add cache_control to content
                if is_cacheable and content:
                    if isinstance(content, str):
                        content = [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    elif isinstance(content, list):
                        # Add cache_control to last content block
                        content = list(content)  # Copy to avoid mutating original
                        if content:
                            last_block = dict(content[-1])
                            last_block["cache_control"] = {"type": "ephemeral"}
                            content[-1] = last_block

                cleaned.append({"role": role, "content": content})
        system_text = "\n\n".join(p for p in system_parts if p).strip() or None

        client = self._get_client()

        async def _call_sdk(msgs: Sequence[Dict[str, Any]], system: str | None, temp: float) -> str:
            max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "16384"))
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": msgs,
                "temperature": temp,
            }
            if system:
                # Use prompt caching for system prompt (90% cheaper on cache hits)
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            try:
                response = await client.messages.create(**kwargs)
            except RateLimitError as e:
                _log.warning("Anthropic rate limit hit for model %s: %s", self.model, e)
                raise
            except APIConnectionError as e:
                _log.error("Anthropic connection error for model %s: %s", self.model, e)
                raise
            except APIError as e:
                _log.error("Anthropic API error for model %s (status %s): %s", self.model, e.status_code, e.message)
                raise

            # SDK returns a list of content blocks; aggregate text blocks.
            parts: List[str] = []
            for block in getattr(response, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts).strip()

        return await _call_sdk(cleaned, system_text, temperature)
