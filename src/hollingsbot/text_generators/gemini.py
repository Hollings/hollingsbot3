# text_generators/gemini.py
"""Gemini 3 API text generator using the google-genai SDK."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, TypedDict, Union

from .base import TextGeneratorAPI

if TYPE_CHECKING:
    from collections.abc import Sequence

_LOG = logging.getLogger(__name__)

# Lazy load the client to avoid import errors if not installed
_CLIENT_CACHE: dict[str, Any] = {}


class _Message(TypedDict):
    role: str
    content: str


class GeminiTextGenerator(TextGeneratorAPI):
    """Text-generation backend for Google Gemini 3 models.

    Uses the google-genai SDK (version 1.51.0+).
    Requires GOOGLE_API_KEY or GEMINI_API_KEY in the environment.

    Accepts either a single string or a list of {role, text, images} messages.
    """

    def __init__(self, model: str = "gemini-3-pro-preview") -> None:
        self.model = model

    def _get_client(self) -> Any:
        """Get or create the Gemini client."""
        if "default" not in _CLIENT_CACHE:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. Install with: pip install google-genai"
                )

            # Check for API key
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable required")

            _CLIENT_CACHE["default"] = genai.Client(api_key=api_key)
        return _CLIENT_CACHE["default"]

    def _convert_messages_to_contents(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[Any]]:
        """Convert internal message format to Gemini content format.

        Returns (system_instruction, contents) where:
        - system_instruction: The system prompt (if any)
        - contents: List of content parts for the conversation
        """
        from google.genai import types

        system_instruction: str | None = None
        contents: list[Any] = []

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("text", "")
            images = msg.get("images", [])

            # Handle system message
            if role == "system":
                system_instruction = text
                continue

            # Map roles: user stays user, assistant becomes model
            gemini_role = "model" if role == "assistant" else "user"

            # Build content parts
            parts: list[Any] = []

            # Add text content
            if text:
                parts.append(types.Part.from_text(text=text))

            # Add image content
            for img in images:
                if isinstance(img, dict):
                    # Handle data URL format: {"data": "base64...", "content_type": "image/jpeg"}
                    if "data" in img:
                        import base64
                        # Extract base64 data (may be prefixed with data:...)
                        data = img["data"]
                        if data.startswith("data:"):
                            # Parse data URL: data:image/jpeg;base64,<data>
                            header, encoded = data.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                        else:
                            encoded = data
                            mime_type = img.get("content_type", "image/jpeg")

                        try:
                            image_bytes = base64.b64decode(encoded)
                            parts.append(
                                types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type=mime_type,
                                )
                            )
                        except Exception as e:
                            _LOG.warning("Failed to decode image: %s", e)

            if parts:
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=parts,
                    )
                )

        return system_instruction, contents

    async def generate(
        self,
        prompt: Union[str, Sequence[dict[str, Any]]],
        *,
        temperature: float = 1.0,
    ) -> str:
        """Generate text using Gemini 3.

        Args:
            prompt: Either a string or a sequence of message dicts with
                    keys: role, text, images
            temperature: Sampling temperature (0.0-2.0)

        Returns:
            Generated text response
        """
        from google.genai import types

        client = self._get_client()

        # Normalize input
        if isinstance(prompt, str):
            messages = [{"role": "user", "text": prompt, "images": []}]
        else:
            messages = list(prompt)

        # Convert to Gemini format
        system_instruction, contents = self._convert_messages_to_contents(messages)

        # Build config
        config = types.GenerateContentConfig(
            temperature=temperature,
            # Use default thinking level for balanced speed/quality
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW  # Use LOW for faster responses
            ),
        )

        # Add system instruction if present
        if system_instruction:
            config.system_instruction = system_instruction

        try:
            # Make the API call
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # Extract text from response
            text = getattr(response, "text", None)
            if text:
                return text.strip()

            # Fallback: try to extract from candidates
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    parts = getattr(candidate.content, "parts", [])
                    text_parts = []
                    for part in parts:
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return "\n".join(text_parts).strip()

            _LOG.warning("Gemini returned empty response for model=%s", self.model)
            return ""

        except Exception as e:
            _LOG.exception("Gemini API error: %s", e)
            raise
