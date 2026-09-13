# text_generators/openrouter_completion.py
"""Raw text-completion backend for OpenRouter.

Unlike the chat generators, this sends the prompt verbatim to OpenRouter's
legacy ``/completions`` endpoint and returns the model's continuation. There is
no system prompt, no role framing, and no CLI-simulation trick: the model just
keeps writing from wherever the prompt stops.

Intended for base-like models (e.g. ``nousresearch/hermes-4-405b``) where the
goal is a "what comes next" snippet rather than an assistant reply.
"""

from __future__ import annotations

import logging

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from .base import TextGeneratorAPI
from .openrouter import _get_openrouter_client

_LOG = logging.getLogger(__name__)

DEFAULT_COMPLETION_MODEL = "nousresearch/hermes-4-405b"


class OpenRouterCompletionGenerator(TextGeneratorAPI):
    """Continue a raw text prompt via OpenRouter's ``/completions`` endpoint.

    Requires OPENROUTER_API_KEY in the environment.
    """

    def __init__(self, model: str = DEFAULT_COMPLETION_MODEL, *, max_tokens: int = 500) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def _get_client(self) -> AsyncOpenAI:
        return _get_openrouter_client()

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's continuation of *prompt* (the prompt itself is not echoed)."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string for raw completions")

        client = self._get_client()
        limit = max_tokens if max_tokens is not None else self.max_tokens

        _LOG.info(
            "OpenRouter completion: model=%s, prompt_len=%d, temp=%.2f, max_tokens=%d",
            self.model,
            len(prompt),
            temperature,
            limit,
        )

        try:
            resp = await client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=limit,
            )
        except RateLimitError as e:
            _LOG.warning("OpenRouter completion rate limit hit for model %s: %s", self.model, e)
            raise
        except APIConnectionError as e:
            _LOG.error("OpenRouter completion connection error for model %s: %s", self.model, e)
            raise
        except APIError as e:
            _LOG.error(
                "OpenRouter completion API error for model %s (status %s): %s",
                self.model,
                getattr(e, "status_code", "unknown"),
                e,
            )
            raise

        choice = resp.choices[0]
        text = choice.text or ""

        _LOG.info(
            "OpenRouter completion result: model=%s, finish_reason=%s, provider=%s, text_len=%d",
            self.model,
            getattr(choice, "finish_reason", None),
            getattr(resp, "provider", None),
            len(text),
        )

        # Only strip the trailing side: leading whitespace is part of the
        # continuation (e.g. " ten-way is a decagon" after a prompt ending in "a").
        return text.rstrip()
