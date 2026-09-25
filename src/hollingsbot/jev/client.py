"""Async client for Jev, TypeSafe's decision model, via OpenRouter's Decisions API.

Jev is not a chat model. A request carries a ``state`` (the situation, as text
or JSON) and a map of named, typed questions; every question is evaluated in
parallel and in isolation against that state, and comes back as a probability
distribution:

- ``choice``: pick one option from ``criteria`` (max 255 options) ->
  ``{"choice", "probabilities", "confidence"}``
- ``noul``: a yes/no question -> ``{"noul": P(yes)}``

Adding questions to a request barely changes its latency (~0.4-0.6 s), so the
writer packs a whole step's worth of questions into one call. Billing is per
input token (output is free). Docs: https://docs.typesafe.ai
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

_LOG = logging.getLogger(__name__)

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
# The tilde is part of the slug: without it OpenRouter answers "model does not exist".
DEFAULT_MODEL = "~typesafe/jev-latest"

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})

Answers = dict[str, dict[str, Any]]


class JevError(RuntimeError):
    """The Decisions API refused a request or kept failing it."""


@dataclass
class Usage:
    """Running totals for one piece of work (e.g. one Discord reply)."""

    calls: int = 0
    input_tokens: int = 0
    cost: float = 0.0
    seconds: float = 0.0

    def record(self, usage: dict[str, Any] | None, seconds: float) -> None:
        usage = usage or {}
        self.calls += 1
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.cost += float(usage.get("cost") or 0.0)
        self.seconds += seconds


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return min(0.75 * 2**attempt, 8.0)


class DecisionsClient:
    """Thin async wrapper over ``POST /api/alpha/decisions``.

    ``transport`` exists for tests (``httpx.MockTransport``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        retries: int = 4,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise JevError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.retries = retries
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
            transport=transport,
        )

    async def ask(self, state: Any, questions: dict[str, dict[str, Any]], usage: Usage | None = None) -> Answers:
        """Evaluate ``questions`` against ``state``; returns the ``answers`` map."""
        payload = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(self.retries + 1):
            last_try = attempt == self.retries
            started = time.monotonic()
            try:
                resp = await self._client.post(DECISIONS_URL, json=payload)
            except httpx.TransportError as exc:
                if last_try:
                    raise JevError(f"Decisions API unreachable: {exc}") from exc
                _LOG.warning("Jev transport error (%s); retrying", exc)
                await asyncio.sleep(_backoff(attempt, None))
                continue

            if resp.status_code in _RETRY_STATUSES and not last_try:
                delay = _backoff(attempt, resp.headers.get("Retry-After"))
                _LOG.warning("Jev answered %s; retrying in %.1fs", resp.status_code, delay)
                await asyncio.sleep(delay)
                continue
            if resp.status_code != 200:
                raise JevError(f"Decisions API answered {resp.status_code}: {resp.text[:300]}")

            body = resp.json()
            if usage is not None:
                usage.record(body.get("usage"), time.monotonic() - started)
            answers = body.get("answers")
            if not isinstance(answers, dict):
                raise JevError(f"Decisions API response has no answers: {str(body)[:300]}")
            return answers
        raise JevError("unreachable")  # the loop always returns or raises

    async def aclose(self) -> None:
        await self._client.aclose()
