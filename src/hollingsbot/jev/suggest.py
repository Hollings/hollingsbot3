"""A small LLM proposes Jev's next word; Jev chooses.

Jev judges well but cannot put words in order. A tiny LLM can: asked to
continue the chat log, it returns its top next tokens with probabilities
(one ~0.4 s call, max_tokens 1, top_logprobs 20). Those become Jev's menu
for the next word, so every option is already a fluent continuation and Jev's
choice among them is what makes the reply Jev's.

LLMs predict word pieces, not words. A piece that begins a word from the chat
or Jev's learned vocabulary becomes that word ("ram" -> "ramen", "sp" ->
"spicy": rare words are exactly the ones that arrive in pieces, and the rare
words worth saying are usually the ones someone just said). A piece that is a
whole dictionary word stays itself; anything else is dropped.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from hollingsbot.jev.client import JevError, Usage
from hollingsbot.jev.text import join_words, words_in

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from hollingsbot.jev.writer import ChatLine

_LOG = logging.getLogger(__name__)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_SUGGEST_MODEL = "meta-llama/llama-3.1-8b-instruct"
# top_logprobs every logprobs provider serves. Deeper lists: only these (2026-09-25, llama-3.1-8b:
# Novita returns 200 in ~0.75 s; CoreWeave caps at 20 and answers deeper asks with an error body).
SHALLOW_TOP = 20
DEEP_PROVIDERS = ("novita",)
_PUNCT_TOKENS = {".": ".", ",": ",", "?": "?", "!": "!", "...": "...", "…": "..."}
_PIECE = re.compile(r"'?[a-z]+(?:'[a-z]+)?")
# Halves of contractions that wordfreq lists as words ("you ll", "i ve", "don t"): never
# accept them as a word of their own; after their other half they merge (see continuation).
_CONTRACTION_PIECES = frozenset(
    {"ll", "ve", "re", "don", "didn", "doesn", "isn", "wasn", "aren", "couldn", "wouldn", "shouldn", "hasn",
     "haven", "weren", "ain"}
)  # fmt: skip


@dataclass(frozen=True)
class Suggestion:
    options: list[tuple[str, float]]  # (word, probability), best first
    # The last word, finished, when the model is still spelling it ("sand" -> "sandwich").
    completes: str | None = None


def continuation(
    last: str, tokens: Sequence[tuple[str, float]], *, whole_words: Iterable[str], heard: Iterable[str]
) -> str | None:
    """``last`` + the model's favourite token, if that spells a real word (pure).

    Pieces that are also words ("sand", "fill", "ton") slip through as options;
    when the next thing the model wants is the rest of that word ("wich", "ing",
    "ight") the reply was mid-word, and the two belong together.
    """
    if not tokens:
        return None
    token, p = max(tokens, key=lambda kv: kv[1])
    piece = token.strip().lower().replace("’", "'")
    if p < 0.3 or token[:1].isspace() or not _PIECE.fullmatch(piece):
        return None
    merged = last + piece  # "don" + "'t", "sand" + "wich"
    known = set(whole_words) | set(heard)
    return merged if merged in known else None


def words_from_tokens(
    tokens: Iterable[tuple[str, float]],
    *,
    whole_words: Sequence[str],
    complete_from: Iterable[str],
    protect_top: int = 1000,
) -> list[tuple[str, float]]:
    """Turn (token, probability) pairs into (word, probability), best first (pure).

    ``complete_from`` (chat + learned words) resolves pieces to the words they
    start; ``whole_words`` (a dictionary, most frequent first) accepts pieces
    that are words already.
    """
    dictionary = set(whole_words)
    protected = set(whole_words[:protect_top])
    targets = sorted(set(complete_from), key=len)
    # Heard words the dictionary lacks ("ramen"): a 3+ letter piece that begins one is
    # taken to mean it, even if the piece is a less common word itself ("ram"), because
    # that is how the model spells a rare word it just read. The most common words
    # ("the" in front of a heard "theremin") are never "completed".
    rare = [t for t in targets if t not in dictionary]
    merged: dict[str, float] = {}
    for token, p in tokens:
        raw = token.strip()
        if raw in _PUNCT_TOKENS:
            word: str | None = _PUNCT_TOKENS[raw]
        else:
            found = words_in(raw)
            if len(found) != 1:
                continue
            piece = found[0]
            if piece in _CONTRACTION_PIECES or raw.startswith(("'", "’")):
                continue  # half a contraction ("don", "'ll"): only ever a continuation
            word = None
            if len(piece) >= 3 and piece not in protected:
                word = next((t for t in rare if t != piece and t.startswith(piece)), None)
            if word is None and (piece in dictionary or piece in targets):
                word = piece
            elif word is None:
                word = next((t for t in targets if t.startswith(piece) and len(piece) >= 2), None)
        if word:
            merged[word] = merged.get(word, 0.0) + p
    return sorted(merged.items(), key=lambda kv: -kv[1])


class NextWordSuggester:
    """``transport`` exists for tests (``httpx.MockTransport``)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_SUGGEST_MODEL,
        top: int = SHALLOW_TOP,
        timeout: float = 20.0,
        retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise JevError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.top = top
        self.retries = retries
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": f"Bearer {key}"}, transport=transport
        )

    def _request(
        self, chat: Sequence[ChatLine], name: str, words: Sequence[str], style: str, top: int
    ) -> dict[str, Any]:
        log = "\n".join(f"{line.speaker}: {line.text}" for line in chat)
        system = f"Continue this Discord chat log. {style} Output only the next words of {name}'s last message."
        # Only providers that actually return logprobs, fastest first. require_parameters does
        # not check how many: a deep list must go where it is served, or it errors.
        provider: dict[str, Any] = {"require_parameters": True, "sort": "latency"}
        if top > SHALLOW_TOP:
            provider["only"] = list(DEEP_PROVIDERS)
        return {
            "model": self.model,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": top,
            "messages": [
                {"role": "system", "content": " ".join(system.split())},
                {"role": "user", "content": f"{log}\n{name}: {join_words(words)}".rstrip()},
            ],
            "provider": provider,
        }

    async def top_tokens(
        self,
        chat: Sequence[ChatLine],
        name: str,
        words: Sequence[str],
        style: str = "",
        usage: Usage | None = None,
        top: int | None = None,
    ) -> list[tuple[str, float]]:
        """The model's top next tokens after ``name: words`` as (token, probability).

        A deep list (``top`` over 20) that fails is asked again at the usual depth:
        fewer pages beat no suggestions.
        """
        want = top or self.top
        try:
            return await self._top_tokens(self._request(chat, name, words, style, want), usage)
        except JevError:
            if want <= SHALLOW_TOP:
                raise
            _LOG.warning("Deep suggestion list (%d) failed; asking for %d", want, SHALLOW_TOP, exc_info=True)
            return await self._top_tokens(self._request(chat, name, words, style, SHALLOW_TOP), usage)

    async def _top_tokens(self, payload: dict[str, Any], usage: Usage | None) -> list[tuple[str, float]]:
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            try:
                resp = await self._client.post(CHAT_URL, json=payload)
            except httpx.TransportError as exc:
                if attempt == self.retries:
                    raise JevError(f"suggester unreachable: {exc}") from exc
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise JevError(f"suggester answered {resp.status_code}: {resp.text[:300]}")
            body = resp.json()
            if usage is not None:
                usage.record(body.get("usage"), time.monotonic() - started)
            try:
                first = body["choices"][0]["logprobs"]["content"][0]
                return [(t["token"], math.exp(t["logprob"])) for t in first["top_logprobs"]]
            except (KeyError, IndexError, TypeError) as exc:
                raise JevError(f"suggester returned no logprobs: {str(body)[:300]}") from exc
        raise JevError("unreachable")

    async def suggest(
        self,
        chat: Sequence[ChatLine],
        name: str,
        words: Sequence[str],
        *,
        whole_words: Iterable[str],
        complete_from: Iterable[str],
        style: str = "",
        usage: Usage | None = None,
        top: int | None = None,
    ) -> Suggestion:
        """``top`` overrides the number of tokens asked for (default: the constructor's)."""
        tokens = await self.top_tokens(chat, name, words, style, usage, top)
        vocab = list(whole_words)
        heard = list(complete_from)
        completes = continuation(words[-1], tokens, whole_words=vocab, heard=heard) if words else None
        return Suggestion(words_from_tokens(tokens, whole_words=vocab, complete_from=heard), completes)

    async def aclose(self) -> None:
        await self._client.aclose()
