"""Turning Jev's picks into text, and chat text into words Jev can pick."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Punctuation Jev can pick as a "word", shown bare. Probed 2026-09-26 (5 states x 2 runs): Jev reads
# the bare marks fine (~0 mid-phrase, more where they fit: 0.29-0.33 vs 0.18-0.19 after a finished
# answer), and the old descriptions ("an ellipsis, trailing off") pulled it toward "..." over ".".
PUNCTUATION: tuple[str, ...] = (".", ",", "?", "!", "...")

_WORD = re.compile(r"[a-z0-9][a-z0-9']*")


def join_words(words: Iterable[str]) -> str:
    """Join picked tokens into text: spaces between words, punctuation attached."""
    out = ""
    for w in words:
        out += w if (w in PUNCTUATION or not out) else " " + w
    return out


def words_in(text: str) -> list[str]:
    """Lowercase words in ``text`` in order (the same shape as vocabulary entries)."""
    # Curly apostrophes (phone keyboards) would otherwise split "don't" in two.
    return _WORD.findall(text.lower().replace("\u2019", "'"))


def context_words(texts: Iterable[str], limit: int) -> list[str]:
    """Distinct words from ``texts`` (earlier texts first), at most ``limit``.

    Pass the chat newest-first so that, when the chat is long, the words that
    survive the limit are the ones from the latest messages.
    """
    seen: dict[str, None] = {}
    for text in texts:
        for w in words_in(text):
            if w not in seen:
                seen[w] = None
                if len(seen) == limit:
                    return list(seen)
    return list(seen)
