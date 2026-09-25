"""Turning Jev's picks into text, and chat text into words Jev can pick."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Punctuation Jev can pick as a "word". The descriptions are what Jev reads.
PUNCTUATION: dict[str, str] = {
    ".": "a period, ending the sentence",
    ",": "a comma",
    "?": "a question mark",
    "!": "an exclamation mark",
    "...": "an ellipsis, trailing off",
}

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
