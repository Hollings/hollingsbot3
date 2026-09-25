"""Jev's learned vocabulary: every word a human says in its channel, remembered.

Jev is born knowing only the ``WriterConfig.vocab_size`` most common English
words. Each human message teaches it the words in it: how often it has heard
each one, when it last did, and who said it first. When Jev writes, the words
it reaches for first are the ones it has heard most, most recently
(:meth:`Lexicon.ranked`); a word nobody says any more fades out of reach but is
never deleted, so it comes back the moment someone uses it again.

It only learns from humans, never from its own replies, so it cannot talk
itself into a rut.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from hollingsbot.jev.db import connect
from hollingsbot.jev.text import words_in
from hollingsbot.jev.vocab import BLOCKED

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS jev_lexicon (
        word TEXT PRIMARY KEY,
        uses INTEGER NOT NULL,
        first_heard TEXT NOT NULL,
        last_heard TEXT NOT NULL,
        taught_by TEXT,
        channel_id INTEGER
    )
    """,
)

MAX_WORD_LEN = 24
# Things that are text but not words: links, custom emoji, mentions' raw ids, code.
_NOT_SPEECH = re.compile(r"https?://\S+|<a?:\w+:\d+>|<[@#][!&]?\d+>|```.*?```|`[^`]*`", re.DOTALL)


def learnable(text: str) -> list[str]:
    """The distinct words in ``text`` Jev may learn, in order."""
    words = words_in(_NOT_SPEECH.sub(" ", text))
    return [w for w in dict.fromkeys(words) if w not in BLOCKED and len(w) <= MAX_WORD_LEN]


@dataclass(frozen=True)
class LexiconSummary:
    total: int  # distinct words ever learned
    newest: list[tuple[str, str]]  # (word, who taught it), newest first
    favorites: list[tuple[str, int]]  # (word, times heard), most heard first


class Lexicon:
    """``db_path`` defaults to the bot DB; ``half_life_days`` is how fast unused words fade."""

    def __init__(self, db_path: str | Path | None = None, *, half_life_days: float = 7.0) -> None:
        self._db_path = db_path
        self.half_life_days = half_life_days

    def _connect(self):
        return connect(self._db_path, _SCHEMA)

    def learn(
        self, text: str, *, speaker: str, channel_id: int | None = None, now: datetime | None = None
    ) -> list[str]:
        """Hear one message. Returns the words Jev had never heard before."""
        words = learnable(text)
        if not words:
            return []
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        with self._connect() as conn:
            marks = ",".join("?" * len(words))
            known = {row[0] for row in conn.execute(f"SELECT word FROM jev_lexicon WHERE word IN ({marks})", words)}
            conn.executemany(
                """
                INSERT INTO jev_lexicon (word, uses, first_heard, last_heard, taught_by, channel_id)
                VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(word) DO UPDATE SET uses = uses + 1, last_heard = excluded.last_heard
                """,
                [(w, stamp, stamp, speaker, channel_id) for w in words],
            )
        return [w for w in words if w not in known]

    def ranked(self, limit: int, *, exclude: Iterable[str] = (), now: datetime | None = None) -> list[str]:
        """Up to ``limit`` learned words, the most heard and most recently heard first."""
        now = now or datetime.now(timezone.utc)
        skip = set(exclude)
        with self._connect() as conn:
            rows = conn.execute("SELECT word, uses, last_heard FROM jev_lexicon").fetchall()
        scored = []
        for word, uses, last_heard in rows:
            if word in skip:
                continue
            age_days = max(0.0, (now - datetime.fromisoformat(last_heard)).total_seconds() / 86400)
            scored.append((uses * 0.5 ** (age_days / self.half_life_days), last_heard, word))
        scored.sort(reverse=True)
        return [word for _, _, word in scored[:limit]]

    def summary(self, *, exclude: Iterable[str] = (), shown: int = 8) -> LexiconSummary:
        """Counts and highlights for humans, leaving out words in ``exclude`` (e.g. the ones Jev was born with)."""
        skip = set(exclude)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT word, uses, first_heard, taught_by FROM jev_lexicon ORDER BY first_heard DESC"
            ).fetchall()
        learned = [r for r in rows if r[0] not in skip]
        newest = [(w, by or "someone") for w, _, _, by in learned[:shown]]
        favorites = [(w, uses) for w, uses, _, _ in sorted(learned, key=lambda r: (-r[1], r[0]))[:shown]]
        return LexiconSummary(len(learned), newest, favorites)
