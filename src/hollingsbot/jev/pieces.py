"""Jev writes in pieces: the LLM's raw next tokens are its menu, not whole words.

The word writer (writer.py) turns the LLM's tokens into whole words before Jev
sees them. Here Jev gets the tokens as they come, word pieces and all ("chang",
"ing", "cr", "..."), and the reply is whatever it strings together. Tried
locally on 2026-09-26 (4 prompts x 2 seeds each way): mostly readable while
Jev picks pieces that are words, and happily lost once it takes a fragment it
can't finish ("har tober h fest val ival events"). The user liked that.

Each piece:

1. The LLM's top next tokens after Jev's reply so far (the word writer's
   suggester call; ``llm_pages`` > 1 asks for 200), minus what the no-repeat
   rule bans, in shuffled pages of ``page_size``; then, with ``own_page``, a
   page of Jev's own words (born, chat and learned) it hasn't passed.
2. One Jev request: which piece goes in the blank (STOP is on the menu from
   ``min_words`` pieces on) and, when another page follows, whether it would
   rather type something that isn't listed (sampled; that turns the page).
3. The pick is sampled (``temperature``, ``top_p``) and joined on.

Joining: chat-mode tokens carry no leading space, so nothing says whether a
piece finishes the last word or starts a new one. It attaches when the two
spell a dictionary or chat word or the start of one ("chang" + "e", "for" +
"ward"); punctuation and apostrophe pieces always attach; anything else gets a
space. That misfires ("SoI", "lay... ing"), which is part of the charm.

No repeats (``no_repeat``): with "never" a piece is off every later menu once
used, so "the the the" and "ng ing ng" spirals can't happen; Jev runs out of
easy pieces and reaches for strange ones instead.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from hollingsbot.jev.client import JevError
from hollingsbot.jev.decoding import pick
from hollingsbot.jev.text import context_words
from hollingsbot.jev.writer import (
    BLANK,
    DEEP_TOP,
    NO_REPEAT_MODES,
    STOP,
    UNITS,
    JevWriter,
    Reply,
    Step,
    _chat_json,
    _newest_first,
    _other,
    _probabilities,
)

if TYPE_CHECKING:
    import random
    from collections.abc import Iterable, Sequence

    from hollingsbot.jev.client import DecisionsClient, Usage
    from hollingsbot.jev.suggest import NextWordSuggester
    from hollingsbot.jev.writer import ChatLine, OnWord, WriterConfig

_LOG = logging.getLogger(__name__)

# The word writer's "none of these" question, for pieces.
PIECE_OTHER_OPTIONS = {"pick": "pick one of `options`", "other": "type something that isn't in `options`"}
_LAST_WORD = re.compile(r"[A-Za-z']+$")


def prefixes(words: Iterable[str]) -> set[str]:
    """Every prefix of every word, lowercase: what a piece may attach to make."""
    out: set[str] = set()
    for word in words:
        w = word.lower()
        out.update(w[:i] for i in range(1, len(w) + 1))
    return out


def join_piece(text: str, piece: str, joins: set[str]) -> str:
    """``text`` + ``piece``, attached or after a space (pure; see the module docstring)."""
    if not text:
        return piece
    if not piece[0].isalnum():  # "...", ",", "'s", "-light"
        return text + piece
    last = _LAST_WORD.search(text)
    if last and (last.group(0) + piece).lower().replace("’", "'") in joins:
        return text + piece
    return f"{text} {piece}"


def clean_piece(token: str) -> str | None:
    """A token as a menu option, or None for one that can't be one (blank, newline, special, broken)."""
    piece = token.strip()
    if not piece or piece == STOP or any(bad in piece for bad in ("\n", "<|", "|>", "�")):
        return None
    return piece


class PieceWriter(JevWriter):
    """Writes one reply per :meth:`write` call from the LLM's raw next pieces (needs a suggester)."""

    def __init__(
        self,
        client: DecisionsClient,
        *,
        name: str = "Jev",
        config: WriterConfig | None = None,
        vocab: Sequence[str] | None = None,
        rng: random.Random | None = None,
        suggester: NextWordSuggester | None = None,
    ) -> None:
        if suggester is None:
            raise ValueError("PieceWriter needs a suggester: the LLM's next tokens are its menu")
        super().__init__(client, name=name, config=config, vocab=vocab, rng=rng, suggester=suggester)
        if self.config.no_repeat not in NO_REPEAT_MODES:
            raise ValueError(f"unknown no_repeat mode {self.config.no_repeat!r}")
        style = f"{self._style_line} " if self._style_line else ""
        self._piece_instructions = (
            f"{name} is a member of this Discord chat and is writing a reply to the latest message. {style}"
            f"`{self._reply_key}` is {name}'s reply so far, typed a piece at a time. The blank ({BLANK}) is "
            "where it would continue, if it continues. Each option is a piece of text: it either finishes the "
            "word right before the blank or starts the next word. Which piece goes in the blank, or is the "
            "message finished?"
        )
        self._piece_other_instructions = f"{name} is choosing the piece for the blank. What does {name} do?"
        self._vocab_prefixes = prefixes(self.vocab)

    async def write(
        self,
        chat: Sequence[ChatLine],
        on_word: OnWord | None = None,
        draft: Reply | None = None,
        learned: Sequence[str] = (),
    ) -> Reply:
        """Write a reply to the last line of ``chat``; ``Reply.words`` holds the pieces."""
        cfg = self.config
        reply = draft if draft is not None else Reply()
        reply.model = self.client.model
        pieces, steps, usage = reply.words, reply.steps, reply.usage
        chat_state = {"chat": _chat_json(chat)}
        own = self._single_pool(chat, learned)
        joins = self._vocab_prefixes | prefixes(context_words(_newest_first(chat), 1000) + list(learned))
        used: list[str] = []  # lowercase, in order
        stop_reason = "max_words"

        while len(pieces) < cfg.max_words:
            pages, suggested = await self._pages_for(chat, reply.text, own, self._banned(used), usage)
            if not pages:
                stop_reason = "no_options"
                break
            if len(pieces) >= cfg.min_words:
                pages = [(src, self._with_stop(opts)) for src, opts in pages]
            page = 0
            source, options = pages[0]
            more = len(pages) > 1
            answers = await self.client.ask(chat_state, self._piece_questions(reply.text, options, more), usage)
            other = _other(answers) if more else 0.0
            while more and self._turns_page(other):
                page += 1
                source, options = pages[page]
                more = page + 1 < len(pages)
                answers = await self.client.ask(chat_state, self._piece_questions(reply.text, options, more), usage)
                other = _other(answers) if more else 0.0

            probs = _probabilities(answers["next"])
            piece = pick({o: probs.get(o, 0.0) for o in options}, self.rng, cfg.temperature, cfg.top_p)
            steps.append(
                Step(
                    piece,
                    probs.get(piece, 0.0),
                    0.0,  # no naturalness check for pieces
                    0.0,
                    1.0,
                    len(options),
                    suggested.get(piece, 0.0),
                    page=page,
                    other=other,
                    source=source,
                )
            )
            if piece == STOP:
                stop_reason = "chose_stop"
                break
            pieces.append(piece)
            used.append(piece.lower())
            reply.text = join_piece(reply.text, piece, joins)
            if on_word is not None:
                await on_word(reply.text)

        reply.stop_reason = stop_reason
        return reply

    def _banned(self, used: Sequence[str]) -> set[str]:
        if self.config.no_repeat == "never":
            return set(used)
        if self.config.no_repeat == "adjacent":
            return set(used[-1:])
        return set()

    async def _pages_for(
        self, chat: Sequence[ChatLine], text: str, own: Sequence[str], banned: set[str], usage: Usage
    ) -> tuple[list[tuple[str, list[str]]], dict[str, float]]:
        """The menus for one piece as (source, options), and the LLM's probability of each piece."""
        cfg = self.config
        try:
            tokens = await self.suggester.top_tokens(
                chat, self.name, [text] if text else [], self._style_line, usage,
                DEEP_TOP if cfg.llm_pages > 1 else None,
            )  # fmt: skip
        except JevError:
            # The LLM is a helper: without it, this piece comes from Jev's own words.
            _LOG.warning("Suggester failed; using Jev's own words for this piece", exc_info=True)
            tokens = []
        suggested: dict[str, float] = {}  # in the LLM's rank order
        for token, p in tokens:
            piece = clean_piece(token)
            if piece is not None and piece.lower() not in banned:
                suggested[piece] = suggested.get(piece, 0.0) + p
        size = max(1, cfg.page_size)
        shown = list(suggested)[: size * max(1, cfg.llm_pages)]
        pages = [("llm", shown[i : i + size]) for i in range(0, len(shown), size)]
        if cfg.own_page or not pages:
            passed = {p.lower() for p in shown} | banned
            pages.append(("own", [w for w in own if w not in passed]))
        pages = [(src, opts) for src, opts in pages if opts]
        if cfg.shuffle:
            for _, options in pages:
                self.rng.shuffle(options)
        return pages, suggested

    def _piece_questions(self, text: str, options: Sequence[str], other: bool) -> dict[str, dict[str, Any]]:
        blank = f"{text}{BLANK}"
        questions: dict[str, dict[str, Any]] = {
            "next": {
                "type": "choice",
                "instructions": {self._reply_key: blank, "question": self._piece_instructions},
                "criteria": {o: self._stop_description if o == STOP else None for o in options},
            }
        }
        if other:
            questions["other"] = {
                "type": "choice",
                "instructions": {
                    self._reply_key: blank,
                    "options": ", ".join(options),
                    "question": self._piece_other_instructions,
                },
                "criteria": PIECE_OTHER_OPTIONS,
            }
        return questions


def make_writer(
    client: DecisionsClient,
    *,
    name: str,
    config: WriterConfig,
    suggester: NextWordSuggester | None,
    rng: random.Random | None = None,
) -> JevWriter:
    """The writer ``config.units`` asks for: whole words (JevWriter) or pieces (PieceWriter)."""
    if config.units not in UNITS:
        raise ValueError(f"unknown units {config.units!r}")
    cls = PieceWriter if config.units == "pieces" else JevWriter
    return cls(client, name=name, config=config, suggester=suggester, rng=rng)
