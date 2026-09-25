"""Make Jev, a decision model that cannot generate text, write a chat reply.

Jev only answers typed questions (pick an option, say yes/no), so the reply is
built one word at a time, two rounds of requests per word:

1. **Buckets + stop check, concurrently.** The vocabulary (words from the chat,
   then ~5000 common English words) is split into buckets of 250, each asked as
   a parallel Choice ("which word goes in the blank?"): one request, ~0.6 s,
   however many buckets. Alongside it, a separate request asks whether to send
   the reply as it stands ("send" vs "keep typing") and whether it still makes
   sense. The stop check's state is the chat alone: showing it the reply with
   its trailing blank reads as "another word is coming" and Jev (a literal
   reader) then almost never says the reply is done.
2. **Runoff.** Each bucket's leaders meet in one final Choice, asked alongside
   a Noul per finalist ("is `next` a natural next word right after `text`?").
   A finalist's score is its runoff probability x fluency^2 x a repetition
   penalty; one is sampled (temperature + nucleus).

The reply ends when Jev says send (always above ``send_at``; between
``send_floor`` and that, with probability P(send)^2, the way an LLM samples its
end token), when it stops making sense, when even the picked word reads as
unnatural, or at ``max_words``.

Why this shape: a bake-off of letter-by-letter, several-letters-per-request,
letter-chunks, one small word list, and this tournament is written up in
README.md (next to this file).
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from hollingsbot.jev.client import DecisionsClient, JevError, Usage
from hollingsbot.jev.decoding import allowed, finalists, pick, repetition_factor
from hollingsbot.jev.text import PUNCTUATION, context_words, join_words
from hollingsbot.jev.vocab import load_vocab

_LOG = logging.getLogger(__name__)

BLANK = "___"
NEXT_WORD_Q = "Is `next` a natural next word right after `text`, in fluent English?"
FIRST_WORD_Q = "Does `text` read as fluent, grammatical English so far? It may be unfinished."
SENSE_Q = "Does `text` make sense as the start of a reply to the latest message in `chat`?"
SEND_OPTIONS = {"send": "send the message as it is", "keep typing": "add more to it first"}


@dataclass(frozen=True)
class ChatLine:
    speaker: str
    text: str


@dataclass(frozen=True)
class WriterConfig:
    vocab_size: int = 5000  # common words offered (the chat's own words come on top)
    bucket_size: int = 250  # Choice questions allow at most 255 options
    finalist_mass: float = 0.6  # a bucket sends its leaders until they hold this much probability...
    finalist_cap: int = 4  # ...or this many words
    fluency_power: float = 2.0  # how hard the naturalness Noul steers the pick
    repeat_penalty: float = 0.3  # score multiplier per earlier use of a content word
    common_repeat_penalty: float = 0.5  # ...and per use of a common word within the last few words
    repeat_window: int = 6
    exempt_top: int = 150  # "common words": the vocabulary's first N (the, a, is, more...)
    send_at: float = 0.6  # always send once Jev puts P(send) at least this high
    send_floor: float = 0.2  # between floor and send_at, send with probability P(send)**send_power
    send_power: float = 2.0  # (T > 0 only; mid-sentence P(send) sits below the floor)
    sense_floor: float = 0.3  # stop once the reply makes less sense than this...
    sense_from: int = 3  # ...judged from this many words on (one word is too little to judge)
    give_up: float = 0.25  # stop if even the picked word is rated less natural than this
    max_words: int = 25
    temperature: float = 0.7
    top_p: float = 0.6


@dataclass
class Step:
    word: str
    choice: float  # runoff probability
    fluency: float  # Noul: natural next word
    send: float  # Choice: P(send the reply as it was before this word)
    sense: float  # Noul: reply so far (before this word) makes sense
    options: int  # runoff size


@dataclass
class Reply:
    """A reply, filled in word by word; readable mid-flight (e.g. after a cancel)."""

    text: str = ""
    words: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    # sent | lost_thread | gave_up | max_words | no_options, or "writing" if cut short
    stop_reason: str = "writing"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    # the stop check that ended the reply: P(send), sense
    final_check: tuple[float, float] | None = None


OnWord = Callable[[str], Awaitable[None]]


class JevWriter:
    """Writes one reply per :meth:`write` call. Safe to reuse across replies."""

    def __init__(
        self,
        client: DecisionsClient,
        *,
        name: str = "Jev",
        config: WriterConfig | None = None,
        vocab: Sequence[str] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.client = client
        self.name = name
        self.config = config or WriterConfig()
        self.vocab = tuple(vocab if vocab is not None else load_vocab())
        self.common = frozenset(self.vocab[: self.config.exempt_top])
        self.rng = rng or random.Random()
        self._reply_key = re.sub(r"\W+", "_", name.lower()).strip("_") + "_reply"
        self._blank_instructions = (
            f"{name} is a member of this Discord chat and is writing a reply to the latest message. "
            f"`{self._reply_key}` is {name}'s reply so far, and the blank ({BLANK}) marks the next word. "
            "Which word goes in the blank?"
        )
        self._send_instructions = (
            f"{name} has typed `text` so far as a reply to the latest message in `chat`. What does {name} do now?"
        )

    # ------------------------------------------------------------------ public

    async def write(
        self,
        chat: Sequence[ChatLine],
        on_word: OnWord | None = None,
        draft: Reply | None = None,
    ) -> Reply:
        """Write a reply to the last line of ``chat``.

        ``on_word`` gets the text so far after each word. The reply is built in
        ``draft`` if given, so a caller that cancels this coroutine can still
        read what was written and what it cost.
        """
        cfg = self.config
        reply = draft if draft is not None else Reply()
        reply.model = self.client.model
        words, steps, usage = reply.words, reply.steps, reply.usage
        bucket_questions = self._bucket_questions(chat)
        chat_state = {"chat": _chat_json(chat)}
        stop_reason = "max_words"

        while len(words) < cfg.max_words:
            state = {**chat_state, self._reply_key: f"{join_words(words)} {BLANK}".strip()}
            if words:
                answers, check = await asyncio.gather(
                    self.client.ask(state, bucket_questions, usage),
                    self.client.ask(chat_state, self._stop_questions(words), usage),
                )
                send = _probabilities(check["send"]).get("send", 0.0)
                sense = _noul(check, "sense")
                reason = self._stop_reason(send, sense, len(words))
                if reason:
                    stop_reason = reason
                    reply.final_check = (send, sense)
                    break
            else:
                answers = await self.client.ask(state, bucket_questions, usage)
                send, sense = 0.0, 1.0

            candidates: list[str] = []
            for answer in answers.values():
                candidates += finalists(_probabilities(answer), cfg.finalist_mass, cfg.finalist_cap)
            if words:
                candidates += list(PUNCTUATION)
            options = allowed(candidates, words)
            if not options:
                stop_reason = "no_options"
                break

            runoff = await self.client.ask(state, self._runoff_questions(words, options), usage)
            probs = _probabilities(runoff["next"])
            fluency = {w: _noul(runoff, f"fit{i}") for i, w in enumerate(options)}
            scores = {
                w: probs.get(w, 0.0)
                * fluency[w] ** cfg.fluency_power
                * repetition_factor(
                    w,
                    words,
                    self.common,
                    penalty=cfg.repeat_penalty,
                    common_penalty=cfg.common_repeat_penalty,
                    window=cfg.repeat_window,
                )
                for w in options
            }
            word = pick(scores, self.rng, cfg.temperature, cfg.top_p)
            steps.append(Step(word, probs.get(word, 0.0), fluency[word], send, sense, len(options)))
            if words and fluency[word] < cfg.give_up:
                stop_reason = "gave_up"
                break
            words.append(word)
            reply.text = join_words(words)
            if on_word is not None:
                await on_word(reply.text)

        reply.stop_reason = stop_reason
        return reply

    def _stop_reason(self, send: float, sense: float, n_words: int) -> str | None:
        cfg = self.config
        if send >= cfg.send_at:
            return "sent"
        if cfg.temperature > 0 and send >= cfg.send_floor and self.rng.random() < send**cfg.send_power:
            return "sent"
        if n_words >= cfg.sense_from and sense < cfg.sense_floor:
            return "lost_thread"
        return None

    # ---------------------------------------------------------------- requests

    def _bucket_questions(self, chat: Sequence[ChatLine]) -> dict[str, dict[str, Any]]:
        size = self.config.bucket_size
        # The chat's own words (and speakers' names) get a bucket of their own, so a topic
        # word like "sushi" is always on the menu even when it isn't a common word.
        newest_first = [line.text for line in reversed(chat)] + [line.speaker for line in reversed(chat)]
        ctx = context_words(newest_first, size)
        in_ctx = set(ctx)
        common = [w for w in self.vocab[: self.config.vocab_size] if w not in in_ctx]
        buckets = [ctx] + [common[i : i + size] for i in range(0, len(common), size)]
        return {
            f"b{i}": {
                "type": "choice",
                "instructions": self._blank_instructions,
                "criteria": dict.fromkeys(bucket),
            }
            for i, bucket in enumerate(b for b in buckets if b)
        }

    def _stop_questions(self, words: Sequence[str]) -> dict[str, dict[str, Any]]:
        text = join_words(words)
        return {
            "send": {
                "type": "choice",
                "instructions": {"text": text, "question": self._send_instructions},
                "criteria": SEND_OPTIONS,
            },
            "sense": {"type": "noul", "instructions": {"text": text, "question": SENSE_Q}},
        }

    def _runoff_questions(self, words: Sequence[str], options: Sequence[str]) -> dict[str, dict[str, Any]]:
        criteria = {w: PUNCTUATION.get(w) for w in options}
        questions: dict[str, dict[str, Any]] = {
            "next": {"type": "choice", "instructions": self._blank_instructions, "criteria": criteria}
        }
        text = join_words(words)
        for i, w in enumerate(options):
            if words:
                instructions = {"text": text, "next": w, "question": NEXT_WORD_Q}
            else:
                instructions = {"text": w, "question": FIRST_WORD_Q}
            questions[f"fit{i}"] = {"type": "noul", "instructions": instructions}
        return questions


def _chat_json(chat: Sequence[ChatLine]) -> list[dict[str, str]]:
    return [{"from": line.speaker, "text": line.text} for line in chat]


def _probabilities(answer: dict[str, Any]) -> dict[str, float]:
    probs = answer.get("probabilities")
    if not isinstance(probs, dict):
        raise JevError(f"choice answer without probabilities: {str(answer)[:200]}")
    return {str(k): float(v or 0.0) for k, v in probs.items()}


def _noul(answers: dict[str, Any], key: str) -> float:
    try:
        return float(answers[key]["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"missing noul answer {key!r}") from exc
