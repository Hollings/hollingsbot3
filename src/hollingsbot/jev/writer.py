"""Make Jev, a decision model that cannot generate text, write a chat reply.

Jev only answers typed questions (pick an option, say yes/no), so the reply is
built one word at a time. How depends on the size of its vocabulary.

**Small vocabulary (the default): one request per word.** Jev is born knowing
100 words; the chat's words and the words it has learned (lexicon.py) fill the
menu up to 250, which fits a single Choice. One request carries that Choice,
a naturalness Noul per option, and the stop check (below). Its state is the
chat alone; the reply with its blank rides in the Choice's own instructions.

**Large vocabulary (``vocab_size`` >= 250): a tournament, three requests per word.**

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

**With a suggester (opt-in, suggest.py): an LLM proposes, Jev chooses.** Each word is
one LLM call plus one Jev request over the LLM's proposals, shuffled so their ranking
can't leak through position. Optionally Jev can turn pages: beside each page it is
asked whether it would rather type a word that isn't listed, and saying so (sampled
with that probability) shows it the LLM's next proposals, then its own menu.

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
import dataclasses
import logging
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hollingsbot.jev.client import DecisionsClient, JevError, Usage
from hollingsbot.jev.decoding import allowed, finalists, pick, repetition_factor
from hollingsbot.jev.suggest import Suggestion
from hollingsbot.jev.text import PUNCTUATION, context_words, join_words
from hollingsbot.jev.vocab import load_vocab

if TYPE_CHECKING:
    from hollingsbot.jev.suggest import NextWordSuggester

_LOG = logging.getLogger(__name__)

BLANK = "___"
NEXT_WORD_Q = "Is `next` a natural next word right after `text`, in fluent English?"
FIRST_WORD_Q = "Does `text` read as fluent, grammatical English so far? It may be unfinished."
SENSE_Q = "Does `text` make sense as the start of a reply to the latest message in `chat`?"
SEND_OPTIONS = {"send": "send the message as it is", "keep typing": "add more to it first"}
# Asked beside a page of suggestions when another page follows. Probed on 27 real states: P(other)
# averaged 0.18 for the LLM's own menu and 0.33 for another chat's, higher for the wrong menu in
# 25/27. A "none of these" option inside the word Choice itself stayed near 0.02 either way.
OTHER_OPTIONS = {"pick": "pick one of `options`", "other": "type a word that isn't in `options`"}
# Top tokens asked of the LLM when Jev can page past the first 20: 100-130 whole words, 60-100
# of them known in strict mode (only some providers serve more than 20; ~0.5 s).
DEEP_TOP = 200


@dataclass(frozen=True)
class ChatLine:
    speaker: str
    text: str


@dataclass(frozen=True)
class WriterConfig:
    # The common words Jev is born knowing (the chat's and learned words come on top). Below
    # bucket_size the whole menu fits one Choice and each word costs a single request (see
    # JevWriter.single_call); 5000 gives the full-vocabulary tournament instead.
    vocab_size: int = 100
    bucket_size: int = 250  # Choice questions allow at most 255 options (5 go to punctuation)
    fluency_check: bool = True  # ask "is `next` natural after `text`?" of every finalist
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
    # Stop once the reply makes less sense than this. None = 0.2 for the big-vocabulary
    # tournament, off for the small vocabulary, whose replies read as half-nonsense even
    # when they are going somewhere (a floor there cut them off at word 4).
    sense_floor: float | None = None
    sense_from: int = 3  # ...judged from this many words on (one word is too little to judge)
    give_up: float = 0.25  # stop if even the picked word is rated less natural than this
    min_words: int = 8  # never send before this many words (sense/give-up can still end it)
    max_words: int = 40
    temperature: float = 0.7
    top_p: float = 0.6
    # How Jev writes, told to both the word and the send questions ("{name}" = its name).
    # This line is what makes replies longer: without it Jev answers "pizza" and sends.
    style: str = "{name} writes long, chatty messages, a few sentences at a time."
    # With a suggester (an LLM proposing the next word, see suggest.py):
    known_only: bool = False  # Jev may only pick suggestions it knows (born, chat or learned words)
    suggest_weight: float = 0.0  # score x P_llm(word)**weight; 0 = Jev alone decides among suggestions
    # Pages (suggester only). Each page is shown in random order, so the LLM's ranking can't
    # leak through position. With a page after it, Jev is also asked whether to pick from this
    # page or "type a word that isn't in `options`"; the latter turns the page: the LLM's next
    # page_size words, then (own_page) Jev's own menu minus every word it already passed.
    shuffle: bool = True
    page_size: int = 20
    llm_pages: int = 1  # >1 asks the LLM for its top 200 tokens: ~3 pages of known words, slower provider
    own_page: bool = False
    page_power: float = 1.0  # turn with probability P(other)**page_power (T=0: when P(other) >= 0.5)


@dataclass
class Step:
    word: str
    choice: float  # runoff probability
    fluency: float  # Noul: natural next word
    send: float  # Choice: P(send the reply as it was before this word)
    sense: float  # Noul: reply so far (before this word) makes sense
    options: int  # runoff size
    suggested: float = 0.0  # the suggester's probability for this word (0 without one)
    page: int = 0  # the page the word was picked from (0 = first; see WriterConfig.llm_pages)
    other: float = 0.0  # Jev's P("a word that isn't on this page") there, 0 if not asked (last page)
    source: str = ""  # "llm" (a suggestion page), "own" (Jev's own menu), "" (no suggester)


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
        suggester: NextWordSuggester | None = None,
    ) -> None:
        self.client = client
        self.suggester = suggester
        self.name = name
        self.config = config or WriterConfig()
        self.vocab = tuple(vocab if vocab is not None else load_vocab())
        self.common = frozenset(self.vocab[: self.config.exempt_top])
        self.rng = rng or random.Random()
        # A small vocabulary fits one Choice: no buckets, no runoff, and the stop check
        # rides in the same request, so each word is one API call instead of three.
        self.single_call = self.config.vocab_size < self.config.bucket_size
        # With a suggester every word is one Jev request too, whatever the vocabulary size.
        self.one_request = self.single_call or suggester is not None
        if self.config.sense_floor is not None:
            self.sense_floor = self.config.sense_floor
        else:
            self.sense_floor = 0.0 if self.one_request else 0.2
        self._reply_key = re.sub(r"\W+", "_", name.lower()).strip("_") + "_reply"
        style_line = self.config.style.replace("{name}", name).strip()
        self._style_line = style_line
        style = f"{style_line} " if style_line else ""
        self._blank_instructions = (
            f"{name} is a member of this Discord chat and is writing a reply to the latest message. {style}"
            f"`{self._reply_key}` is {name}'s reply so far, and the blank ({BLANK}) marks the next word. "
            "Which word goes in the blank?"
        )
        self._send_instructions = (
            f"{style}{name} has typed `text` so far as a reply to the latest message in `chat`. "
            f"What does {name} do now?"
        )
        self._other_instructions = f"{name} is choosing the word for the blank. What does {name} do?"

    # ------------------------------------------------------------------ public

    async def write(
        self,
        chat: Sequence[ChatLine],
        on_word: OnWord | None = None,
        draft: Reply | None = None,
        learned: Sequence[str] = (),
    ) -> Reply:
        """Write a reply to the last line of ``chat``.

        ``on_word`` gets the text so far after each word. The reply is built in
        ``draft`` if given, so a caller that cancels this coroutine can still
        read what was written and what it cost. ``learned`` is extra vocabulary,
        best first (the lexicon's ranking): it fills whatever room the chat's own
        words leave, ahead of the built-in words.
        """
        cfg = self.config
        reply = draft if draft is not None else Reply()
        reply.model = self.client.model
        words, steps, usage = reply.words, reply.steps, reply.usage
        chat_state = {"chat": _chat_json(chat)}
        if self.one_request:
            pool = self._single_pool(chat, learned)
        else:
            bucket_questions = self._bucket_questions(chat, learned)
        if self.suggester is not None:
            heard = context_words(_newest_first(chat), 1000) + list(learned)
            known = set(self.vocab[: cfg.vocab_size]) | set(heard)
        stop_reason = "max_words"
        merges = 0  # times the last word has been finished off (reset for each new word)

        while len(words) < cfg.max_words:
            send, sense = 0.0, 1.0
            suggested: dict[str, float] = {}
            page, other, source = 0, 0.0, ""
            if self.one_request:
                if self.suggester is not None:
                    try:
                        suggestion = await self.suggester.suggest(
                            chat, self.name, words, whole_words=self.vocab, complete_from=heard,
                            style=self._style_line, usage=usage, top=DEEP_TOP if cfg.llm_pages > 1 else None,
                        )  # fmt: skip
                    except JevError:
                        # The LLM is a helper: without it, this word comes from Jev's own menu.
                        _LOG.warning("Suggester failed; using Jev's own menu for this word", exc_info=True)
                        suggestion = Suggestion([])
                    if suggestion.completes and merges < 3:
                        # The model is still spelling the last word ("sand" -> "sandwich"):
                        # finish it and ask again, no decision needed from Jev.
                        words[-1] = suggestion.completes
                        steps[-1] = dataclasses.replace(steps[-1], word=suggestion.completes)
                        reply.text = join_words(words)
                        merges += 1
                        if on_word is not None:
                            await on_word(reply.text)
                        continue
                    suggested = dict(suggestion.options)
                    pages = self._pages(suggestion.options, known, pool, words)
                else:
                    pages = [("", allowed(pool + (list(PUNCTUATION) if words else []), words))]
                pages = [(src, opts) for src, opts in pages if opts]
                if not pages:
                    stop_reason = "no_options"
                    break
                # One request: the word itself, each option's fit, the stop check, and (when
                # another page follows) whether Jev would rather type a word that isn't here.
                source, options = pages[0]
                more = len(pages) > 1
                answers = await self.client.ask(chat_state, self._single_questions(words, options, other=more), usage)
                if words:
                    send, sense = _probabilities(answers["send"]).get("send", 0.0), _noul(answers, "sense")
                    if reason := self._stop_reason(send, sense, len(words)):
                        stop_reason = reason
                        reply.final_check = (send, sense)
                        break
                other = _other(answers) if more else 0.0
                while more and self._turns_page(other):
                    page += 1
                    source, options = pages[page]
                    more = page + 1 < len(pages)
                    questions = self._single_questions(words, options, stop=False, other=more)
                    answers = await self.client.ask(chat_state, questions, usage)
                    other = _other(answers) if more else 0.0
                runoff = answers
            else:
                state = {**chat_state, self._reply_key: f"{join_words(words)} {BLANK}".strip()}
                if words:
                    answers, check = await asyncio.gather(
                        self.client.ask(state, bucket_questions, usage),
                        self.client.ask(chat_state, self._stop_questions(words), usage),
                    )
                    send, sense = _probabilities(check["send"]).get("send", 0.0), _noul(check, "sense")
                    if reason := self._stop_reason(send, sense, len(words)):
                        stop_reason = reason
                        reply.final_check = (send, sense)
                        break
                else:
                    answers = await self.client.ask(state, bucket_questions, usage)

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
            if cfg.fluency_check:
                fluency = {w: _noul(runoff, f"fit{i}") for i, w in enumerate(options)}
            else:
                fluency = dict.fromkeys(options, 1.0)
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
                * (max(suggested.get(w, 0.0), 1e-4) ** cfg.suggest_weight if suggested else 1.0)
                for w in options
            }
            word = pick(scores, self.rng, cfg.temperature, cfg.top_p)
            steps.append(
                Step(
                    word,
                    probs.get(word, 0.0),
                    fluency[word],
                    send,
                    sense,
                    len(options),
                    suggested.get(word, 0.0),
                    page=page,
                    other=other,
                    source=source,
                )
            )
            if words and fluency[word] < cfg.give_up:
                stop_reason = "gave_up"
                break
            words.append(word)
            merges = 0
            reply.text = join_words(words)
            if on_word is not None:
                await on_word(reply.text)

        reply.stop_reason = stop_reason
        return reply

    def _stop_reason(self, send: float, sense: float, n_words: int) -> str | None:
        cfg = self.config
        if n_words >= cfg.min_words:
            if send >= cfg.send_at:
                return "sent"
            if cfg.temperature > 0 and send >= cfg.send_floor and self.rng.random() < send**cfg.send_power:
                return "sent"
        if n_words >= cfg.sense_from and sense < self.sense_floor:
            return "lost_thread"
        return None

    def _turns_page(self, other: float) -> bool:
        """Sampled like the send decision: P(other) is Jev's own odds that its word isn't here."""
        if self.config.temperature <= 0:
            return other >= 0.5
        return self.rng.random() < other**self.config.page_power

    # ---------------------------------------------------------------- requests

    def _bucket_questions(self, chat: Sequence[ChatLine], learned: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
        size = self.config.bucket_size
        # The chat's own words (and speakers' names) lead, so a topic word like "sushi" is
        # always on the menu even when it isn't a common word; then learned words.
        ctx = context_words(_newest_first(chat), size)
        extra = ctx + [w for w in dict.fromkeys(learned) if w not in set(ctx)]
        in_extra = set(extra)
        common = [w for w in self.vocab[: self.config.vocab_size] if w not in in_extra]
        buckets = [extra[i : i + size] for i in range(0, len(extra), size)]
        buckets += [common[i : i + size] for i in range(0, len(common), size)]
        return {
            f"b{i}": {
                "type": "choice",
                "instructions": self._blank_instructions,
                "criteria": dict.fromkeys(bucket),
            }
            for i, bucket in enumerate(b for b in buckets if b)
        }

    def _single_pool(self, chat: Sequence[ChatLine], learned: Sequence[str] = ()) -> list[str]:
        """Single-call menu, at most ``bucket_size`` words.

        The built-in words always make it. The room left goes first to the chat's
        own words (newest message first), then to learned words in their ranking.
        """
        cfg = self.config
        base = list(self.vocab[: cfg.vocab_size])
        if len(base) >= cfg.bucket_size:
            # Only reached with a suggester (without one a born vocabulary this big means the
            # tournament): this is just its fallback menu, so chat and learned words go first.
            heard = context_words(_newest_first(chat), cfg.bucket_size) + list(learned)
            return list(dict.fromkeys(heard + base))[: cfg.bucket_size]
        in_base = set(base)
        room = max(0, cfg.bucket_size - len(base))
        ctx = [w for w in context_words(_newest_first(chat), room + len(base)) if w not in in_base][:room]
        taken = in_base | set(ctx)
        fill = [w for w in dict.fromkeys(learned) if w not in taken][: room - len(ctx)]
        return ctx + fill + base

    def _pages(
        self, ranked: Sequence[tuple[str, float]], known: set[str], pool: Sequence[str], words: Sequence[str]
    ) -> list[tuple[str, list[str]]]:
        """The menus Jev may turn through for one word, in order, as (source, options).

        The LLM's proposals in pages of ``page_size`` (``llm_pages`` of them), then with
        ``own_page`` Jev's own menu minus every word already shown. With no usable
        proposal at all, the own menu is the only page.
        """
        cfg = self.config
        proposals = allowed([w for w, _ in ranked if not cfg.known_only or w in known or w in PUNCTUATION], words)
        size = max(1, cfg.page_size)
        shown = proposals[: size * max(1, cfg.llm_pages)]
        pages = [("llm", shown[i : i + size]) for i in range(0, len(shown), size)]
        if cfg.own_page or not pages:
            passed = set(shown)
            own = [w for w in allowed(list(pool) + (list(PUNCTUATION) if words else []), words) if w not in passed]
            pages.append(("own", own))
        if cfg.shuffle:
            for _, options in pages:
                self.rng.shuffle(options)
        return pages

    def _single_questions(
        self, words: Sequence[str], options: Sequence[str], *, stop: bool = True, other: bool = False
    ) -> dict[str, dict[str, Any]]:
        # The state is the chat alone (the stop check must not see a blank), so the reply
        # with its blank travels in the word question's own instructions.
        questions = self._runoff_questions(words, options)
        blank = f"{join_words(words)} {BLANK}".strip()
        questions["next"]["instructions"] = {self._reply_key: blank, "question": self._blank_instructions}
        if other:
            questions["other"] = {
                "type": "choice",
                "instructions": {
                    self._reply_key: blank,
                    "options": ", ".join(options),
                    "question": self._other_instructions,
                },
                "criteria": OTHER_OPTIONS,
            }
        if words and stop:
            questions.update(self._stop_questions(words))
        return questions

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
        if not self.config.fluency_check:
            return questions
        text = join_words(words)
        for i, w in enumerate(options):
            if words:
                instructions = {"text": text, "next": w, "question": NEXT_WORD_Q}
            else:
                instructions = {"text": w, "question": FIRST_WORD_Q}
            questions[f"fit{i}"] = {"type": "noul", "instructions": instructions}
        return questions


def _newest_first(chat: Sequence[ChatLine]) -> list[str]:
    """Texts then speakers' names, newest first: the order chat words claim menu room."""
    return [line.text for line in reversed(chat)] + [line.speaker for line in reversed(chat)]


def _chat_json(chat: Sequence[ChatLine]) -> list[dict[str, str]]:
    return [{"from": line.speaker, "text": line.text} for line in chat]


def _probabilities(answer: dict[str, Any]) -> dict[str, float]:
    probs = answer.get("probabilities")
    if not isinstance(probs, dict):
        raise JevError(f"choice answer without probabilities: {str(answer)[:200]}")
    return {str(k): float(v or 0.0) for k, v in probs.items()}


def _other(answers: dict[str, Any]) -> float:
    return _probabilities(answers["other"]).get("other", 0.0)


def _noul(answers: dict[str, Any], key: str) -> float:
    try:
        return float(answers[key]["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"missing noul answer {key!r}") from exc
