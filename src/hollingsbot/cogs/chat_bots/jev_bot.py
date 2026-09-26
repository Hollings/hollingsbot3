"""JevBot - TypeSafe's Jev decision model, chatting one word at a time.

Jev cannot generate text; hollingsbot.jev.writer makes it pick a reply word by
word from its own vocabulary, one API call per word. (Optionally a small LLM
proposes each next word and Jev only chooses, see hollingsbot.jev.suggest; off
by default because Jev then mostly filters another model's text.) The whole
reply is written behind a typing indicator (shown under the bot's own name:
webhooks cannot type), held while a human is mid-message, then posted once
through a "Jev" webhook. Someone speaking first cancels it unposted.

Jev is born knowing only the most common words and learns every word a human
says in its channels (hollingsbot.jev.lexicon); `!jev` shows what it knows.

Beyond its own channels, `!spawn jev [N]` brings Jev into any channel: it answers
the chat there at once, then every message, other bots' included (Wendy, temp
bots: the reply count ends any back-and-forth), until it has posted N replies
(the first included) and leaves like a temp bot. In its own channels it answers
humans only. `!despawn jev` sends it away early. Visits are kept in the DB
(hollingsbot.jev.spawns) across restarts.

Config (env):
    JEV_BOT_CHANNELS          comma-separated channel IDs Jev answers in (every human message)
    JEV_BOT_NAME              display name, also the name Jev is told it has (default "Jev")
    JEV_CONTEXT_MESSAGES      chat messages Jev sees, the latest included (default 5)
    JEV_DAILY_BUDGET_USD      stop replying for the rest of the UTC day past this spend (default 2.00)
    JEV_SUGGEST_MODEL         "on" or an OpenRouter model slug: an LLM proposes next words and Jev
                              chooses (default off: Jev alone picks from its own vocabulary)
    JEV_KNOWN_ONLY            1 = Jev only says proposals it knows (born or learned) (default 1 with a
                              suggester: rare words must be taught first); 0 = any proposal
    JEV_VOCAB_SIZE            words Jev is born knowing (default 1000 with a suggester, 100 without;
                              without one, >= 250 means the full-vocabulary tournament)
    JEV_LLM_PAGES             with a suggester: pages of the LLM's words Jev can turn through by
                              saying it would rather type a word that isn't listed (default 1)
    JEV_OWN_PAGE              with a suggester: 1 = after the LLM's pages, a last page of Jev's own
                              vocabulary, minus what it passed (default 0)
    JEV_SHUFFLE               0 = show the LLM's words in its own rank order (default 1: shuffled)
    JEV_FLUENCY_CHECK         0 to skip the per-option naturalness check (cheaper, more scrambled)
    JEV_MIN_WORDS / JEV_MAX_WORDS   reply length bounds (defaults 8 / 40)
    JEV_STOP                  how a reply ends: threshold (default: send when P(send) is high),
                              sample (send sampled like a word), choice (STOP on the word menu)
    JEV_STYLE                 how Jev writes, "{name}" = its name (default: long, chatty messages;
                              set to a single space for Jev's natural one-word answers)
    JEV_TEMPERATURE / JEV_TOP_P   sampling (defaults 0.7 / 0.6, see WriterConfig)
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import discord

from hollingsbot.cogs import chat_utils
from hollingsbot.cogs.chat_bots.temp_bot.names import departure_message
from hollingsbot.jev import ChatLine, DecisionsClient, JevError, JevWriter, Reply, WriterConfig
from hollingsbot.jev.ledger import JevLedger
from hollingsbot.jev.lexicon import Lexicon
from hollingsbot.jev.spawns import JevSpawns
from hollingsbot.jev.suggest import DEFAULT_SUGGEST_MODEL, NextWordSuggester
from hollingsbot.jev.writer import STOP_MODES
from hollingsbot.settings import parse_id_set
from hollingsbot.utils.discord_utils import get_display_name

if TYPE_CHECKING:
    from discord.ext import commands

    from hollingsbot.cogs.conversation import ConversationTurn

_LOG = logging.getLogger(__name__)

AVATAR_FILE = Path(__file__).resolve().parents[2] / "assets" / "jev_avatar.png"
_NAME_PREFIX = re.compile(r"^<[^>]+>:\s*")
_MAX_LINE_CHARS = 500
CUT_OFF_MARK = " \u2014"  # em dash: the reply broke off mid-way (an API error)

OVER_BUDGET_REACTION = "\N{SLEEPING SYMBOL}"
ERROR_REACTION = "\N{WARNING SIGN}"
SPAWNED_REACTION = "\N{WHITE HEAVY CHECK MARK}"  # spawned into a channel with nothing to answer yet

SPAWN_REPLIES_DEFAULT = 10
SPAWN_REPLIES_MAX = 20  # the same cap as a temp bot's `!spawn`


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        _LOG.warning("%s is not a number; using %s", name, default)
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_stop_mode(default: str) -> str:
    raw = os.getenv("JEV_STOP", "").strip().lower()
    if not raw:
        return default
    if raw not in STOP_MODES:
        _LOG.warning("JEV_STOP=%r is not one of %s; using %s", raw, ", ".join(STOP_MODES), default)
        return default
    return raw


@dataclass(frozen=True)
class JevBotSettings:
    channels: frozenset[int]
    name: str = "Jev"
    context_messages: int = 5
    daily_budget: float = 2.0
    writer: WriterConfig = field(default_factory=WriterConfig)
    suggest_model: str | None = None  # None = Jev picks from its own vocabulary alone

    @classmethod
    def from_env(cls) -> JevBotSettings:
        base = WriterConfig()
        # Off by default: with an LLM proposing every word, Jev is only a filter on another
        # model's text. Opt in with JEV_SUGGEST_MODEL=on (or a model slug).
        model = os.getenv("JEV_SUGGEST_MODEL", "off").strip()
        if model.lower() in ("1", "on", "true", "yes"):
            model = DEFAULT_SUGGEST_MODEL
        suggest_model = None if model.lower() in ("", "0", "off", "none", "false") else model
        # With a suggester, Jev is born with the 1000 commonest words (grammar is always
        # available) and may only say proposals it knows, so rarer words must be taught.
        born = 1000 if suggest_model else base.vocab_size
        writer = dataclasses.replace(
            base,
            vocab_size=max(1, int(_env_float("JEV_VOCAB_SIZE", born))),
            known_only=_env_flag("JEV_KNOWN_ONLY", bool(suggest_model)),
            fluency_check=_env_flag("JEV_FLUENCY_CHECK", True),
            min_words=int(_env_float("JEV_MIN_WORDS", base.min_words)),
            max_words=int(_env_float("JEV_MAX_WORDS", base.max_words)),
            temperature=_env_float("JEV_TEMPERATURE", base.temperature),
            top_p=_env_float("JEV_TOP_P", base.top_p),
            style=os.getenv("JEV_STYLE", base.style).strip(),
            llm_pages=max(1, int(_env_float("JEV_LLM_PAGES", base.llm_pages))),
            own_page=_env_flag("JEV_OWN_PAGE", base.own_page),
            shuffle=_env_flag("JEV_SHUFFLE", base.shuffle),
            stop=_env_stop_mode(base.stop),
        )
        return cls(
            channels=frozenset(parse_id_set(os.getenv("JEV_BOT_CHANNELS"))),
            name=os.getenv("JEV_BOT_NAME", "Jev").strip() or "Jev",
            context_messages=max(1, int(_env_float("JEV_CONTEXT_MESSAGES", 5))),
            daily_budget=_env_float("JEV_DAILY_BUDGET_USD", 2.0),
            writer=writer,
            suggest_model=suggest_model,
        )

    def reachable_learned(self, learned: int) -> int:
        """How many learned words Jev can use at once (for `!jev`)."""
        if self.suggest_model:
            return learned  # proposals are checked against everything it knows
        return min(learned, max(0, self.writer.bucket_size - self.writer.vocab_size))

    @property
    def learned_limit(self) -> int:
        """How many ranked learned words to hand the writer each reply."""
        return 100_000 if self.suggest_model else self.writer.bucket_size


def _is_human(message: discord.Message) -> bool:
    return not message.author.bot and message.webhook_id is None


def chat_lines(history: list[ConversationTurn], count: int) -> list[ChatLine]:
    """The last ``count`` turns as (speaker, text), the way Jev reads the chat."""
    lines = []
    for turn in history[-count:]:
        text = _NAME_PREFIX.sub("", turn.content or "", count=1).strip()
        if turn.images:
            text = f"{text} [image]".strip()
        if len(text) > _MAX_LINE_CHARS:
            text = text[: _MAX_LINE_CHARS - 1] + "\u2026"
        lines.append(ChatLine(turn.author_name or "someone", text or "[no text]"))
    return lines


class JevBot:
    """Answers every human message in its channels, and every message wherever it's spawned (bots too)."""

    def __init__(
        self, bot: commands.Bot, coordinator: Any, typing_tracker: Any, settings: JevBotSettings | None = None
    ):
        self.bot = bot
        self.coordinator = coordinator
        self.typing_tracker = typing_tracker
        self.settings = settings or JevBotSettings.from_env()
        self.whitelist_channels: set[int] = set(self.settings.channels)
        self.ledger = JevLedger()
        self.lexicon = Lexicon()
        self.spawns = JevSpawns()
        self._spawned: dict[int, int] | None = None  # channel -> replies left; loaded on first use
        self._writer: JevWriter | None = None
        self._webhooks: dict[int, discord.Webhook | None] = {}
        self._active: dict[int, asyncio.Task] = {}
        self._cleanups: set[asyncio.Task] = set()
        w = self.settings.writer
        _LOG.info(
            "JevBot initialized (name=%s, channels=%s, suggest=%s, born=%d, known_only=%s, llm_pages=%d, "
            "own_page=%s, max_words=%d, stop=%s)",
            self.settings.name,
            sorted(self.whitelist_channels),
            self.settings.suggest_model or "off",
            w.vocab_size,
            w.known_only,
            w.llm_pages,
            w.own_page,
            w.max_words,
            w.stop,
        )

    # ------------------------------------------------------------ coordinator API

    def claims_channel(self, channel_id: int) -> bool:
        return self._answers_in(channel_id)

    async def receive_message(self, message: discord.Message, history: list[ConversationTurn]) -> dict | None:
        if not self._should_respond(message):
            return None
        if _is_human(message):  # it learns from people, not from the bots it talks to
            await self._learn_from(message)
        if not history or history[-1].message_id != message.id:
            _LOG.warning("JevBot: latest history turn is not message %s; skipping", message.id)
            return None
        if await self._over_budget(message):
            return None
        return await self._answer(message, chat_lines(history, self.settings.context_messages))

    async def _answer(self, message: discord.Message, chat: list[ChatLine]) -> dict | None:
        """Reply to ``chat`` in ``message``'s channel, replacing a reply already under way there.

        Returns the posted reply (None if nothing was posted); a newer message cancels it.
        """
        channel = message.channel
        await self._cancel_generation(channel.id)
        task = asyncio.create_task(self._write_and_send(message, chat))
        self._active[channel.id] = task
        try:
            result = await task
        finally:
            if self._active.get(channel.id) is task:
                self._active.pop(channel.id, None)
        if result is not None:
            # Shielded: a message arriving right after the post must not leave the count half-done.
            await asyncio.shield(self._count_reply(channel))
        return result

    async def _over_budget(self, message: discord.Message) -> bool:
        """Past today's budget: react instead of replying."""
        spent = await asyncio.to_thread(self.ledger.spent_today)
        if spent < self.settings.daily_budget:
            return False
        _LOG.info("JevBot over daily budget ($%.4f >= $%.2f)", spent, self.settings.daily_budget)
        with contextlib.suppress(discord.HTTPException):
            await message.add_reaction(OVER_BUDGET_REACTION)
        return True

    async def _cancel_generation(self, channel_id: int) -> None:
        task = self._active.pop(channel_id, None)
        if task and not task.done():
            task.cancel()
            # The task's own cleanup is shielded; no need to wait for it here.
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.2)

    async def _learn_from(self, message: discord.Message) -> None:
        """Every human message in Jev's channels teaches it the words in it."""
        speaker = get_display_name(message.author)
        text = chat_utils.clean_mentions(message, self.bot)
        try:
            new = await asyncio.to_thread(self.lexicon.learn, text, speaker=speaker, channel_id=message.channel.id)
        except Exception:
            _LOG.exception("JevBot could not learn from message %s", message.id)
            return
        if new:
            _LOG.info("Jev learned %s from %s", new, speaker)

    # ------------------------------------------------------------------ spawning

    @property
    def spawned(self) -> dict[int, int]:
        """Channels Jev was spawned into -> replies it has left there (read from the DB once)."""
        if self._spawned is None:
            self._spawned = self.spawns.active()
            if self._spawned:
                _LOG.info("Jev is still visiting %s (channel: replies left)", self._spawned)
        return self._spawned

    async def spawn(self, ctx: commands.Context, replies: int) -> None:
        """`!spawn jev N`: answer in this channel, starting now, until Jev has posted ``replies`` replies."""
        channel, name = ctx.channel, self.settings.name
        if not isinstance(channel, discord.TextChannel):
            await ctx.send("This command only works in text channels.")
            return
        if not 1 <= replies <= SPAWN_REPLIES_MAX:
            await ctx.send(f"**{name}** can stay for 1 to {SPAWN_REPLIES_MAX} replies.")
            return
        if channel.id in self.whitelist_channels:
            await ctx.send(f"**{name}** already lives in this channel.")
            return
        visiting = channel.id in self.spawned
        await asyncio.to_thread(self.spawns.start, channel.id, replies, by=get_display_name(ctx.author))
        self.spawned[channel.id] = replies
        _LOG.info("Jev spawned in %s by %s for %d replies", channel.id, ctx.author, replies)
        if visiting:
            await ctx.send(f"**{name}** is already here; it has {replies} replies left now.")
            return
        await self._join(ctx.message)

    async def despawn(self, channel: discord.abc.Messageable) -> bool:
        """`!despawn jev`: end the visit now, dropping any reply under way; False if Jev isn't visiting."""
        if channel.id not in self.spawned:
            return False
        await self._cancel_generation(channel.id)
        await self._leave(channel, announce=False)
        return True

    async def _join(self, message: discord.Message) -> None:
        """Jev's first reply after being spawned (by ``message``): to whatever was being said."""
        channel = message.channel
        history = await self.coordinator.recent_history(channel)
        if not history:
            with contextlib.suppress(discord.HTTPException):
                await message.add_reaction(SPAWNED_REACTION)
            return
        if await self._over_budget(message):
            return
        answer = asyncio.ensure_future(self._answer(message, chat_lines(history, self.settings.context_messages)))
        try:
            await asyncio.wait({answer})
        except asyncio.CancelledError:
            answer.cancel()
            raise
        if answer.cancelled():
            return  # someone spoke before it was done: Jev answers them instead
        result = answer.result()
        if result is not None:  # the coordinator only records replies to messages it handed out
            await self.coordinator._add_response_to_history(
                channel.id, result["message_id"], result["text"], result["webhook_id"], result["bot_name"]
            )

    async def _count_reply(self, channel: discord.abc.Messageable) -> None:
        """A reply posted where Jev was spawned uses one up; after the last one, Jev leaves."""
        if channel.id not in self.spawned:
            return
        left = await asyncio.to_thread(self.spawns.use_reply, channel.id)
        if left:
            self.spawned[channel.id] = left
        else:
            await self._leave(channel, announce=True)

    async def _leave(self, channel: discord.abc.Messageable, *, announce: bool) -> None:
        self.spawned.pop(channel.id, None)
        await asyncio.to_thread(self.spawns.end, channel.id)
        _LOG.info("Jev left %s", channel.id)
        if announce:
            webhook = await self._webhook_for(channel)
            with contextlib.suppress(discord.HTTPException):
                await self._post(channel, webhook, departure_message(self.settings.name))

    # ------------------------------------------------------------------ gating

    def _answers_in(self, channel_id: int) -> bool:
        return channel_id in self.whitelist_channels or channel_id in self.spawned

    def _should_respond(self, message: discord.Message) -> bool:
        if not self._answers_in(message.channel.id):
            return False
        if not _is_human(message) and not self._answers_bot(message):
            return False
        if chat_utils.should_ignore_message(message.content):
            return False
        return bool(message.content.strip() or message.attachments)

    def _answers_bot(self, message: discord.Message) -> bool:
        """Other bots and webhooks get answers only where Jev was spawned.

        There its reply count ends a back-and-forth with another bot; in its own
        channels nothing would, so two bots answering each other would never stop.
        Never itself (its webhook is claimed with the coordinator, this is a
        backstop) or the bot account it runs as (command output, not conversation).
        """
        if message.channel.id not in self.spawned or message.author.id == self.bot.user.id:
            return False
        own = self._webhooks.get(message.channel.id)
        return own is None or message.webhook_id != own.id

    # ------------------------------------------------------------------ writing

    def _get_writer(self) -> JevWriter:
        if self._writer is None:
            model = self.settings.suggest_model
            self._writer = JevWriter(
                DecisionsClient(),
                name=self.settings.name,
                config=self.settings.writer,
                suggester=NextWordSuggester(model=model) if model else None,
            )
        return self._writer

    async def _write_and_send(self, message: discord.Message, chat: list[ChatLine]) -> dict | None:
        """Write the whole reply behind a typing indicator, then post it once."""
        channel = message.channel
        webhook = await self._webhook_for(channel)
        name = self.settings.name
        draft = Reply()

        try:
            learned = await asyncio.to_thread(self.lexicon.ranked, self.settings.learned_limit)
            async with channel.typing():
                reply = await self._get_writer().write(chat, draft=draft, learned=learned)
                # Don't talk over someone mid-message (if they send, this task is cancelled).
                await self.typing_tracker.wait_until_quiet(channel.id, self.bot.user.id)
        except asyncio.CancelledError:
            # Someone spoke first. Nothing was posted, but the words Jev picked were paid for.
            cleanup = asyncio.create_task(self._log_interrupted(channel.id, chat, draft))
            self._cleanups.add(cleanup)
            cleanup.add_done_callback(self._cleanups.discard)
            raise
        except (JevError, discord.HTTPException):
            _LOG.exception("JevBot failed to write a reply in channel %s", channel.id)
            await asyncio.to_thread(
                self.ledger.record, draft, chat, channel_id=channel.id, message_id=None, stop_reason="error"
            )
            with contextlib.suppress(discord.HTTPException):
                if draft.text:
                    await self._post(channel, webhook, draft.text + CUT_OFF_MARK)
                await message.add_reaction(ERROR_REACTION)
            return None

        _LOG.info(
            "Jev wrote %r (%d words, %s, %d calls, $%.4f, %.1fs)",
            reply.text,
            len(reply.words),
            reply.stop_reason,
            reply.usage.calls,
            reply.usage.cost,
            reply.usage.seconds,
        )
        sent: discord.Message | None = None
        try:
            sent = await self._post(channel, webhook, reply.text)
        except discord.HTTPException:
            _LOG.exception("JevBot could not post its reply in channel %s", channel.id)
        finally:
            await asyncio.to_thread(
                self.ledger.record, reply, chat, channel_id=channel.id, message_id=sent.id if sent else None
            )
        if sent is None:
            return None
        return {
            "message_id": sent.id,
            "text": reply.text,
            "webhook_id": webhook.id if webhook is not None else None,
            "bot_name": name,
        }

    async def _post(
        self, channel: discord.abc.Messageable, webhook: discord.Webhook | None, text: str
    ) -> discord.Message | None:
        """Post ``text`` as Jev (its webhook, or the bot itself without one); nothing if empty."""
        if not text:
            return None
        if webhook is not None:
            return await webhook.send(text, username=self.settings.name, wait=True)
        return await channel.send(text)

    async def _log_interrupted(self, channel_id: int, chat: list[ChatLine], draft: Reply) -> None:
        try:
            await asyncio.to_thread(
                self.ledger.record, draft, chat, channel_id=channel_id, message_id=None, stop_reason="interrupted"
            )
        except Exception:
            _LOG.exception("JevBot failed to log an interrupted reply")

    # ------------------------------------------------------------------ webhook

    async def _webhook_for(self, channel: discord.abc.Messageable) -> discord.Webhook | None:
        """Find or create this bot's "Jev" webhook in ``channel`` (None if not allowed)."""
        if channel.id in self._webhooks:
            return self._webhooks[channel.id]
        webhook: discord.Webhook | None = None
        if isinstance(channel, discord.TextChannel):
            try:
                mine = [
                    w
                    for w in await channel.webhooks()
                    if w.name == self.settings.name and w.user is not None and w.user.id == self.bot.user.id and w.token
                ]
                if mine:
                    webhook = mine[0]
                else:
                    avatar = AVATAR_FILE.read_bytes() if AVATAR_FILE.exists() else None
                    webhook = await channel.create_webhook(
                        name=self.settings.name, avatar=avatar, reason="Jev chat bot"
                    )
            except discord.Forbidden:
                _LOG.warning("JevBot has no Manage Webhooks in %s; posting as the bot", channel.id)
            except discord.HTTPException:
                _LOG.exception("JevBot could not set up a webhook in %s; posting as the bot", channel.id)
                return None  # not cached: try again next message
        if webhook is not None:
            self.coordinator.claim_webhook(webhook.id)
        self._webhooks[channel.id] = webhook
        return webhook
