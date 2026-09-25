"""JevBot - TypeSafe's Jev decision model, chatting one word at a time.

Jev cannot generate text; hollingsbot.jev.writer makes it pick a reply word by
word (one Decisions API call per word with the default small vocabulary). The
reply is posted through a "Jev" webhook as soon as the first word is picked and
edited as the rest arrive, so the channel watches it think.

Jev is born knowing only the most common words and learns every word a human
says in its channels (hollingsbot.jev.lexicon); `!jev` shows what it knows.

Config (env):
    JEV_BOT_CHANNELS          comma-separated channel IDs Jev answers in (every human message)
    JEV_BOT_NAME              display name, also the name Jev is told it has (default "Jev")
    JEV_CONTEXT_MESSAGES      chat messages Jev sees, the latest included (default 5)
    JEV_DAILY_BUDGET_USD      stop replying for the rest of the UTC day past this spend (default 2.00)
    JEV_VOCAB_SIZE            words Jev is born knowing (default 100; >= 250 = full-vocabulary tournament)
    JEV_FLUENCY_CHECK         0 to skip the per-option naturalness check (~7x cheaper, more scrambled)
    JEV_MIN_WORDS / JEV_MAX_WORDS   reply length bounds (defaults 8 / 40)
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
from hollingsbot.jev import ChatLine, DecisionsClient, JevError, JevWriter, Reply, WriterConfig
from hollingsbot.jev.ledger import JevLedger
from hollingsbot.jev.lexicon import Lexicon
from hollingsbot.settings import parse_id_set
from hollingsbot.utils.discord_utils import get_display_name
from hollingsbot.utils.live_message import LiveMessage

if TYPE_CHECKING:
    from discord.ext import commands

    from hollingsbot.cogs.conversation import ConversationTurn

_LOG = logging.getLogger(__name__)

AVATAR_FILE = Path(__file__).resolve().parents[2] / "assets" / "jev_avatar.png"
_NAME_PREFIX = re.compile(r"^<[^>]+>:\s*")
_MAX_LINE_CHARS = 500
INTERRUPTED_MARK = " \u2014"  # em dash: someone spoke before Jev finished

OVER_BUDGET_REACTION = "\N{SLEEPING SYMBOL}"
ERROR_REACTION = "\N{WARNING SIGN}"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        _LOG.warning("%s is not a number; using %s", name, default)
        return default


@dataclass(frozen=True)
class JevBotSettings:
    channels: frozenset[int]
    name: str = "Jev"
    context_messages: int = 5
    daily_budget: float = 2.0
    writer: WriterConfig = field(default_factory=WriterConfig)

    @classmethod
    def from_env(cls) -> JevBotSettings:
        base = WriterConfig()
        writer = dataclasses.replace(
            base,
            vocab_size=max(1, int(_env_float("JEV_VOCAB_SIZE", base.vocab_size))),
            fluency_check=os.getenv("JEV_FLUENCY_CHECK", "1").strip().lower() not in ("0", "false", "no", "off"),
            min_words=int(_env_float("JEV_MIN_WORDS", base.min_words)),
            max_words=int(_env_float("JEV_MAX_WORDS", base.max_words)),
            temperature=_env_float("JEV_TEMPERATURE", base.temperature),
            top_p=_env_float("JEV_TOP_P", base.top_p),
            style=os.getenv("JEV_STYLE", base.style).strip(),
        )
        return cls(
            channels=frozenset(parse_id_set(os.getenv("JEV_BOT_CHANNELS"))),
            name=os.getenv("JEV_BOT_NAME", "Jev").strip() or "Jev",
            context_messages=max(1, int(_env_float("JEV_CONTEXT_MESSAGES", 5))),
            daily_budget=_env_float("JEV_DAILY_BUDGET_USD", 2.0),
            writer=writer,
        )


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
    """Answers every human message in its channels, via the chat coordinator."""

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
        self._writer: JevWriter | None = None
        self._webhooks: dict[int, discord.Webhook | None] = {}
        self._active: dict[int, asyncio.Task] = {}
        self._cleanups: set[asyncio.Task] = set()
        _LOG.info("JevBot initialized (name=%s, channels=%s)", self.settings.name, sorted(self.whitelist_channels))

    # ------------------------------------------------------------ coordinator API

    def claims_channel(self, channel_id: int) -> bool:
        return channel_id in self.whitelist_channels

    async def receive_message(self, message: discord.Message, history: list[ConversationTurn]) -> dict | None:
        if not self._should_respond(message):
            return None
        await self._learn_from(message)
        if not history or history[-1].message_id != message.id:
            _LOG.warning("JevBot: latest history turn is not message %s; skipping", message.id)
            return None

        spent = await asyncio.to_thread(self.ledger.spent_today)
        if spent >= self.settings.daily_budget:
            _LOG.info("JevBot over daily budget ($%.4f >= $%.2f)", spent, self.settings.daily_budget)
            with contextlib.suppress(discord.HTTPException):
                await message.add_reaction(OVER_BUDGET_REACTION)
            return None

        chat = chat_lines(history, self.settings.context_messages)
        await self._cancel_generation(message.channel.id)
        task = asyncio.create_task(self._write_and_send(message, chat))
        self._active[message.channel.id] = task
        try:
            return await task
        finally:
            if self._active.get(message.channel.id) is task:
                self._active.pop(message.channel.id, None)

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

    # ------------------------------------------------------------------ gating

    def _should_respond(self, message: discord.Message) -> bool:
        if message.channel.id not in self.whitelist_channels:
            return False
        # Humans only: answering bots or webhooks (including itself) is how loops start.
        if message.author.bot or message.webhook_id is not None:
            return False
        if chat_utils.should_ignore_message(message.content):
            return False
        return bool(message.content.strip() or message.attachments)

    # ------------------------------------------------------------------ writing

    def _get_writer(self) -> JevWriter:
        if self._writer is None:
            self._writer = JevWriter(DecisionsClient(), name=self.settings.name, config=self.settings.writer)
        return self._writer

    async def _write_and_send(self, message: discord.Message, chat: list[ChatLine]) -> dict | None:
        channel = message.channel
        webhook = await self._webhook_for(channel)
        name = self.settings.name
        draft = Reply()

        if webhook is not None:
            live = LiveMessage(
                send=lambda text: webhook.send(text, username=name, wait=True),
                edit=lambda msg, text: webhook.edit_message(msg.id, content=text),
            )
            on_word = live.update
        else:
            # No webhook permission: post once at the end as the bot itself. Streaming
            # would put a one-word message into history via on_message.
            live = LiveMessage(send=lambda text: channel.send(text), edit=lambda msg, text: msg.edit(content=text))
            on_word = None

        try:
            learned = await asyncio.to_thread(self.lexicon.ranked, self.settings.writer.bucket_size)
            if webhook is None:
                async with channel.typing():
                    reply = await self._get_writer().write(chat, draft=draft, learned=learned)
            else:
                reply = await self._get_writer().write(chat, on_word=on_word, draft=draft, learned=learned)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._finish_interrupted(channel, live, webhook, chat, draft))
            self._cleanups.add(cleanup)
            cleanup.add_done_callback(self._cleanups.discard)
            raise
        except (JevError, discord.HTTPException):
            _LOG.exception("JevBot failed to write a reply in channel %s", channel.id)
            await asyncio.to_thread(
                self.ledger.record, draft, chat, channel_id=channel.id, message_id=None, stop_reason="error"
            )
            with contextlib.suppress(discord.HTTPException):
                await live.finish(draft.text + INTERRUPTED_MARK if draft.text else "")
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
            sent = await live.finish(reply.text)
        except discord.HTTPException:
            _LOG.exception("JevBot could not post its final text in channel %s", channel.id)
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

    async def _finish_interrupted(
        self,
        channel: discord.abc.Messageable,
        live: LiveMessage,
        webhook: discord.Webhook | None,
        chat: list[ChatLine],
        draft: Reply,
    ) -> None:
        """Someone spoke mid-reply: mark the partial message cut off and remember it."""
        try:
            sent = None
            if live.message is not None:
                sent = await live.finish(draft.text + INTERRUPTED_MARK)
            await asyncio.to_thread(
                self.ledger.record,
                draft,
                chat,
                channel_id=channel.id,
                message_id=sent.id if sent else None,
                stop_reason="interrupted",
            )
            if sent is not None:
                await self.coordinator._add_response_to_history(
                    channel.id,
                    sent.id,
                    draft.text + INTERRUPTED_MARK,
                    webhook.id if webhook else None,
                    self.settings.name,
                )
        except Exception:
            _LOG.exception("JevBot failed to wrap up an interrupted reply")

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
