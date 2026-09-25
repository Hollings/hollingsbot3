"""Tests for JevBot (Discord side) and the coordinator hooks it relies on."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from hollingsbot.cogs.chat_bots import jev_bot as jev_bot_mod
from hollingsbot.cogs.chat_bots.jev_bot import INTERRUPTED_MARK, JevBot, JevBotSettings, chat_lines
from hollingsbot.cogs.chat_coordinator import ChatCoordinator
from hollingsbot.cogs.conversation import ConversationTurn
from hollingsbot.cogs.jev_commands import describe
from hollingsbot.jev.client import Usage
from hollingsbot.jev.ledger import JevLedger
from hollingsbot.jev.lexicon import Lexicon, LexiconSummary
from hollingsbot.jev.writer import ChatLine, Reply

CHANNEL = 1473033550805598253


class FakeWriter:
    """Says ``words`` one at a time; optionally hangs after ``hang_after`` words."""

    def __init__(self, words, hang_after=None):
        self.words = words
        self.hang_after = hang_after
        self.chats: list[list[ChatLine]] = []
        self.learned: list[list[str]] = []

    async def write(self, chat, on_word=None, draft=None, learned=()):
        self.chats.append(chat)
        self.learned.append(list(learned))
        reply = draft if draft is not None else Reply()
        for i, w in enumerate(self.words):
            if self.hang_after is not None and i == self.hang_after:
                await asyncio.Event().wait()
            reply.words.append(w)
            reply.text = " ".join(reply.words)
            reply.usage.record({"input_tokens": 1000, "cost": 0.002}, 0.5)
            if on_word:
                await on_word(reply.text)
        reply.stop_reason = "complete"
        return reply


def make_webhook():
    webhook = MagicMock()
    webhook.id = 555
    sent = MagicMock(spec=discord.Message)
    sent.id = 999
    webhook.send = AsyncMock(return_value=sent)
    webhook.edit_message = AsyncMock()
    return webhook, sent


def make_channel(webhook):
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = CHANNEL
    channel.webhooks = AsyncMock(return_value=[])
    channel.create_webhook = AsyncMock(return_value=webhook)
    return channel


def make_message(channel, content="Jev what is the capital of France?", *, bot=False, webhook_id=None, mid=42):
    message = MagicMock(spec=discord.Message)
    message.id = mid
    message.channel = channel
    message.content = content
    message.attachments = []
    message.webhook_id = webhook_id
    message.mentions = []
    message.guild = None
    message.author = MagicMock()
    message.author.bot = bot
    message.author.nick = None
    message.author.global_name = "Hollings"
    message.author.name = "hollings"
    message.add_reaction = AsyncMock()
    return message


def turn(message, name="Hollings"):
    return ConversationTurn(
        role="user", content=f"<{name}>: {message.content}", message_id=message.id, author_name=name
    )


@pytest.fixture
def jev(temp_db, mock_bot):
    coordinator = MagicMock()
    coordinator._add_response_to_history = AsyncMock()
    bot = JevBot(mock_bot, coordinator, MagicMock(), JevBotSettings(channels=frozenset({CHANNEL}), daily_budget=1.0))
    bot.ledger = JevLedger(temp_db)
    bot.lexicon = Lexicon(temp_db)
    return bot


async def test_learns_every_word_a_human_says_and_writes_with_them(jev):
    webhook, _ = make_webhook()
    message = make_message(make_channel(webhook), "I love quokkas <:blob:123> https://x.com/y")
    jev._writer = FakeWriter(["quokkas"])
    await jev.receive_message(message, [turn(message)])
    summary = jev.lexicon.summary()
    assert {w for w, _ in summary.newest} == {"i", "love", "quokkas"}
    assert dict(summary.newest)["quokkas"] == "Hollings"
    assert "quokkas" in jev._writer.learned[0]


async def test_ignored_messages_teach_nothing(jev):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)
    for message in (
        make_message(channel, "zebra from a bot", bot=True),
        make_message(channel, "zebra from a webhook", webhook_id=7),
        make_message(channel, "!zebra command"),
    ):
        await jev.receive_message(message, [turn(message)])
    assert jev.lexicon.summary().total == 0


def test_jev_command_text():
    empty = LexiconSummary(0, [], [])
    assert "hasn't learned any yet" in describe(empty, name="Jev", born=100, menu=250)
    summary = LexiconSummary(3, [("quokka", "Hollings"), ("pizza", "Mallory")], [("pizza", 4), ("quokka", 1)])
    text = describe(summary, name="Jev", born=100, menu=250)
    assert "learned **3** more" in text
    assert "*quokka* (Hollings)" in text and "*pizza* x4" in text
    assert "reach 3 learned words" in text


async def test_replies_through_a_new_jev_webhook_and_logs_the_reply(jev):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)
    message = make_message(channel)
    jev._writer = FakeWriter(["paris"])

    result = await jev.receive_message(message, [turn(message)])

    assert result == {"message_id": 999, "text": "paris", "webhook_id": 555, "bot_name": "Jev"}
    channel.create_webhook.assert_awaited_once()
    assert channel.create_webhook.await_args.kwargs["name"] == "Jev"
    jev.coordinator.claim_webhook.assert_called_once_with(555)
    webhook.send.assert_awaited_once_with("paris", username="Jev", wait=True)
    assert jev._writer.chats[0][-1] == ChatLine("Hollings", "Jev what is the capital of France?")
    assert jev.ledger.spent_today() == pytest.approx(0.002)


async def test_reuses_its_existing_webhook(jev, mock_bot):
    webhook, _ = make_webhook()
    webhook.name = "Jev"
    webhook.user = MagicMock(id=mock_bot.user.id)
    webhook.token = "t"
    channel = make_channel(webhook)
    channel.webhooks = AsyncMock(return_value=[webhook])
    jev._writer = FakeWriter(["hi"])
    message = make_message(channel)
    await jev.receive_message(message, [turn(message)])
    channel.create_webhook.assert_not_awaited()


async def test_interrupted_reply_is_marked_logged_and_remembered(jev):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)
    message = make_message(channel, "Jev tell me a joke")
    jev._writer = FakeWriter(["knock", "knock", "who"], hang_after=2)

    task = asyncio.create_task(jev.receive_message(message, [turn(message)]))
    while webhook.send.await_count == 0 or not jev._writer.chats:
        await asyncio.sleep(0)
    for _ in range(20):
        await asyncio.sleep(0)
    await jev._cancel_generation(CHANNEL)
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(*list(jev._cleanups))

    final_text = "knock knock" + INTERRUPTED_MARK
    assert webhook.edit_message.await_args_list[-1].kwargs["content"] == final_text
    jev.coordinator._add_response_to_history.assert_awaited_once_with(CHANNEL, 999, final_text, 555, "Jev")
    assert jev.ledger.spent_today() == pytest.approx(0.004)


async def test_over_budget_reacts_instead_of_replying(jev):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)
    spent = Reply("x", ["x"], [], "complete", Usage(calls=1, cost=5.0))
    jev.ledger.record(spent, [], channel_id=CHANNEL, message_id=None)
    jev._writer = FakeWriter(["hi"])
    message = make_message(channel)

    assert await jev.receive_message(message, [turn(message)]) is None
    message.add_reaction.assert_awaited_once_with(jev_bot_mod.OVER_BUDGET_REACTION)
    assert jev._writer.chats == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bot": True},
        {"webhook_id": 123},
        {"content": "!help"},
        {"content": "   "},
    ],
)
async def test_ignores_bots_webhooks_commands_and_empty(jev, kwargs):
    webhook, _ = make_webhook()
    message = make_message(make_channel(webhook), **kwargs)
    jev._writer = FakeWriter(["hi"])
    assert await jev.receive_message(message, [turn(message)]) is None
    assert jev._writer.chats == []


async def test_ignores_other_channels(jev):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)
    channel.id = 1
    message = make_message(channel)
    jev._writer = FakeWriter(["hi"])
    assert await jev.receive_message(message, [turn(message)]) is None
    assert not jev.claims_channel(1) and jev.claims_channel(CHANNEL)


def test_chat_lines_strip_name_prefix_mark_images_and_trim():
    image_turn = ConversationTurn(role="user", content="<Mallory>: look", images=[MagicMock()], author_name="Mallory")
    bot_turn = ConversationTurn(role="user", content="paris", author_name="Jev")
    long_turn = ConversationTurn(role="user", content="<Hollings>: " + "a" * 900, author_name="Hollings")
    old = ConversationTurn(role="user", content="<X>: too old", author_name="X")
    lines = chat_lines([old, image_turn, bot_turn, long_turn], count=3)
    assert lines[0] == ChatLine("Mallory", "look [image]")
    assert lines[1] == ChatLine("Jev", "paris")
    assert lines[2].speaker == "Hollings" and len(lines[2].text) == 500


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("JEV_BOT_CHANNELS", f"{CHANNEL}, 7")
    monkeypatch.setenv("JEV_BOT_NAME", "Jevvy")
    monkeypatch.setenv("JEV_CONTEXT_MESSAGES", "3")
    monkeypatch.setenv("JEV_DAILY_BUDGET_USD", "0.5")
    monkeypatch.setenv("JEV_TOP_P", "0.8")
    monkeypatch.setenv("JEV_MIN_WORDS", "10")
    monkeypatch.setenv("JEV_STYLE", "  Jev rambles.  ")
    s = JevBotSettings.from_env()
    assert s.channels == {CHANNEL, 7}
    assert (s.name, s.context_messages, s.daily_budget, s.writer.top_p) == ("Jevvy", 3, 0.5, 0.8)
    assert (s.writer.min_words, s.writer.style) == (10, "Jev rambles.")


def test_settings_default_to_no_channels(monkeypatch):
    monkeypatch.delenv("JEV_BOT_CHANNELS", raising=False)
    assert JevBotSettings.from_env().channels == frozenset()


# ---------------------------------------------------------------- coordinator


def test_claiming_bot_goes_first_only_in_its_channel(mock_bot):
    coordinator = ChatCoordinator(mock_bot)
    others = [MagicMock(spec=["receive_message"]) for _ in range(4)]
    claimer = MagicMock(spec=["receive_message", "claims_channel"])
    claimer.claims_channel = lambda cid: cid == CHANNEL
    coordinator.bots = [*others, claimer]
    for _ in range(10):
        assert coordinator._bots_in_turn_order(CHANNEL)[0] is claimer
    assert set(map(id, coordinator._bots_in_turn_order(1))) == set(map(id, coordinator.bots))


async def test_claimed_webhook_messages_skip_history_and_bots(mock_bot):
    coordinator = ChatCoordinator(mock_bot)
    responder = MagicMock()
    responder.receive_message = AsyncMock(return_value=None)
    coordinator.bots = [responder]
    coordinator.claim_webhook(555)
    message = make_message(make_channel(make_webhook()[0]), "knock", bot=True, webhook_id=555)
    message.guild = MagicMock()
    await coordinator.on_message(message)
    assert coordinator.channel_histories == {}
    responder.receive_message.assert_not_awaited()
