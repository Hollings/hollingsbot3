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
from hollingsbot.jev.client import Usage
from hollingsbot.jev.ledger import JevLedger
from hollingsbot.jev.writer import ChatLine, Reply

CHANNEL = 1473033550805598253


class FakeWriter:
    """Says ``words`` one at a time; optionally hangs after ``hang_after`` words."""

    def __init__(self, words, hang_after=None):
        self.words = words
        self.hang_after = hang_after
        self.chats: list[list[ChatLine]] = []

    async def write(self, chat, on_word=None, draft=None):
        self.chats.append(chat)
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
    message.author = MagicMock()
    message.author.bot = bot
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
    return bot


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
    s = JevBotSettings.from_env()
    assert s.channels == {CHANNEL, 7}
    assert (s.name, s.context_messages, s.daily_budget, s.writer.top_p) == ("Jevvy", 3, 0.5, 0.8)


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
