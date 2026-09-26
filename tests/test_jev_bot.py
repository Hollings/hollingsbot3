"""Tests for JevBot (Discord side) and the coordinator hooks it relies on."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from hollingsbot.cogs.chat_bots import jev_bot as jev_bot_mod
from hollingsbot.cogs.chat_bots.jev_bot import CUT_OFF_MARK, JevBot, JevBotSettings, chat_lines
from hollingsbot.cogs.chat_coordinator import ChatCoordinator
from hollingsbot.cogs.conversation import ConversationTurn
from hollingsbot.cogs.jev_commands import describe
from hollingsbot.jev.client import JevError, Usage
from hollingsbot.jev.ledger import JevLedger
from hollingsbot.jev.lexicon import Lexicon, LexiconSummary
from hollingsbot.jev.spawns import JevSpawns
from hollingsbot.jev.writer import ChatLine, Reply, WriterConfig

CHANNEL = 1473033550805598253
AWAY = 1553193310439608371  # a channel outside JEV_BOT_CHANNELS, where `!spawn jev` brings it


class FakeWriter:
    """Says ``words`` one at a time; optionally hangs (or fails) after that many words."""

    def __init__(self, words, hang_after=None, fail_after=None):
        self.words = words
        self.hang_after = hang_after
        self.fail_after = fail_after
        self.chats: list[list[ChatLine]] = []
        self.learned: list[list[str]] = []

    async def write(self, chat, on_word=None, draft=None, learned=()):
        self.chats.append(chat)
        self.learned.append(list(learned))
        reply = draft if draft is not None else Reply()
        for i, w in enumerate(self.words):
            if self.hang_after is not None and i == self.hang_after:
                await asyncio.Event().wait()
            if self.fail_after is not None and i == self.fail_after:
                raise JevError("decisions API down")
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


class TypingSpy:
    """Stands in for ``channel.typing()``: remembers whether the indicator is on."""

    def __init__(self):
        self.active = False
        self.times = 0

    async def __aenter__(self):
        self.active = True
        self.times += 1

    async def __aexit__(self, *exc):
        self.active = False


def make_channel(webhook):
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = CHANNEL
    channel.webhooks = AsyncMock(return_value=[])
    channel.create_webhook = AsyncMock(return_value=webhook)
    channel.typing_spy = TypingSpy()
    channel.typing = MagicMock(return_value=channel.typing_spy)
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
    message.reference = None
    return message


def turn(message, name="Hollings"):
    return ConversationTurn(
        role="user", content=f"<{name}>: {message.content}", message_id=message.id, author_name=name
    )


@pytest.fixture
def temp_bots(monkeypatch):
    """The active temp bots Jev sees (by webhook_id); empty unless a test adds some."""
    active: list[dict] = []
    monkeypatch.setattr(jev_bot_mod, "get_temp_bots_for_channel", lambda channel_id: active)
    return active


@pytest.fixture
def jev(temp_db, mock_bot, temp_bots):
    coordinator = MagicMock()
    coordinator._add_response_to_history = AsyncMock()
    coordinator.recent_history = AsyncMock(return_value=[])
    typing_tracker = MagicMock()
    typing_tracker.wait_until_quiet = AsyncMock()
    settings = JevBotSettings(channels=frozenset({CHANNEL}), daily_budget=1.0)
    bot = JevBot(mock_bot, coordinator, typing_tracker, settings)
    bot.ledger = JevLedger(temp_db)
    bot.lexicon = Lexicon(temp_db)
    bot.spawns = JevSpawns(temp_db)
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
    assert "hasn't learned any yet" in describe(empty, name="Jev", born=1000, reach=0)
    summary = LexiconSummary(3, [("quokka", "Hollings"), ("pizza", "Mallory")], [("pizza", 4), ("quokka", 1)])
    text = describe(summary, name="Jev", born=1000, reach=3)
    assert "born knowing 1000 words" in text and "learned **3** more" in text
    assert "*quokka* (Hollings)" in text and "*pizza* x4" in text
    assert "at a time" not in text  # everything learned is usable
    assert "reach 2 learned words at a time" in describe(summary, name="Jev", born=100, reach=2)


def test_reachable_learned_depends_on_mode():
    llm = JevBotSettings(channels=frozenset(), suggest_model="m")
    alone = JevBotSettings(channels=frozenset(), writer=WriterConfig(vocab_size=100))
    assert llm.reachable_learned(500) == 500
    assert alone.reachable_learned(500) == 150 and alone.reachable_learned(20) == 20


def test_suggester_defaults(monkeypatch):
    for var in ("JEV_SUGGEST_MODEL", "JEV_KNOWN_ONLY", "JEV_VOCAB_SIZE"):
        monkeypatch.delenv(var, raising=False)
    alone = JevBotSettings.from_env()  # default: Jev alone, no LLM
    assert alone.suggest_model is None and (alone.writer.vocab_size, alone.writer.known_only) == (100, False)
    monkeypatch.setenv("JEV_SUGGEST_MODEL", "on")
    on = JevBotSettings.from_env()
    assert on.suggest_model == "meta-llama/llama-3.1-8b-instruct"
    assert (on.writer.vocab_size, on.writer.known_only) == (1000, True)
    monkeypatch.setenv("JEV_SUGGEST_MODEL", "some/model")
    monkeypatch.setenv("JEV_KNOWN_ONLY", "0")
    free = JevBotSettings.from_env()
    assert free.suggest_model == "some/model" and not free.writer.known_only


async def test_replies_through_a_new_jev_webhook_and_logs_the_reply(jev):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)
    message = make_message(channel)
    jev._writer = FakeWriter(["paris"])

    result = await jev.receive_message(message, [turn(message)])

    assert result == {"message_id": 999, "text": "paris", "webhook_id": 555, "bot_name": "Jev"}
    channel.create_webhook.assert_awaited_once()
    assert channel.create_webhook.await_args.kwargs["name"] == "Jev"
    webhook.send.assert_awaited_once_with("paris", username="Jev", wait=True)
    webhook.edit_message.assert_not_awaited()  # one finished message, never edited
    assert jev._writer.chats[0][-1] == ChatLine("Hollings", "Jev what is the capital of France?")
    assert jev.ledger.spent_today() == pytest.approx(0.002)


async def test_types_while_writing_then_posts_the_whole_reply_once(jev, mock_bot):
    webhook, sent = make_webhook()
    channel = make_channel(webhook)
    message = make_message(channel)
    writer = FakeWriter(["it's", "paris", "obviously"])
    jev._writer = writer
    events = []
    write = writer.write

    async def spying_write(*args, **kwargs):
        events.append(("write", channel.typing_spy.active, kwargs.get("on_word")))
        return await write(*args, **kwargs)

    writer.write = spying_write
    jev.typing_tracker.wait_until_quiet.side_effect = lambda *a: events.append(("wait", *a))
    webhook.send.side_effect = lambda text, **kw: events.append(("send", text, channel.typing_spy.active)) or sent

    await jev.receive_message(message, [turn(message)])

    assert events == [
        ("write", True, None),  # typing on, no per-word callback
        ("wait", CHANNEL, mock_bot.user.id),  # hold while a human is mid-message
        ("send", "it's paris obviously", False),  # then one post, typing off
    ]
    assert channel.typing_spy.times == 1


async def test_api_error_posts_what_it_had_cut_off_and_reacts(jev):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)
    message = make_message(channel)
    jev._writer = FakeWriter(["well", "the", "capital"], fail_after=2)

    assert await jev.receive_message(message, [turn(message)]) is None
    webhook.send.assert_awaited_once_with("well the" + CUT_OFF_MARK, username="Jev", wait=True)
    message.add_reaction.assert_awaited_once_with(jev_bot_mod.ERROR_REACTION)


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


async def test_interrupted_reply_is_never_posted_but_is_logged(jev, temp_db):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)
    message = make_message(channel, "Jev tell me a joke")
    jev._writer = FakeWriter(["knock", "knock", "who"], hang_after=2)

    task = asyncio.create_task(jev.receive_message(message, [turn(message)]))
    while not jev._writer.chats:
        await asyncio.sleep(0)
    for _ in range(20):
        await asyncio.sleep(0)
    await jev._cancel_generation(CHANNEL)
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(*list(jev._cleanups))

    webhook.send.assert_not_awaited()  # someone spoke first: Jev starts over on their message
    jev.coordinator._add_response_to_history.assert_not_awaited()
    assert jev.ledger.spent_today() == pytest.approx(0.004)  # the two words it picked were paid for
    with contextlib.closing(sqlite3.connect(temp_db)) as conn:
        rows = conn.execute("SELECT reply, stop_reason, message_id FROM jev_replies").fetchall()
    assert rows == [("knock knock", "interrupted", None)]


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
async def test_its_own_channels_ignore_bots_webhooks_commands_and_empty(jev, kwargs):
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
    monkeypatch.setenv("JEV_LLM_PAGES", "3")
    monkeypatch.setenv("JEV_OWN_PAGE", "1")
    monkeypatch.setenv("JEV_SHUFFLE", "0")
    s = JevBotSettings.from_env()
    assert s.channels == {CHANNEL, 7}
    assert (s.name, s.context_messages, s.daily_budget, s.writer.top_p) == ("Jevvy", 3, 0.5, 0.8)
    assert (s.writer.min_words, s.writer.style) == (10, "Jev rambles.")
    assert (s.writer.llm_pages, s.writer.own_page, s.writer.shuffle) == (3, True, False)


def test_stop_mode_from_env(monkeypatch):
    monkeypatch.setenv("JEV_STOP", " Choice ")
    assert JevBotSettings.from_env().writer.stop == "choice"
    monkeypatch.setenv("JEV_STOP", "whenever")  # a typo must not break every reply
    assert JevBotSettings.from_env().writer.stop == "threshold"
    monkeypatch.delenv("JEV_STOP")
    assert JevBotSettings.from_env().writer.stop == "threshold"


def test_units_from_env(monkeypatch):
    for var in ("JEV_UNITS", "JEV_NO_REPEAT", "JEV_SUGGEST_MODEL"):
        monkeypatch.delenv(var, raising=False)
    words = JevBotSettings.from_env()
    assert (words.writer.units, words.writer.no_repeat, words.suggest_model) == ("words", "never", None)
    monkeypatch.setenv("JEV_UNITS", "Pieces")
    monkeypatch.setenv("JEV_NO_REPEAT", "adjacent")
    pieces = JevBotSettings.from_env()
    assert (pieces.writer.units, pieces.writer.no_repeat) == ("pieces", "adjacent")
    assert pieces.suggest_model == "meta-llama/llama-3.1-8b-instruct"  # pieces are the LLM's tokens
    assert pieces.copy_named("Jev2").writer.units == "pieces"  # every Jev writes the same way
    monkeypatch.setenv("JEV_UNITS", "letters")  # a typo keeps the default rather than breaking replies
    monkeypatch.setenv("JEV_NO_REPEAT", "sometimes")
    typo = JevBotSettings.from_env()
    assert (typo.writer.units, typo.writer.no_repeat) == ("words", "never")


def test_paging_defaults_off_and_shuffle_on(monkeypatch):
    for var in ("JEV_LLM_PAGES", "JEV_OWN_PAGE", "JEV_SHUFFLE"):
        monkeypatch.delenv(var, raising=False)
    w = JevBotSettings.from_env().writer
    assert (w.llm_pages, w.own_page, w.shuffle) == (1, False, True)


def test_settings_default_to_no_channels(monkeypatch):
    monkeypatch.delenv("JEV_BOT_CHANNELS", raising=False)
    assert JevBotSettings.from_env().channels == frozenset()


# ---------------------------------------------------------------- !spawn jev


def away_channel(webhook):
    channel = make_channel(webhook)
    channel.id = AWAY
    return channel


def make_ctx(channel, content="!spawn jev 3"):
    ctx = MagicMock()
    ctx.channel = channel
    ctx.message = make_message(channel, content, mid=77)
    ctx.author = ctx.message.author
    ctx.send = AsyncMock()
    return ctx


async def test_spawned_jev_answers_the_chat_then_every_message_until_its_replies_run_out(jev):
    webhook, _ = make_webhook()
    channel = away_channel(webhook)
    earlier = make_message(channel, "anyone want to see my kidney stone", mid=1)
    jev.coordinator.recent_history = AsyncMock(return_value=[turn(earlier)])
    jev._writer = FakeWriter(["yes"])
    assert not jev.claims_channel(AWAY)

    await jev.spawn(make_ctx(channel), 2)

    # First reply, at once: to what was being said before the spawn.
    assert jev._writer.chats[0][-1] == ChatLine("Hollings", "anyone want to see my kidney stone")
    webhook.send.assert_awaited_once_with("yes", username="Jev", wait=True)
    jev.coordinator._add_response_to_history.assert_awaited_once_with(AWAY, 999, "yes", 555, "Jev")
    assert jev.claims_channel(AWAY) and jev.spawned == {AWAY: 1}

    message = make_message(channel, "jev you there", mid=2)
    assert await jev.receive_message(message, [turn(message)]) is not None
    # That was the last reply: Jev says goodbye like a temp bot and stops answering.
    goodbye = webhook.send.await_args_list[-1]
    assert goodbye.args[0].startswith("*[Jev ") and goodbye.kwargs["username"] == "Jev"
    assert jev.spawned == {} and jev.spawns.active() == {} and not jev.claims_channel(AWAY)
    later = make_message(channel, "jev?", mid=3)
    assert await jev.receive_message(later, [turn(later)]) is None


async def test_spawned_jev_and_other_bots_answer_each_other_until_its_replies_run_out(jev):
    webhook, _ = make_webhook()
    channel = away_channel(webhook)
    jev._writer = FakeWriter(["hey"])
    await jev.spawn(make_ctx(channel), 3)  # a quiet channel: no first reply

    answered = []
    for i in range(6):  # Wendy (a bot account) and a temp bot (a webhook) take turns talking to Jev
        other = make_message(channel, f"zebra number {i}", bot=True, webhook_id=123 if i % 2 else None, mid=10 + i)
        answered.append(await jev.receive_message(other, [turn(other, "Wendy")]) is not None)

    assert answered == [True, True, True, False, False, False]  # its 3 replies, then it's gone
    assert webhook.send.await_args_list[-1].args[0].startswith("*[Jev ")
    assert jev.spawned == {}
    assert jev.lexicon.summary().total == 0  # it learns from people, not from bots


async def test_spawned_jev_never_answers_itself_or_the_bot_account_it_runs_as(jev, mock_bot):
    webhook, _ = make_webhook()
    channel = away_channel(webhook)
    jev._writer = FakeWriter(["hi"])
    await jev.spawn(make_ctx(channel), 5)
    jev._webhooks[AWAY] = webhook  # as after its first post
    itself = make_message(channel, "hi", bot=True, webhook_id=webhook.id, mid=20)
    dog = make_message(channel, "Jev is here for 5 more replies", bot=True, mid=21)
    dog.author.id = mock_bot.user.id

    for message in (itself, dog):
        assert await jev.receive_message(message, [turn(message)]) is None
    assert jev._writer.chats == []


def jev_copy(jev, name, temp_db):
    copy = JevBot(jev.bot, jev.coordinator, jev.typing_tracker, jev.settings.copy_named(name))
    copy.ledger, copy.lexicon = jev.ledger, jev.lexicon
    copy.spawns = JevSpawns(temp_db, bot=name)
    return copy


def posted_by(channel, name, webhook_id, mid):
    message = make_message(channel, f"something {name} said", bot=True, webhook_id=webhook_id, mid=mid)
    message.author.name = name  # a webhook message's author is the name it posted under
    return message


async def test_two_jevs_in_a_channel_answer_each_other_until_both_run_out(jev, temp_db):
    webhook, _ = make_webhook()
    webhook2, _ = make_webhook()
    webhook2.id = 556
    channel = away_channel(webhook)
    jev2 = jev_copy(jev, "Jev2", temp_db)
    jev._writer, jev2._writer = FakeWriter(["hi"]), FakeWriter(["yo"])
    jev._webhooks[AWAY], jev2._webhooks[AWAY] = webhook, webhook2  # each posts through its own
    await jev.spawn(make_ctx(channel), 2)  # a quiet channel: no first replies
    await jev2.spawn(make_ctx(channel, "!spawn jev2 3"), 3)
    assert (jev.spawned, jev2.spawned) == ({AWAY: 2}, {AWAY: 3})  # one channel, two visits

    # Relay each post to both Jevs (the coordinator's job): only the other one answers it.
    said, answers = posted_by(channel, "Jev", 555, 100), []
    for mid in range(101, 110):
        replies = [(bot, await bot.receive_message(said, [turn(said, said.author.name)])) for bot in (jev, jev2)]
        answered = [(bot, result) for bot, result in replies if result is not None]
        if not answered:
            break
        [(bot, result)] = answered
        answers.append(bot.settings.name)
        said = posted_by(channel, bot.settings.name, result["webhook_id"], mid)

    assert answers == ["Jev2", "Jev", "Jev2", "Jev", "Jev2"]  # 2 + 3 replies, then both are gone
    assert (jev.spawned, jev2.spawned) == ({}, {})
    assert webhook2.send.await_args_list[0].kwargs["username"] == "Jev2"


async def test_a_jev_never_answers_a_post_under_its_own_name(jev):
    webhook, _ = make_webhook()
    channel = away_channel(webhook)
    jev._writer = FakeWriter(["hi"])
    await jev.spawn(make_ctx(channel), 5)  # its webhook isn't known yet (as right after a restart)
    itself = posted_by(channel, "Jev", 777, 30)
    assert await jev.receive_message(itself, [turn(itself, "Jev")]) is None
    assert jev._writer.chats == []


async def test_its_own_channels_answer_temp_bots_but_never_wendy(jev, temp_bots):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)  # CHANNEL: one of Jev's own
    jev._writer = FakeWriter(["arr"])
    temp_bots.append({"webhook_id": 321, "name": "Pirate"})
    pirate = posted_by(channel, "Pirate", 321, 40)
    wendy = posted_by(channel, "Wendy's Mobile Oracle", 654, 41)

    assert await jev.receive_message(pirate, [turn(pirate, "Pirate")]) is not None  # it runs out of replies
    assert await jev.receive_message(wendy, [turn(wendy, "Wendy")]) is None  # it never would
    temp_bots.clear()  # the pirate left
    later = posted_by(channel, "Pirate", 321, 42)
    assert await jev.receive_message(later, [turn(later, "Pirate")]) is None


async def test_its_own_channels_answer_a_jev_spawned_there(jev, temp_db):
    webhook, _ = make_webhook()
    webhook2, _ = make_webhook()
    webhook2.id = 556
    channel = make_channel(webhook)  # CHANNEL: Jev's own; Jev2 can still be spawned into it
    jev2 = jev_copy(jev, "Jev2", temp_db)
    jev.coordinator.bots = [jev, jev2]
    jev._writer = FakeWriter(["hi"])
    said = posted_by(channel, "Jev2", 556, 50)
    assert await jev.receive_message(said, [turn(said, "Jev2")]) is None  # Jev2 isn't visiting yet

    await jev2.spawn(make_ctx(channel, "!spawn jev2 3"), 3)
    jev2._webhooks[CHANNEL] = webhook2  # as after its first post
    said = posted_by(channel, "Jev2", 556, 51)
    assert await jev.receive_message(said, [turn(said, "Jev2")]) is not None
    assert jev.spawned == {}  # nothing to count: this is Jev's own channel


async def test_a_reply_to_another_bots_message_is_left_to_that_bot(jev):
    webhook, _ = make_webhook()
    channel = make_channel(webhook)
    jev._writer = FakeWriter(["hey"])

    def replying_to(author_name, *, bot, webhook_id, mid):
        replied = make_message(channel, "earlier", bot=bot, webhook_id=webhook_id, mid=mid)
        replied.author.name = author_name
        message = make_message(channel, "what do you mean", mid=mid + 100)
        message.reference = MagicMock(resolved=replied)
        return message

    to_pirate = replying_to("Pirate", bot=True, webhook_id=321, mid=60)
    assert await jev.receive_message(to_pirate, [turn(to_pirate)]) is None
    to_jev = replying_to("Jev", bot=True, webhook_id=555, mid=61)
    assert await jev.receive_message(to_jev, [turn(to_jev)]) is not None
    to_human = replying_to("mallory", bot=False, webhook_id=None, mid=62)
    assert await jev.receive_message(to_human, [turn(to_human)]) is not None


def test_copies_from_env(monkeypatch):
    monkeypatch.delenv("JEV_COPIES", raising=False)
    s = JevBotSettings.from_env()
    assert s.copies == ("Jev2",)
    monkeypatch.setenv("JEV_COPIES", "Jev2, Jev3, jev, Big Jev, JEV3,")  # the main name, a space, a repeat
    assert JevBotSettings.from_env().copies == ("Jev2", "Jev3")
    copy = s.copy_named("Jev2")
    assert (copy.name, copy.channels, copy.copies) == ("Jev2", frozenset(), ())
    assert copy.writer == s.writer and copy.suggest_model == s.suggest_model  # the same brain


async def test_spawned_into_a_quiet_channel_waits_for_someone_to_talk(jev):
    webhook, _ = make_webhook()
    channel = away_channel(webhook)
    jev._writer = FakeWriter(["hi"])
    ctx = make_ctx(channel)

    await jev.spawn(ctx, 3)

    ctx.message.add_reaction.assert_awaited_once_with(jev_bot_mod.SPAWNED_REACTION)
    webhook.send.assert_not_awaited()
    assert jev.spawned == {AWAY: 3}


async def test_a_first_reply_cut_off_by_someone_talking_uses_no_reply(jev):
    webhook, _ = make_webhook()
    channel = away_channel(webhook)
    earlier = make_message(channel, "so anyway", mid=1)
    jev.coordinator.recent_history = AsyncMock(return_value=[turn(earlier)])
    jev._writer = FakeWriter(["knock", "knock"], hang_after=1)

    spawning = asyncio.create_task(jev.spawn(make_ctx(channel), 3))
    while not jev._writer.chats:
        await asyncio.sleep(0)
    await jev._cancel_generation(AWAY)  # what the coordinator does when a human speaks
    await spawning  # the command itself finishes quietly
    await asyncio.gather(*list(jev._cleanups))

    webhook.send.assert_not_awaited()
    jev.coordinator._add_response_to_history.assert_not_awaited()
    assert jev.spawned == {AWAY: 3}


async def test_spawn_refuses_jevs_own_channels_and_bad_counts(jev):
    webhook, _ = make_webhook()
    ctx = make_ctx(make_channel(webhook))  # CHANNEL is in JEV_BOT_CHANNELS
    await jev.spawn(ctx, 5)
    assert "already lives in this channel" in ctx.send.await_args.args[0]
    for bad in (0, jev_bot_mod.SPAWN_REPLIES_MAX + 1):
        ctx = make_ctx(away_channel(webhook))
        await jev.spawn(ctx, bad)
        assert "1 to 20 replies" in ctx.send.await_args.args[0]
    assert jev.spawned == {} and jev.spawns.active() == {}


async def test_spawning_again_resets_the_count_without_a_new_reply(jev):
    webhook, _ = make_webhook()
    channel = away_channel(webhook)
    jev._writer = FakeWriter(["hi"])
    await jev.spawn(make_ctx(channel), 3)
    jev.coordinator.recent_history = AsyncMock(return_value=[turn(make_message(channel, "hey", mid=1))])
    ctx = make_ctx(channel)

    await jev.spawn(ctx, 7)

    assert "7 replies left" in ctx.send.await_args.args[0]
    assert jev.spawned == {AWAY: 7} and jev.spawns.active() == {AWAY: 7}
    assert jev._writer.chats == []


async def test_despawn_ends_the_visit_at_once(jev):
    webhook, _ = make_webhook()
    channel = away_channel(webhook)
    await jev.spawn(make_ctx(channel), 3)

    assert await jev.despawn(channel) is True
    assert jev.spawned == {} and jev.spawns.active() == {}
    webhook.send.assert_not_awaited()  # `!despawn` says so itself; no goodbye line
    assert await jev.despawn(channel) is False


async def test_a_visit_survives_a_restart(jev, mock_bot, temp_db):
    webhook, _ = make_webhook()
    await jev.spawn(make_ctx(away_channel(webhook)), 4)
    reborn = JevBot(mock_bot, jev.coordinator, jev.typing_tracker, jev.settings)
    reborn.spawns = JevSpawns(temp_db)
    assert reborn.claims_channel(AWAY) and reborn.spawned == {AWAY: 4}


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


async def test_recent_history_warms_the_channel_and_hands_out_a_copy(mock_bot):
    coordinator = ChatCoordinator(mock_bot)
    coordinator._ensure_channel_warm = AsyncMock()
    said = ConversationTurn(role="user", content="<A>: hi", message_id=1, author_name="A")
    coordinator._history_for_channel(CHANNEL).append(said)
    channel = make_channel(make_webhook()[0])

    got = await coordinator.recent_history(channel)

    coordinator._ensure_channel_warm.assert_awaited_once_with(channel)
    assert got == [said]
    got.clear()
    assert list(coordinator._history_for_channel(CHANNEL)) == [said]


async def test_a_jev_post_goes_to_the_other_bots_and_into_history_once(mock_bot):
    """Jev's own posts reach the other bots (another Jev answers them), recorded once in history."""
    coordinator = ChatCoordinator(mock_bot)
    coordinator._ensure_channel_warm = AsyncMock()
    message = make_message(make_channel(make_webhook()[0]), "knock", bot=True, webhook_id=555, mid=5)
    message.guild = MagicMock()
    posted = turn(message, "Jev")
    coordinator._prepare_full_turn = AsyncMock(return_value=posted)
    other_jev = MagicMock()
    other_jev.receive_message = AsyncMock(return_value=None)
    coordinator.bots = [other_jev]

    await coordinator.on_message(message)
    # The coordinator handling the message Jev was answering records the same post from its reply.
    await coordinator._add_response_to_history(CHANNEL, 5, "knock", 555, "Jev")

    other_jev.receive_message.assert_awaited_once()
    assert other_jev.receive_message.await_args.args[1][-1] is posted
    assert list(coordinator._history_for_channel(CHANNEL)) == [posted]
