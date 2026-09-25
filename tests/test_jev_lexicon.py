"""Tests for Jev's learned vocabulary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hollingsbot.jev.lexicon import Lexicon, learnable

T0 = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)


def test_learnable_skips_links_emoji_code_blocked_and_huge_tokens():
    text = "Pizza pizza! see https://example.com/x <:blob:123456> `rm -rf` tranny aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa don't"
    assert learnable(text) == ["pizza", "see", "don't"]


def test_learn_reports_only_new_words_and_keeps_first_teacher(temp_db):
    lex = Lexicon(temp_db)
    assert lex.learn("hi Jev I love pizza", speaker="Hollings", channel_id=1, now=T0) == [
        "hi",
        "jev",
        "i",
        "love",
        "pizza",
    ]
    assert lex.learn("pizza is life", speaker="Mallory", now=T0) == ["is", "life"]
    summary = lex.summary()
    assert summary.total == 7
    assert dict(summary.newest)["pizza"] == "Hollings"
    assert summary.favorites[0] == ("pizza", 2)


def test_a_message_counts_each_word_once(temp_db):
    lex = Lexicon(temp_db)
    lex.learn("lol lol lol", speaker="A", now=T0)
    assert lex.summary().favorites == [("lol", 1)]


def test_ranked_prefers_heard_often_and_recently(temp_db):
    lex = Lexicon(temp_db, half_life_days=7)
    for _ in range(4):
        lex.learn("sushi", speaker="A", now=T0 - timedelta(days=30))  # 4 uses, a month ago
    lex.learn("tacos", speaker="A", now=T0)  # 1 use, just now
    lex.learn("burrito", speaker="A", now=T0 - timedelta(days=1))
    lex.learn("burrito", speaker="A", now=T0)  # 2 uses, fresh
    assert lex.ranked(10, now=T0) == ["burrito", "tacos", "sushi"]
    assert lex.ranked(2, now=T0) == ["burrito", "tacos"]
    assert lex.ranked(10, exclude={"tacos"}, now=T0) == ["burrito", "sushi"]


def test_summary_can_leave_out_the_words_jev_was_born_with(temp_db):
    lex = Lexicon(temp_db)
    lex.learn("the quokka", speaker="A", now=T0)
    summary = lex.summary(exclude={"the"})
    assert summary.total == 1 and summary.newest == [("quokka", "A")]


def test_empty_lexicon(temp_db):
    lex = Lexicon(temp_db)
    assert lex.ranked(5) == []
    assert lex.summary().total == 0
    assert lex.learn("   ", speaker="A") == []
