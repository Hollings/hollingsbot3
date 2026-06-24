"""Tests for the temp-bot fallback name/departure generators.

These are pure functions used when the LLM identity generator is unavailable, so
they must always produce well-formed output regardless of randomness.
"""

from __future__ import annotations

from hollingsbot.cogs.chat_bots.temp_bot.names import (
    _ADJECTIVES,
    _DEPARTURE_PHRASES,
    _NOUNS,
    departure_message,
    generate_bot_name,
)


def test_generate_bot_name_is_adjective_space_noun():
    for _ in range(50):
        name = generate_bot_name()
        adjective, noun = name.split(" ", 1)
        assert adjective in _ADJECTIVES
        assert noun in _NOUNS


def test_generate_bot_name_nonempty():
    assert generate_bot_name().strip()


def test_departure_message_wraps_name_and_phrase():
    for _ in range(50):
        msg = departure_message("Silent Cipher")
        assert msg.startswith("*[Silent Cipher ")
        assert msg.endswith("]*")
        phrase = msg[len("*[Silent Cipher ") : -len("]*")]
        assert phrase in _DEPARTURE_PHRASES


def test_departure_message_preserves_special_chars_in_name():
    msg = departure_message("Bot (v2)")
    assert msg.startswith("*[Bot (v2) ")
    assert msg.endswith("]*")
