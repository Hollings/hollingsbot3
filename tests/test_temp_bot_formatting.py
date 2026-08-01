"""Tests for temp bot output formatting.

Some models imitate the ``<DisplayName>: text`` history format and prepend
their own name to replies. Webhook messages already display the bot's name,
so `_strip_self_name_prefix` removes any leading self-name tag.
"""

from __future__ import annotations

from hollingsbot.cogs.chat_bots.temp_bot.manager import _strip_self_name_prefix


def test_strips_angle_bracket_prefix_without_colon():
    # Observed in production: "<Jeff> Wow, this place is still active?"
    assert _strip_self_name_prefix("<Jeff> Wow, this place is still active?", "Jeff") == (
        "Wow, this place is still active?"
    )


def test_strips_bare_name_colon_prefix():
    # Observed in production: "Jeff: Wow, Randy wandered off..."
    assert _strip_self_name_prefix("Jeff: Wow, Randy wandered off", "Jeff") == "Wow, Randy wandered off"


def test_strips_angle_bracket_colon_prefix():
    assert _strip_self_name_prefix("<Jeff>: hello", "Jeff") == "hello"


def test_case_insensitive():
    assert _strip_self_name_prefix("jeff: hello", "Jeff") == "hello"


def test_strips_repeated_prefixes():
    assert _strip_self_name_prefix("<Jeff> Jeff: hello", "Jeff") == "hello"


def test_name_with_regex_chars():
    assert _strip_self_name_prefix("Mr. Snuffles: soup time", "Mr. Snuffles") == "soup time"


def test_bare_name_without_colon_untouched():
    # "Jeff here!" is legitimate content, not a name tag
    assert _strip_self_name_prefix("Jeff here, reporting for duty", "Jeff") == "Jeff here, reporting for duty"


def test_other_names_untouched():
    assert _strip_self_name_prefix("<Randy>: hello", "Jeff") == "<Randy>: hello"


def test_mid_message_names_untouched():
    text = "I told you.\nJeff: that's me quoting myself"
    assert _strip_self_name_prefix(text, "Jeff") == text


def test_no_prefix_unchanged():
    assert _strip_self_name_prefix("just a normal message", "Jeff") == "just a normal message"
