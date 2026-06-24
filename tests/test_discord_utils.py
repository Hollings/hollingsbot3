"""Tests for hollingsbot.utils.discord_utils.get_display_name.

Uses lightweight stand-in objects rather than real discord.py objects so the
fallback hierarchy (nick > global_name > name) can be exercised exhaustively.
"""

from __future__ import annotations

from types import SimpleNamespace

from hollingsbot.utils.discord_utils import get_display_name


def test_member_nick_takes_priority():
    user = SimpleNamespace(nick="Nicky", global_name="GlobalName", name="username")
    assert get_display_name(user) == "Nicky"


def test_falls_back_to_global_name_when_no_nick():
    user = SimpleNamespace(nick=None, global_name="GlobalName", name="username")
    assert get_display_name(user) == "GlobalName"


def test_falls_back_to_username_when_no_nick_or_global():
    user = SimpleNamespace(nick=None, global_name=None, name="username")
    assert get_display_name(user) == "username"


def test_user_object_without_nick_attr_uses_global_name():
    # discord.User has no `nick` attribute at all.
    user = SimpleNamespace(global_name="GlobalName", name="username")
    assert get_display_name(user) == "GlobalName"


def test_user_object_without_nick_or_global_uses_name():
    user = SimpleNamespace(name="username")
    assert get_display_name(user) == "username"


def test_empty_nick_is_skipped():
    # Empty string is falsy -> should fall through to next option.
    user = SimpleNamespace(nick="", global_name="", name="username")
    assert get_display_name(user) == "username"
