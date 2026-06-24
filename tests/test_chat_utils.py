"""Tests for hollingsbot.cogs.chat_utils helpers."""

from __future__ import annotations

import pytest

from hollingsbot.cogs.chat_utils import webhook_id_from_url


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://discord.com/api/webhooks/123456789012345678/abcDEF-token_123", 123456789012345678),
        ("https://discord.com/api/v10/webhooks/987654321098765432/some.Token", 987654321098765432),
        # Trailing slash
        ("https://discord.com/api/webhooks/111222333444555666/tok/", 111222333444555666),
        # Query params (e.g. thread_id) must be stripped before parsing
        ("https://discord.com/api/webhooks/222333444555666777/tok?thread_id=42", 222333444555666777),
        # ID-only form (no token segment)
        ("https://discord.com/api/webhooks/333444555666777888", 333444555666777888),
    ],
)
def test_extracts_webhook_id(url, expected):
    assert webhook_id_from_url(url) == expected


def test_returns_none_for_unparseable():
    assert webhook_id_from_url("https://example.com/not/a/webhook") is None
    assert webhook_id_from_url("") is None


def test_token_with_digit_run_does_not_confuse_id():
    """The ID must come from the path segment, not a digit run in the token.

    Regression: ownership used to be a substring test of str(id) against the
    full URL, so a webhook_id appearing inside another webhook's secret token
    could cause a false match. Exact ID extraction prevents that.
    """
    # A different bot's id (999) happens to be a substring of this token.
    url = "https://discord.com/api/webhooks/123456789012345678/x999xToken"
    assert webhook_id_from_url(url) == 123456789012345678
    assert webhook_id_from_url(url) != 999
