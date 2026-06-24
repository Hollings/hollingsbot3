"""Tests for hollingsbot.settings ID-parsing helpers."""

from __future__ import annotations

from hollingsbot import settings


class TestParseIdSet:
    def test_none_returns_empty(self):
        assert settings.parse_id_set(None) == set()

    def test_empty_string_returns_empty(self):
        assert settings.parse_id_set("") == set()

    def test_whitespace_only_returns_empty(self):
        assert settings.parse_id_set("  ,  ") == set()

    def test_parses_and_strips(self):
        assert settings.parse_id_set("1, 2 , 3") == {1, 2, 3}

    def test_ignores_non_numeric_tokens(self):
        assert settings.parse_id_set("1,abc,2,,3x") == {1, 2}

    def test_deduplicates(self):
        assert settings.parse_id_set("5,5,5") == {5}


class TestGetAdminUserIds:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("ADMIN_USER_IDS", "10, 20")
        assert settings.get_admin_user_ids() == {10, 20}

    def test_missing_env_is_empty(self, monkeypatch):
        monkeypatch.delenv("ADMIN_USER_IDS", raising=False)
        assert settings.get_admin_user_ids() == set()
