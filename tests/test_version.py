"""Tests for hollingsbot.version (git-binary-free SHA resolution) and the presence cog."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from hollingsbot import version
from hollingsbot.cogs.presence import PresenceCog

if TYPE_CHECKING:
    from pathlib import Path

SHA = "2e0c5d0a1b2c3d4e5f60718293a4b5c6d7e8f901"
OTHER = "ffffffffffffffffffffffffffffffffffffffff"


@pytest.fixture(autouse=True)
def _clean_tree(monkeypatch):
    """Keep the real git binary out of it unless a test opts in."""
    monkeypatch.setattr(version, "_is_dirty", lambda cwd: False)


def _make_repo(root: Path, head: str) -> Path:
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text(head + "\n", "utf8")
    return git_dir


class TestRunningSha:
    def test_loose_branch_ref(self, tmp_path):
        git_dir = _make_repo(tmp_path, "ref: refs/heads/main")
        (git_dir / "refs" / "heads" / "main").write_text(SHA + "\n", "utf8")
        assert version.running_sha(tmp_path) == SHA[:7]

    def test_packed_ref(self, tmp_path):
        git_dir = _make_repo(tmp_path, "ref: refs/heads/main")
        (git_dir / "packed-refs").write_text(
            f"# pack-refs with: peeled fully-peeled sorted\n{OTHER} refs/heads/other\n{SHA} refs/heads/main\n",
            "utf8",
        )
        assert version.running_sha(tmp_path) == SHA[:7]

    def test_loose_ref_beats_stale_packed_ref(self, tmp_path):
        git_dir = _make_repo(tmp_path, "ref: refs/heads/main")
        (git_dir / "packed-refs").write_text(f"{OTHER} refs/heads/main\n", "utf8")
        (git_dir / "refs" / "heads" / "main").write_text(SHA + "\n", "utf8")
        assert version.running_sha(tmp_path) == SHA[:7]

    def test_detached_head(self, tmp_path):
        _make_repo(tmp_path, SHA)
        assert version.running_sha(tmp_path) == SHA[:7]

    def test_found_from_nested_directory(self, tmp_path):
        _make_repo(tmp_path, SHA)
        nested = tmp_path / "src" / "pkg"
        nested.mkdir(parents=True)
        assert version.running_sha(nested) == SHA[:7]

    def test_gitdir_pointer_file(self, tmp_path):
        real = tmp_path / "elsewhere"
        real.mkdir()
        (real / "HEAD").write_text(SHA + "\n", "utf8")
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / ".git").write_text("gitdir: ../elsewhere\n", "utf8")
        assert version.running_sha(checkout) == SHA[:7]

    def test_unknown_without_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(version, "_find_git_dir", lambda start: None)
        assert version.running_sha(tmp_path) == "unknown"

    def test_unknown_when_branch_has_no_commit(self, tmp_path):
        _make_repo(tmp_path, "ref: refs/heads/main")
        assert version.running_sha(tmp_path) == "unknown"

    def test_dirty_tree_gets_plus(self, tmp_path, monkeypatch):
        _make_repo(tmp_path, SHA)
        monkeypatch.setattr(version, "_is_dirty", lambda cwd: True)
        assert version.running_sha(tmp_path) == SHA[:7] + "+"

    def test_resolves_this_repo(self):
        assert re.fullmatch(r"[0-9a-f]{7}\+?", version.running_sha())


def test_presence_text_format(monkeypatch):
    monkeypatch.setattr(version, "running_sha", lambda start=None: "abc1234")
    assert re.fullmatch(r"vabc1234 · up [A-Z][a-z]{2} \d{2} \d{2}:\d{2}", version.presence_text())


class TestPresenceCog:
    def _bot(self, ready: bool) -> MagicMock:
        bot = MagicMock()
        bot.change_presence = AsyncMock()
        bot.is_ready.return_value = ready
        return bot

    async def test_on_ready_sets_custom_activity(self, monkeypatch):
        monkeypatch.setattr("hollingsbot.cogs.presence.presence_text", lambda: "vabc1234 · up Sep 17 14:02")
        bot = self._bot(ready=True)
        cog = PresenceCog(bot)
        await cog.on_ready()
        activity = bot.change_presence.await_args.kwargs["activity"]
        assert isinstance(activity, discord.CustomActivity)
        assert activity.name == "vabc1234 · up Sep 17 14:02"

    async def test_text_resolved_once_across_reconnects(self, monkeypatch):
        calls = []
        monkeypatch.setattr("hollingsbot.cogs.presence.presence_text", lambda: calls.append(1) or "v1")
        bot = self._bot(ready=True)
        cog = PresenceCog(bot)
        await cog.on_ready()
        await cog.on_ready()
        assert len(calls) == 1
        assert bot.change_presence.await_count == 2

    async def test_cog_load_waits_for_ready(self, monkeypatch):
        monkeypatch.setattr("hollingsbot.cogs.presence.presence_text", lambda: "v1")
        bot = self._bot(ready=False)
        await PresenceCog(bot).cog_load()
        bot.change_presence.assert_not_awaited()

    async def test_failure_is_swallowed(self, monkeypatch):
        monkeypatch.setattr("hollingsbot.cogs.presence.presence_text", lambda: "v1")
        bot = self._bot(ready=True)
        bot.change_presence.side_effect = RuntimeError("gateway down")
        await PresenceCog(bot).on_ready()  # must not raise
