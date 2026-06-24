"""Tests for the summarization package: prompts, Summarizer, and SummaryCache."""

from __future__ import annotations

from pathlib import Path

import pytest

import hollingsbot.prompt_db as prompt_db
from hollingsbot.summarization import summary_cache as sc_mod
from hollingsbot.summarization.summarizer import (
    Summarizer,
    build_level_1_prompt,
    build_level_2_prompt,
)
from hollingsbot.summarization.summary_cache import (
    GROUP_SIZE,
    CachedMessage,
    MessageGroup,
    SummaryCache,
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _msg(mid: int, name: str = "Alice", content: str = "hi", ts: int = 2_000_000_000) -> CachedMessage:
    return CachedMessage(
        channel_id=1,
        message_id=mid,
        author_id=10,
        author_name=name,
        content=content,
        timestamp=ts,
    )


def _group(
    start: int,
    end: int,
    summary: str | None,
    *,
    count: int = 5,
    level: int = 1,
    channel: int = 1,
) -> MessageGroup:
    return MessageGroup(
        id=None,
        channel_id=channel,
        level=level,
        start_message_id=start,
        end_message_id=end,
        summary_text=summary,
        message_count=count,
    )


class TestPromptBuilders:
    def test_level_1_includes_messages(self):
        prompt = build_level_1_prompt([_msg(1, "Bob", "hello"), _msg(2, "Sue", "world")])
        assert "Bob: hello" in prompt
        assert "Sue: world" in prompt
        assert "ONE sentence" in prompt

    def test_level_2_bullets_summaries(self):
        prompt = build_level_2_prompt(["talked about cats", "discussed dogs"])
        assert "- talked about cats" in prompt
        assert "- discussed dogs" in prompt
        assert "2 sentences" in prompt


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------
class _FakeLLM:
    def __init__(self, reply: str = "  a summary  "):
        self.reply = reply
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class TestSummarizer:
    async def test_summarize_messages_strips(self):
        llm = _FakeLLM("  done  ")
        out = await Summarizer(llm).summarize_messages([_msg(1)])
        assert out == "done"
        assert len(llm.prompts) == 1

    async def test_summarize_messages_empty(self):
        llm = _FakeLLM()
        out = await Summarizer(llm).summarize_messages([])
        assert out == "[No messages]"
        assert llm.prompts == []  # LLM not called

    async def test_summarize_groups(self):
        llm = _FakeLLM("combined")
        groups = [
            _group(1, 5, "s1"),
            _group(6, 10, "s2"),
        ]
        out = await Summarizer(llm).summarize_groups(groups)
        assert out == "combined"

    async def test_summarize_groups_no_summaries(self):
        llm = _FakeLLM()
        groups = [_group(1, 5, None)]
        out = await Summarizer(llm).summarize_groups(groups)
        assert out == "[No summaries to combine]"
        assert llm.prompts == []


# ---------------------------------------------------------------------------
# SummaryCache (real sqlite via temp db)
# ---------------------------------------------------------------------------
@pytest.fixture
def cache(temp_db, monkeypatch) -> SummaryCache:
    monkeypatch.setattr(prompt_db, "DB_PATH", Path(temp_db))
    monkeypatch.setattr(sc_mod, "DB_PATH", Path(temp_db))
    prompt_db.init_db()
    return SummaryCache()


# Timestamp after the SUMMARIZATION_CUTOFF so messages are visible by default.
TS = 2_000_000_000


class TestMessageCaching:
    def test_cache_and_get_recent_order(self, cache):
        for mid in (3, 1, 2):
            cache.cache_message(_msg(mid, ts=TS))
        recent = cache.get_recent_messages(1, count=10)
        assert [m.message_id for m in recent] == [1, 2, 3]  # chronological

    def test_get_recent_limit(self, cache):
        for mid in range(1, 6):
            cache.cache_message(_msg(mid, ts=TS))
        recent = cache.get_recent_messages(1, count=2)
        assert [m.message_id for m in recent] == [4, 5]

    def test_insert_or_replace(self, cache):
        cache.cache_message(_msg(1, content="first", ts=TS))
        cache.cache_message(_msg(1, content="second", ts=TS))
        msgs = cache.get_all_messages_ordered(1)
        assert len(msgs) == 1
        assert msgs[0].content == "second"

    def test_cutoff_filters_old_messages(self, cache):
        cache.cache_message(_msg(1, content="old", ts=100))  # before cutoff
        cache.cache_message(_msg(2, content="new", ts=TS))
        assert [m.message_id for m in cache.get_all_messages_ordered(1)] == [2]
        # include_old returns everything
        assert len(cache.get_all_messages_ordered(1, include_old=True)) == 2


class TestMessageGroups:
    def test_save_and_query_by_level(self, cache):
        gid = cache.save_message_group(_group(1, 5, "summary"))
        assert isinstance(gid, int)
        groups = cache.get_groups_by_level(1, 1)
        assert len(groups) == 1
        assert groups[0].summary_text == "summary"
        assert groups[0].id == gid

    def test_summarized_filter(self, cache):
        cache.save_message_group(_group(1, 5, "has"))
        cache.save_message_group(_group(6, 10, None))
        assert len(cache.get_groups_by_level(1, 1)) == 2
        summarized = cache.get_summarized_groups(1, 1)
        assert len(summarized) == 1
        assert summarized[0].summary_text == "has"

    def test_ordered_by_start_id(self, cache):
        cache.save_message_group(_group(11, 15, "b"))
        cache.save_message_group(_group(1, 5, "a"))
        groups = cache.get_groups_by_level(1, 1)
        assert [g.start_message_id for g in groups] == [1, 11]


class TestClearPoints:
    def test_set_and_get(self, cache):
        assert cache.get_clear_point(1) is None
        cache.set_clear_point(1, 42)
        assert cache.get_clear_point(1) == 42

    def test_replace(self, cache):
        cache.set_clear_point(1, 10)
        cache.set_clear_point(1, 20)
        assert cache.get_clear_point(1) == 20


class TestLevel1Chunking:
    def test_needs_buffer_of_group_size(self, cache):
        # Exactly GROUP_SIZE messages -> nothing to summarize (all in buffer)
        for mid in range(1, GROUP_SIZE + 1):
            cache.cache_message(_msg(mid, ts=TS))
        assert cache.get_messages_needing_level1_summary(1) == []

    def test_returns_chunk_when_enough(self, cache):
        # 2 * GROUP_SIZE messages -> first GROUP_SIZE summarizable, last GROUP_SIZE buffered
        for mid in range(1, 2 * GROUP_SIZE + 1):
            cache.cache_message(_msg(mid, ts=TS))
        chunks = cache.get_messages_needing_level1_summary(1)
        assert len(chunks) == 1
        assert len(chunks[0]) == GROUP_SIZE
        assert [m.message_id for m in chunks[0]] == list(range(1, GROUP_SIZE + 1))

    def test_skips_already_covered(self, cache):
        for mid in range(1, 2 * GROUP_SIZE + 1):
            cache.cache_message(_msg(mid, ts=TS))
        cache.save_message_group(_group(1, GROUP_SIZE, "done", count=GROUP_SIZE))
        assert cache.get_messages_needing_level1_summary(1) == []


class TestLevel2Grouping:
    def test_needs_group_size_level1(self, cache):
        for i in range(GROUP_SIZE - 1):
            cache.save_message_group(_group(i * 5 + 1, i * 5 + 5, f"s{i}"))
        assert cache.get_level1_groups_needing_level2(1) == []

    def test_returns_chunk_of_level1(self, cache):
        for i in range(GROUP_SIZE):
            cache.save_message_group(_group(i * 5 + 1, i * 5 + 5, f"s{i}"))
        chunks = cache.get_level1_groups_needing_level2(1)
        assert len(chunks) == 1
        assert len(chunks[0]) == GROUP_SIZE


class TestHierarchicalContext:
    def test_includes_recent_and_summaries(self, cache):
        for mid in range(1, 4):
            cache.cache_message(_msg(mid, ts=TS))
        cache.save_message_group(_group(1, 5, "l1"))
        ctx = cache.get_hierarchical_context(1, raw_message_count=2)
        assert [m.message_id for m in ctx["raw_messages"]] == [2, 3]
        assert len(ctx["level_1_groups"]) == 1
        assert ctx["total_message_coverage"] == 2 + 5

    def test_respects_clear_point(self, cache):
        for mid in range(1, 6):
            cache.cache_message(_msg(mid, ts=TS))
        cache.set_clear_point(1, 3)
        ctx = cache.get_hierarchical_context(1, raw_message_count=10)
        assert all(m.message_id > 3 for m in ctx["raw_messages"])
