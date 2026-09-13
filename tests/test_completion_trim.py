"""Tests for raw-completion trimming."""

from __future__ import annotations

from hollingsbot.cogs.chat_bots.completion_trim import trim_completion

PROMPT = "a three-man group is called a guyangle\nan eight-way is a cocktagon\na"


def test_drops_dangling_last_line():
    full = PROMPT + " nine-man is a nonagon\na ten-guy is a decag"
    assert trim_completion(full, len(PROMPT)) == PROMPT + " nine-man is a nonagon"


def test_single_line_continuation_is_kept_whole():
    # The only newlines are inside the prompt; never cut there.
    full = PROMPT + " nine-man is a nonagon"
    assert trim_completion(full, len(PROMPT)) == full


def test_hard_caps_at_limit_then_cuts_to_newline():
    full = PROMPT + " line one\n" + ("x" * 5000)
    out = trim_completion(full, len(PROMPT), limit=2000)
    assert out == PROMPT + " line one"
    assert len(out) <= 2000


def test_hard_cap_without_newline_after_prompt():
    full = PROMPT + " " + ("y" * 5000)
    out = trim_completion(full, len(PROMPT), limit=2000)
    assert len(out) == 2000
    assert out.startswith(PROMPT)


def test_only_whitespace_added_collapses_to_prompt():
    full = PROMPT + "\n\n\n"
    assert trim_completion(full, len(PROMPT)) == PROMPT


def test_trailing_whitespace_stripped():
    full = PROMPT + " done   \nfrag"
    assert trim_completion(full, len(PROMPT)) == PROMPT + " done"
