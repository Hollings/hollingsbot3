"""Tests for tool-call parsing (hollingsbot.tools.parser).

These exercise the pure string-parsing helpers that sit on the LLM tool-call
critical path. No tools are actually executed.
"""

from __future__ import annotations

from hollingsbot.tools.parser import (
    _extract_args,
    _split_respecting_quotes,
    parse_arguments,
    parse_tool_calls,
)


class TestParseToolCalls:
    def test_single_call(self):
        calls = parse_tool_calls("TOOL_CALL: roll(sides=6)")
        assert len(calls) == 1
        assert calls[0].tool_name == "roll"
        assert calls[0].raw_args == "sides=6"
        assert calls[0].full_match == "TOOL_CALL: roll(sides=6)"

    def test_tool_name_lowercased(self):
        calls = parse_tool_calls("TOOL_CALL: Roll(sides=6)")
        assert calls[0].tool_name == "roll"

    def test_keyword_case_insensitive(self):
        calls = parse_tool_calls("tool_call: roll(sides=6)")
        assert len(calls) == 1

    def test_multiple_calls(self):
        text = "TOOL_CALL: a(x=1) then TOOL_CALL: b(y=2)"
        calls = parse_tool_calls(text)
        assert [c.tool_name for c in calls] == ["a", "b"]
        assert [c.raw_args for c in calls] == ["x=1", "y=2"]

    def test_no_calls(self):
        assert parse_tool_calls("just some plain text") == []

    def test_parens_inside_quotes_preserved(self):
        calls = parse_tool_calls('TOOL_CALL: say(content="hi (there, friend)")')
        assert len(calls) == 1
        assert calls[0].raw_args == 'content="hi (there, friend)"'

    def test_unclosed_paren_is_skipped(self):
        # No closing paren -> _extract_args returns None -> call dropped.
        assert parse_tool_calls("TOOL_CALL: roll(sides=6") == []


class TestExtractArgs:
    def test_balanced(self):
        raw, end = _extract_args("a=1)", 0)
        assert raw == "a=1"
        assert end == 4

    def test_nested_parens(self):
        text = "f(g(x))"
        raw, _ = _extract_args(text, 2)  # start just after first '('
        assert raw == "g(x)"

    def test_unclosed_returns_none(self):
        raw, end = _extract_args("a=1", 0)
        assert raw is None
        assert end == 0


class TestParseArguments:
    def test_empty(self):
        assert parse_arguments("") == {}
        assert parse_arguments("   ") == {}

    def test_single_pair(self):
        assert parse_arguments("max_value=100") == {"max_value": "100"}

    def test_multiple_pairs(self):
        assert parse_arguments("name=Alice, age=30") == {"name": "Alice", "age": "30"}

    def test_quoted_value_with_comma_and_parens(self):
        assert parse_arguments('content="text (a, b)", slot=1') == {
            "content": "text (a, b)",
            "slot": "1",
        }

    def test_single_quotes_stripped(self):
        assert parse_arguments("name='Bob'") == {"name": "Bob"}

    def test_part_without_equals_skipped(self):
        assert parse_arguments("standalone, key=val") == {"key": "val"}


class TestSplitRespectingQuotes:
    def test_plain_commas(self):
        assert _split_respecting_quotes("a, b, c") == ["a", "b", "c"]

    def test_comma_inside_quotes(self):
        assert _split_respecting_quotes('x="a, b", y=c') == ['x="a, b"', "y=c"]

    def test_no_comma(self):
        assert _split_respecting_quotes("solo") == ["solo"]
