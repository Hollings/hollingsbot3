"""Parse and execute tool calls from LLM responses."""

from __future__ import annotations

import re
import logging
from typing import NamedTuple

from . import AVAILABLE_TOOLS

_LOG = logging.getLogger(__name__)

# Global context for tool execution (set by the calling code)
_tool_context: dict[str, object] = {}


def set_current_context(context: dict[str, object]) -> None:
    """Set the current execution context for tools."""
    global _tool_context
    _tool_context = context.copy()


def get_current_context() -> dict[str, object]:
    """Get the current execution context for tools."""
    return _tool_context.copy()


class ToolCall(NamedTuple):
    """Represents a parsed tool call."""
    tool_name: str
    raw_args: str
    full_match: str


def parse_tool_calls(text: str) -> list[ToolCall]:
    """
    Parse tool calls from LLM response text.

    Format: TOOL_CALL: tool_name(arg1=value1, arg2=value2)

    Returns list of ToolCall objects.

    Handles parentheses inside quoted strings properly.
    """
    # Find all TOOL_CALL: occurrences
    pattern = r'TOOL_CALL:\s*(\w+)\('
    matches = re.finditer(pattern, text, re.IGNORECASE)

    calls: list[ToolCall] = []
    for match in matches:
        tool_name = match.group(1).lower()
        start_pos = match.end()  # Position after the opening '('

        # Parse arguments manually to handle nested parentheses and quotes
        raw_args, end_pos = _extract_args(text, start_pos)

        if raw_args is not None:
            full_match = text[match.start():end_pos]
            calls.append(ToolCall(tool_name, raw_args, full_match))

    return calls


def _extract_args(text: str, start: int) -> tuple[str | None, int]:
    """
    Extract arguments from a tool call, handling quotes and nested parentheses.

    Args:
        text: Full text containing the tool call
        start: Position after the opening parenthesis

    Returns:
        tuple: (raw_args_string, end_position) or (None, start) if parsing fails
    """
    in_quote = False
    quote_char = None
    paren_depth = 0
    i = start

    while i < len(text):
        char = text[i]

        # Handle quote state
        if char in ('"', "'"):
            if not in_quote:
                in_quote = True
                quote_char = char
            elif char == quote_char:
                # Check if it's escaped (simple check for preceding backslash)
                if i > 0 and text[i-1] != '\\':
                    in_quote = False
                    quote_char = None

        # Only count parentheses outside of quotes
        elif not in_quote:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                if paren_depth == 0:
                    # Found the closing parenthesis
                    return text[start:i], i + 1
                else:
                    paren_depth -= 1

        i += 1

    # Didn't find closing parenthesis
    return None, start


def parse_arguments(raw_args: str) -> dict[str, str]:
    """
    Parse argument string into dict, respecting quotes.

    Examples:
    - "max_value=100" -> {"max_value": "100"}
    - "name=Alice, age=30" -> {"name": "Alice", "age": "30"}
    - 'content="text (a, b)", slot=1' -> {"content": "text (a, b)", "slot": "1"}
    - "" -> {}
    """
    if not raw_args.strip():
        return {}

    args: dict[str, str] = {}

    # Parse respecting quotes - split by commas that aren't inside quotes
    parts = _split_respecting_quotes(raw_args)

    for part in parts:
        part = part.strip()
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()
        # Remove outer quotes if present
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        args[key] = value

    return args


def _split_respecting_quotes(text: str) -> list[str]:
    """Split text by commas, but not commas inside quotes."""
    parts = []
    current = []
    in_quote = False
    quote_char = None

    for i, char in enumerate(text):
        if char in ('"', "'"):
            if not in_quote:
                in_quote = True
                quote_char = char
            elif char == quote_char:
                # Check if escaped
                if i > 0 and text[i-1] != '\\':
                    in_quote = False
                    quote_char = None
            current.append(char)
        elif char == ',' and not in_quote:
            # Found a separator
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)

    # Add final part
    if current:
        parts.append(''.join(current))

    return [p.strip() for p in parts]


def execute_tool_call(tool_call: ToolCall) -> tuple[str, str]:
    """
    Execute a tool call and return (result, error).

    Returns:
        tuple: (result_text, error_text)
        - If successful: (result, "")
        - If error: ("", error_message)
    """
    tool_name = tool_call.tool_name

    # Check if tool exists
    if tool_name not in AVAILABLE_TOOLS:
        return "", f"Unknown tool: {tool_name}"

    tool = AVAILABLE_TOOLS[tool_name]

    # Parse arguments
    try:
        args = parse_arguments(tool_call.raw_args)
    except Exception as exc:
        _LOG.exception("Failed to parse tool arguments: %s", tool_call.raw_args)
        return "", f"Failed to parse arguments: {exc}"

    # Execute tool
    try:
        result = tool.function(**args)
        return str(result), ""
    except TypeError as exc:
        # Wrong arguments
        _LOG.warning("Tool %s called with invalid arguments: %s", tool_name, exc)
        return "", f"Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        _LOG.exception("Tool %s execution failed", tool_name)
        return "", f"Tool execution failed: {exc}"
