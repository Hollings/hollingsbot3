"""Parse and execute tool calls from LLM responses."""

from __future__ import annotations

import re
import logging
from typing import NamedTuple

from . import AVAILABLE_TOOLS

_LOG = logging.getLogger(__name__)


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
    """
    pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
    matches = re.finditer(pattern, text, re.IGNORECASE)

    calls: list[ToolCall] = []
    for match in matches:
        tool_name = match.group(1).lower()
        raw_args = match.group(2).strip()
        full_match = match.group(0)
        calls.append(ToolCall(tool_name, raw_args, full_match))

    return calls


def parse_arguments(raw_args: str) -> dict[str, str]:
    """
    Parse argument string into dict.

    Examples:
    - "max_value=100" -> {"max_value": "100"}
    - "name=Alice, age=30" -> {"name": "Alice", "age": "30"}
    - "" -> {}
    """
    if not raw_args.strip():
        return {}

    args: dict[str, str] = {}

    # Simple parsing: split by comma, then by =
    parts = [p.strip() for p in raw_args.split(',')]
    for part in parts:
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()
        # Remove quotes if present
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        args[key] = value

    return args


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
