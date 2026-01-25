"""Tool calling system for LLM chat bot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tokens import check_tokens, give_token

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["AVAILABLE_TOOLS", "Tool", "get_tool_definitions_text"]


@dataclass
class Tool:
    """Represents a callable tool that the LLM can use."""

    name: str
    description: str
    parameters: dict[str, str]  # param_name -> description
    function: Callable[..., str]
    channel_message: str | None = None  # Optional message to append to bot's response
    is_async: bool = False  # Whether the function is async


# Registry of available tools (reduced set - assistant handles memory/search/images)
AVAILABLE_TOOLS: dict[str, Tool] = {
    "give_token": Tool(
        name="give_token",
        description="Give one token to a user as a reward or gift. Tokens are tracked in the database.",
        parameters={
            "user": "The user to give a token to. Can be a @mention (e.g., <@123456>), user ID (e.g., 123456), or display name (e.g., John)"
        },
        function=give_token,
        channel_message="*hands over a token*",
    ),
    "check_tokens": Tool(
        name="check_tokens",
        description="Check how many tokens a user has.",
        parameters={"user": "The user to check. Can be a @mention, user ID, or display name"},
        function=check_tokens,
        channel_message=None,
    ),
    # Note: "assistant" tool removed - when using claude-cli provider,
    # Wendy IS Claude Code and has native access to WebSearch, WebFetch, Read
}


def get_tool_definitions_text() -> str:
    """Generate text description of all available tools for system prompt."""
    if not AVAILABLE_TOOLS:
        return ""

    lines = [
        "# Available Tools",
        "",
        "You have access to the following tools. To use a tool, include a line in your response with the format:",
        "TOOL_CALL: tool_name(param1=value1, param2=value2)",
        "",
        "The tool will be executed and you'll receive the result. You can then continue your response.",
        "",
        "Available tools:",
        "",
    ]

    for tool in AVAILABLE_TOOLS.values():
        lines.append(f"## {tool.name}")
        lines.append(f"{tool.description}")
        if tool.parameters:
            lines.append("")
            lines.append("Parameters:")
            for param_name, param_desc in tool.parameters.items():
                lines.append(f"- {param_name}: {param_desc}")
        else:
            lines.append("No parameters required.")
        lines.append("")

        # Example usage
        if tool.parameters:
            example_params = ", ".join(f"{p}=..." for p in tool.parameters)
            lines.append(f"Example: TOOL_CALL: {tool.name}({example_params})")
        else:
            lines.append(f"Example: TOOL_CALL: {tool.name}()")
        lines.append("")

    return "\n".join(lines)
