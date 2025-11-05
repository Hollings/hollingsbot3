"""Tool calling system for LLM chat bot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .notebook import save_memory

__all__ = ["Tool", "AVAILABLE_TOOLS", "get_tool_definitions_text"]


@dataclass
class Tool:
    """Represents a callable tool that the LLM can use."""
    name: str
    description: str
    parameters: dict[str, str]  # param_name -> description
    function: Callable[..., str]
    channel_message: str | None = None  # Optional message to append to bot's response


# Registry of all available tools
AVAILABLE_TOOLS: dict[str, Tool] = {
    "save_memory": Tool(
        name="save_memory",
        description="Save a note to yourself in one of 5 persistent memory slots. This is for your own use - write reminders to your future self about facts, preferences, ongoing tasks, or other context you want to remember. Use this actively and often whenever you learn something worth remembering. When all slots are full, overwrite the least relevant one. These notes persist across conversations and history clears.",
        parameters={
            "slot": "Slot number (1-5) where to store your note",
            "content": "Your note to yourself (should be a clear, concise reminder)"
        },
        function=save_memory,
        channel_message="*the bot will remember that*",
    ),
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
        ""
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
            example_params = ", ".join(f"{p}=..." for p in tool.parameters.keys())
            lines.append(f"Example: TOOL_CALL: {tool.name}({example_params})")
        else:
            lines.append(f"Example: TOOL_CALL: {tool.name}()")
        lines.append("")

    return "\n".join(lines)
