"""Notebook tool for storing persistent memory slots across conversations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_LOG = logging.getLogger(__name__)


class NotebookManager:
    """Manages 5 memory slots that persist across bot restarts."""

    NUM_SLOTS = 5

    def __init__(self, state_path: Path) -> None:
        """Initialize the notebook manager.

        Args:
            state_path: Path to the JSON file storing notebook state
        """
        self.state_path = state_path
        self.slots: list[str] = self._load_state()

    def _load_state(self) -> list[str]:
        """Load notebook slots from disk."""
        if not self.state_path.exists():
            return [""] * self.NUM_SLOTS

        try:
            data = json.loads(self.state_path.read_text("utf-8"))
            if isinstance(data, dict) and "slots" in data:
                slots = data["slots"]
                if isinstance(slots, list) and len(slots) == self.NUM_SLOTS:
                    return slots
        except Exception:
            _LOG.exception("Failed to load notebook state from %s", self.state_path)

        return [""] * self.NUM_SLOTS

    def _save_state(self) -> None:
        """Save notebook slots to disk atomically."""
        payload = {"slots": self.slots}
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        tmp_path.replace(self.state_path)

    def save_memory(self, slot: int, content: str) -> str:
        """Save content to a memory slot.

        Args:
            slot: Slot number (1-5)
            content: Content to save

        Returns:
            Success or error message
        """
        try:
            slot_num = int(slot)
        except (ValueError, TypeError):
            return f"Error: slot must be a number between 1 and {self.NUM_SLOTS}"

        if slot_num < 1 or slot_num > self.NUM_SLOTS:
            return f"Error: slot must be between 1 and {self.NUM_SLOTS}"

        # Convert to 0-indexed
        idx = slot_num - 1

        # Store the content
        self.slots[idx] = str(content).strip()
        self._save_state()

        _LOG.info("Saved memory to slot %d: %s", slot_num, content[:50])
        return f"Memory saved to slot {slot_num}"

    def get_notebook_text(self) -> str:
        """Get formatted notebook text for inclusion in system prompt.

        Returns:
            Formatted text showing all slots and their contents
        """
        lines = [
            "# Your Notebook",
            "",
            "You have a persistent notebook with 5 memory slots for your own use. These are",
            "notes from yourself to yourself - reminders you've written for future reference.",
            "The notebook persists even when conversation history is cleared.",
            "",
            "**USE THIS OFTEN.** Actively save important information: facts about the user,",
            "their preferences, ongoing tasks, conversation context, or anything you want to",
            "remember. Pay attention to what's currently stored and update it as you learn new",
            "things. When all slots are full, overwrite the least relevant one.",
            "",
            "Current notebook contents:",
            ""
        ]

        for i, content in enumerate(self.slots, start=1):
            if content:
                lines.append(f"{i}. {content}")
            else:
                lines.append(f"{i}. [empty]")

        lines.extend([
            "",
            "To update a slot, use: TOOL_CALL: save_memory(slot=1, content=\"your note to yourself\")",
            ""
        ])

        return "\n".join(lines)


# Global notebook manager instance (will be initialized by llm_chat cog)
_notebook_manager: NotebookManager | None = None


def initialize_notebook(state_path: Path) -> None:
    """Initialize the global notebook manager.

    Args:
        state_path: Path to the JSON file storing notebook state
    """
    global _notebook_manager
    _notebook_manager = NotebookManager(state_path)
    _LOG.info("Notebook manager initialized with state file: %s", state_path)


def get_notebook_manager() -> NotebookManager | None:
    """Get the global notebook manager instance.

    Returns:
        NotebookManager instance or None if not initialized
    """
    return _notebook_manager


def save_memory(slot: str, content: str) -> str:
    """Tool function to save content to a memory slot.

    Args:
        slot: Slot number (1-5)
        content: Content to save

    Returns:
        Success or error message
    """
    if _notebook_manager is None:
        return "Error: Notebook not initialized"

    return _notebook_manager.save_memory(slot, content)
