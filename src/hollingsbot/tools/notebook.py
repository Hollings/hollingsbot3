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
        # Pending changes awaiting confirmation: slot_num -> new_content
        self._pending: dict[int, str] = {}

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

    MAX_SLOT_LENGTH = 500

    def save_memory(self, slot: int, content: str) -> str:
        """Stage a memory slot overwrite (requires confirmation).

        Args:
            slot: Slot number (1-5)
            content: New content to write to the slot

        Returns:
            Confirmation prompt for the LLM to respond to
        """
        try:
            slot_num = int(slot)
        except (ValueError, TypeError):
            return f"Error: slot must be a number between 1 and {self.NUM_SLOTS}"

        if slot_num < 1 or slot_num > self.NUM_SLOTS:
            return f"Error: slot must be between 1 and {self.NUM_SLOTS}"

        content = str(content).strip()

        # Check length limit
        if len(content) > self.MAX_SLOT_LENGTH:
            return f"Error: content is {len(content)} chars, max is {self.MAX_SLOT_LENGTH}"

        # Get current content
        idx = slot_num - 1
        current = self.slots[idx]

        # Stage the change
        self._pending[slot_num] = content

        # Build confirmation prompt
        if current:
            old_preview = current[:100] + ("..." if len(current) > 100 else "")
            new_preview = content[:100] + ("..." if len(content) > 100 else "")
            return f"Replacing slot {slot_num} content '{old_preview}' with '{new_preview}', are you sure? Y/N"
        else:
            new_preview = content[:100] + ("..." if len(content) > 100 else "")
            return f"Writing to empty slot {slot_num}: '{new_preview}', are you sure? Y/N"

    def confirm_memory(self, slot: int, confirm: str) -> str:
        """Confirm or cancel a pending memory slot change.

        Args:
            slot: Slot number (1-5)
            confirm: 'Y' to confirm, 'N' to cancel

        Returns:
            Success or cancellation message
        """
        try:
            slot_num = int(slot)
        except (ValueError, TypeError):
            return f"Error: slot must be a number between 1 and {self.NUM_SLOTS}"

        if slot_num < 1 or slot_num > self.NUM_SLOTS:
            return f"Error: slot must be between 1 and {self.NUM_SLOTS}"

        # Check for pending change
        if slot_num not in self._pending:
            return f"Error: no pending change for slot {slot_num}"

        confirm = str(confirm).strip().upper()

        if confirm == "Y":
            # Commit the change
            content = self._pending.pop(slot_num)
            idx = slot_num - 1
            self.slots[idx] = content
            self._save_state()

            _LOG.info("Confirmed slot %d: %s", slot_num, content[:50])
            return f"Slot {slot_num} saved ({len(content)}/{self.MAX_SLOT_LENGTH} chars): {content}"
        elif confirm == "N":
            # Cancel the change
            self._pending.pop(slot_num)
            _LOG.info("Cancelled pending change for slot %d", slot_num)
            return f"Cancelled - slot {slot_num} unchanged"
        else:
            return "Error: please respond with Y or N"

    def get_notebook_text(self) -> str:
        """Get formatted notebook text for inclusion in system prompt.

        Returns:
            Formatted text showing all slots and their contents
        """
        lines = [
            "# Your Notebook",
            "",
            f"You have 5 memory slots (max {self.MAX_SLOT_LENGTH} chars each). These persist across restarts.",
            "",
            "Current notebook contents:",
            ""
        ]

        for i, content in enumerate(self.slots, start=1):
            if content:
                lines.append(f"{i}. {content} ({len(content)}/{self.MAX_SLOT_LENGTH})")
            else:
                lines.append(f"{i}. [empty]")

        lines.extend([
            "",
            "To save: save_memory(slot=1, content=\"your text here\")",
            "You will be asked to confirm with Y/N before the change is saved.",
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
    """Tool function to stage a memory slot overwrite.

    Args:
        slot: Slot number (1-5)
        content: New content to write to the slot

    Returns:
        Confirmation prompt for the LLM
    """
    if _notebook_manager is None:
        return "Error: Notebook not initialized"

    return _notebook_manager.save_memory(slot, content)


def confirm_memory(slot: str, confirm: str) -> str:
    """Tool function to confirm or cancel a pending memory change.

    Args:
        slot: Slot number (1-5)
        confirm: 'Y' to confirm, 'N' to cancel

    Returns:
        Success or cancellation message
    """
    if _notebook_manager is None:
        return "Error: Notebook not initialized"

    return _notebook_manager.confirm_memory(slot, confirm)
