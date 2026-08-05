"""Persistent debug log storage for temp bot LLM responses.

Wraps a JSON file on disk that maps ``message_id -> log_dict``. Used to power
the ``/debug`` slash command (or similar tooling) so we can recover the full
conversation context that produced a given message after the fact.

Capped at ``max_size`` entries — when full, the oldest entries (by stored
``timestamp``) are dropped.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from hollingsbot.utils.file_utils import atomic_write_json

if TYPE_CHECKING:
    from pathlib import Path

_LOG = logging.getLogger(__name__)


def _redact_conversation(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop base64 image payloads before persisting.

    Each data_url can be several MB; with up to ``max_size`` logs each holding
    a full conversation, keeping them would balloon the file to hundreds of MB
    and it gets rewritten on every temp bot reply.
    """
    redacted: list[dict[str, Any]] = []
    for entry in conversation:
        entry = dict(entry)
        images = entry.get("images")
        if isinstance(images, list) and images:
            entry["images"] = [
                {
                    **{k: v for k, v in img.items() if k != "data_url"},
                    "data_url_bytes": len(img.get("data_url") or ""),
                }
                if isinstance(img, dict)
                else img
                for img in images
            ]
        redacted.append(entry)
    return redacted


class DebugLogStore:
    """Bounded, disk-backed store of LLM response debug logs."""

    def __init__(self, file_path: Path, max_size: int = 100) -> None:
        self._file_path = file_path
        self._max_size = max_size
        self._logs: dict[int, dict[str, Any]] = {}
        self._load()

    # ---- persistence ------------------------------------------------------

    def _load(self) -> None:
        """Load logs from disk, replacing any in-memory state."""
        if not self._file_path.exists():
            _LOG.info("No existing temp bot debug logs file found, starting fresh")
            return

        try:
            with self._file_path.open("r") as f:
                data = json.load(f)
            # Convert string keys back to integers
            self._logs = {int(k): v for k, v in data.items()}
            _LOG.info("Loaded %d temp bot debug logs from disk", len(self._logs))
        except Exception as exc:
            _LOG.exception("Failed to load temp bot debug logs: %s", exc)
            self._logs = {}

    def _save(self) -> None:
        """Persist current logs to disk (atomically, so a crash can't corrupt them)."""
        try:
            # Convert integer keys to strings for JSON serialization
            data = {str(k): v for k, v in self._logs.items()}
            atomic_write_json(self._file_path, data, indent=None)
            _LOG.debug("Saved %d temp bot debug logs to disk", len(self._logs))
        except Exception as exc:
            _LOG.exception("Failed to save temp bot debug logs: %s", exc)

    # ---- public api -------------------------------------------------------

    def store(
        self,
        message_id: int,
        conversation: list[dict[str, Any]],
        response_text: str,
        llm_debug: dict[str, Any],
        tool_debug: list[dict[str, Any]],
    ) -> None:
        """Store debug log for a response, evicting oldest entries past the cap."""
        self._logs[message_id] = {
            "conversation": _redact_conversation(conversation),
            "response_text": response_text,
            "llm_debug": llm_debug,
            "tool_debug": tool_debug,
            "timestamp": time.time(),
        }

        # Trim old logs if we exceed the limit
        if len(self._logs) > self._max_size:
            sorted_ids = sorted(
                self._logs.keys(),
                key=lambda mid: self._logs[mid].get("timestamp", 0),
            )
            to_remove = sorted_ids[: len(self._logs) - self._max_size]
            for mid in to_remove:
                del self._logs[mid]

        self._save()

    def get(self, message_id: int) -> dict[str, Any] | None:
        """Retrieve debug log for a message ID, or ``None`` if not present."""
        return self._logs.get(message_id)
