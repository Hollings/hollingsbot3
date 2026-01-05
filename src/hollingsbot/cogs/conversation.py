"""Conversation data structures for chat system."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ImageAttachment:
    """Represents an image attachment with metadata and encoded data."""
    name: str
    url: str
    data_url: str  # base64 encoded data URL
    width: int | None = None
    height: int | None = None
    size: int | None = None

    def to_payload(self) -> dict[str, str]:
        """Convert to API payload format."""
        return {"data_url": self.data_url}


@dataclass(slots=True)
class ConversationTurn:
    """Represents a single turn in the conversation history.

    This is the RAW turn stored in the coordinator's history with real names/IDs.
    """
    role: str  # "user" or "assistant"
    content: str
    images: list[ImageAttachment] = field(default_factory=list)
    message_id: int | None = None
    author_id: int | None = None
    author_name: str | None = None
    webhook_id: int | None = None  # None for main bot, webhook ID for temp bots


@dataclass(slots=True)
class ModelTurn:
    """Lightweight turn for sending to LLM API."""
    role: str
    text: str
    images: list[ImageAttachment] = field(default_factory=list)
