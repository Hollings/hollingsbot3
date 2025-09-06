from __future__ import annotations

from abc import ABC, abstractmethod


class TextGeneratorAPI(ABC):
    """Abstract base class for text generator providers."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Return generated text for the given prompt."""
        raise NotImplementedError
