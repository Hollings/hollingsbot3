from __future__ import annotations

from abc import ABC, abstractmethod

class ImageGeneratorAPI(ABC):
    """Abstract base class for image generator providers."""

    @abstractmethod
    async def generate(self, prompt: str) -> bytes:
        """Return image bytes generated from the given prompt."""
        raise NotImplementedError
