from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

__all__: Final = ["ImageGeneratorAPI"]


class ImageGeneratorAPI(ABC):
    """Async interface for all image-generation providers.

    Sub-classes **must** implement :meth:`generate` to return raw image
    bytes for a given prompt.  A default no-op :meth:`aclose` is
    provided so that callers can safely ``await generator.aclose()``
    regardless of whether the implementation needs explicit teardown.
    """

    @abstractmethod
    async def generate(self, prompt: str) -> bytes: ...

    async def aclose(self) -> None:  # noqa: D401 – “Close …”
        """Release any open resources (optional)."""
        return None