from __future__ import annotations

import os
from typing import Any, AsyncIterator, Final, Sequence

import aiohttp
import replicate

from .base import ImageGeneratorAPI


class ReplicateImageGenerator(ImageGeneratorAPI):
    """Image generator backed by the `replicate` Python client."""

    _DEFAULT_MODEL: Final[str] = "black-forest-labs/flux-schnell"

    def __init__(self, model: str | None = None, *, api_token: str | None = None) -> None:
        api_token = api_token or os.getenv("REPLICATE_API_TOKEN", "")
        if not api_token:
            raise RuntimeError(
                "Missing REPLICATE_API_TOKEN environment variable or argument."
            )

        self.client = replicate.Client(api_token=api_token)
        self.model: str = model or self._DEFAULT_MODEL
        self._session: aiohttp.ClientSession | None = None

    # --------------------------------------------------------------------- public

    async def generate(self, prompt: str) -> bytes:
        """Return raw PNG/JPEG bytes for *prompt*."""
        raw_output = await self.client.async_run(
            self.model,
            input={"prompt": prompt, "disable_safety_checker": True},
        )
        return await self._to_bytes(raw_output)

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -------------------------------------------------------------------- private

    async def _to_bytes(self, data: Any) -> bytes:  # noqa: C901 – deliberate
        """Normalise *data* returned by `replicate.run` into ``bytes``."""

        # 1. Already bytes/bytearray
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

        # 2. replicate.helpers.FileOutput (stream)
        if hasattr(replicate.helpers, "FileOutput") and isinstance(
            data, replicate.helpers.FileOutput
        ):
            return await data.aread()

        # 3. Async iterator (streaming responses)
        if isinstance(data, AsyncIterator) or hasattr(data, "__aiter__"):
            async for chunk in data:
                return await self._to_bytes(chunk)

        # 4. Sequence of outputs – recurse into the first element
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            if not data:
                raise RuntimeError("Model produced no output.")
            return await self._to_bytes(data[0])

        # 5. URL string – download it
        if isinstance(data, str) and data.startswith(("http://", "https://")):
            return await self._download(data)

        raise RuntimeError(f"Unsupported Replicate output type: {type(data)!r}")

    async def _download(self, url: str) -> bytes:
        """Download *url* and return its payload."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        async with self._session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to download {url!s} – HTTP {resp.status}")
            return await resp.read()
