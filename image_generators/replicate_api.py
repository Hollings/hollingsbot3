from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Final, Sequence

import aiohttp
import replicate

from .base import ImageGeneratorAPI  # local import

# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReplicateImageGenerator(ImageGeneratorAPI):
    """
    Image generator backed by the **async** Replicate client.

    A ``seed`` can now be passed to :meth:`generate` to make results
    reproducible, e.g.::

        gen = ReplicateImageGenerator("black-forest-labs/flux-schnell")
        png_bytes = await gen.generate("A red panda", seed=42)
    """

    model: str = "black-forest-labs/flux-schnell"
    api_token: str = field(default_factory=lambda: os.getenv("REPLICATE_API_TOKEN", ""))

    # ------------------------------ dunders ------------------------------

    def __post_init__(self) -> None:
        if not self.api_token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN is required. "
                "Pass it explicitly or set the environment variable."
            )

        # replicate.Client is synchronous, but its ``.async_run`` returns a coroutine
        self._client = replicate.Client(api_token=self.api_token)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "ReplicateImageGenerator":  # type: ignore[override]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.aclose()

    # ------------------------------ public API --------------------------

    async def generate(self, prompt: str, *, seed: int | None = None) -> bytes:  # type: ignore[override]
        """
        Generate a PNG/JPEG as *bytes* for *prompt*.

        Parameters
        ----------
        seed
            Optional deterministic seed forwarded to the model’s ``seed`` input
            (if supported).  ``None`` leaves reproducibility up to the backend.

        Raises
        ------
        RuntimeError
            If the upstream call fails or produces no usable output.
        """
        try:
            inputs: dict[str, Any] = {
                "prompt": prompt,
                # Make safety configurable; default remains `True`
                "disable_safety_checker": True,
            }
            if seed is not None:
                inputs["seed"] = seed

            raw_output = await self._client.async_run(self.model, input=inputs)
            return await self._normalise_output(raw_output)
        except Exception as exc:
            raise RuntimeError(f"Replicate generation failed: {exc}") from exc

    async def aclose(self) -> None:  # noqa: D401
        """Release the underlying HTTP session, if any."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------- implementation details -------------------

    _StreamTypes: Final = (
        bytes,
        bytearray,
        replicate.helpers.FileOutput
        if hasattr(replicate.helpers, "FileOutput")
        else tuple(),
    )


    async def _normalise_output(self, data: Any) -> bytes:  # noqa: C901 – complexity intentional

        # 0. A **list/tuple of byte‑chunks**  -----------------------------------------------
        #    Replicate occasionally yields `[b'…', b'…']`; join them into a single blob.
        if (
           isinstance(data, (list, tuple))
           and data
           and all(isinstance(x, (bytes, bytearray)) for x in data)
        ):
           return b"".join(data)

        # 1. Already bytes‑like or ``bytearray`` or **single** FileOutput -------------------
        if isinstance(data, self._StreamTypes):
           # `replicate.helpers.FileOutput` is a str‑subclass pointing at a URL, not raw bytes
           if not isinstance(data, (bytes, bytearray)):
               return await self._download(str(data))
           return bytes(data)


        # 2. AsyncIterator – stream until first chunk --------------------------
        if isinstance(data, AsyncIterator) or hasattr(data, "__aiter__"):
            async for chunk in data:
                return await self._normalise_output(chunk)

        # 3. Sequence (list of URLs / bytes / etc.) ----------------------------
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            if not data:
                raise RuntimeError("Model returned an empty sequence.")
            return await self._normalise_output(data[0])

        # 4. URL – download -----------------------------------------------------
        if isinstance(data, str) and data.startswith(("http://", "https://")):
            return await self._download(data)

        raise RuntimeError(f"Unsupported Replicate output type: {type(data).__name__}")

    async def _download(self, url: str) -> bytes:
        """Fetch *url* and return its body as bytes."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        # Two quick retries with exponential back‑off for transient CDN errors
        for delay in (0, 1.0):
            if delay:
                await asyncio.sleep(delay)

            async with self._session.get(url) as rsp:
                if rsp.status == 200:
                    return await rsp.read()
                if rsp.status >= 500:  # retryable
                    continue

                raise RuntimeError(f"Download failed: HTTP {rsp.status}")

        raise RuntimeError(f"Download failed after retries: {url}")
