from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Final, Sequence, BinaryIO
from tempfile import NamedTemporaryFile

import aiohttp
import replicate

from .base import ImageGeneratorAPI  # local import


@dataclass(slots=True)
class ReplicateImageGenerator(ImageGeneratorAPI):
    """
    Async Replicate-backed generator.

    Supports:
      - Text to image: generate(prompt=..., seed=...)
      - Image editing: generate(prompt=..., image_input=[...], output_format="png")
    """

    model: str = "black-forest-labs/flux-schnell"
    api_token: str = field(default_factory=lambda: os.getenv("REPLICATE_API_TOKEN", ""))

    def __post_init__(self) -> None:
        if not self.api_token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN is required. "
                "Pass it explicitly or set the environment variable."
            )
        self._client = replicate.Client(api_token=self.api_token)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "ReplicateImageGenerator":  # type: ignore[override]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.aclose()

    async def generate(  # type: ignore[override]
        self,
        prompt: str,
        *,
        seed: int | None = None,
        image_input: Sequence[Any] | None = None,
        output_format: str | None = None,
    ) -> bytes:
        """
        Generate or edit an image. When image_input is provided, pass files/URLs
        to models that expect an `image_input` array (e.g., google/nano-banana).
        """
        try:
            inputs: dict[str, Any] = {"prompt": prompt}

            if image_input:
                prepared, cleanup = self._prepare_image_inputs(image_input)
                try:
                    inputs["image_input"] = prepared
                    if output_format:
                        inputs["output_format"] = output_format
                    # Avoid unsupported fields for some Google models
                    if seed is not None and not self.model.startswith("google/"):
                        inputs["seed"] = seed
                    raw_output = await self._client.async_run(self.model, input=inputs)
                finally:
                    self._cleanup_files(cleanup)
            else:
                inputs["disable_safety_checker"] = True
                if seed is not None:
                    inputs["seed"] = seed
                raw_output = await self._client.async_run(self.model, input=inputs)

            return await self._normalise_output(raw_output)
        except Exception as exc:
            raise RuntimeError(f"Replicate generation failed: {exc}") from exc

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    _StreamTypes: Final = (
        bytes,
        bytearray,
        replicate.helpers.FileOutput
        if hasattr(replicate.helpers, "FileOutput")
        else tuple(),
    )

    async def _normalise_output(self, data: Any) -> bytes:
        if (
            isinstance(data, (list, tuple))
            and data
            and all(isinstance(x, (bytes, bytearray)) for x in data)
        ):
            return b"".join(data)

        if isinstance(data, self._StreamTypes):
            if not isinstance(data, (bytes, bytearray)):
                return await self._download(str(data))
            return bytes(data)

        if isinstance(data, AsyncIterator) or hasattr(data, "__aiter__"):
            async for chunk in data:
                return await self._normalise_output(chunk)

        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            if not data:
                raise RuntimeError("Model returned an empty sequence.")
            return await self._normalise_output(data[0])

        if isinstance(data, str) and data.startswith(("http://", "https://")):
            return await self._download(data)

        raise RuntimeError(f"Unsupported Replicate output type: {type(data).__name__}")

    async def _download(self, url: str) -> bytes:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        for delay in (0, 1.0):
            if delay:
                await asyncio.sleep(delay)
            async with self._session.get(url) as rsp:
                if rsp.status == 200:
                    return await rsp.read()
                if rsp.status >= 500:
                    continue
                raise RuntimeError(f"Download failed: HTTP {rsp.status}")
        raise RuntimeError(f"Download failed after retries: {url}")

    # ---- helpers for editing ----

    def _prepare_image_inputs(
        self, items: Sequence[Any]
    ) -> tuple[list[Any], list[tuple[BinaryIO, str | None]]]:
        prepared: list[Any] = []
        cleanup: list[tuple[BinaryIO, str | None]] = []
        for item in items:
            if isinstance(item, str) and item.startswith(("http://", "https://", "data:")):
                prepared.append(item)
                continue
            if isinstance(item, (bytes, bytearray)):
                tmp = NamedTemporaryFile(prefix="nb_", suffix=".png", delete=False)
                tmp.write(item)
                tmp.flush()
                tmp.close()
                fh = open(tmp.name, "rb")
                prepared.append(fh)
                cleanup.append((fh, tmp.name))
                continue
            if hasattr(item, "read"):
                prepared.append(item)
                continue
            try:
                path = os.fspath(item)
                fh = open(path, "rb")
                prepared.append(fh)
                cleanup.append((fh, None))
            except TypeError as exc:
                raise RuntimeError(f"Unsupported image_input element: {type(item)}") from exc
        if not prepared:
            raise RuntimeError("image_input is empty.")
        return prepared, cleanup

    def _cleanup_files(self, cleanup: list[tuple[BinaryIO, str | None]]) -> None:
        for fh, path in cleanup:
            try:
                fh.close()
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
