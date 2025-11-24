from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
import logging
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
        self._log = logging.getLogger(__name__)
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

    def _supports_seed(self) -> bool:
        m = self.model.lower()
        # Restrict seed to models known to accept it (e.g., FLUX variants, prunaai).
        return ("black-forest-labs/" in m) or ("flux" in m) or ("prunaai/" in m)

    def _supports_disable_safety(self) -> bool:
        m = self.model.lower()
        # Safety toggle is known for many FLUX models; avoid sending elsewhere.
        return ("black-forest-labs/" in m) or ("flux" in m)

    def _is_seedream(self) -> bool:
        m = self.model.lower()
        return ("bytedance/seedream-4" in m) or ("seedream-4" in m)

    async def generate(  # type: ignore[override]
        self,
        prompt: str,
        *,
        seed: int | None = None,
        image_input: Sequence[Any] | None = None,
        mask: str | None = None,
        output_format: str | None = None,
    ) -> bytes:
        """
        Generate, edit, or outpaint an image.
        - When image_input is provided, pass files/URLs to models that expect an `image_input` array (e.g., google/nano-banana).
        - When both image_input and mask are provided, use for outpainting with flux-fill-dev model.
        """
        try:
            inputs: dict[str, Any] = {"prompt": prompt}
            # Prefer maximum resolution and sequential generation by default for Seedream-4
            if self._is_seedream():
                inputs["size"] = "4K"
                inputs["sequential_image_generation"] = "auto"

            if image_input:
                prepared, cleanup = self._prepare_image_inputs(image_input)
                try:
                    # For flux-fill-dev (outpainting), use 'image' and 'mask' parameters
                    if mask and "flux-fill" in self.model.lower():
                        # flux-fill-dev expects 'image' (not 'image_input') and 'mask'
                        inputs["image"] = prepared[0] if prepared else None
                        # Prepare mask as single input
                        mask_prepared, mask_cleanup = self._prepare_image_inputs([mask])
                        try:
                            inputs["mask"] = mask_prepared[0] if mask_prepared else None
                        finally:
                            self._cleanup_files(mask_cleanup)
                    else:
                        # Standard edit mode with image_input array
                        inputs["image_input"] = prepared

                    if output_format:
                        inputs["output_format"] = output_format
                    # For Seedream, cap max_images so (input + generated) <= 15, default target = 4
                    if self._is_seedream() and inputs.get("sequential_image_generation") == "auto":
                        allowed = max(0, 15 - len(prepared))
                        if allowed >= 1:
                            inputs["max_images"] = min(4, allowed)
                        else:
                            # No room to generate; disable grouping to avoid API error
                            inputs["sequential_image_generation"] = "disabled"
                    # Add seed only for models that are known to support it
                    if seed is not None and self._supports_seed():
                        inputs["seed"] = seed
                    self._log.info(
                        "Replicate run (single) model=%s keys=%s", self.model, sorted(inputs.keys())
                    )
                    raw_output = await self._client.async_run(self.model, input=inputs)
                finally:
                    self._cleanup_files(cleanup)
            else:
                if self._supports_disable_safety():
                    inputs["disable_safety_checker"] = True
                # For Seedream default to up to 4 images when sequential generation is auto
                if self._is_seedream() and inputs.get("sequential_image_generation") == "auto":
                    inputs["max_images"] = 4
                if seed is not None and self._supports_seed():
                    inputs["seed"] = seed
                self._log.info(
                    "Replicate run (single) model=%s keys=%s", self.model, sorted(inputs.keys())
                )
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

    async def _collect_all(self, data: Any) -> list[bytes]:
        """Recursively collect all image bytes from Replicate outputs.

        Supports lists/tuples of URLs or file-like objects, async iterators that yield
        results, or single URL/bytes. Non-image scalars are ignored.
        """
        out: list[bytes] = []

        # Bytes directly
        if isinstance(data, (bytes, bytearray)):
            out.append(bytes(data))
            return out

        # URL or FileOutput-like
        if isinstance(data, str) and data.startswith(("http://", "https://")):
            out.append(await self._download(data))
            return out

        if isinstance(data, self._StreamTypes) and not isinstance(data, (bytes, bytearray)):
            out.append(await self._download(str(data)))
            return out

        # Async iterator / stream
        if isinstance(data, AsyncIterator) or hasattr(data, "__aiter__"):
            async for item in data:  # type: ignore[assignment]
                out.extend(await self._collect_all(item))
            return out

        # Sequences (lists/tuples)
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            for item in data:
                out.extend(await self._collect_all(item))
            return out

        # Fallback: nothing recognized
        return out

    async def generate_many(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        image_input: Sequence[Any] | None = None,
        mask: str | None = None,
        output_format: str | None = None,
    ) -> list[bytes]:
        """Generate one or more images and return all results as a list of bytes.

        - For multi-image-capable models like Seedream-4 (with
          sequential_image_generation='auto'), this returns all images.
        - For single-image models, the list has a single element.
        - When both image_input and mask are provided, use for outpainting with flux-fill-dev model.
        """
        inputs: dict[str, Any] = {"prompt": prompt}
        if self._is_seedream():
            inputs["size"] = "4K"
            inputs["sequential_image_generation"] = "auto"
            inputs.setdefault("max_images", 4)

        if image_input:
            prepared, cleanup = self._prepare_image_inputs(image_input)
            try:
                # For flux-fill-dev (outpainting), use 'image' and 'mask' parameters
                if mask and "flux-fill" in self.model.lower():
                    # flux-fill-dev expects 'image' (not 'image_input') and 'mask'
                    inputs["image"] = prepared[0] if prepared else None
                    # Prepare mask as single input
                    mask_prepared, mask_cleanup = self._prepare_image_inputs([mask])
                    try:
                        inputs["mask"] = mask_prepared[0] if mask_prepared else None
                    finally:
                        self._cleanup_files(mask_cleanup)
                else:
                    # Standard edit mode with image_input array
                    inputs["image_input"] = prepared

                if output_format:
                    inputs["output_format"] = output_format
                if self._is_seedream() and inputs.get("sequential_image_generation") == "auto":
                    allowed = max(0, 15 - len(prepared))
                    if allowed >= 1:
                        inputs["max_images"] = min(inputs.get("max_images", 4), allowed)
                    else:
                        inputs["sequential_image_generation"] = "disabled"
                if seed is not None and self._supports_seed():
                    inputs["seed"] = seed
                self._log.info(
                    "Replicate run (many) model=%s keys=%s", self.model, sorted(inputs.keys())
                )
                raw_output = await self._client.async_run(self.model, input=inputs)
            finally:
                self._cleanup_files(cleanup)
        else:
            if self._supports_disable_safety():
                inputs["disable_safety_checker"] = True
            if self._is_seedream() and inputs.get("sequential_image_generation") == "auto":
                inputs.setdefault("max_images", 4)
            if seed is not None and self._supports_seed():
                inputs["seed"] = seed
            self._log.info(
                "Replicate run (many) model=%s keys=%s", self.model, sorted(inputs.keys())
            )
            raw_output = await self._client.async_run(self.model, input=inputs)

        results = await self._collect_all(raw_output)
        # Ensure at least one image (raise if empty to keep callers aware)
        if not results:
            # fall back to the single-image normalizer to raise a clearer error
            await self._normalise_output(raw_output)
        return results

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
