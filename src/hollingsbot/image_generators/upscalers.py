from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import aiohttp
import replicate
from PIL import Image


@dataclass(slots=True)
class RealESRGANUpscaler:
    """
    AI upscaler backed by Replicate's nightmareai/real-esrgan model.

    - Accepts raw image bytes.
    - Calls the ESRGAN model to upscale (default 2x).
    - Optionally re-encodes output JPEG to approximately match a target byte size.
    """

    # Use a pinned model version to avoid 404 and ensure consistent outputs
    model: str = (
        os.getenv(
            "REALESRGAN_MODEL",
            "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa",
        )
    )
    api_token: str = field(default_factory=lambda: os.getenv("REPLICATE_API_TOKEN", ""))
    # Internal client/session are explicit fields to be compatible with slots
    _client: Any | None = field(default=None, init=False, repr=False)
    _session: aiohttp.ClientSession | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_token:
            # Degrade gracefully (no token -> no upscale), but make it explicit
            self._client = None
        else:
            self._client = replicate.Client(api_token=self.api_token)

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _download(self, url: str) -> bytes:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        async with self._session.get(url) as rsp:
            rsp.raise_for_status()
            return await rsp.read()

    def _encode_jpeg_to_target(self, im: Image.Image, target_bytes: int) -> bytes:
        """
        Encode `im` (RGB) to JPEG aiming for ~target_bytes using a small binary search on quality.
        Falls back to closest size if an exact match isn't possible.
        """
        # Sanity guard
        target = max(16_000, min(target_bytes, 200_000_000))
        low, high = 25, 95
        best = None
        best_delta = None
        for _ in range(10):
            q = (low + high) // 2
            buf = BytesIO()
            try:
                im.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
            except Exception:
                buf.seek(0)
                buf.truncate(0)
                im.save(buf, format="JPEG", quality=q)
            data = buf.getvalue()
            delta = abs(len(data) - target)
            if best is None or delta < best_delta:  # type: ignore[operator]
                best, best_delta = data, delta
            # Adjust bounds
            if len(data) > target:
                high = max(low, q - 1)
            else:
                low = min(high, q + 1)
            # Early exit if close enough
            if delta <= max(10_000, int(0.05 * target)):
                return data
        return best or data  # type: ignore[name-defined]

    async def upscale(
        self,
        image_bytes: bytes,
        *,
        target_bytes: int | None = None,
        scale: int | None = None,
        face_enhance: bool = False,
    ) -> bytes:
        """
        Upscale image via Real-ESRGAN and return bytes.

        - If `target_bytes` is provided, re-encode the upscaled image as JPEG
          to approximately match that size.
        """
        # If no token, skip
        if not self._client:
            return image_bytes

        # Prepare inputs for Replicate run
        inputs: dict[str, Any] = {}
        # Default to a modest 2x upscale to avoid excessive sizes/cost
        if scale is None:
            scale = 2
        inputs["scale"] = int(scale)
        if face_enhance:
            inputs["face_enhance"] = True

        # Replicate can take a file handle for the input image under the key "image"
        # Use a small async helper to run and normalize outputs.
        async def _run(img: bytes) -> bytes:
            import tempfile

            # Create temp file and track its path for guaranteed cleanup
            tmp_path = None
            fh = None
            try:
                # Create temp file - use context manager for initial write
                fd, tmp_path = tempfile.mkstemp(prefix="esrgan_", suffix=".png")
                os.write(fd, img)
                os.close(fd)

                # Open for reading
                fh = open(tmp_path, "rb")
                out = await self._client.async_run(self.model, input={"image": fh, **inputs})  # type: ignore[union-attr]
            finally:
                # Always close file handle first
                if fh is not None:
                    with contextlib.suppress(OSError):
                        fh.close()
                # Then delete the temp file
                if tmp_path is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_path)

            # Normalize output: could be a URL string, bytes-like, blob with .url(), or list
            if isinstance(out, (bytes, bytearray)):
                return bytes(out)
            if isinstance(out, str) and out.startswith(("http://", "https://")):
                return await self._download(out)
            # Blob-like object with .url property or method
            try:
                if hasattr(out, "url"):
                    u = out.url() if callable(out.url) else out.url  # type: ignore[misc]
                    if isinstance(u, str) and u.startswith(("http://", "https://")):
                        return await self._download(u)
            except Exception:
                pass
            # Some versions return simple lists (e.g., single URL)
            if isinstance(out, (list, tuple)) and out:
                first = out[0]
                if isinstance(first, str) and first.startswith(("http://", "https://")):
                    return await self._download(first)
                if isinstance(first, (bytes, bytearray)):
                    return bytes(first)
            # Fallback: encode original to JPEG moderately if model format unknown
            return img

        upscaled = await _run(image_bytes)

        # Optionally re-encode to match ~target_bytes
        if target_bytes and target_bytes > 0:
            try:
                im = Image.open(BytesIO(upscaled))
                # Flatten to RGB to ensure consistent JPEG sizing
                if im.mode in ("RGBA", "LA"):
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    if im.mode == "LA":
                        rgb = im.convert("RGBA")
                        bg.paste(rgb, mask=rgb.split()[-1])
                    else:
                        bg.paste(im, mask=im.split()[-1])
                    rgb = bg
                else:
                    rgb = im.convert("RGB")
                return self._encode_jpeg_to_target(rgb, target_bytes)
            except Exception:
                return upscaled

        return upscaled
