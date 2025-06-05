from __future__ import annotations

import os
from typing import Any, AsyncIterator

import replicate

from .base import ImageGeneratorAPI


class ReplicateImageGenerator(ImageGeneratorAPI):
    """Generate images using Replicate's API."""

    def __init__(self, model: str = "black-forest-labs/flux-schnell") -> None:
        self.client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
        self.model = model

    async def _extract_bytes(self, output: Any) -> bytes:
        """Normalize different output types from replicate.run."""
        if isinstance(output, replicate.helpers.FileOutput):
            return await output.aread()
        if isinstance(output, bytes):
            return output
        if isinstance(output, list):
            if not output:
                raise RuntimeError("No output from model")
            return await self._extract_bytes(output[0])
        if isinstance(output, AsyncIterator) or hasattr(output, "__aiter__"):
            async for item in output:
                return await self._extract_bytes(item)
        raise RuntimeError("Unexpected output type from Replicate")

    async def generate(self, prompt: str) -> bytes:
        output = await self.client.async_run(
            self.model, input={"prompt": prompt}
        )
        return await self._extract_bytes(output)
