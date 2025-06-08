from __future__ import annotations

import asyncio
from typing import Any

from .base import TextGeneratorAPI


class HuggingFaceTextGenerator(TextGeneratorAPI):
    """Generate text using a HuggingFace pipeline."""

    def __init__(self, model: str = "gpt2-large") -> None:
        self.model = model
        self._pipeline: Any | None = None

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline  # imported lazily
            try:
                import torch  # noqa: WPS433 - optional
                device = 0 if torch.cuda.is_available() else -1
            except ModuleNotFoundError:
                device = -1
            self._pipeline = pipeline("text-generation", model=self.model, device=device)
        return self._pipeline

    async def generate(self, prompt: str) -> str:
        pipe = self._ensure_pipeline()

        def _run(text: str) -> str:
            data = pipe(text, max_new_tokens=500)
            result = data[0]["generated_text"]
            return result[:2000].strip()

        return await asyncio.to_thread(_run, prompt)
