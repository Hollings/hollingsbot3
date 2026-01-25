# text_generators/huggingface.py
from __future__ import annotations

import asyncio
import threading
from typing import Any

from .base import TextGeneratorAPI

# --------------------------------------------------------------------------- #
# Model-wide cache so multiple tasks/processes don’t reload the same pipeline #
# --------------------------------------------------------------------------- #
_PIPELINE_CACHE: dict[str, Any] = {}
_LOCKS: dict[str, threading.Lock] = {}


class HuggingFaceTextGenerator(TextGeneratorAPI):
    """Generate text using a Hugging Face *text-generation* pipeline.

    The underlying model pipeline is cached globally, so repeated calls (even
    across different generator instances) reuse a single loaded model rather
    than re-loading it for every request.
    """

    def __init__(self, model: str = "gpt2-large") -> None:
        self.model = model

    # ------------------------------------------------------------------ helpers

    def _ensure_pipeline(self) -> Any:
        """Load (or retrieve) the HF pipeline for ``self.model``."""
        if self.model in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[self.model]

        # One lock **per** model to avoid serialising unrelated models.
        lock = _LOCKS.setdefault(self.model, threading.Lock())
        with lock:
            # Double-check once inside the lock.
            if self.model in _PIPELINE_CACHE:
                return _PIPELINE_CACHE[self.model]

            # Lazy import so that our tests can monkey-patch ``transformers``.
            from transformers import pipeline  # type: ignore

            try:
                import torch

                device = 0 if torch.cuda.is_available() else -1
            except ModuleNotFoundError:
                device = -1

            pipe = pipeline(
                "text-generation",
                model=self.model,
                device=device,
            )
            _PIPELINE_CACHE[self.model] = pipe
            return pipe

    # ------------------------------------------------------------------ public

    async def generate(self, prompt: str) -> str:
        """Return the model’s response for *prompt* (max 500 new tokens)."""
        pipe = self._ensure_pipeline()

        def _run(inp: str) -> str:  # heavy CPU/GPU work – run in executor
            data = pipe(inp, max_new_tokens=500)
            return data[0]["generated_text"][:2000].strip()

        return await asyncio.to_thread(_run, prompt)
