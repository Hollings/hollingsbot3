# text_generators/openai_chatgpt.py
from __future__ import annotations

from typing import Dict, Sequence, TypedDict, Union, List

from openai import AsyncOpenAI

from .base import TextGeneratorAPI

_CLIENT_CACHE: Dict[str, AsyncOpenAI] = {}


class _Message(TypedDict):
    role: str
    content: str


class OpenAIChatTextGenerator(TextGeneratorAPI):
    """Text-generation backend for OpenAI ChatGPT models (default: gpt-4o).

    Requires OPENAI_API_KEY in the environment.
    Accepts either a single string or a list of {role, content} messages.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model

    def _get_client(self) -> AsyncOpenAI:
        if "default" not in _CLIENT_CACHE:
            _CLIENT_CACHE["default"] = AsyncOpenAI()  # picks up OPENAI_API_KEY
        return _CLIENT_CACHE["default"]

    async def generate(
        self,
        prompt: Union[str, Sequence[_Message]],
        *,
        temperature: float = 1.0,
    ) -> str:
        if isinstance(prompt, str):
            messages: List[_Message] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, Sequence):
            if not all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
                raise TypeError("Each message must be a dict with 'role' and 'content' keys")
            messages = list(prompt)  # type: ignore[arg-type]
        else:
            raise TypeError("prompt must be a string or a sequence of message dicts")

        client = self._get_client()
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,      # type: ignore[arg-type]
            temperature=temperature,
        )
        choice = resp.choices[0]
        return (choice.message.content or "").strip()
