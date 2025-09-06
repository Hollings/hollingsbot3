# text_generators/openai_chatgpt.py
from __future__ import annotations

from typing import Dict, Sequence, TypedDict, Union, List, Any
import logging

from openai import AsyncOpenAI

from .base import TextGeneratorAPI

_CLIENT_CACHE: Dict[str, AsyncOpenAI] = {}
_LOG = logging.getLogger(__name__)


class _Message(TypedDict):
    role: str
    content: str


class OpenAIChatTextGenerator(TextGeneratorAPI):
    """Text-generation backend for OpenAI models.

    - Defaults to "gpt-5" and uses the Responses API with low reasoning effort.
    - Falls back to Chat Completions for non-gpt-5 models (e.g., gpt-4o).

    Requires OPENAI_API_KEY in the environment.
    Accepts either a single string or a list of {role, content} messages.
    """

    def __init__(self, model: str = "gpt-5") -> None:
        self.model = model

    def _get_client(self) -> AsyncOpenAI:
        if "default" not in _CLIENT_CACHE:
            _CLIENT_CACHE["default"] = AsyncOpenAI()  # picks up OPENAI_API_KEY
        return _CLIENT_CACHE["default"]

    def _is_gpt5(self) -> bool:
        name = (self.model or "").lower()
        return name.startswith("gpt-5") or name == "gpt-5"

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Best-effort conversion of OpenAI-style content into plain text.

        - Strings are returned as-is.
        - Lists of {type: text|image_url, ...} are flattened with placeholders for images.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                try:
                    t = item.get("type")
                    if t == "text":
                        parts.append(item.get("text", ""))
                    elif t == "image_url":
                        url = item.get("image_url", {}).get("url") or item.get("image_url")
                        parts.append(f"[image: {url}]")
                    else:
                        parts.append(str(item))
                except Exception:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)
        return str(content)

    @staticmethod
    def _to_responses_easy_input(messages: List[_Message]) -> tuple[str | None, List[Dict[str, Any]]]:
        """Build Responses API EasyInputMessageParam list preserving roles and content.

        Returns (instructions, input_messages).
        - Takes the first system message as instructions.
        - Emits the remaining messages as EasyInput messages with their roles
          (user/assistant/system/developer) and raw OpenAI-style content.
        """
        instructions: str | None = None
        out: List[Dict[str, Any]] = []

        def _to_resp_content_list(content: Any, role_for_types: str) -> List[Dict[str, Any]]:
            # Normalize to a list of {type: input_text|input_image|input_file, ...}
            if isinstance(content, str):
                tname = "output_text" if role_for_types == "assistant" else "input_text"
                return [{"type": tname, "text": content}]
            items: List[Dict[str, Any]] = []
            if isinstance(content, list):
                for it in content:
                    try:
                        t = it.get("type")
                    except Exception:
                        t = None
                    if role_for_types == "assistant":
                        # Assistant content must be output_* per Responses types
                        if t in ("output_text",):
                            items.append(it)
                        elif t in ("input_text", "text"):
                            items.append({"type": "output_text", "text": it.get("text", "")})
                        else:
                            # Fallback: stringify as output_text
                            items.append({"type": "output_text", "text": str(it)})
                        continue

                    # User/system/developer inputs
                    if t in ("input_text", "input_image", "input_file"):
                        items.append(it)
                    elif t == "text":
                        items.append({"type": "input_text", "text": it.get("text", "")})
                    elif t == "image_url":
                        url = it.get("image_url")
                        if isinstance(url, dict):
                            url = url.get("url")
                        if url:
                            items.append({"type": "input_image", "image_url": url, "detail": "auto"})
                    elif t == "image":
                        # If any generic image object with a URL sneaks through, map to input_image
                        src = it.get("source") if isinstance(it, dict) else None
                        url = None
                        if isinstance(src, dict):
                            url = src.get("url")
                        if url:
                            items.append({"type": "input_image", "image_url": url, "detail": "auto"})
                    else:
                        # Fallback to text
                        items.append({"type": "input_text", "text": str(it)})
                return items
            # Unknown type -> coerce to input_text
            return [{"type": "input_text", "text": str(content)}]

        for m in messages:
            role = (m.get("role") or "user").lower()
            content = m.get("content")
            # First system message becomes instructions
            if role == "system" and instructions is None:
                instructions = OpenAIChatTextGenerator._content_to_text(content)
                continue
            # Convert to Responses input content list
            content_list = _to_resp_content_list(content, role)
            msg: Dict[str, Any] = {"role": role, "content": content_list, "type": "message"}
            out.append(msg)

        if not out:
            # Defensive: ensure at least one message exists
            out = [{"role": "user", "type": "message", "content": ""}]

        return instructions, out

    async def generate(
        self,
        prompt: Union[str, Sequence[_Message]],
        *,
        temperature: float = 1.0,
    ) -> str:
        # Normalize input into a list of messages for unified handling
        if isinstance(prompt, str):
            messages: List[_Message] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, Sequence):
            if not all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
                raise TypeError("Each message must be a dict with 'role' and 'content' keys")
            messages = list(prompt)  # type: ignore[arg-type]
        else:
            raise TypeError("prompt must be a string or a sequence of message dicts")

        client = self._get_client()

        # gpt-5 path: use Responses API with reasoning={effort: low}; preserve roles and images
        if self._is_gpt5():
            instructions, input_messages = self._to_responses_easy_input(messages)
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": input_messages,
                "reasoning": {"effort": "medium"},
            }
            if instructions:
                kwargs["instructions"] = instructions

            try:
                _LOG.debug(
                    "OpenAI Responses: input_messages=%d (first roles: %s; first content types: %s)",
                    len(input_messages),
                    ", ".join(m.get("role", "?") for m in input_messages[:3]),
                    ", ".join(
                        (c.get("type", "?")) for c in (input_messages[0].get("content", []) or [])
                    ) if input_messages else "",
                )
            except Exception:
                pass

            resp = await client.responses.create(**kwargs)

            # Extract text similar to our SVG generator helper
            text = getattr(resp, "output_text", None)
            if not text:
                parts: List[str] = []
                output = getattr(resp, "output", None)
                if output:
                    for item in output:
                        for c in getattr(item, "content", []) or []:
                            tt = getattr(c, "text", None)
                            if tt:
                                parts.append(tt)
                text = "\n".join(parts)
            return (text or "").strip()

        # Fallback for non-gpt-5 models: Chat Completions with temperature
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,      # type: ignore[arg-type]
            temperature=temperature,
        )
        choice = resp.choices[0]
        return (choice.message.content or "").strip()
