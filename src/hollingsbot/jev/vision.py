"""Choose a text reply or locate one requested region in the prepared image."""

from __future__ import annotations

import base64
import io
import math
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from hollingsbot.jev.client import JevError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hollingsbot.jev.client import DecisionsClient
    from hollingsbot.jev.writer import ChatLine, Reply


def _probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value) if math.isfinite(value) and 0 <= value <= 1 else 0.0


def _percentage(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not 0 <= value <= 1:
        return "unavailable"
    return f"{value:.0%}"


def _edges(answer: dict) -> tuple[float, float, float, float] | None:
    box = answer.get("box")
    if not isinstance(box, dict):
        return None
    values = [box.get(key) for key in ("left", "top", "right", "bottom")]
    if any(
        isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values
    ):
        return None
    left, top, right, bottom = values
    return (left, top, right, bottom) if left < right and top < bottom else None


def annotate(data_url: str, edges: tuple[float, float, float, float]) -> bytes:
    """Draw on the exact prepared canvas Jev saw, including its padding."""
    with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as original:
        image = original.convert("RGB")
        left, top, right, bottom = edges
        coords = (
            round(left * (image.width - 1)),
            round(top * (image.height - 1)),
            round(right * (image.width - 1)),
            round(bottom * (image.height - 1)),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(coords, outline="white", width=6)
        draw.rectangle(coords, outline="red", width=3)
        output = io.BytesIO()
        image.save(output, "PNG")
        return output.getvalue()


async def image_response(
    client: DecisionsClient, chat: Sequence[ChatLine], draft: Reply
) -> tuple[str, bytes | None] | None:
    """None means continue word selection; a tuple is a completed visual reply.

    Route only image-bearing chats. Require a clear localization intent, then
    report presence alongside Box because Box always returns coordinates.
    Both calls share the writer's usage accounting and cancellation lifecycle.
    """
    image = next((line.images[0] for line in reversed(chat) if line.images), None)
    if image is None:
        return None
    draft.model = client.model
    state = {
        "chat": [{"from": line.speaker, "text": line.text} for line in chat],
        "image": {"type": "image_url", "image_url": {"url": image}},
    }
    route = await client.ask(
        state,
        {
            "response": {
                "type": "choice",
                "instructions": (
                    "Which response best satisfies the final post in `chat`? Earlier posts are background. "
                    "Choose box only when the user asks to locate, point to, mark, highlight, or draw a box "
                    "around a particular visible object or region in `image`. For descriptions, identification, "
                    "reading text, general questions, or ambiguous requests, choose text."
                ),
                "criteria": {"text": "A normal text reply", "box": "The image with one requested region boxed"},
            }
        },
        draft.usage,
    )
    answer = route.get("response", {})
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise JevError("image route answer without probabilities")
    if _probability(probabilities.get("box")) < 0.7:
        return None
    answers = await client.ask(
        state,
        {
            "target": {
                "type": "box",
                "instructions": "the object or region in `image` that the final post in `chat` asks to locate",
            },
            "present": {
                "type": "noul",
                "instructions": "Is the particular object or region requested by the final post in `chat` visible in `image`?",
            },
        },
        draft.usage,
    )
    target = answers.get("target", {})
    edges = _edges(target)
    if edges is None:
        draft.text = "Jev returned invalid box coordinates, so I couldn't draw the box."
        draft.stop_reason = "box_invalid"
        return draft.text, None
    confidence = _percentage(target.get("confidence"))
    presence = _percentage(answers.get("present", {}).get("noul"))
    draft.text = f"Box confidence: {confidence}. Target presence: {presence}."
    draft.stop_reason = "box"
    return draft.text, annotate(image, edges)
