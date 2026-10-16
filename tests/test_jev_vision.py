"""Routing, presence gating, normalized coordinates and rendered boxes."""

import base64
import io

import pytest
from PIL import Image

from hollingsbot.jev.client import JevError
from hollingsbot.jev.vision import image_response
from hollingsbot.jev.writer import ChatLine, Reply


def image_url():
    output = io.BytesIO()
    Image.new("RGB", (512, 512), "white").save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class Client:
    model = "typesafe/jev-1.14"

    def __init__(self, route=0.99, present=0.99, confidence=0.9, box=None):
        self.route = route
        self.present = present
        self.confidence = confidence
        self.box = box if box is not None else {"left": 0.25, "top": 0.25, "right": 0.75, "bottom": 0.75}
        self.requests = []

    async def ask(self, state, questions, usage):
        self.requests.append((state, questions))
        usage.record({"cost": 0.00005, "input_tokens": 1200}, 0.1)
        if "response" in questions:
            return {"response": {"probabilities": {"text": 1 - self.route, "box": self.route}}}
        return {"target": {"box": self.box, "confidence": self.confidence}, "present": {"noul": self.present}}


async def test_no_image_skips_router():
    client = Client()
    assert await image_response(client, [ChatLine("H", "hello")], Reply()) is None
    assert not client.requests


@pytest.mark.parametrize("probability", [0.1, 0.5, 0.69])
async def test_text_or_ambiguous_route_continues_normal_writer(probability):
    client = Client(route=probability)
    draft = Reply()
    assert await image_response(client, [ChatLine("H", "describe this", (image_url(),))], draft) is None
    assert len(client.requests) == draft.usage.calls == 1
    assert draft.usage.cost > 0  # router spend carries into the word writer


async def test_box_and_presence_share_request_and_render_on_exact_canvas():
    client, draft = Client(), Reply()
    chat = [ChatLine("H", "box the object", (image_url(),))]
    text, png = await image_response(client, chat, draft)
    assert text == draft.text
    assert draft.stop_reason == "box"
    assert draft.model == client.model
    assert draft.usage.calls == 2
    assert draft.usage.cost == pytest.approx(0.0001)
    state, questions = client.requests[1]
    assert questions["target"]["type"] == "box"
    assert "criteria" not in questions["target"]
    assert questions["present"]["type"] == "noul"
    assert state["image"]["image_url"]["url"] == chat[0].images[0]
    assert "images" not in state["chat"][0]  # image is sent only once
    with Image.open(io.BytesIO(png)) as image:
        assert image.size == (512, 512)
        assert image.getpixel((128, 128)) == (255, 0, 0)
        assert image.getpixel((256, 256)) == (255, 255, 255)


@pytest.mark.parametrize(
    "overrides",
    [
        {"present": 0.1},
        {"confidence": 0.1},
        {"box": {"left": 0.8, "right": 0.2, "top": 0, "bottom": 1}},
        {"box": {"left": -0.1, "right": 0.9, "top": 0, "bottom": 1}},
        {"box": {"left": float("nan"), "right": 1, "top": 0, "bottom": 1}},
        {"box": {}},
    ],
)
async def test_absent_uncertain_or_invalid_target_never_draws_a_box(overrides):
    draft = Reply()
    text, png = await image_response(Client(**overrides), [ChatLine("H", "find cat", (image_url(),))], draft)
    assert png is None and "couldn't confidently locate" in text
    assert draft.stop_reason == "box_not_found"


async def test_api_failure_propagates_with_usage_kept():
    class Failing(Client):
        async def ask(self, state, questions, usage):
            if self.requests:
                raise JevError("unavailable")
            return await super().ask(state, questions, usage)

    draft = Reply()
    with pytest.raises(JevError):
        await image_response(Failing(), [ChatLine("H", "box it", (image_url(),))], draft)
    assert draft.usage.calls == 1
