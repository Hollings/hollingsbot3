"""Image preparation, bounded history and actual Decisions wire serialization."""

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from hollingsbot.cogs.chat_bots.jev_bot import chat_lines
from hollingsbot.cogs.conversation import ConversationTurn, ImageAttachment
from hollingsbot.jev.client import DecisionsClient
from hollingsbot.jev.images import prepare_image
from hollingsbot.jev.pieces import PieceWriter
from hollingsbot.jev.writer import ChatLine, JevWriter, WriterConfig, _chat_json


def data_url(size=(20, 100), fmt="PNG", color="red"):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(output.getvalue()).decode()


@pytest.mark.parametrize("size", [(20, 100), (2000, 100), (1000, 1000)])
def test_prepares_small_and_large_images_on_valid_canvas(size):
    prepared = prepare_image(data_url(size))
    assert prepared.startswith("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(prepared.split(",", 1)[1]))) as image:
        assert image.size == (512, 512)
        assert image.mode == "RGB"
        assert image.getpixel((256, 256))[0] > 240
        if size[0] != size[1]:
            assert image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize(
    "url", [None, "https://example.com/a.png", "data:image/png;base64,!", "data:image/png;base64,YQ=="]
)
def test_unusable_images_are_skipped(url):
    assert prepare_image(url) is None


def test_only_newest_usable_image_in_selected_history_and_can_disable():
    def turn(color):
        return ConversationTurn(
            "user", color, images=[ImageAttachment("image", "", data_url(color=color))], author_name="H"
        )

    history = [turn("red"), turn("blue"), ConversationTurn("user", "what is this?")]
    lines = chat_lines(history, 3)
    assert not lines[0].images and len(lines[1].images) == 1 and not lines[2].images
    assert lines[0].text == "red [image]"
    assert not any(line.images for line in chat_lines(history, 3, include_images=False))
    assert not any(line.images for line in chat_lines(history, 1))


async def test_image_is_nested_in_chat_on_decisions_wire():
    image = prepare_image(data_url())
    turn = ConversationTurn("user", "look", images=[ImageAttachment("x", "", image)], author_name="H")
    state = {"chat": _chat_json(chat_lines([turn], 1))}

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "typesafe/jev-1.14"
        assert body["state"]["chat"][0]["images"][0]["type"] == "image_url"
        assert body["state"]["chat"][0]["images"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return httpx.Response(200, json={"answers": {"q": {"noul": 1}}})

    client = DecisionsClient("test", transport=httpx.MockTransport(handler))
    try:
        assert await client.ask(state, {"q": {"type": "noul", "instructions": "Is it red?"}}) == {"q": {"noul": 1}}
    finally:
        await client.aclose()


@pytest.mark.parametrize("units", ["words", "pieces"])
async def test_image_spend_stops_both_writers_after_current_word(units):
    class CostlyClient:
        model = "test"
        calls = 0

        async def ask(self, state, questions, usage):
            self.calls += 1
            assert state["chat"][0]["images"][0]["image_url"]["url"] == "data:image/jpeg;base64,test"
            usage.record({"cost": 0.02}, 0)
            return {
                key: {"probabilities": {option: float(i == 0) for i, option in enumerate(q["criteria"])}}
                if q["type"] == "choice"
                else {"noul": 1.0}
                for key, q in questions.items()
            }

    class Suggester:
        async def top_tokens(self, *args, **kwargs):
            return [("red", 1.0)]

    client = CostlyClient()
    cfg = WriterConfig(units=units, min_words=8, max_words=10, vocab_size=2, fluency_check=False)
    if units == "pieces":
        writer = PieceWriter(client, config=cfg, suggester=Suggester())
    else:
        writer = JevWriter(client, config=cfg, vocab=["red", "blue"])
    reply = await writer.write([ChatLine("H", "look", ("data:image/jpeg;base64,test",))])
    assert client.calls == 1
    assert len(reply.words) == 1
    assert reply.stop_reason == "image_budget"
