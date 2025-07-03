import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock

from cogs.image_gen_cog import ImageGenCog, GeneratorSpec, THINKING, SUCCESS, FAILURE


@pytest.fixture()
def cog():
    cfg = {"!": {"api": "replicate", "model": "test"}}
    return ImageGenCog(bot=None, config=cfg)


def _make_message():
    msg = MagicMock()
    msg.author.bot = False
    msg.author.id = 42
    msg.guild = object()
    msg.channel = MagicMock()
    msg.channel.send = AsyncMock()
    msg.add_reaction = AsyncMock()
    msg.clear_reaction = AsyncMock()
    msg.id = 1
    return msg


def test_split_prompt_simple(cog):
    spec = cog._prefix_map["!"]
    assert cog._split_prompt("!hello") == ("hello", spec)
    assert cog._split_prompt("something else") is None


def test_split_prompt_longest_prefix():
    cfg = {"!": {"api": "a", "model": "x"}, "!!": {"api": "b", "model": "y"}}
    c = ImageGenCog(bot=None, config=cfg)
    spec = c._prefix_map["!!"]
    assert c._split_prompt("!!hi") == ("hi", spec)


def test_build_filename(cog):
    spec = GeneratorSpec(api="my/api", model="modelA")
    name = cog._build_filename("Hello world!", spec, 5)
    assert name == "hello_world_my-api-modela_seed_5.png"

    long_name = cog._build_filename("a" * 40, spec, None)
    assert long_name.startswith("a" * 32)
    assert long_name.endswith("_my-api-modela_seed_rand.png")


@pytest.mark.asyncio
async def test_handle_generation_success(monkeypatch):
    spec = GeneratorSpec(api="replicate", model="test-model")
    cog = ImageGenCog(bot=None, config={"!": {"api": "replicate", "model": "test"}})

    async def fake_run(pid, api, model, prompt, seed, *, poll_interval=0.5):
        return base64.b64encode(f"img-{prompt}".encode()).decode()

    cog._run_task = fake_run
    monkeypatch.setattr("cogs.image_gen_cog.add_prompt", lambda *a, **k: 1)
    monkeypatch.setattr("cogs.image_gen_cog.add_caption", lambda b, t: b)

    msg = _make_message()

    await cog._handle_generation(msg, "{7}hi <x,y>", spec)

    assert msg.channel.send.call_count == 2
    filenames = [c.kwargs["file"].filename for c in msg.channel.send.call_args_list]
    assert filenames == [
        "hi_x_replicate-test-model_seed_7.png",
        "hi_y_replicate-test-model_seed_7.png",
    ]

    msg.add_reaction.assert_any_call(THINKING)
    msg.clear_reaction.assert_called_with(THINKING)
    msg.add_reaction.assert_any_call(SUCCESS)
