"""
End‑to‑end and unit tests for `cogs.image_gen_cog.ImageGenCog`.

The suite is deliberately split between:

* **Behaviour‑level (black‑box)** checks that exercise the cog’s public
  contract (`_handle_generation` acts as our “service boundary” in lieu
  of a full Discord dispatch).
* **Unit‑level (white‑box)** checks for tricky helper behaviour such as
  filename building.  These remain private helpers but are isolated and
  parametrised for clarity.

Keeping them together makes it easy for reviewers to trace intent while
still giving refactors confidence.
"""

import base64
import sys
from pathlib import Path
from typing import Optional

import pytest
from unittest.mock import AsyncMock, MagicMock

from cogs.image_gen_cog import (
    FAILURE,
    SUCCESS,
    THINKING,
    GeneratorSpec,
    ImageGenCog,
)

# --------------------------------------------------------------------------- #
# Test constants
# --------------------------------------------------------------------------- #
PROMPT_SIMPLE = "!hello"
PROMPT_COMPLEX = "{7}hi <x,y>"
PROMPT_INVALID_CHARS = "my:prompt/with*bad|chars"

SEED = 7
MSG_ID = 1
USER_ID = 42

# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def ensure_project_root():
    """
    Keep the original path hack for now so the tests can resolve the package
    no matter where they’re executed from.  When the project is fully
    packaged this fixture can be removed.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def cog():
    """Return a freshly initialised `ImageGenCog` for each test."""
    cfg = {"!": {"api": "replicate", "model": "test"}}
    return ImageGenCog(bot=None, config=cfg)


@pytest.fixture()
def make_message():
    """
    Factory for a realistic Discord‐like message object with async helpers
    already mocked.
    """

    def _factory(content: str, *, author_id: int = USER_ID):
        msg = MagicMock()
        msg.content = content
        msg.author.bot = False
        msg.author.id = author_id
        msg.guild = object()
        msg.channel = MagicMock()
        msg.channel.send = AsyncMock()
        msg.add_reaction = AsyncMock()
        msg.clear_reaction = AsyncMock()
        msg.id = MSG_ID
        return msg

    return _factory


# --------------------------------------------------------------------------- #
# Private helper tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected_body", "prefix"),
    [
        (PROMPT_SIMPLE, "hello", "!"),
        ("something else", None, None),
    ],
)
def test_split_prompt(cog, raw: str, expected_body: Optional[str], prefix: Optional[str]):
    """
    `_split_prompt` should return `(stripped_prompt, GeneratorSpec)` for valid
    prefixes and `None` for non‐matching inputs.
    """
    result = cog._split_prompt(raw)
    if expected_body is None:
        assert result is None
    else:
        spec = cog._prefix_map[prefix]
        assert result == (expected_body, spec)


def test_split_prompt_longest_prefix():
    """Longest prefix wins when multiple potential prefixes match."""
    cfg = {"!": {"api": "a", "model": "x"}, "!!": {"api": "b", "model": "y"}}
    c = ImageGenCog(bot=None, config=cfg)
    spec = c._prefix_map["!!"]
    assert c._split_prompt("!!hi") == ("hi", spec)


def test_build_filename_basic(cog):
    """Filename contains slugified prompt, api/model, and seed."""
    spec = GeneratorSpec(api="my/api", model="ModelA")
    assert (
        cog._build_filename("Hello world!", spec, SEED)
        == "hello_world_my-api-modela_seed_7.png"
    )


def test_build_filename_truncation_and_random_seed(cog):
    """Long prompt gets truncated, seed defaults to 'rand'."""
    spec = GeneratorSpec(api="my/api", model="ModelA")
    long_name = cog._build_filename("a" * 40, spec, None)
    assert long_name.startswith("a" * 32)
    assert long_name.endswith("_my-api-modela_seed_rand.png")


def test_build_filename_sanitises_invalid_chars(cog):
    """Invalid filesystem characters are removed or replaced."""
    spec = GeneratorSpec(api="api", model="m")
    name = cog._build_filename(PROMPT_INVALID_CHARS, spec, SEED)
    assert ":" not in name and "/" not in name and "*" not in name and "|" not in name


# --------------------------------------------------------------------------- #
# Behaviour / async flow tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handle_generation_success(monkeypatch, cog, make_message):
    """
    Successful generation:
    * THINKING reaction added then cleared
    * Two images returned (one per coordinate)
    * SUCCESS emoji reacts at the end
    """
    spec = GeneratorSpec(api="replicate", model="test-model")

    async def fake_run(pid, api, model, prompt, seed, *, poll_interval=0.5):
        # Encode something unique so we can later inspect the payload if needed
        return base64.b64encode(f"img-{prompt}".encode()).decode()

    cog._run_task = fake_run
    monkeypatch.setattr("cogs.image_gen_cog.add_prompt", lambda *a, **k: 1)
    monkeypatch.setattr("cogs.image_gen_cog.add_caption", lambda b, t: b)

    msg = make_message(PROMPT_COMPLEX)

    await cog._handle_generation(msg, PROMPT_COMPLEX, spec)

    # Two sends: x and y coordinate images
    assert msg.channel.send.call_count == 2
    filenames = [c.kwargs["file"].filename for c in msg.channel.send.call_args_list]
    assert filenames == [
        "hi_x_replicate-test-model_seed_7.png",
        "hi_y_replicate-test-model_seed_7.png",
    ]

    msg.add_reaction.assert_any_call(THINKING)
    msg.clear_reaction.assert_called_with(THINKING)
    msg.add_reaction.assert_any_call(SUCCESS)


@pytest.mark.asyncio
async def test_handle_generation_failure(monkeypatch, cog, make_message):
    """
    If `_run_task` raises an exception:
    * THINKING is cleared
    * FAILURE emoji is added
    * No files are sent
    """

    async def fake_run(*_a, **_kw):
        raise RuntimeError("boom")

    # Monkeypatch the task runner to fail
    cog._run_task = fake_run
    monkeypatch.setattr("cogs.image_gen_cog.add_prompt", lambda *a, **k: 1)

    msg = make_message(PROMPT_SIMPLE)

    spec = cog._prefix_map["!"]
    await cog._handle_generation(msg, PROMPT_SIMPLE, spec)

    msg.clear_reaction.assert_called_with(THINKING)
    msg.add_reaction.assert_any_call(FAILURE)
    msg.channel.send.assert_not_called()
