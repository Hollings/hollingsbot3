"""Tests for the image cog's small Claude helpers (thematic emoji, aspect ratio).

They run through a real ``AsyncAnthropic`` client against the ``fake_anthropic``
fixture, so an SDK signature change breaks them the way it breaks production:
under anthropic 1.x the ``temperature=`` keyword made every one of these calls
raise ``TypeError``, which the helpers swallowed into their silent fallbacks.
"""

from __future__ import annotations

import pytest

from hollingsbot.cogs.image_gen_cog import THINKING, ImageGenCog


class _BareCog(ImageGenCog):
    """ImageGenCog without __init__ (which needs a bot + DB); the helpers use neither."""

    def __init__(self) -> None:
        pass


class TestThematicEmoji:
    @pytest.mark.asyncio
    async def test_shortcode_becomes_the_emoji(self, fake_anthropic):
        fake_anthropic.reply = ":fire:"

        assert await _BareCog()._get_thematic_emoji("a house on fire") == "\N{FIRE}"

        body = fake_anthropic.requests[-1]
        assert body["model"] == "claude-haiku-4-5"
        assert body["temperature"] == 0.7
        assert "a house on fire" in body["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_bare_name_is_normalised(self, fake_anthropic):
        fake_anthropic.reply = "art"

        assert await _BareCog()._get_thematic_emoji("a painting") == "\N{ARTIST PALETTE}"

    @pytest.mark.asyncio
    async def test_unknown_shortcode_falls_back(self, fake_anthropic):
        fake_anthropic.reply = ":definitely_not_an_emoji:"

        assert await _BareCog()._get_thematic_emoji("anything") == THINKING
        assert len(fake_anthropic.requests) == 1  # fell back on the reply, not a failed call

    @pytest.mark.asyncio
    async def test_api_failure_falls_back(self, fake_anthropic):
        fake_anthropic.status = 500

        assert await _BareCog()._get_thematic_emoji("anything") == THINKING
        assert len(fake_anthropic.requests) == 1  # it did reach the API


class TestAspectRatio:
    @pytest.mark.asyncio
    async def test_picked_ratio_is_used(self, fake_anthropic):
        fake_anthropic.reply = "16:9"

        assert await _BareCog()._get_aspect_ratio_for_prompt("an epic mountain vista") == "16:9"

        body = fake_anthropic.requests[-1]
        assert body["model"] == "claude-haiku-4-5"
        assert "temperature" not in body

    @pytest.mark.asyncio
    async def test_api_failure_defaults_to_landscape(self, fake_anthropic):
        fake_anthropic.status = 500

        assert await _BareCog()._get_aspect_ratio_for_prompt("anything") == "3:2"
        assert len(fake_anthropic.requests) == 1  # it did reach the API
