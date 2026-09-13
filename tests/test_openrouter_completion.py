"""Tests for the OpenRouter raw-completion generator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hollingsbot.text_generators import get_text_generator
from hollingsbot.text_generators.openrouter_completion import (
    DEFAULT_COMPLETION_MODEL,
    OpenRouterCompletionGenerator,
)


@pytest.fixture
def generator():
    return OpenRouterCompletionGenerator(model="nousresearch/hermes-4-405b")


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.completions = MagicMock()
    return client


def _response(text: str | None, finish_reason: str = "length") -> MagicMock:
    resp = MagicMock()
    choice = MagicMock()
    choice.text = text
    choice.finish_reason = finish_reason
    resp.choices = [choice]
    resp.provider = "Nebius"
    return resp


class TestRegistry:
    def test_factory_returns_completion_generator(self):
        gen = get_text_generator("openrouter-completion", "nousresearch/hermes-4-405b")
        assert isinstance(gen, OpenRouterCompletionGenerator)
        assert gen.model == "nousresearch/hermes-4-405b"

    def test_default_model(self):
        assert OpenRouterCompletionGenerator().model == DEFAULT_COMPLETION_MODEL


class TestGenerate:
    @pytest.mark.asyncio
    async def test_sends_raw_prompt_to_completions_endpoint(self, generator, mock_client):
        mock_client.completions.create = AsyncMock(return_value=_response(" nine-man is a nonagon\n"))
        prompt = "a three-man group is called a guyangle\na"

        with patch.object(generator, "_get_client", return_value=mock_client):
            out = await generator.generate(prompt, temperature=0.9)

        kwargs = mock_client.completions.create.call_args.kwargs
        assert kwargs["model"] == "nousresearch/hermes-4-405b"
        assert kwargs["prompt"] == prompt  # no system prompt, no role wrapping
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 500
        # Leading whitespace is part of the continuation; only the tail is stripped.
        assert out == " nine-man is a nonagon"

    @pytest.mark.asyncio
    async def test_max_tokens_override(self, mock_client):
        generator = OpenRouterCompletionGenerator(max_tokens=123)
        mock_client.completions.create = AsyncMock(return_value=_response("x"))
        with patch.object(generator, "_get_client", return_value=mock_client):
            await generator.generate("p")
            assert mock_client.completions.create.call_args.kwargs["max_tokens"] == 123
            await generator.generate("p", max_tokens=7)
            assert mock_client.completions.create.call_args.kwargs["max_tokens"] == 7

    @pytest.mark.asyncio
    async def test_empty_choice_text_returns_empty_string(self, generator, mock_client):
        mock_client.completions.create = AsyncMock(return_value=_response(None))
        with patch.object(generator, "_get_client", return_value=mock_client):
            assert await generator.generate("p") == ""

    @pytest.mark.asyncio
    async def test_rejects_message_lists(self, generator):
        with pytest.raises(TypeError):
            await generator.generate([{"role": "user", "content": "hi"}])  # type: ignore[arg-type]
