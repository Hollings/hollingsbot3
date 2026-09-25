"""Tests for Anthropic text generator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hollingsbot.text_generators.anthropic import AnthropicTextGenerator, supports_temperature


@pytest.fixture
def generator():
    """Create an Anthropic generator instance."""
    return AnthropicTextGenerator(model="claude-3-sonnet-20240229")


@pytest.fixture
def mock_client():
    """Create a mock AsyncAnthropic client."""
    mock = MagicMock()
    mock.messages = MagicMock()
    return mock


class TestAnthropicTextGeneratorInit:
    """Tests for initialization."""

    def test_default_model(self):
        """Test default model is set."""
        gen = AnthropicTextGenerator()
        assert gen.model == "claude-4o"

    def test_custom_model(self):
        """Test custom model can be set."""
        gen = AnthropicTextGenerator(model="claude-3-opus-20240229")
        assert gen.model == "claude-3-opus-20240229"


class TestGenerate:
    """Tests for the generate method."""

    @pytest.mark.asyncio
    async def test_generate_with_string_prompt(self, generator, mock_client):
        """Test generation with a simple string prompt."""
        # Mock response
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Hello, world!"
        mock_response.content = [mock_block]

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.object(generator, "_get_client", return_value=mock_client):
            result = await generator.generate("Hello")

        assert result == "Hello, world!"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == generator.model
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_with_message_list(self, generator, mock_client):
        """Test generation with a list of messages."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Response"
        mock_response.content = [mock_block]

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        with patch.object(generator, "_get_client", return_value=mock_client):
            result = await generator.generate(messages)

        assert result == "Response"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert len(call_kwargs["messages"]) == 3

    @pytest.mark.asyncio
    async def test_generate_extracts_system_messages(self, generator, mock_client):
        """Test that system messages are extracted to top-level parameter."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Response"
        mock_response.content = [mock_block]

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        with patch.object(generator, "_get_client", return_value=mock_client):
            await generator.generate(messages)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        # System should be a top-level parameter, not in messages
        assert "system" in call_kwargs
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_with_temperature(self, generator, mock_client):
        """A temperature for a model that accepts one rides in extra_body.

        anthropic 1.x dropped ``temperature`` from ``messages.create()``'s
        signature, so it must never be passed as a keyword of its own.
        """
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Response"
        mock_response.content = [mock_block]

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.object(generator, "_get_client", return_value=mock_client):
            await generator.generate("Hello", temperature=0.5)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert call_kwargs["extra_body"] == {"temperature": 0.5}

    @pytest.mark.asyncio
    async def test_generate_without_temperature_sends_none(self, generator, mock_client):
        """No temperature argument means no sampling parameter in the request."""
        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.object(generator, "_get_client", return_value=mock_client):
            await generator.generate("Hello")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "extra_body" not in call_kwargs

    @pytest.mark.asyncio
    async def test_temperature_dropped_for_models_that_reject_it(self, mock_client):
        """Models that 400 on sampling parameters are sent no temperature."""
        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        gen = AnthropicTextGenerator(model="claude-sonnet-5")

        with patch.object(gen, "_get_client", return_value=mock_client):
            await gen.generate("Hello", temperature=0.7)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "extra_body" not in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_invalid_prompt_type_raises(self, generator):
        """Test that invalid prompt type raises TypeError."""
        with pytest.raises(TypeError, match="must be either a string or a sequence"):
            await generator.generate(12345)

    @pytest.mark.asyncio
    async def test_generate_invalid_message_format_raises(self, generator):
        """Test that invalid message format raises TypeError."""
        messages = [{"not_role": "user"}]  # Missing 'role' key
        with pytest.raises(TypeError, match="must be a dict with 'role' and 'content'"):
            await generator.generate(messages)

    @pytest.mark.asyncio
    async def test_generate_handles_multiple_content_blocks(self, generator, mock_client):
        """Test generation with multiple content blocks in response."""
        mock_response = MagicMock()
        mock_block1 = MagicMock()
        mock_block1.text = "Part 1 "
        mock_block2 = MagicMock()
        mock_block2.text = "Part 2"
        mock_response.content = [mock_block1, mock_block2]

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.object(generator, "_get_client", return_value=mock_client):
            result = await generator.generate("Hello")

        assert result == "Part 1 Part 2"

    @pytest.mark.asyncio
    async def test_generate_handles_empty_response(self, generator, mock_client):
        """Test generation with empty response."""
        mock_response = MagicMock()
        mock_response.content = []

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.object(generator, "_get_client", return_value=mock_client):
            result = await generator.generate("Hello")

        assert result == ""


class TestSupportsTemperature:
    """Which model ids may be sent a temperature."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-5",
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-20250514",
            "claude-opus-4-0",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ],
    )
    def test_accepting_models(self, model):
        assert supports_temperature(model)

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-opus-5-5",
            "claude-sonnet-5",
            "claude-fable-5-1",
            "claude-haiku-5",  # unknown future models default to "no"
            "claude-4o",
        ],
    )
    def test_rejecting_models(self, model):
        assert not supports_temperature(model)


class TestRealSDK:
    """Requests built by the installed anthropic SDK, not a mock.

    These are the regression net for the 1.x upgrade: ``temperature=`` as a
    ``messages.create()`` keyword raised ``TypeError`` on every call in prod
    while the ``AsyncMock`` tests above kept passing.
    """

    @pytest.mark.asyncio
    async def test_temperature_reaches_the_request_body(self, fake_anthropic):
        fake_anthropic.reply = "hi"
        gen = AnthropicTextGenerator(model="claude-haiku-4-5")

        assert await gen.generate("Hello", temperature=0.7) == "hi"

        body = fake_anthropic.requests[-1]
        assert body["model"] == "claude-haiku-4-5"
        assert body["temperature"] == 0.7
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_default_request_has_no_temperature(self, fake_anthropic):
        gen = AnthropicTextGenerator(model="claude-haiku-4-5")

        await gen.generate("Hello")

        assert "temperature" not in fake_anthropic.requests[-1]

    @pytest.mark.asyncio
    async def test_rejecting_model_request_has_no_temperature(self, fake_anthropic):
        gen = AnthropicTextGenerator(model="claude-sonnet-5")

        await gen.generate("Hello", temperature=0.7)

        assert "temperature" not in fake_anthropic.requests[-1]

    @pytest.mark.asyncio
    async def test_system_prompt_and_caching_survive_serialisation(self, fake_anthropic):
        gen = AnthropicTextGenerator(model="claude-haiku-4-5")

        await gen.generate([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}])

        body = fake_anthropic.requests[-1]
        assert body["system"] == [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}]
        assert [m["role"] for m in body["messages"]] == ["user"]


class TestErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_propagates(self, generator, mock_client):
        """Test that rate limit errors are logged and re-raised."""
        from anthropic import RateLimitError

        mock_client.messages.create = AsyncMock(
            side_effect=RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
        )

        with patch.object(generator, "_get_client", return_value=mock_client), pytest.raises(RateLimitError):
            await generator.generate("Hello")

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self, generator, mock_client):
        """Test that connection errors are logged and re-raised."""
        from anthropic import APIConnectionError

        # APIConnectionError requires a request object
        mock_request = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=APIConnectionError(message="Connection failed", request=mock_request)
        )

        with patch.object(generator, "_get_client", return_value=mock_client):
            with pytest.raises(APIConnectionError):
                await generator.generate("Hello")

    @pytest.mark.asyncio
    async def test_api_error_propagates(self, generator, mock_client):
        """Test that API errors are logged and re-raised."""
        from anthropic import APIStatusError

        # Create a proper API error with required params
        mock_response = MagicMock()
        mock_response.status_code = 500

        error = APIStatusError(
            message="Internal error",
            response=mock_response,
            body={"error": {"message": "Internal error"}},
        )

        mock_client.messages.create = AsyncMock(side_effect=error)

        with patch.object(generator, "_get_client", return_value=mock_client), pytest.raises(APIStatusError):
            await generator.generate("Hello")


class TestCaching:
    """Tests for prompt caching behavior."""

    @pytest.mark.asyncio
    async def test_cacheable_message_adds_cache_control(self, generator, mock_client):
        """Test that cacheable messages get cache_control added."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Response"
        mock_response.content = [mock_block]

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "Cached content", "_cacheable": True},
        ]

        with patch.object(generator, "_get_client", return_value=mock_client):
            await generator.generate(messages)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        msg = call_kwargs["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0].get("cache_control") is not None

    @pytest.mark.asyncio
    async def test_system_prompt_has_cache_control(self, generator, mock_client):
        """Test that system prompts have cache_control."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Response"
        mock_response.content = [mock_block]

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]

        with patch.object(generator, "_get_client", return_value=mock_client):
            await generator.generate(messages)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert system[0].get("cache_control") == {"type": "ephemeral"}
