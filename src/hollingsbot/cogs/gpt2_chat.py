from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import TYPE_CHECKING

from celery.exceptions import TimeoutError as CeleryTimeoutError
from discord.ext import commands

from hollingsbot.tasks import generate_text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import discord

logger = logging.getLogger(__name__)


class GPT2Chat(commands.Cog):
    """Respond to messages in a designated channel using GPT-2 (via Celery).

    If multiple messages arrive faster than the model can reply, only the most
    recent prompt per channel will be answered. Older, still-running generation
    tasks are allowed to finish but their results are discarded.
    """

    # Constants
    _MAX_DISCORD_LEN: int = 2_000
    _IGNORED_KEYWORD: str = "enhance"
    _DEFAULT_TIMEOUT: int = 180
    _TEMP_MIN: float = 0.5
    _TEMP_MAX: float = 1.5
    _TEMP_MODE: float = 1.0

    def __init__(
        self,
        bot: commands.Bot,
        *,
        channel_id: int | None = None,
        api: str = "huggingface",
        model: str = "gpt2-medium",
        task_func: Callable[[str, str, str, float], Awaitable[str]] | None = None,
        timeout: int | None = None,
    ) -> None:
        """Initialize the GPT2Chat cog.

        Args:
            bot: The Discord bot instance.
            channel_id: Optional channel ID to restrict responses to. If None,
                reads from GPT2_CHANNEL_ID environment variable.
            api: Text generation API to use (default: "huggingface").
            model: Model name to use for generation (default: "gpt2-medium").
            task_func: Optional custom task function for dependency injection.
                Primarily used for testing.
            timeout: Maximum time in seconds to wait for generation. If None,
                reads from GPT2_RESPONSE_TIMEOUT environment variable.
        """
        self.bot = bot

        # Channel configuration
        self.channel_id = self._resolve_channel_id(channel_id)

        # Generation configuration
        self.api = api
        self.model = model
        self.task_func = task_func or self._celery_task
        self.timeout = self._resolve_timeout(timeout)

        # Track the newest message ID per channel to ignore stale responses
        self._latest: dict[int, int] = {}

        logger.info(
            "GPT2Chat initialized: channel_id=%s, api=%s, model=%s, timeout=%ds",
            self.channel_id,
            self.api,
            self.model,
            self.timeout,
        )

    # Configuration helpers

    def _resolve_channel_id(self, channel_id: int | None) -> int | None:
        """Resolve channel ID from parameter or environment variable.

        Args:
            channel_id: Explicit channel ID or None.

        Returns:
            Resolved channel ID or None if not configured.
        """
        if channel_id is not None:
            return channel_id

        env_channel_id = os.getenv("GPT2_CHANNEL_ID")
        return int(env_channel_id) if env_channel_id else None

    def _resolve_timeout(self, timeout: int | None) -> int:
        """Resolve timeout from parameter or environment variable.

        Args:
            timeout: Explicit timeout value or None.

        Returns:
            Resolved timeout in seconds.
        """
        if timeout is not None:
            return timeout

        env_timeout = os.getenv("GPT2_RESPONSE_TIMEOUT", str(self._DEFAULT_TIMEOUT))
        return int(env_timeout)

    # Message filtering

    def _is_ignored_keyword(self, message: discord.Message) -> bool:
        """Check if message contains an ignored keyword.

        Args:
            message: Discord message to check.

        Returns:
            True if message should be ignored due to keyword.
        """
        return message.content.lower() == self._IGNORED_KEYWORD

    def _is_bot_message(self, message: discord.Message) -> bool:
        """Check if message was sent by a bot.

        Args:
            message: Discord message to check.

        Returns:
            True if message author is a bot.
        """
        return message.author.bot

    def _is_target_channel(self, message: discord.Message) -> bool:
        """Check if message is in the target channel.

        Args:
            message: Discord message to check.

        Returns:
            True if message is in target channel or if no channel restriction exists.
        """
        if self.channel_id is None:
            return True

        return getattr(message.channel, "id", None) == self.channel_id

    def _should_respond(self, message: discord.Message) -> bool:
        """Determine if the bot should respond to a message.

        Args:
            message: Discord message to evaluate.

        Returns:
            True if all response criteria are met.
        """
        if self._is_ignored_keyword(message):
            return False
        if self._is_bot_message(message):
            return False
        return self._is_target_channel(message)

    # Text generation

    async def _celery_task(self, api: str, model: str, prompt: str, temperature: float) -> str:
        """Execute text generation task via Celery.

        Args:
            api: Text generation API name.
            model: Model identifier.
            prompt: Input text prompt.
            temperature: Sampling temperature.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If task exceeds timeout.
        """
        task = generate_text.apply_async((api, model, prompt, temperature), queue="text")
        try:
            # Run the potentially blocking task.get call in a thread
            return await asyncio.to_thread(task.get, timeout=self.timeout)
        except CeleryTimeoutError as exc:
            error_msg = f"Model did not respond within {self.timeout}s. Please try again."
            logger.warning(
                "Celery task timeout for prompt: %s (timeout=%ds)",
                prompt[:50],
                self.timeout,
            )
            raise RuntimeError(error_msg) from exc

    def _generate_temperature(self) -> float:
        """Generate a random temperature value for text generation.

        Uses triangular distribution biased toward the mode value.

        Returns:
            Random temperature between TEMP_MIN and TEMP_MAX.
        """
        return random.triangular(self._TEMP_MIN, self._TEMP_MAX, self._TEMP_MODE)

    async def _generate(self, prompt: str) -> str:
        """Generate text response with error handling.

        Args:
            prompt: Input text prompt.

        Returns:
            Generated text or error message if generation fails.
        """
        temperature = self._generate_temperature()

        try:
            response = await self.task_func(self.api, self.model, prompt, temperature)
            logger.debug(
                "Generated response for prompt: %s (temp=%.2f)",
                prompt[:50],
                temperature,
            )
            return response
        except RuntimeError as exc:
            # User-facing timeout or runtime errors
            logger.error("Generation failed: %s", exc)
            return f"⚠️ {exc}"
        except Exception as exc:
            # Unexpected errors
            logger.exception("Unexpected error during text generation: %s", exc)
            return f"⚠️ Error generating response: {exc}"

    # Event listeners

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages and generate responses when appropriate.

        Args:
            message: Discord message that triggered the event.
        """
        if not self._should_respond(message):
            return

        prompt = message.content.strip()
        if not prompt:
            return

        channel_id = getattr(message.channel, "id", None)
        if channel_id is None:
            logger.warning("Message has no channel ID: %s", message.id)
            return

        # Mark this as the newest prompt for this channel
        self._latest[channel_id] = message.id

        logger.info(
            "Generating response for message %s in channel %s",
            message.id,
            channel_id,
        )

        reply = await self._generate(prompt)

        if reply:
            truncated_reply = reply[: self._MAX_DISCORD_LEN]
            await message.channel.send(truncated_reply)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(GPT2Chat(bot))
