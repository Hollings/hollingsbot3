"""LlamaBot - Llama 3.1 405B via OpenRouter with webhook support."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import discord

from hollingsbot.cogs import chat_utils
from hollingsbot.tasks import generate_text

if TYPE_CHECKING:
    from discord.ext import commands

_LOG = logging.getLogger(__name__)


class GenerationJob:
    """Tracks active generation state."""

    def __init__(self):
        self.task: asyncio.Task | None = None
        self.result: Any = None


class LlamaBot:
    """Llama 3.1 405B via OpenRouter with webhook support."""

    def __init__(self, bot: commands.Bot, coordinator: Any, typing_tracker: Any):
        self.bot = bot
        self.coordinator = coordinator
        self.typing_tracker = typing_tracker

        # Configuration - using Loom technique for Gemini completions
        self.text_timeout = int(os.getenv("TEXT_TIMEOUT", "180"))
        self.default_provider = "openrouter-loom"
        self.default_model = "google/gemini-3-flash-preview"

        # Channel whitelist - only respond in configured channels
        whitelist_str = os.getenv("LLAMA_BOT_CHANNELS", "")
        self.whitelist_channels: set[int] = set()
        if whitelist_str:
            for cid_str in whitelist_str.split(","):
                with suppress(ValueError):
                    self.whitelist_channels.add(int(cid_str.strip()))

        # Webhook configuration per channel
        self.channel_webhooks: dict[int, str] = {}
        self.channel_names: dict[int, str] = {}
        self._load_webhook_config()

        # Active generations (per channel)
        self._active_generations: dict[int, GenerationJob] = {}

        _LOG.info(
            "LlamaBot initialized (provider=%s, model=%s, channels=%s)",
            self.default_provider,
            self.default_model,
            list(self.whitelist_channels),
        )

    def _load_webhook_config(self) -> None:
        """Load webhook URLs and bot names for configured channels."""
        for channel_id in self.whitelist_channels:
            webhook_key = f"LLAMA_BOT_WEBHOOK_{channel_id}"
            name_key = f"LLAMA_BOT_NAME_{channel_id}"

            webhook_url = os.getenv(webhook_key)
            bot_name = os.getenv(name_key, "Llama")

            if webhook_url:
                self.channel_webhooks[channel_id] = webhook_url
                self.channel_names[channel_id] = bot_name
                _LOG.info("LlamaBot webhook configured: channel=%s, name='%s'", channel_id, bot_name)

    # ==================== Message Handling ====================

    async def receive_message(
        self,
        message: discord.Message,
        history: list,
    ) -> dict[str, Any] | None:
        """Receive message and potentially respond."""
        if not await self._should_respond(message):
            return None

        _LOG.info(f"LlamaBot will respond to message from {message.author}")

        # Just use the raw message content
        raw_text = message.content.strip()
        if not raw_text:
            return None

        # Cancel any existing generation
        await self._cancel_generation(message.channel.id)

        # Generate response
        job = GenerationJob()
        task = self.bot.loop.create_task(
            self._generate_and_send(message, raw_text, job)
        )
        job.task = task
        self._active_generations[message.channel.id] = job

        try:
            return await task
        except Exception:
            _LOG.exception("Failed to generate response")
            return None

    async def _should_respond(self, message: discord.Message) -> bool:
        """Check if bot should respond."""
        # Never respond to own messages
        if message.author.bot and message.author.id == self.bot.user.id and not message.webhook_id:
            return False

        # Check channel whitelist
        bot_mentioned = await self._check_bot_mentioned(message)
        if not self._channel_allowed(message.channel, mentioned=bot_mentioned):
            return False

        # Ignore commands
        if chat_utils.should_ignore_message(message.content):
            return False

        # Ignore empty
        return not (not message.content.strip() and not message.attachments)

    async def _check_bot_mentioned(self, message: discord.Message) -> bool:
        """Check if bot was explicitly mentioned."""
        if self.bot.user not in message.mentions:
            return False
        if message.reference and message.reference.resolved:
            replied_message = message.reference.resolved
            if isinstance(replied_message, discord.Message) and replied_message.author == self.bot.user:
                return False
        return True

    def _channel_allowed(self, channel: discord.abc.Messageable, mentioned: bool = False) -> bool:
        """Check if channel is allowed."""
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            return False
        if mentioned:
            return True
        if not self.whitelist_channels:
            return False
        return channel_id in self.whitelist_channels

    # ==================== Generation ====================

    async def _generate_and_send(
        self,
        message: discord.Message,
        prompt: str,
        job: GenerationJob,
    ) -> dict[str, Any] | None:
        """Generate and send response."""
        channel = message.channel

        try:
            text = await self._run_generation(prompt, job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOG.exception("Generation failed: %s", exc)
            return None
        finally:
            if self._active_generations.get(channel.id) is job:
                self._active_generations.pop(channel.id, None)

        if not text.strip():
            return None

        # Prepend the prompt if not already included in response
        if text.strip().startswith(prompt.strip()):
            full_text = text
        else:
            full_text = prompt + " " + text

        # Wait for typing to clear
        await self._wait_for_typing_to_clear(channel.id)

        # Send to Discord
        sent_messages = await self._send_to_discord(channel, full_text)
        if not sent_messages:
            return None

        channel_id = channel.id
        webhook_url = self.channel_webhooks.get(channel_id)
        bot_name = self.channel_names.get(channel_id, "Llama")

        webhook_id = None
        if sent_messages and webhook_url and sent_messages[0].webhook_id:
            webhook_id = sent_messages[0].webhook_id

        return {
            "message_id": sent_messages[0].id,
            "text": full_text,
            "webhook_id": webhook_id,
            "bot_name": bot_name,
        }

    async def _run_generation(
        self,
        prompt: str,
        job: GenerationJob,
    ) -> str:
        """Run LLM generation via Celery with raw text prompt."""
        _LOG.info(f"Sending raw prompt to {self.default_provider}/{self.default_model}: {prompt[:100]}...")

        async_result = generate_text.apply_async(
            (self.default_provider, self.default_model, prompt),
            kwargs={"temperature": 1.0},
        )
        job.result = async_result
        start = time.monotonic()

        while True:
            if async_result.ready():
                break
            if (time.monotonic() - start) > self.text_timeout:
                async_result.revoke(terminate=True)
                raise TimeoutError(f"timed out after {self.text_timeout:.0f}s")
            await asyncio.sleep(0.5)

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            functools.partial(async_result.get, timeout=0.1),
        )

        return str(text)

    async def _cancel_generation(self, channel_id: int) -> None:
        """Cancel active generation."""
        job = self._active_generations.get(channel_id)
        if not job:
            return

        if job.task and not job.task.done():
            job.task.cancel()
            if job.result:
                job.result.revoke(terminate=True)
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(job.task, timeout=0.5)
            self._active_generations.pop(channel_id, None)

    async def _wait_for_typing_to_clear(self, channel_id: int) -> None:
        """Wait for human typing to stop."""
        max_wait = 10.0
        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < max_wait:
            if not self.typing_tracker.is_human_typing(channel_id, self.bot.user.id):
                return
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    async def _send_to_discord(
        self,
        channel: discord.abc.Messageable,
        text: str,
    ) -> list[discord.Message]:
        """Send response to Discord."""
        sent: list[discord.Message] = []

        channel_id = channel.id
        webhook_url = self.channel_webhooks.get(channel_id)
        bot_name = self.channel_names.get(channel_id, "Llama")

        webhook = None
        if webhook_url:
            try:
                webhook = discord.Webhook.from_url(webhook_url, client=self.bot.http)
            except Exception:
                _LOG.exception("Failed to create webhook")

        # Truncate to 2000 chars for Discord
        if len(text) > 2000:
            text = text[:1997] + "..."

        if webhook:
            msg = await webhook.send(text, username=bot_name, wait=True)
        else:
            msg = await channel.send(text)
        sent.append(msg)

        return sent
