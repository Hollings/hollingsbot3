"""Chat coordinator - maintains history and broadcasts events to bots."""

import asyncio
import io
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque

import discord
from discord.ext import commands

try:
    import cairosvg  # type: ignore
except Exception:
    cairosvg = None  # type: ignore

from hollingsbot.cogs import chat_utils
from hollingsbot.cogs.conversation import ConversationTurn, ImageAttachment, ModelTurn
from hollingsbot.cogs.typing_tracker import TypingTracker
from hollingsbot.utils.discord_utils import get_display_name

# Summarization imports
from hollingsbot.summarization import (
    SummaryCache,
    CachedMessage,
    Summarizer,
    SummaryWorker,
)
from hollingsbot.text_generators.anthropic import AnthropicTextGenerator

_LOG = logging.getLogger(__name__)


class _SummarizerLLM:
    """Adapter to use AnthropicTextGenerator with Summarizer."""

    def __init__(self, model: str = "claude-haiku-4-5"):
        self._gen = AnthropicTextGenerator(model=model)

    async def generate(self, prompt: str) -> str:
        return await self._gen.generate(prompt)


# Response callback type: bot instance, response text, webhook_id for temp bots
ResponseCallback = Callable[[object, str, int | None], None]


class ChatCoordinator(commands.Cog):
    """Central coordinator for multi-bot chat system.

    Maintains single source of truth for conversation history and broadcasts
    messages to registered bot instances.
    """

    def __init__(self, bot: commands.Bot):
        self.bot = bot

        # History management
        history_limit = int(os.getenv("LLM_HISTORY_LIMIT", "50"))
        self.channel_histories: dict[int, Deque[ConversationTurn]] = {}
        self.history_limit = history_limit
        self._history_locks: dict[int, asyncio.Lock] = {}
        self._warmed_channels: set[int] = set()

        # Message processing locks (prevent concurrent on_message per channel)
        self._processing_locks: dict[int, asyncio.Lock] = {}

        # Typing tracker for typing-aware responses
        self.typing_tracker = TypingTracker()

        # Registered bots
        self.bots: list[object] = []

        # Track active message processing per channel
        self._active_message_tasks: dict[int, asyncio.Task] = {}

        # Summarization setup (uses shared DB from prompt_db)
        self.summary_enabled = os.getenv("LLM_SUMMARY_ENABLED", "0") == "1"
        if self.summary_enabled:
            self.summary_cache = SummaryCache()  # Uses DB_PATH from prompt_db
            summary_model = os.getenv("LLM_SUMMARY_MODEL", "claude-haiku-4-5")
            self.summarizer = Summarizer(_SummarizerLLM(summary_model))
            self.summary_worker = SummaryWorker(self.summary_cache, self.summarizer)
            self.summary_char_limit = int(os.getenv("LLM_SUMMARY_CHAR_LIMIT", "8000"))
            _LOG.info("Summarization enabled with model=%s", summary_model)
        else:
            self.summary_cache = None
            self.summarizer = None
            self.summary_worker = None
            self.summary_char_limit = 8000
            _LOG.info("Summarization disabled")

        _LOG.info("ChatCoordinator initialized with history_limit=%d", history_limit)

    def register_bot(self, bot_instance: object) -> None:
        """Register a bot instance to receive message events."""
        self.bots.append(bot_instance)
        bot_name = bot_instance.__class__.__name__
        _LOG.info(f"Registered bot: {bot_name}")

    def unregister_bot(self, bot_instance: object) -> None:
        """Unregister a bot instance."""
        if bot_instance in self.bots:
            self.bots.remove(bot_instance)
            bot_name = bot_instance.__class__.__name__
            _LOG.info(f"Unregistered bot: {bot_name}")

    # ==================== Generation Cancellation ====================

    async def _cancel_all_generations(self, channel_id: int) -> None:
        """Cancel all active bot generations in a channel.

        This is called when a new message arrives to ensure we only respond
        to the latest message.
        """
        # Cancel previous message processing task for this channel
        old_task = self._active_message_tasks.get(channel_id)
        if old_task and not old_task.done():
            _LOG.info(f"Cancelling previous message processing in channel {channel_id}")
            old_task.cancel()
            try:
                await asyncio.wait_for(old_task, timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Ask each bot to cancel its generation for this channel
        for bot_instance in self.bots:
            if hasattr(bot_instance, "_cancel_generation"):
                try:
                    await bot_instance._cancel_generation(channel_id)
                except Exception:
                    _LOG.exception(f"Error cancelling generation for {bot_instance.__class__.__name__}")

    # ==================== History Management ====================

    def _history_for_channel(self, channel_id: int) -> Deque[ConversationTurn]:
        """Get or create history deque for a channel."""
        if channel_id not in self.channel_histories:
            self.channel_histories[channel_id] = deque(maxlen=self.history_limit)
        return self.channel_histories[channel_id]

    def _lock_for_channel(self, channel_id: int) -> asyncio.Lock:
        """Get or create async lock for a channel."""
        lock = self._history_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._history_locks[channel_id] = lock
        return lock

    def _processing_lock_for_channel(self, channel_id: int) -> asyncio.Lock:
        """Get or create message processing lock for a channel."""
        lock = self._processing_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._processing_locks[channel_id] = lock
        return lock

    async def _ensure_channel_warm(self, channel: discord.abc.Messageable) -> None:
        """Warm up channel history by backfilling from Discord."""
        channel_id = channel.id
        if channel_id in self._warmed_channels:
            return

        _LOG.info("Warming channel %s history", channel_id)
        self._warmed_channels.add(channel_id)

        if not isinstance(channel, (discord.TextChannel, discord.Thread, discord.DMChannel)):
            return

        try:
            messages = []
            async for msg in channel.history(limit=self.history_limit * 3):
                messages.append(msg)
            messages.reverse()

            lock = self._lock_for_channel(channel_id)
            async with lock:
                history = self._history_for_channel(channel_id)
                for msg in messages:
                    # Skip other bots (except main bot and temp bots)
                    if msg.author.bot and msg.author.id != self.bot.user.id:
                        # Check if it's a temp bot webhook
                        if not msg.webhook_id:
                            continue

                    # Skip bot commands and image generation prompts
                    if chat_utils.should_ignore_message(msg.content):
                        continue

                    # Build lightweight turn for history warming
                    turn = await self._build_history_turn(msg)
                    if turn:
                        history.append(turn)

            _LOG.info("Warmed channel %s with %d messages", channel_id, len(messages))
        except Exception:
            _LOG.exception("Failed to warm channel %s", channel_id)

    async def _build_history_turn(self, message: discord.Message) -> ConversationTurn | None:
        """Build a lightweight turn for history warming (placeholders for attachments)."""
        # Always store as "user" - bots will translate to their own perspective
        role = "user"

        display = get_display_name(message.author)
        base_text = chat_utils.clean_mentions(message, self.bot).strip()

        # For history warming, use placeholders instead of full text attachments
        placeholders = [
            f"[uploaded file {att.filename} removed]"
            for att in message.attachments
            if chat_utils.is_text_attachment(att)
        ]

        # Collect images (full processing even for history)
        images = await chat_utils.collect_image_attachments(message)

        # Build simple content
        content = f"<{display}>: {base_text}"
        for placeholder in placeholders:
            content += f"\n{placeholder}"

        return ConversationTurn(
            role=role,
            content=content,
            images=images,
            message_id=message.id,
            author_id=message.author.id,
            author_name=display,
            webhook_id=message.webhook_id,
        )

    # ==================== Message Processing ====================

    @commands.Cog.listener()
    async def on_typing(
        self, channel: discord.abc.Messageable, user: discord.User | discord.Member, when
    ) -> None:
        """Handle typing events and forward to the typing tracker."""
        await self.typing_tracker.on_typing(channel, user, when)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages and let bots respond."""
        # Ignore DMs
        if not message.guild:
            return

        # Ignore empty messages
        if not message.content.strip() and not message.attachments:
            return

        # Ignore bot commands and image generation prompts
        if chat_utils.should_ignore_message(message.content):
            return

        # Ignore !spawn commands (they stay visible in Discord but not in bot history)
        if message.content.startswith("!spawn "):
            return

        channel_id = message.channel.id

        # Cancel any active generations ONLY for human messages (not bot responses)
        if not message.author.bot and not message.webhook_id:
            await self._cancel_all_generations(channel_id)

        # Ensure channel is warmed
        await self._ensure_channel_warm(message.channel)

        # Prepare the full turn and add to raw history
        lock = self._lock_for_channel(channel_id)
        async with lock:
            turn = await self._prepare_full_turn(message)
            if turn:
                history = self._history_for_channel(channel_id)

                # Check if this message is already in history (avoid duplicates)
                already_in_history = any(
                    t.message_id == message.id for t in history if t.message_id is not None
                )

                if not already_in_history:
                    history.append(turn)
                    _LOG.info(
                        f"Added to history: role={turn.role}, author_name='{turn.author_name}', "
                        f"content_preview={turn.content[:80]}..."
                    )

                    # Cache message for summarization and trigger background summarization
                    if self.summary_enabled and self.summary_cache:
                        self._cache_message_for_summary(turn, channel_id)
                        await self.trigger_summarization(channel_id)
                else:
                    _LOG.info(f"Message {message.id} already in history, skipping")

                # Get snapshot for bots to use
                snapshot = list(history)

        # Try to get a bot response (wrapped in task for cancellation)
        if turn:
            task = asyncio.create_task(self._try_bot_responses(channel_id, message, snapshot))
            self._active_message_tasks[channel_id] = task
            try:
                await task
            except asyncio.CancelledError:
                _LOG.info(f"Message processing cancelled for channel {channel_id}")
            finally:
                # Clean up task reference if it's still ours
                if self._active_message_tasks.get(channel_id) is task:
                    self._active_message_tasks.pop(channel_id, None)

    async def _try_bot_responses(self, channel_id: int, message: discord.Message, snapshot: list[ConversationTurn]) -> None:
        """Try to get a response from bots (can be cancelled)."""
        # Wendy always gets triggered first - she's the main character
        # Other bots run after, with only one responding
        wendy = None
        other_bots = []

        for bot_instance in self.bots:
            if bot_instance.__class__.__name__ == "WendyBot":
                wendy = bot_instance
            else:
                other_bots.append(bot_instance)

        # Always trigger Wendy (she checks messages herself via check_messages.sh)
        if wendy and hasattr(wendy, "receive_message"):
            try:
                _LOG.info("Triggering WendyBot...")
                response_data = await wendy.receive_message(message, snapshot)
                if response_data:
                    await self._add_response_to_history(
                        channel_id,
                        response_data["message_id"],
                        response_data["text"],
                        response_data.get("webhook_id"),
                        response_data["bot_name"],
                    )
                    _LOG.info("WendyBot responded")
            except asyncio.CancelledError:
                _LOG.info("WendyBot generation cancelled")
                raise
            except Exception:
                _LOG.exception("Error calling WendyBot")

        # Then try other bots (temp bots, etc.) - stop after first response
        import random
        random.shuffle(other_bots)

        for bot_instance in other_bots:
            try:
                if hasattr(bot_instance, "receive_message"):
                    _LOG.info(f"Trying {bot_instance.__class__.__name__}...")
                    response_data = await bot_instance.receive_message(message, snapshot)

                    if response_data:
                        await self._add_response_to_history(
                            channel_id,
                            response_data["message_id"],
                            response_data["text"],
                            response_data.get("webhook_id"),
                            response_data["bot_name"],
                        )
                        _LOG.info(f"{bot_instance.__class__.__name__} responded, stopping")
                        break
                    else:
                        _LOG.info(f"{bot_instance.__class__.__name__} declined to respond")
            except asyncio.CancelledError:
                _LOG.info(f"{bot_instance.__class__.__name__} generation cancelled")
                raise
            except Exception:
                _LOG.exception(f"Error calling {bot_instance.__class__.__name__}")

    async def _prepare_full_turn(self, message: discord.Message) -> ConversationTurn | None:
        """Prepare a full conversation turn with all content."""
        # Always store as "user" - bots will translate to their own perspective
        role = "user"

        display = get_display_name(message.author)

        # Build reply hint and collect reply images
        hint, reply_images = await chat_utils.build_reply_hint(
            message, self.bot, self.channel_histories
        )

        base_text = chat_utils.clean_mentions(message, self.bot).strip()

        # Collect text attachments (full content + placeholders)
        text_blocks, placeholders = await chat_utils.collect_text_attachments_full(message)

        # Build message text
        prefixed = chat_utils.build_user_message_text(display, hint, base_text)

        # For history, use placeholders
        history_text = prefixed
        for placeholder in placeholders:
            history_text += f"\n{placeholder}"

        # Extract URL metadata
        _, _, history_metadata = await chat_utils.extract_url_images(base_text)
        if history_metadata:
            history_text += f"\n\n{history_metadata}"

        # Collect images for LLM context
        current_images = await chat_utils.collect_image_attachments(message)
        merged_images = reply_images + current_images

        # Save all attachments (any file type) for Wendy to access via check_messages.sh
        if message.attachments:
            await chat_utils.save_attachments_for_wendy(message)

        return ConversationTurn(
            role=role,
            content=history_text,
            images=merged_images,
            message_id=message.id,
            author_id=message.author.id,
            author_name=display,
            webhook_id=message.webhook_id,
        )

    # ==================== Response History Management ====================

    async def _add_response_to_history(
        self,
        channel_id: int,
        message_id: int,
        text: str,
        webhook_id: int | None,
        bot_name: str,
    ) -> None:
        """Add a bot response to the channel history."""
        lock = self._lock_for_channel(channel_id)
        async with lock:
            history = self._history_for_channel(channel_id)

            # Check if already in history
            already_in_history = any(
                t.message_id == message_id for t in history if t.message_id is not None
            )

            if not already_in_history:
                # Store as "user" - each bot will translate to their own perspective
                turn = ConversationTurn(
                    role="user",
                    content=text,
                    images=[],
                    message_id=message_id,
                    author_id=self.bot.user.id,
                    author_name=bot_name,
                    webhook_id=webhook_id,
                )
                history.append(turn)
                _LOG.info(
                    f"Added bot response to history: bot={bot_name}, "
                    f"message_id={message_id}, text={text[:80]}..."
                )

                # Cache for summarization and trigger background summarization
                if self.summary_enabled and self.summary_cache:
                    self._cache_message_for_summary(turn, channel_id)
                    await self.trigger_summarization(channel_id)

    async def _send_response_to_channel(
        self,
        channel: discord.TextChannel,
        text: str,
        svg_files: list[discord.File],
        webhook_id: int | None = None,
    ) -> list[discord.Message]:
        """Send response to Discord channel, handling webhooks and long messages."""
        sent: list[discord.Message] = []

        # If webhook_id provided, send via webhook
        webhook: discord.Webhook | None = None
        webhook_name = None
        if webhook_id:
            from hollingsbot.prompt_db import get_temp_bot_by_webhook_id
            temp_bot = get_temp_bot_by_webhook_id(webhook_id)
            if temp_bot:
                try:
                    webhook = await self.bot.fetch_webhook(webhook_id)
                    webhook_name = temp_bot["name"]
                except (discord.NotFound, discord.Forbidden):
                    _LOG.warning(f"Failed to fetch webhook {webhook_id}, falling back")
                    webhook = None

        # Handle long messages
        if len(text) > 2000:
            timestamp = int(time.time())
            filename = f"response_{timestamp}.txt"
            file = discord.File(io.BytesIO(text.encode("utf-8")), filename=filename)

            if webhook:
                msg = await webhook.send(
                    "Response too long, attached as file:",
                    file=file,
                    username=webhook_name,
                    wait=True,
                )
            else:
                msg = await channel.send("Response too long, attached as file:", file=file)
            sent.append(msg)
        else:
            # Send text
            if webhook:
                msg = await webhook.send(text, username=webhook_name, wait=True)
            else:
                msg = await channel.send(text)
            sent.append(msg)

        # Send SVG files
        if svg_files:
            for svg_file in svg_files:
                if webhook:
                    msg = await webhook.send(file=svg_file, username=webhook_name, wait=True)
                else:
                    msg = await channel.send(file=svg_file)
                sent.append(msg)

        return sent

    # ==================== SVG Extraction ====================

    async def _extract_and_convert_svgs(self, text: str) -> list[discord.File]:
        """Extract SVG code from text and convert to PNG files."""
        svg_files: list[discord.File] = []

        if not cairosvg:
            return svg_files

        # Pattern for SVG code blocks: ```svg\n...\n```
        pattern = r'```svg\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for idx, svg_code in enumerate(matches):
            try:
                png_bytes = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
                filename = f"diagram_{idx + 1}.png"
                svg_files.append(discord.File(io.BytesIO(png_bytes), filename=filename))
            except Exception:
                _LOG.exception("Failed to convert SVG to PNG")

        # Also check for raw SVG tags
        raw_pattern = r'<svg[\s\S]*?</svg>'
        raw_matches = re.findall(raw_pattern, text, re.IGNORECASE)

        for idx, svg_code in enumerate(raw_matches):
            try:
                png_bytes = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
                filename = f"svg_{idx + 1}.png"
                svg_files.append(discord.File(io.BytesIO(png_bytes), filename=filename))
            except Exception:
                _LOG.exception("Failed to convert raw SVG to PNG")

        return svg_files

    def _clean_svgs_from_text(self, text: str) -> str:
        """Remove SVG code blocks and raw SVG tags from text."""
        # Remove SVG code blocks
        text = re.sub(r'```svg\s*\n.*?\n```', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove raw SVG tags
        text = re.sub(r'<svg[\s\S]*?</svg>', '', text, flags=re.IGNORECASE)
        return text.strip()

    # ==================== Summarization ====================

    def _cache_message_for_summary(self, turn: ConversationTurn, channel_id: int) -> None:
        """Cache a conversation turn for summarization."""
        if not self.summary_cache or not turn.message_id:
            return

        try:
            cached = CachedMessage(
                channel_id=channel_id,
                message_id=turn.message_id,
                author_id=turn.author_id or 0,
                author_name=turn.author_name or "Unknown",
                content=turn.content,
                timestamp=int(time.time()),
                has_images=bool(turn.images),
                has_attachments=False,
            )
            self.summary_cache.cache_message(cached)
            _LOG.debug("Cached message %d for summarization", turn.message_id)
        except Exception:
            _LOG.exception("Failed to cache message for summarization")

    async def trigger_summarization(self, channel_id: int) -> None:
        """Trigger background summarization for a channel (non-blocking)."""
        if not self.summary_enabled or not self.summary_worker:
            return

        try:
            # Run in background, don't wait
            asyncio.create_task(self.summary_worker.trigger_summarization(channel_id))
        except Exception:
            _LOG.exception("Failed to trigger summarization for channel %d", channel_id)

    def get_summarized_context(self, channel_id: int, raw_message_count: int | None = None) -> dict:
        """Get hierarchical summaries + recent messages for LLM context building.

        Args:
            channel_id: The channel to get context for
            raw_message_count: Number of raw messages to include (default: 7)

        Returns:
            dict with keys:
                - raw_messages: list[CachedMessage] - recent messages (count varies)
                - level_1_groups: list[MessageGroup] - up to 5 summaries of 5 messages each
                - level_2_groups: list[MessageGroup] - up to 5 summaries of 25 messages each
                - total_message_coverage: int - total messages represented
                - has_summaries: bool - whether any summaries are available
        """
        if not self.summary_enabled or not self.summary_cache:
            return {
                "raw_messages": [],
                "level_1_groups": [],
                "level_2_groups": [],
                "total_message_coverage": 0,
                "has_summaries": False,
            }

        try:
            context = self.summary_cache.get_hierarchical_context(channel_id, raw_message_count)
            has_summaries = bool(context["level_1_groups"]) or bool(context["level_2_groups"])

            return {
                "raw_messages": context["raw_messages"],
                "level_1_groups": context["level_1_groups"],
                "level_2_groups": context["level_2_groups"],
                "total_message_coverage": context["total_message_coverage"],
                "has_summaries": has_summaries,
            }
        except Exception:
            _LOG.exception("Failed to get summarized context")
            return {
                "raw_messages": [],
                "level_1_groups": [],
                "level_2_groups": [],
                "total_message_coverage": 0,
                "has_summaries": False,
            }


async def setup(bot: commands.Bot) -> None:
    """Setup function for loading the cog."""
    import os
    coordinator = ChatCoordinator(bot)
    await bot.add_cog(coordinator)

    # Import and register bots
    from hollingsbot.cogs.chat_bots.wendy_bot import WendyBot
    from hollingsbot.cogs.chat_bots.temp_bot import TempBotManager
    from hollingsbot.cogs.chat_bots.grok_bot import GrokBot
    from hollingsbot.cogs.chat_bots.llama_bot import LlamaBot
    from hollingsbot.cogs.chat_bots.gemini_bot import GeminiBot

    # Initialize bots with typing tracker
    wendy = WendyBot(bot, coordinator, coordinator.typing_tracker)
    temp_bot_manager = TempBotManager(bot, coordinator, coordinator.typing_tracker)
    grok = GrokBot(bot, coordinator, coordinator.typing_tracker)
    llama = LlamaBot(bot, coordinator, coordinator.typing_tracker)
    gemini = GeminiBot(bot, coordinator, coordinator.typing_tracker)

    # Register with coordinator
    coordinator.register_bot(wendy)
    coordinator.register_bot(temp_bot_manager)
    coordinator.register_bot(grok)
    coordinator.register_bot(llama)
    coordinator.register_bot(gemini)

    _LOG.info("Chat system initialized with all bots")
