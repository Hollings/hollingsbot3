"""GeminiBot - Gemini 3-powered LLM chat bot with webhook support for custom names."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from contextlib import suppress

import discord
from discord.ext import commands

from hollingsbot.cogs import chat_utils
from hollingsbot.cogs.conversation import ConversationTurn, ModelTurn
from hollingsbot.tasks import generate_llm_chat_response
from hollingsbot.utils.svg_utils import extract_render_and_strip_svgs

_LOG = logging.getLogger(__name__)


class GenerationJob:
    """Tracks active generation state."""

    def __init__(self):
        self.task: asyncio.Task | None = None
        self.result: Any = None


class GeminiBot:
    """Gemini 3-powered LLM chat bot with webhook support for per-channel custom names."""

    def __init__(self, bot: commands.Bot, coordinator: Any, typing_tracker: Any):
        self.bot = bot
        self.coordinator = coordinator
        self.typing_tracker = typing_tracker

        # Configuration - hardcoded to use Gemini
        self.text_timeout = int(os.getenv("TEXT_TIMEOUT", "180"))
        self.max_turns_sent = 10  # GeminiBot uses reduced history
        self.default_provider = "gemini"
        self.default_model = "gemini-3-pro-preview"

        # Channel whitelist - only respond in configured channels
        whitelist_str = os.getenv("GEMINI_BOT_CHANNELS", "")
        self.whitelist_channels: set[int] = set()
        if whitelist_str:
            for cid_str in whitelist_str.split(","):
                with suppress(ValueError):
                    self.whitelist_channels.add(int(cid_str.strip()))

        # Webhook configuration per channel
        self.channel_webhooks: dict[int, str] = {}  # channel_id -> webhook_url
        self.channel_names: dict[int, str] = {}  # channel_id -> bot_name
        self._load_webhook_config()

        # Model preferences and system prompts (per user)
        self.state_file = Path("generated/gemini_bot_state.json")
        self.model_preferences: dict[str, dict[str, dict[str, str]]] = {}
        self.user_system_prompts: dict[str, dict[str, str]] = {}
        self._load_state()

        # Available models
        self._model_lookup: set[tuple[str, str]] = set()
        self._load_available_models()

        # No tools or notebook for GeminiBot (keep it simple)
        # System prompt (base only, no tools)
        self.base_system_prompt = self._load_base_system_prompt()
        self.system_prompt = self.base_system_prompt

        # Active generations (per channel)
        self._active_generations: dict[int, GenerationJob] = {}

        # Debug log storage (message_id -> log data)
        self._response_logs: dict[int, dict[str, Any]] = {}
        self._response_log_max_size = 100

        # Register commands
        self._register_commands()

        _LOG.info(
            "GeminiBot initialized (provider=%s, model=%s, channels=%s)",
            self.default_provider,
            self.default_model,
            list(self.whitelist_channels),
        )

    def _load_webhook_config(self) -> None:
        """Load webhook URLs and bot names for configured channels from environment."""
        for channel_id in self.whitelist_channels:
            # Look for GEMINI_BOT_WEBHOOK_{channel_id} and GEMINI_BOT_NAME_{channel_id}
            webhook_key = f"GEMINI_BOT_WEBHOOK_{channel_id}"
            name_key = f"GEMINI_BOT_NAME_{channel_id}"

            webhook_url = os.getenv(webhook_key)
            bot_name = os.getenv(name_key, "Gemini")  # Default to "Gemini" if not specified

            if webhook_url:
                self.channel_webhooks[channel_id] = webhook_url
                self.channel_names[channel_id] = bot_name
                _LOG.info(
                    "GeminiBot webhook configured: channel=%s, name='%s'",
                    channel_id,
                    bot_name,
                )
            else:
                _LOG.warning(
                    "No webhook configured for GeminiBot channel %s (set %s)",
                    channel_id,
                    webhook_key,
                )

    # ==================== Debug Log Storage ====================

    def _store_response_log(
        self,
        message_id: int,
        conversation: list[dict[str, Any]],
        response_text: str,
        llm_debug: dict[str, Any],
        tool_debug: list[dict[str, Any]],
    ) -> None:
        """Store debug log for a response, maintaining size limit."""
        log_data = {
            "conversation": conversation,
            "response_text": response_text,
            "llm_debug": llm_debug,
            "tool_debug": tool_debug,
            "timestamp": time.time(),
        }
        self._response_logs[message_id] = log_data

        # Trim old logs if we exceed the limit
        if len(self._response_logs) > self._response_log_max_size:
            # Sort by timestamp and remove oldest
            sorted_ids = sorted(
                self._response_logs.keys(),
                key=lambda mid: self._response_logs[mid].get("timestamp", 0),
            )
            to_remove = sorted_ids[: len(self._response_logs) - self._response_log_max_size]
            for mid in to_remove:
                del self._response_logs[mid]

    def get_response_log(self, message_id: int) -> dict[str, Any] | None:
        """Retrieve debug log for a message ID."""
        return self._response_logs.get(message_id)

    # ==================== Available Models ====================

    def _load_available_models(self) -> None:
        """Load available models from JSON file."""
        try:
            models_file = Path("src/hollingsbot/available_models.json")
            with models_file.open("r") as f:
                data = json.load(f)
            for provider, models in data.items():
                provider_lower = provider.lower()
                for model_id in models:
                    self._model_lookup.add((provider_lower, model_id))
            _LOG.info("Loaded %d available models", len(self._model_lookup))
        except Exception:
            _LOG.exception("Failed to load available models")

    def _load_base_system_prompt(self) -> str:
        """Load base system prompt from GeminiBot-specific file."""
        try:
            from pathlib import Path
            gemini_prompt_file = Path("config/gemini_bot_system_prompt.txt")
            if gemini_prompt_file.exists():
                return gemini_prompt_file.read_text(encoding="utf-8").strip()
            else:
                _LOG.warning("gemini_bot_system_prompt.txt not found, using default")
                return "You are a helpful AI assistant powered by Google Gemini 3."
        except Exception:
            _LOG.exception("Failed to load GeminiBot system prompt")
            return "You are a helpful AI assistant powered by Google Gemini 3."

    def _build_full_system_prompt(self, user_override: str | None = None) -> str:
        """Build system prompt (GeminiBot uses simple prompts without tools/notebook)."""
        return user_override or self.base_system_prompt

    # ==================== State Management ====================

    def _load_state(self) -> None:
        """Load model preferences and system prompts from state file."""
        if not self.state_file.exists():
            return
        try:
            with self.state_file.open("r") as f:
                state = json.load(f)
            self.model_preferences = state.get("model_preferences", {})
            self.user_system_prompts = state.get("user_system_prompts", {})
            _LOG.info("Loaded state from %s", self.state_file)
        except Exception:
            _LOG.exception("Failed to load state from %s", self.state_file)

    def _save_state(self) -> None:
        """Save model preferences and system prompts to state file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "model_preferences": self.model_preferences,
                "user_system_prompts": self.user_system_prompts,
            }
            with self.state_file.open("w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            _LOG.exception("Failed to save state to %s", self.state_file)

    # ==================== Message Handling ====================

    async def receive_message(
        self,
        message: discord.Message,
        history: list[ConversationTurn],
    ) -> dict[str, Any] | None:
        """Receive message event from coordinator and potentially respond.

        Returns:
            dict with keys: message_id, text, webhook_id (None for main bot), bot_name
            Or None if bot declines to respond
        """
        _LOG.info(f"GeminiBot received message from {message.author}: {message.content[:50]}...")

        # Check if we should respond to this message
        if not await self._should_respond(message):
            _LOG.info(f"GeminiBot decided not to respond to message from {message.author}")
            return None

        _LOG.info(f"GeminiBot will respond to message from {message.author}")

        # Get user's preferred model
        provider, model = self._get_model_for_user(
            getattr(message.guild, "id", None), message.author.id
        )

        # Build current turn (already in history, just extract for payload building)
        current_turn = self._extract_current_turn(message, history)
        if not current_turn:
            _LOG.warning("Failed to extract current turn from history")
            return None

        # Translate history to this bot's perspective
        translated_history = self._translate_history(history[:-1])  # Exclude current turn

        # Get user's custom system prompt if any
        user_system_prompt = self._get_user_system_prompt(
            getattr(message.guild, "id", None), message.author.id
        )
        full_system_prompt = self._build_full_system_prompt(user_system_prompt)

        # Build conversation payload
        conversation = self._build_conversation_payload(
            translated_history, current_turn, full_system_prompt
        )

        # Cancel any existing generation in this channel
        await self._cancel_generation(message.channel.id)

        # Generate and send response
        job = GenerationJob()
        task = self.bot.loop.create_task(
            self._generate_and_send(
                message, conversation, provider, model, job
            )
        )
        job.task = task
        self._active_generations[message.channel.id] = job

        # Wait for generation to complete
        try:
            return await task
        except Exception:
            _LOG.exception("Failed to generate response")
            return None

    async def _should_respond(self, message: discord.Message) -> bool:
        """Check if bot should respond to this message."""
        # Never respond to own messages (webhook_id=None for main bot)
        if message.author.bot and message.author.id == self.bot.user.id and not message.webhook_id:
            _LOG.info("Ignoring own message")
            return False

        # Check channel whitelist (with mention override)
        bot_mentioned = await self._check_bot_mentioned(message)
        if not self._channel_allowed(message.channel, mentioned=bot_mentioned):
            _LOG.info(f"Channel {message.channel.id} not allowed (mentioned={bot_mentioned})")
            return False

        # Ignore commands and image generation
        if chat_utils.should_ignore_message(message.content):
            _LOG.info("Ignoring command/image-gen message")
            return False

        # Ignore empty messages
        if not message.content.strip() and not message.attachments:
            _LOG.info("Ignoring empty message")
            return False

        return True

    async def _check_bot_mentioned(self, message: discord.Message) -> bool:
        """Check if bot was explicitly mentioned (not via reply)."""
        if self.bot.user not in message.mentions:
            return False

        # Check if this is a reply mention or an explicit mention
        if message.reference and message.reference.resolved:
            replied_message = message.reference.resolved
            if isinstance(replied_message, discord.Message) and replied_message.author == self.bot.user:
                # This is a reply to the bot's message, not an explicit mention
                return False

        return True

    def _channel_allowed(self, channel: discord.abc.Messageable, mentioned: bool = False) -> bool:
        """Check if LLM chat is enabled for this channel."""
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            return False

        # If bot is mentioned, allow response in any channel
        if mentioned:
            return True

        if not self.whitelist_channels:
            # Empty whitelist disables the feature entirely
            return False
        return channel_id in self.whitelist_channels

    def _extract_current_turn(
        self, message: discord.Message, history: list[ConversationTurn]
    ) -> ModelTurn | None:
        """Extract current turn from history (should be the last item)."""
        if not history:
            return None

        last_turn = history[-1]
        if last_turn.message_id != message.id:
            _LOG.warning("Last turn in history doesn't match current message")
            return None

        return ModelTurn(role=last_turn.role, text=last_turn.content, images=last_turn.images)

    # ==================== History Translation ====================

    def _translate_history(self, history: list[ConversationTurn]) -> list[ConversationTurn]:
        """Translate raw history to this bot's perspective.

        GeminiBot sees:
        - Own past webhook messages -> "assistant"
        - All other messages (users, Wendy, temp bots) -> "user"
        """
        # Get GeminiBot's webhook IDs
        gemini_webhook_ids = set(self.channel_webhooks.values())

        translated = []
        for turn in history:
            # Check if this is from GeminiBot itself (has one of our webhook IDs)
            is_own_message = (
                turn.webhook_id and
                any(str(turn.webhook_id) in wh_url for wh_url in gemini_webhook_ids)
            )

            if is_own_message:
                # This is from GeminiBot -> "assistant"
                role = "assistant"
            else:
                # Everything else (users, Wendy, temp bots) -> "user"
                role = "user"

            # Create new turn with translated role
            translated.append(
                ConversationTurn(
                    role=role,
                    content=turn.content,
                    images=turn.images,
                    message_id=turn.message_id,
                    author_id=turn.author_id,
                    author_name=turn.author_name,
                    webhook_id=turn.webhook_id,
                )
            )

        return translated

    def _build_conversation_payload(
        self,
        history: list[ConversationTurn],
        current_turn: ModelTurn,
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """Build conversation payload for LLM."""
        # Take recent history
        recent = history[-self.max_turns_sent :]

        # Build payload
        conversation: list[dict[str, Any]] = [
            {
                "role": "system",
                "text": system_prompt,
                "images": [],
            }
        ]

        for turn in recent:
            # Strip display name prefix from ALL messages for GeminiBot
            # (prevents Gemini from seeing role-play patterns like "<Name>: text")
            text = self._strip_display_name_prefix(turn.content)

            conversation.append(
                {
                    "role": turn.role,
                    "text": text,
                    "images": [img.to_payload() for img in turn.images],
                }
            )

        # Add current turn (strip prefix here too)
        conversation.append(
            {
                "role": current_turn.role,
                "text": self._strip_display_name_prefix(current_turn.text),
                "images": [img.to_payload() for img in current_turn.images],
            }
        )

        return conversation

    def _strip_display_name_prefix(self, text: str) -> str:
        """Strip display name prefix from text (e.g., '<DisplayName>: text' -> 'text')."""
        # Match pattern: <anything>: text
        match = re.match(r'^<[^>]+>:\s*', text)
        if match:
            return text[match.end():]
        return text

    # ==================== Generation ====================

    async def _generate_and_send(
        self,
        message: discord.Message,
        conversation: list[dict[str, Any]],
        provider: str,
        model: str,
        job: GenerationJob,
    ) -> dict[str, Any] | None:
        """Generate LLM response, execute tools, and send to Discord.

        Returns dict with message_id, text, webhook_id, bot_name or None on failure.
        """
        channel = message.channel
        llm_debug: dict[str, Any] = {}
        tool_debug: list[dict[str, Any]] = []
        response_text = ""
        stored_conversation = conversation.copy()

        try:
            # Generate initial response
            text, llm_debug, stored_conversation = await self._run_generation(provider, model, conversation, job)
            response_text = text

            # GeminiBot doesn't support tools - keep it simple
            tool_debug = []

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            import traceback
            error_traceback = traceback.format_exc()
            _LOG.error(
                "Generation FAILED for channel %s (provider=%s model=%s): %s",
                channel.id,
                provider,
                model,
                exc,
            )
            _LOG.error("Full traceback:\n%s", error_traceback)

            # Store debug log for request message even on error
            llm_debug["status"] = "error"
            llm_debug["error_message"] = str(exc)
            llm_debug["error_traceback"] = error_traceback
            self._store_response_log(
                message.id,
                stored_conversation,
                response_text or "",
                llm_debug,
                tool_debug,
            )

            # Don't send error to Discord (triggers message loop)
            return None
        finally:
            if self._active_generations.get(channel.id) is job:
                self._active_generations.pop(channel.id, None)

        # Check if response is empty
        if not text.strip():
            _LOG.warning("LLM returned EMPTY response (likely conversation history issue) in channel %s", channel.id)
            return None

        # Extract and convert SVGs if present (using shared utility)
        clean_text, svg_tuples = extract_render_and_strip_svgs(text)
        svg_files = [discord.File(fp=buf, filename=name) for name, buf in svg_tuples]

        # Wait for human to finish typing (if applicable)
        await self._wait_for_typing_to_clear(channel.id)

        # Send to Discord
        sent_messages = await self._send_to_discord(channel, clean_text, svg_files)
        if not sent_messages:
            return None

        # Store debug log for both request and response messages
        log_data_args = (
            stored_conversation,
            response_text,
            llm_debug,
            tool_debug,
        )

        # Store for request message
        self._store_response_log(message.id, *log_data_args)

        # Store for response message
        self._store_response_log(sent_messages[0].id, *log_data_args)

        # Get bot name (use configured name if webhook is being used)
        channel_id = channel.id
        webhook_url = self.channel_webhooks.get(channel_id)
        bot_name = self.channel_names.get(channel_id, "Gemini")

        # Extract webhook_id from sent message if it was sent via webhook
        webhook_id = None
        if sent_messages and webhook_url and sent_messages[0].webhook_id:
            webhook_id = sent_messages[0].webhook_id

        return {
            "message_id": sent_messages[0].id,
            "text": clean_text,
            "webhook_id": webhook_id,
            "bot_name": bot_name,
        }

    async def _run_generation(
        self,
        provider: str,
        model: str,
        conversation: list[dict[str, Any]],
        job: GenerationJob,
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        """Run LLM generation via Celery and return (text, debug_info, conversation)."""
        # Debug logging
        _LOG.info(f"GeminiBot calling Celery with {len(conversation)} turns")
        for i, turn in enumerate(conversation):
            text_preview = turn.get('text', '')[:100].replace('\n', '\\n')
            _LOG.info(f"Turn {i}: role={turn.get('role')}, text_preview={text_preview}, images={len(turn.get('images', []))}")

        # Check for role alternation issues
        alternation_issues = 0
        for i in range(1, len(conversation)):
            if conversation[i]['role'] == conversation[i-1]['role'] and conversation[i]['role'] != 'system':
                alternation_issues += 1
                _LOG.warning(f"Role alternation issue: turns {i-1} and {i} both have role={conversation[i]['role']}")

        if alternation_issues > 0:
            _LOG.warning(f"Found {alternation_issues} role alternation issues - this may cause empty responses!")

        async_result = generate_llm_chat_response.apply_async(
            (provider, model, conversation),
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
        result = await loop.run_in_executor(
            None,
            functools.partial(async_result.get, timeout=0.1),
        )

        # Extract text and debug info from result
        if isinstance(result, dict):
            text = str(result.get("text", ""))
            debug_info = result.get("debug", {})
            conv = result.get("conversation", conversation)
        else:
            text = str(result)
            debug_info = {}
            conv = conversation

        # Log for debugging
        if not text or not text.strip():
            _LOG.warning(f"Empty text from Celery: result={result}")

        return text, debug_info, conv

    async def _cancel_generation(self, channel_id: int) -> None:
        """Cancel any active generation in the channel."""
        job = self._active_generations.get(channel_id)
        if not job:
            return

        if job.task and not job.task.done():
            # Cancel the local asyncio task first
            job.task.cancel()

            # Revoke the Celery task immediately
            if job.result:
                job.result.revoke(terminate=True)
                _LOG.info("Revoked Celery task for channel %s (terminate=True)", channel_id)

            # Now wait for local task cleanup
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(job.task, timeout=0.5)

            self._active_generations.pop(channel_id, None)
            _LOG.info("Cancelled active generation in channel %s", channel_id)

    async def _wait_for_typing_to_clear(self, channel_id: int) -> None:
        """Wait for human typing to stop before sending response."""
        max_wait = 10.0  # seconds
        poll_interval = 0.5  # seconds
        elapsed = 0.0

        while elapsed < max_wait:
            if not self.typing_tracker.is_human_typing(channel_id, self.bot.user.id):
                return

            _LOG.debug(
                "Human is typing in channel %s, waiting %.1fs before sending...",
                channel_id,
                max_wait - elapsed,
            )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        _LOG.debug("Typing wait timeout reached for channel %s, sending response", channel_id)

    # ==================== Discord Message Sending ====================

    async def _send_to_discord(
        self,
        channel: discord.abc.Messageable,
        text: str,
        svg_files: list[discord.File],
    ) -> list[discord.Message]:
        """Send response to Discord channel via webhook (with custom name) or regular message."""
        import io
        sent: list[discord.Message] = []

        # Try to get webhook for this channel
        channel_id = channel.id
        webhook_url = self.channel_webhooks.get(channel_id)
        bot_name = self.channel_names.get(channel_id, "Gemini")

        webhook = None
        if webhook_url:
            try:
                # Parse webhook URL to get ID and token
                webhook = discord.Webhook.from_url(webhook_url, client=self.bot.http)
            except Exception:
                _LOG.exception("Failed to create webhook from URL for channel %s", channel_id)

        # Handle long messages
        if len(text) > 2000:
            timestamp = int(time.time())
            filename = f"response_{timestamp}.txt"
            file = discord.File(io.BytesIO(text.encode("utf-8")), filename=filename)

            if webhook:
                msg = await webhook.send(
                    "Response too long, attached as file:",
                    file=file,
                    username=bot_name,
                    wait=True,
                )
            else:
                msg = await channel.send("Response too long, attached as file:", file=file)
            sent.append(msg)
        else:
            # Send text
            if webhook:
                msg = await webhook.send(text, username=bot_name, wait=True)
            else:
                msg = await channel.send(text)
            sent.append(msg)

        # Send SVG files
        if svg_files:
            for svg_file in svg_files:
                if webhook:
                    msg = await webhook.send(file=svg_file, username=bot_name, wait=True)
                else:
                    msg = await channel.send(file=svg_file)
                sent.append(msg)

        return sent

    # ==================== Model Preferences ====================

    def _is_valid_model(self, provider: str, model: str) -> bool:
        """Check if provider/model combination is available."""
        key = (provider.lower(), model)
        return key in self._model_lookup

    def _get_model_for_user(self, guild_id: int | None, user_id: int) -> tuple[str, str]:
        """Get user's preferred model or return default."""
        gid = str(guild_id or 0)
        uid = str(user_id)
        entry = self.model_preferences.get(gid, {}).get(uid)
        if isinstance(entry, dict):
            provider = entry.get("provider")
            model = entry.get("model")
            if isinstance(provider, str) and isinstance(model, str) and self._is_valid_model(provider, model):
                return provider.lower(), model
        return self.default_provider, self.default_model

    def _set_model_for_user(self, guild_id: int | None, user_id: int, provider: str, model: str) -> None:
        """Save user's model preference."""
        gid = str(guild_id or 0)
        uid = str(user_id)
        guild_entry = self.model_preferences.setdefault(gid, {})
        guild_entry[uid] = {"provider": provider.lower(), "model": model}
        self._save_state()

    def _get_user_system_prompt(self, guild_id: int | None, user_id: int) -> str | None:
        """Get user's custom system prompt."""
        gid = str(guild_id or 0)
        uid = str(user_id)
        return self.user_system_prompts.get(gid, {}).get(uid)

    def _set_user_system_prompt(self, guild_id: int | None, user_id: int, prompt: str) -> None:
        """Save user's custom system prompt."""
        gid = str(guild_id or 0)
        uid = str(user_id)
        guild_entry = self.user_system_prompts.setdefault(gid, {})
        guild_entry[uid] = prompt
        self._save_state()

    def _clear_user_system_prompt(self, guild_id: int | None, user_id: int) -> None:
        """Clear user's custom system prompt."""
        gid = str(guild_id or 0)
        uid = str(user_id)
        guild_entry = self.user_system_prompts.get(gid, {})
        if uid in guild_entry:
            del guild_entry[uid]
            self._save_state()

    # ==================== Commands ====================

    def _register_commands(self) -> None:
        """Register bot commands."""
        pass

    async def handle_model_command(self, ctx: commands.Context, provider: str | None = None, model: str | None = None) -> None:
        """Handle !model command to set user's preferred model."""
        if provider is None or model is None:
            current_provider, current_model = self._get_model_for_user(
                getattr(ctx.guild, "id", None), ctx.author.id
            )
            await ctx.send(f"Current model: {current_provider}/{current_model}")
            return

        if not self._is_valid_model(provider, model):
            await ctx.send(f"Invalid model: {provider}/{model}")
            return

        self._set_model_for_user(getattr(ctx.guild, "id", None), ctx.author.id, provider, model)
        await ctx.send(f"Model set to: {provider}/{model}")

    async def handle_system_command(self, ctx: commands.Context, *, prompt: str | None = None) -> None:
        """Handle !system command to set custom system prompt."""
        if prompt is None:
            user_prompt = self._get_user_system_prompt(
                getattr(ctx.guild, "id", None), ctx.author.id
            )
            if user_prompt:
                await ctx.send(f"Custom system prompt: {user_prompt[:500]}...")
            else:
                await ctx.send("Using default system prompt")
            return

        if prompt.lower() == "reset":
            self._clear_user_system_prompt(getattr(ctx.guild, "id", None), ctx.author.id)
            await ctx.send("Reset to default system prompt")
            return

        self._set_user_system_prompt(getattr(ctx.guild, "id", None), ctx.author.id, prompt)
        await ctx.send("Custom system prompt set")

    async def handle_clear_command(self, ctx: commands.Context) -> None:
        """Handle !clear command to clear channel history."""
        channel_id = ctx.channel.id
        lock = self.coordinator._lock_for_channel(channel_id)
        async with lock:
            self.coordinator.channel_histories[channel_id] = self.coordinator.channel_histories.get(channel_id).__class__(
                maxlen=self.coordinator.history_limit
            )
        await ctx.send("Channel history cleared")

    async def handle_cancel_command(self, ctx: commands.Context) -> None:
        """Handle !cancel command to cancel active generation."""
        await self._cancel_generation(ctx.channel.id)
        await ctx.send("Generation cancelled")
