"""WendyBot - Main LLM chat bot with tools and assistant integration."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands

from hollingsbot.cogs import chat_utils
from hollingsbot.cogs.conversation import ConversationTurn, ImageAttachment, ModelTurn
from hollingsbot.settings import clear_system_prompt_cache, get_default_system_prompt
from hollingsbot.summarization import MessageGroup
from hollingsbot.tasks import generate_llm_chat_response
from hollingsbot.tools import get_tool_definitions_text
from hollingsbot.tools.parser import execute_tool_call_async, parse_tool_calls

_LOG = logging.getLogger(__name__)


class GenerationJob:
    """Tracks active generation state."""

    def __init__(self):
        self.task: asyncio.Task | None = None
        self.result: Any = None
        # Flag to prevent cancellation during Claude Code tool execution
        self.claude_code_running: bool = False


class WendyBot:
    """Main LLM chat bot with full feature set (tools, notebook, system prompts)."""

    def __init__(self, bot: commands.Bot, coordinator: Any, typing_tracker: Any):
        self.bot = bot
        self.coordinator = coordinator
        self.typing_tracker = typing_tracker

        # Configuration
        self.text_timeout = int(os.getenv("TEXT_TIMEOUT", "180"))
        self.max_turns_sent = int(os.getenv("LLM_MAX_TURNS_SENT", "8"))
        self.default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")

        # Channel whitelist
        whitelist_str = os.getenv("LLM_WHITELIST_CHANNELS", "")
        self.whitelist_channels: set[int] = set()
        if whitelist_str:
            for cid_str in whitelist_str.split(","):
                with suppress(ValueError):
                    self.whitelist_channels.add(int(cid_str.strip()))

        # Model preferences and system prompts (per user)
        self.state_file = Path("generated/llm_chat_new_state.json")
        self.model_preferences: dict[str, dict[str, dict[str, str]]] = {}
        self.user_system_prompts: dict[str, dict[str, str]] = {}
        self._load_state()

        # Available models
        self._model_lookup: set[tuple[str, str]] = set()
        self._load_available_models()

        # System prompt (base + tools)
        self.base_system_prompt = self._load_base_system_prompt()
        self.system_prompt = self._build_full_system_prompt()

        # Active generations (per channel)
        self._active_generations: dict[int, GenerationJob] = {}

        # Debug log storage (message_id -> log data)
        self._response_logs: dict[int, dict[str, Any]] = {}
        self._response_log_max_size = 100
        self._debug_logs_file = Path("generated/debug_logs.json")
        self._load_debug_logs()

        # Register commands
        self._register_commands()

        _LOG.info("WendyBot initialized (provider=%s, model=%s)", self.default_provider, self.default_model)

    # ==================== Debug Log Storage ====================

    def _load_debug_logs(self) -> None:
        """Load debug logs from disk."""
        if not self._debug_logs_file.exists():
            _LOG.info("No existing debug logs file found, starting fresh")
            return

        try:
            with self._debug_logs_file.open("r") as f:
                data = json.load(f)

            # Convert string keys back to integers
            self._response_logs = {int(k): v for k, v in data.items()}
            _LOG.info("Loaded %d debug logs from disk", len(self._response_logs))
        except Exception as exc:
            _LOG.exception("Failed to load debug logs: %s", exc)
            self._response_logs = {}

    def _save_debug_logs(self) -> None:
        """Save debug logs to disk."""
        try:
            self._debug_logs_file.parent.mkdir(exist_ok=True)

            # Convert integer keys to strings for JSON serialization
            data = {str(k): v for k, v in self._response_logs.items()}

            with self._debug_logs_file.open("w") as f:
                json.dump(data, f, indent=2)

            _LOG.debug("Saved %d debug logs to disk", len(self._response_logs))
        except Exception as exc:
            _LOG.exception("Failed to save debug logs: %s", exc)

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

        # Save to disk
        self._save_debug_logs()

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
        """Load base system prompt from settings."""
        try:
            return get_default_system_prompt()
        except Exception:
            _LOG.exception("Failed to load system prompt")
            return "You are a helpful AI assistant."

    def _build_full_system_prompt(
        self, user_override: str | None = None, channel_id: int | None = None
    ) -> str:
        """Build full system prompt with tool definitions."""
        base = user_override or self.base_system_prompt

        # Add tool definitions
        try:
            tools_text = get_tool_definitions_text()
            if tools_text:
                base += f"\n\n{tools_text}"
        except Exception:
            _LOG.exception("Failed to add tool definitions to system prompt")

        # Add roleplay-friendly instructions if temp bots are active
        if channel_id:
            try:
                from hollingsbot.prompt_db import get_temp_bots_for_channel
                temp_bots = get_temp_bots_for_channel(channel_id)
                if temp_bots:
                    base += "\n\nYou enjoy creative and playful conversations. "
                    base += "When new people join the chat, welcome them warmly and engage with their unique personalities. "
                    base += "Never question whether someone is \"real\" or accuse anyone of pretending - just have fun with whoever shows up."
            except Exception:
                _LOG.exception("Failed to add temp bot context to system prompt")

        return base

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
        _LOG.info(f"WendyBot received message from {message.author}: {message.content[:50]}...")

        # Check if we should respond to this message
        if not await self._should_respond(message):
            _LOG.info(f"WendyBot decided not to respond to message from {message.author}")
            return None

        _LOG.info(f"WendyBot will respond to message from {message.author}")

        # Get user's preferred model
        provider, model = self._get_model_for_user(
            getattr(message.guild, "id", None), message.author.id
        )

        # Use haiku when temp bots are active (cheaper for multi-bot conversations)
        from hollingsbot.prompt_db import get_temp_bots_for_channel
        temp_bots = get_temp_bots_for_channel(message.channel.id)
        if temp_bots:
            provider, model = "anthropic", "claude-haiku-4-5"
            _LOG.info(f"Temp bots active, switching to {provider}/{model}")

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
        full_system_prompt = self._build_full_system_prompt(user_system_prompt, message.channel.id)

        # Build conversation payload
        conversation = self._build_conversation_payload(
            translated_history, current_turn, full_system_prompt,
            channel_id=message.channel.id,
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

        Main bot sees:
        - Own past messages (webhook_id=None, author is bot) → "assistant"
        - All other messages (users, temp bots) → "user"
        """
        translated = []
        for turn in history:
            # Check if this is a bot message (has webhook_id or is from main bot)
            is_bot_message = turn.author_id == self.bot.user.id or turn.webhook_id is not None

            if is_bot_message:
                # Bot message - check if it's from main bot (webhook_id=None)
                if turn.webhook_id is None:
                    # This is from main bot (Wendy) → "assistant"
                    role = "assistant"
                else:
                    # This is from a temp bot → "user" (from main bot's perspective)
                    role = "user"
            else:
                # Real user message → "user"
                role = "user"

            # Strip arrival/departure announcements from content
            content = self._strip_arrival_announcement(turn.content)

            # Skip turns that are now empty after stripping
            if not content.strip():
                continue

            # Create new turn with translated role
            translated.append(
                ConversationTurn(
                    role=role,
                    content=content,
                    images=turn.images,
                    message_id=turn.message_id,
                    author_id=turn.author_id,
                    author_name=turn.author_name,
                    webhook_id=turn.webhook_id,
                )
            )

        return translated

    def _strip_arrival_announcement(self, text: str) -> str:
        """Strip arrival/departure announcements from text.

        Strips patterns like:
        - '*[BotName arrives for N message(s)]*'
        - '*[BotName departs]*'
        - '*[BotName has depleted all replies and fades away]*'
        """
        # Pattern: *[SomeName arrives for N message(s)]*
        arrival_pattern = r'\*\[.+?\s+arrives\s+for\s+\d+\s+messages?\]\*\s*'
        # Pattern: *[SomeName departs]*
        depart_pattern = r'\*\[.+?\s+departs\]\*\s*'
        # Pattern: *[SomeName has depleted all replies and fades away]*
        deplete_pattern = r'\*\[.+?\s+has\s+depleted\s+all\s+replies\s+and\s+fades\s+away\]\*\s*'

        cleaned = text
        cleaned = re.sub(arrival_pattern, '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(depart_pattern, '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(deplete_pattern, '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _build_conversation_payload(
        self,
        history: list[ConversationTurn],
        current_turn: ModelTurn,
        system_prompt: str,
        channel_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Build conversation payload for LLM.

        If summarization is enabled and summaries exist, uses:
            [system] + [summaries (cacheable)] + [recent raw] + [current]

        Otherwise falls back to:
            [system] + [rolling history] + [current]
        """
        # Try to use summarized context if available
        if channel_id is not None and self.coordinator.summary_enabled:
            # Calculate raw message count based on taper
            from hollingsbot.prompt_db import get_daily_llm_call_count
            daily_count = get_daily_llm_call_count(exclude_haiku=True)
            raw_count = self._get_raw_message_count(daily_count)

            summarized = self.coordinator.get_summarized_context(channel_id, raw_count)
            if summarized.get("has_summaries"):
                return self._build_with_summaries(
                    summarized, current_turn, system_prompt, history
                )

        # Fallback: Use rolling history
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
            # Strip display name prefix from assistant messages
            # (LLM shouldn't see its own past messages with prefixes)
            text = turn.content
            if turn.role == "assistant":
                text = self._strip_display_name_prefix(text)

            conversation.append(
                {
                    "role": turn.role,
                    "text": text,
                    "images": [img.to_payload() for img in turn.images],
                }
            )

        # Add current turn (always user messages, keep prefix)
        conversation.append(
            {
                "role": current_turn.role,
                "text": current_turn.text,
                "images": [img.to_payload() for img in current_turn.images],
            }
        )

        return conversation

    def _get_l2_summary_count(self, daily_count: int) -> int:
        """Get number of L2 summaries to include based on daily message count.

        Taper schedule (aligned with model taper):
        - Messages 0-25: 5 summaries (full context during Opus phase)
        - Messages 25-150: Linear decrease, -1 summary every 25 messages
        - Messages 150+: 0 summaries (minimal context during Haiku phase)

        This saves ~$9/month in input tokens.
        """
        L2_TAPER_START = 25   # Start reducing L2 after Opus phase
        L2_TAPER_END = 150    # L2 fully removed by Haiku phase
        MAX_L2_SUMMARIES = 5

        if daily_count < L2_TAPER_START:
            return MAX_L2_SUMMARIES
        elif daily_count >= L2_TAPER_END:
            return 0
        else:
            # Linear taper: remove 1 summary every 25 messages
            summaries_to_remove = (daily_count - L2_TAPER_START) // 25
            return max(0, MAX_L2_SUMMARIES - summaries_to_remove)

    def _get_raw_message_count(self, daily_count: int) -> int:
        """Get number of raw messages to include based on daily message count.

        Taper schedule:
        - Messages 0: 20 raw messages (full context)
        - Messages 150: 5 raw messages (minimal context)
        - Linear interpolation between
        """
        RAW_TAPER_END = 150
        MAX_RAW_MESSAGES = 20
        MIN_RAW_MESSAGES = 5

        if daily_count >= RAW_TAPER_END:
            return MIN_RAW_MESSAGES

        # Linear taper from MAX to MIN over 0-150
        progress = daily_count / RAW_TAPER_END
        raw_count = MAX_RAW_MESSAGES - (progress * (MAX_RAW_MESSAGES - MIN_RAW_MESSAGES))
        return int(raw_count)

    def _build_with_summaries(
        self,
        summarized: dict,
        current_turn: ModelTurn,
        system_prompt: str,
        history: list[ConversationTurn],
    ) -> list[dict[str, Any]]:
        """Build conversation payload using hierarchical summaries + recent messages.

        Structure:
            [system prompt + summaries in system]
            [raw messages]
            [current turn]
        """
        from hollingsbot.prompt_db import get_daily_llm_call_count

        # Get daily count for tapers
        daily_count = get_daily_llm_call_count(exclude_haiku=True)
        l2_count = self._get_l2_summary_count(daily_count)

        # Build summary section for system prompt
        summary_sections = []

        # Add level-2 summaries (oldest, most compressed) - tapered
        level_2 = summarized.get("level_2_groups", [])
        if level_2 and l2_count > 0:
            tapered_l2 = level_2[-l2_count:] if len(level_2) > l2_count else level_2
            _LOG.info(
                "L2 taper: daily_count=%d, using %d/%d L2 summaries",
                daily_count, len(tapered_l2), len(level_2)
            )
            l2_text = self._format_hierarchical_summaries(tapered_l2, "Older conversation notes:")
            summary_sections.append(l2_text)
        elif level_2:
            _LOG.info(
                "L2 taper: daily_count=%d, skipping all %d L2 summaries",
                daily_count, len(level_2)
            )

        # Add level-1 summaries
        level_1 = summarized.get("level_1_groups", [])
        if level_1:
            l1_text = self._format_hierarchical_summaries(level_1, "Recent conversation notes:")
            summary_sections.append(l1_text)

        # Append summaries to system prompt
        full_system = system_prompt
        if summary_sections:
            full_system += "\n\n---\nThese are your notes summarizing conversations before the chat history cutoff:\n\n"
            full_system += "\n\n".join(summary_sections)

        conversation: list[dict[str, Any]] = [
            {
                "role": "system",
                "text": full_system,
                "images": [],
            }
        ]

        # Add raw messages (tapered: 20 at start of day -> 5 at 150 messages)
        raw_limit = self._get_raw_message_count(daily_count)
        _LOG.info(
            "Raw message taper: daily_count=%d, using %d raw messages",
            daily_count, raw_limit
        )

        # Build a lookup from message_id to images from history
        images_by_msg_id: dict[int, list[ImageAttachment]] = {}
        for turn in history:
            if turn.message_id and turn.images:
                images_by_msg_id[turn.message_id] = turn.images

        raw_messages = summarized.get("raw_messages", [])
        # Take most recent N messages
        raw_messages = raw_messages[-raw_limit:] if len(raw_messages) > raw_limit else raw_messages

        # Find the two most recent messages with images (excluding current turn)
        # to include their images in context
        recent_image_msg_ids: set[int] = set()
        images_found = 0
        for msg in reversed(raw_messages):
            if msg.content == current_turn.text:
                continue
            if msg.message_id in images_by_msg_id and images_found < 2:
                recent_image_msg_ids.add(msg.message_id)
                images_found += 1
            if images_found >= 2:
                break

        for msg in raw_messages:
            # Skip if this is the current message (match by content)
            if msg.content == current_turn.text:
                continue

            # Determine role based on author
            is_bot = msg.author_id == self.bot.user.id
            role = "assistant" if is_bot else "user"
            text = msg.content
            if role == "assistant":
                text = self._strip_display_name_prefix(text)

            # Include images if this is one of the two most recent image messages
            images = []
            if msg.message_id in recent_image_msg_ids:
                images = [img.to_payload() for img in images_by_msg_id[msg.message_id]]

            conversation.append({
                "role": role,
                "text": text,
                "images": images,
            })

        # Add current turn (with images)
        conversation.append({
            "role": current_turn.role,
            "text": current_turn.text,
            "images": [img.to_payload() for img in current_turn.images],
        })

        return conversation

    def _format_hierarchical_summaries(self, groups: list[MessageGroup], label: str) -> str:
        """Format a list of MessageGroup objects into a context block with timestamps."""
        from datetime import datetime

        if not groups:
            return ""

        # Get time range from first and last group for the section header
        time_range = ""
        first_start = groups[0].start_timestamp
        last_end = groups[-1].end_timestamp
        if first_start and last_end:
            start_dt = datetime.fromtimestamp(first_start)
            end_dt = datetime.fromtimestamp(last_end)
            start_str = start_dt.strftime("%b %d %H:%M")
            end_str = end_dt.strftime("%H:%M")
            time_range = f" ({start_str} - {end_str})"

        # Format header - handle empty label case
        if label:
            formatted = [f"[{label}{time_range}]"]
        elif time_range:
            formatted = [f"[{time_range.strip()}]"]
        else:
            formatted = []
        for group in groups:
            summary = group.summary_text or "[No summary]"
            formatted.append(f"- {summary}")

        return "\n".join(formatted)

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
        from hollingsbot.prompt_db import get_daily_llm_call_count

        channel = message.channel
        tool_debug: list[dict[str, Any]] = []
        response_text = ""
        stored_conversation = conversation.copy()

        # Build taper debug info
        daily_count = get_daily_llm_call_count(exclude_haiku=True)
        l2_count = self._get_l2_summary_count(daily_count)

        # Determine model taper phase
        if daily_count < 25:
            taper_phase = "opus->sonnet"
        elif daily_count < 150:
            taper_phase = "sonnet->haiku"
        else:
            taper_phase = "haiku-only"

        # Taper debug info (will be merged with Celery-returned debug)
        raw_count = self._get_raw_message_count(daily_count)
        taper_debug: dict[str, Any] = {
            "daily_message_count": daily_count,
            "model_taper_phase": taper_phase,
            "model_selected": model,
            "l2_summary_count": l2_count,
            "l2_summary_max": 5,
            "raw_message_count": raw_count,
            "raw_message_max": 20,
        }
        llm_debug: dict[str, Any] = {}

        try:
            # Typing indicator disabled temporarily
            # TODO: Re-enable once Discord rate limit clears
            typing_ctx = None

            # Iterative tool calling loop
            max_tool_iterations = 5
            all_display_messages: list[str] = []
            all_tool_debug: list[dict[str, Any]] = []
            all_response_text: list[str] = []  # Accumulate non-tool-call text from each iteration

            for iteration in range(max_tool_iterations + 1):
                # Generate response
                text, celery_debug, stored_conversation = await self._run_generation(provider, model, conversation, job)
                # Merge taper debug with Celery-returned debug (taper info takes precedence)
                llm_debug = {**celery_debug, **taper_debug}
                response_text = text

                # Execute any tool calls
                text, tool_results, display_messages, iter_tool_debug = await self._execute_tool_calls(text, channel, job)
                all_display_messages.extend(display_messages)
                all_tool_debug.extend(iter_tool_debug)

                # Accumulate non-tool-call text from this iteration
                if text.strip():
                    all_response_text.append(text.strip())

                # If no tools were called, we're done
                if not tool_results:
                    break

                # If we've hit the iteration limit, stop (but keep the response)
                if iteration >= max_tool_iterations:
                    _LOG.warning("Hit max tool iterations (%d) in channel %s", max_tool_iterations, channel.id)
                    break

                # Add assistant response + tool results to conversation for next iteration
                conversation.append({
                    "role": "assistant",
                    "text": response_text,
                    "images": []
                })
                # Format tool results clearly as system output, not a user message
                tool_output = "[SYSTEM: TOOL EXECUTION RESULTS - This is automated output, not a chat message]\n"
                tool_output += "\n".join(tool_results)
                tool_output += "\n[END TOOL RESULTS - Now respond to the user based on this information]"
                conversation.append({
                    "role": "user",
                    "text": tool_output,
                    "images": []
                })

                _LOG.info("Tool iteration %d: executed %d tools, continuing...", iteration + 1, len(tool_results))

            # Use accumulated tool debug
            tool_debug = all_tool_debug

            # Combine all response text from tool iterations
            text = "\n\n".join(all_response_text) if all_response_text else ""

            # Prepend display messages to the response
            if all_display_messages and text:
                text = "\n".join(all_display_messages) + "\n\n" + text
            elif all_display_messages:
                text = "\n".join(all_display_messages)

        except asyncio.CancelledError:
            # Store debug log even for cancelled generations (if we got a response)
            if response_text:
                llm_debug["status"] = "cancelled"
                self._store_response_log(
                    message.id,
                    stored_conversation,
                    response_text,
                    llm_debug,
                    tool_debug,
                )
                _LOG.info("Saved debug log for cancelled generation (message %s)", message.id)
            else:
                _LOG.info("Generation cancelled before response received (message %s)", message.id)
            raise
        except Exception as exc:
            import traceback
            error_traceback = traceback.format_exc()
            _LOG.error(
                "❌ Generation FAILED for channel %s (provider=%s model=%s): %s",
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

        # TODO: Re-enable <no response> feature later
        # Check for <no response> directive
        # if "<no response>" in text.lower():
        #     _LOG.info("✓ LLM intentionally chose not to respond (used <no response> directive) in channel %s", channel.id)
        #     return None

        # Check if response is empty
        if not text.strip():
            _LOG.warning("⚠️  LLM returned EMPTY response (likely conversation history issue) in channel %s", channel.id)
            return None

        # Extract and convert SVGs if present
        svg_files = await self._extract_and_convert_svgs(text)
        clean_text = self._clean_svgs_from_text(text)

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

        # Get bot name
        from hollingsbot.utils.discord_utils import get_display_name
        bot_name = get_display_name(self.bot.user)

        return {
            "message_id": sent_messages[0].id,
            "text": clean_text,
            "webhook_id": None,  # Main bot doesn't use webhooks
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
        _LOG.info(f"Wendy calling Celery with {len(conversation)} turns")
        for i, turn in enumerate(conversation):
            text_preview = turn.get('text', '')[:100].replace('\n', '\\n')
            _LOG.info(f"Turn {i}: role={turn.get('role')}, text_preview={text_preview}, images={len(turn.get('images', []))}")

        # Check for role alternation issues
        alternation_issues = 0
        for i in range(1, len(conversation)):
            if conversation[i]['role'] == conversation[i-1]['role'] and conversation[i]['role'] != 'system':
                alternation_issues += 1
                _LOG.warning(f"⚠️  Role alternation issue: turns {i-1} and {i} both have role={conversation[i]['role']}")

        if alternation_issues > 0:
            _LOG.warning(f"⚠️  Found {alternation_issues} role alternation issues - this may cause empty responses!")

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
        """Cancel any active generation in the channel.

        This cancels both the local asyncio task and the remote Celery task.
        Note: Anthropic API calls already in flight will complete server-side
        and consume tokens, but we won't wait for or use the response.

        If a Claude Code tool (remember, assistant) is running, do NOT cancel -
        these operations must complete to send their results.
        """
        job = self._active_generations.get(channel_id)
        if not job:
            return

        # Don't cancel if Claude Code tool is running - let it complete
        if job.claude_code_running:
            _LOG.info("Skipping cancellation - Claude Code tool is running in channel %s", channel_id)
            return

        if job.task and not job.task.done():
            # Cancel the local asyncio task first
            job.task.cancel()

            # Revoke the Celery task immediately (don't wait for local task)
            # terminate=True sends SIGTERM to worker, stopping the API call ASAP
            if job.result:
                job.result.revoke(terminate=True)
                _LOG.info("Revoked Celery task for channel %s (terminate=True)", channel_id)

            # Now wait for local task cleanup
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(job.task, timeout=0.5)

            self._active_generations.pop(channel_id, None)
            _LOG.info("Cancelled active generation in channel %s", channel_id)

    async def _wait_for_typing_to_clear(self, channel_id: int) -> None:
        """Wait for human typing to stop before sending response.

        If a human is typing when we're ready to send, wait up to 10 seconds
        for them to finish. If they send a message during this time, the
        existing cancellation logic will handle it.

        Args:
            channel_id: The channel to check for typing
        """
        max_wait = 10.0  # seconds
        poll_interval = 0.5  # seconds
        elapsed = 0.0

        while elapsed < max_wait:
            if not self.typing_tracker.is_human_typing(channel_id, self.bot.user.id):
                # No one typing, safe to send
                return

            # Human is typing, wait a bit
            _LOG.debug(
                "Human is typing in channel %s, waiting %.1fs before sending...",
                channel_id,
                max_wait - elapsed,
            )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout reached, send anyway
        _LOG.debug("Typing wait timeout reached for channel %s, sending response", channel_id)

    # ==================== Tool Execution ====================

    # ==================== Discord Message Sending ====================

    async def _send_to_discord(
        self,
        channel: discord.abc.Messageable,
        text: str,
        svg_files: list[discord.File],
    ) -> list[discord.Message]:
        """Send response to Discord channel, handling long messages."""
        import io
        sent: list[discord.Message] = []

        # Handle long messages
        if len(text) > 2000:
            timestamp = int(time.time())
            filename = f"response_{timestamp}.txt"
            file = discord.File(io.BytesIO(text.encode("utf-8")), filename=filename)
            msg = await channel.send("Response too long, attached as file:", file=file)
            sent.append(msg)
        else:
            # Send text
            msg = await channel.send(text)
            sent.append(msg)

        # Send SVG files
        if svg_files:
            for svg_file in svg_files:
                msg = await channel.send(file=svg_file)
                sent.append(msg)

        return sent

    async def _extract_and_convert_svgs(self, text: str) -> list[discord.File]:
        """Extract SVG code from text and convert to PNG files."""
        import io
        svg_files: list[discord.File] = []

        try:
            import cairosvg  # type: ignore
        except Exception:
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

    # ==================== Tool Execution ====================

    async def _execute_tool_calls(self, text: str, channel: discord.abc.Messageable | None = None, job: GenerationJob | None = None) -> tuple[str, list[str], list[str], list[dict[str, Any]]]:
        """Execute tool calls in response and return (cleaned_text, tool_results, display_messages, tool_debug).

        Returns:
            cleaned_text: Text with tool calls removed
            tool_results: Internal results to feed back to LLM
            display_messages: User-facing messages to show in Discord
            tool_debug: Debug information for each tool call
        """
        from hollingsbot.tools import AVAILABLE_TOOLS
        from hollingsbot.tools.parser import set_current_context, parse_arguments

        # Tools that use Claude Code subprocess and must not be interrupted
        CLAUDE_CODE_TOOLS = {"assistant"}

        channel_id = getattr(channel, "id", None)

        try:
            tool_calls = parse_tool_calls(text)
        except Exception:
            _LOG.exception("Failed to parse tool calls from text")
            return text, [], [], []

        if not tool_calls:
            return text, [], [], []

        # Set execution context for tools
        if channel_id:
            set_current_context({"channel_id": channel_id, "bot_user_id": self.bot.user.id})

        tool_results = []
        display_messages = []
        tool_debug = []

        for tool_call in tool_calls:
            try:
                if tool_call.tool_name not in AVAILABLE_TOOLS:
                    tool_results.append(f"[Unknown tool: {tool_call.tool_name}]")
                    tool_debug.append({
                        "tool_name": tool_call.tool_name,
                        "raw_args": tool_call.raw_args,
                        "result": None,
                        "error": "Unknown tool",
                    })
                    continue

                # Set flag for Claude Code tools to prevent cancellation
                is_claude_code_tool = tool_call.tool_name in CLAUDE_CODE_TOOLS
                if is_claude_code_tool and job:
                    job.claude_code_running = True

                try:
                    result, error = await execute_tool_call_async(tool_call)
                finally:
                    # Clear the flag after execution
                    if is_claude_code_tool and job:
                        job.claude_code_running = False
                tool_debug.append({
                    "tool_name": tool_call.tool_name,
                    "raw_args": tool_call.raw_args,
                    "result": result,
                    "error": error,
                })

                if error:
                    tool_results.append(f"[Tool error: {tool_call.tool_name}] {error}")
                elif result:
                    tool_results.append(f"[Tool: {tool_call.tool_name}] {result}")

                    # Get tool definition for channel_message
                    tool_def = AVAILABLE_TOOLS.get(tool_call.tool_name)

                    # Generate display message for specific tools
                    if tool_call.tool_name == "assistant":
                        args = parse_arguments(tool_call.raw_args)
                        task = args.get("task", args.get("request", ""))
                        # Show what wendy asked her assistant
                        display_msg = f'*asking assistant: "{task[:200]}{"..." if len(task) > 200 else ""}"*\n'
                        display_msg += f"```\n{result[:800]}{'...' if len(result) > 800 else ''}\n```"
                        display_messages.append(display_msg)

                        # Check if assistant generated any images in workspace and post them
                        if channel:
                            await self._post_assistant_images(channel, result)

                    elif tool_call.tool_name == "give_token":
                        # Show the token message with user mention
                        display_messages.append(f"*{result}*")
                    elif tool_def and tool_def.channel_message:
                        # Use tool's configured channel message (only add once)
                        if tool_def.channel_message not in display_messages:
                            display_messages.append(tool_def.channel_message)

            except Exception as exc:
                _LOG.exception("Tool execution failed: %s", tool_call)
                tool_results.append(f"[Tool error] {exc}")
                tool_debug.append({
                    "tool_name": tool_call.tool_name,
                    "raw_args": tool_call.raw_args,
                    "result": None,
                    "error": str(exc),
                })

        # Remove tool call markers from text (use the full_match from parsed calls)
        cleaned = text
        for tool_call in tool_calls:
            cleaned = cleaned.replace(tool_call.full_match, '')
        return cleaned.strip(), tool_results, display_messages, tool_debug

    async def _post_assistant_images(self, channel: discord.abc.Messageable, result: str) -> None:
        """Post any images generated by the assistant to Discord.

        Scans the assistant result for file paths in the workspace and posts them.
        """
        import io
        from pathlib import Path

        # Pattern to find image paths in assistant workspace
        workspace = Path(os.getenv("ASSISTANT_WORKSPACE", "/data/wendy_assistant"))
        image_pattern = re.compile(r'/data/wendy_assistant/[^\s\'"]+\.(?:png|jpg|jpeg|gif|webp)', re.IGNORECASE)
        matches = image_pattern.findall(result)

        for image_path_str in matches:
            image_path = Path(image_path_str)
            if image_path.exists() and image_path.is_file():
                try:
                    image_bytes = image_path.read_bytes()
                    timestamp = int(time.time())
                    filename = f"assistant_{timestamp}_{image_path.name}"
                    file = discord.File(io.BytesIO(image_bytes), filename=filename)
                    await channel.send(file=file)
                    _LOG.info("Posted assistant-generated image: %s", image_path)
                except Exception as exc:
                    _LOG.exception("Failed to post assistant image %s: %s", image_path, exc)

    # ==================== Model Preferences ====================

    def _is_valid_model(self, provider: str, model: str) -> bool:
        """Check if provider/model combination is available."""
        key = (provider.lower(), model)
        return key in self._model_lookup

    def _get_tapered_model(self) -> tuple[str, str]:
        """Select model using smooth daily taper algorithm.

        Taper thresholds (based on actual usage data):
        - Messages 0-25: Opus fades from 100% to 0%, Sonnet fills in
        - Messages 25-150: Sonnet fades from 100% to 0%, Haiku fills in
        - Messages 150+: 100% Haiku

        This saves ~40% on API costs while maintaining quality on quiet days.
        """
        from hollingsbot.prompt_db import get_daily_llm_call_count

        daily_count = get_daily_llm_call_count(exclude_haiku=True)

        # Taper breakpoints
        OPUS_END = 25      # Opus fades out by message 25
        SONNET_END = 150   # Sonnet fades out by message 150

        if daily_count < OPUS_END:
            # Phase 1: Opus fading to Sonnet
            # At 0 = 100% opus, at OPUS_END = 0% opus
            opus_prob = 1.0 - (daily_count / OPUS_END)
            roll = random.random()
            if roll < opus_prob:
                _LOG.info(
                    "Taper: msg #%d, phase=opus->sonnet, prob=%.0f%%, roll=%.2f -> opus",
                    daily_count, opus_prob * 100, roll
                )
                return ("anthropic", "claude-opus-4-5")
            _LOG.info(
                "Taper: msg #%d, phase=opus->sonnet, prob=%.0f%%, roll=%.2f -> sonnet",
                daily_count, opus_prob * 100, roll
            )
            return ("anthropic", "claude-sonnet-4-5-20250929")

        elif daily_count < SONNET_END:
            # Phase 2: Sonnet fading to Haiku
            # At OPUS_END = 100% sonnet, at SONNET_END = 0% sonnet
            sonnet_prob = 1.0 - ((daily_count - OPUS_END) / (SONNET_END - OPUS_END))
            roll = random.random()
            if roll < sonnet_prob:
                _LOG.info(
                    "Taper: msg #%d, phase=sonnet->haiku, prob=%.0f%%, roll=%.2f -> sonnet",
                    daily_count, sonnet_prob * 100, roll
                )
                return ("anthropic", "claude-sonnet-4-5-20250929")
            _LOG.info(
                "Taper: msg #%d, phase=sonnet->haiku, prob=%.0f%%, roll=%.2f -> haiku",
                daily_count, sonnet_prob * 100, roll
            )
            return ("anthropic", "claude-haiku-4-5")

        else:
            # Phase 3: All Haiku
            _LOG.info("Taper: msg #%d, phase=haiku-only -> haiku", daily_count)
            return ("anthropic", "claude-haiku-4-5")

    def _get_model_for_user(self, guild_id: int | None, user_id: int) -> tuple[str, str]:
        """Get user's preferred model or return default.

        Priority:
        1. User's explicit model preference (if set)
        2. Daily taper algorithm (automatic cost optimization)
        """
        gid = str(guild_id or 0)
        uid = str(user_id)
        entry = self.model_preferences.get(gid, {}).get(uid)
        if isinstance(entry, dict):
            provider = entry.get("provider")
            model = entry.get("model")
            if isinstance(provider, str) and isinstance(model, str) and self._is_valid_model(provider, model):
                return provider.lower(), model

        # No user preference - use daily taper
        return self._get_tapered_model()

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
        # These will be registered as methods on the bot instance
        # The coordinator or a separate command cog should handle actual command registration
        pass

    async def handle_model_command(self, ctx: commands.Context, provider: str | None = None, model: str | None = None) -> None:
        """Handle !model command to set user's preferred model."""
        if provider is None or model is None:
            # Show current model
            current_provider, current_model = self._get_model_for_user(
                getattr(ctx.guild, "id", None), ctx.author.id
            )
            await ctx.send(f"Current model: {current_provider}/{current_model}")
            return

        # Validate model
        if not self._is_valid_model(provider, model):
            await ctx.send(f"Invalid model: {provider}/{model}")
            return

        # Set model
        self._set_model_for_user(getattr(ctx.guild, "id", None), ctx.author.id, provider, model)
        await ctx.send(f"Model set to: {provider}/{model}")

    async def handle_system_command(self, ctx: commands.Context, *, prompt: str | None = None) -> None:
        """Handle !system command to set custom system prompt."""
        if prompt is None:
            # Show current system prompt
            user_prompt = self._get_user_system_prompt(
                getattr(ctx.guild, "id", None), ctx.author.id
            )
            if user_prompt:
                await ctx.send(f"Custom system prompt: {user_prompt[:500]}...")
            else:
                await ctx.send("Using default system prompt")
            return

        if prompt.lower() == "reset":
            # Clear custom system prompt
            self._clear_user_system_prompt(getattr(ctx.guild, "id", None), ctx.author.id)
            await ctx.send("Reset to default system prompt")
            return

        # Set custom system prompt
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


# Suppress import errors
from contextlib import suppress
