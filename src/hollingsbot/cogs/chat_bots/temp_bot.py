"""Temp bot system - manages temporary LLM bots that share conversation history."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
import re
import time
from contextlib import suppress
from typing import Any

import discord
from discord.ext import commands, tasks

from hollingsbot.cogs.conversation import ConversationTurn, ModelTurn
from hollingsbot.image_generators import generate_avatar
from hollingsbot.prompt_db import (
    DB_PATH,
    create_temp_bot,
    decrement_temp_bot_replies,
    increment_temp_bot_replies,
    delete_temp_bot,
    get_depleted_temp_bots,
    get_temp_bot_by_name,
    get_temp_bot_by_webhook_id,
    get_temp_bots_for_channel,
    get_temp_bot_previous_messages,
    get_messages_since_bot_left,
)
from hollingsbot.tasks import generate_llm_chat_response
from hollingsbot.text_generators import AnthropicTextGenerator

_LOG = logging.getLogger(__name__)

# Configuration constants
MIN_REPLY_COUNT = 1
MAX_REPLY_COUNT = 20
MAX_PROMPT_LENGTH = 500
CLEANUP_INTERVAL_SECONDS = 60
RESPONSE_PROBABILITY = 0.5  # 50% chance temp bot responds
MAX_TEMP_BOTS_PER_CHANNEL = 3  # Maximum concurrent temp bots per channel
MAX_CANCELLATIONS_BEFORE_FORCE = 2  # After this many cancellations, force the response through
MIN_GENERATION_TIME_BEFORE_CANCEL = 2.0  # Don't cancel if generation started less than this many seconds ago

# Name generation - occult, abstract names
_ADJECTIVES = [
    "Veiled", "Forgotten", "Obscured", "Liminal", "Spectral", "Cryptic", "Arcane",
    "Fractured", "Ephemeral", "Aberrant", "Eldritch", "Nameless", "Silent",
    "Distant", "Hollow", "Echoing", "Wandering", "Shrouded", "Hidden", "Fading"
]
_NOUNS = [
    "Cipher", "Sigil", "Phantom", "Revenant", "Threshold", "Abyss", "Echo",
    "Vestige", "Whisper", "Shadow", "Oracle", "Glyph", "Omen", "Ritual",
    "Fragment", "Veil", "Specter", "Herald", "Watcher", "Void"
]


def generate_bot_name() -> str:
    """Generate a random occult bot name."""
    return f"{random.choice(_ADJECTIVES)} {random.choice(_NOUNS)}"


class GenerationJob:
    """Tracks active temp bot generation state."""

    def __init__(self, webhook_id: int | None = None):
        self.task: asyncio.Task | None = None
        self.result: Any = None
        self.webhook_id: int | None = webhook_id  # For refunding on cancellation
        self.start_time: float = time.monotonic()  # When generation started


class TempBotManager:
    """Manages temporary LLM bots - spawning, responding, and cleanup."""

    def __init__(self, bot: commands.Bot, coordinator: Any, typing_tracker: Any):
        self.bot = bot
        self.coordinator = coordinator
        self.typing_tracker = typing_tracker

        # Configuration
        self.text_timeout = int(os.getenv("TEXT_TIMEOUT", "180"))
        self.max_turns_sent = int(os.getenv("LLM_MAX_TURNS_SENT", "8"))
        self.base_system_prompt = self._load_base_system_prompt()

        # Active generations (per channel)
        self._active_generations: dict[int, GenerationJob] = {}

        # Track webhook_ids with active generations (to prevent cleanup during generation)
        self._generating_webhooks: set[int] = set()

        # Track cancellation counts per channel (reset on successful post)
        self._cancellation_counts: dict[int, int] = {}

        # Debug log storage (message_id -> log data) - shared with WendyBot
        from pathlib import Path
        self._debug_logs_file = Path("generated/debug_logs.json")
        self._response_logs: dict[int, dict[str, Any]] = {}
        self._response_log_max_size = 100
        self._load_debug_logs()

        # Register commands
        self._register_commands()

        # Start cleanup task
        self.cleanup_task.start()

        _LOG.info("TempBotManager initialized")

    # ==================== Debug Log Storage ====================

    def _load_debug_logs(self) -> None:
        """Load debug logs from disk."""
        if not self._debug_logs_file.exists():
            _LOG.info("No existing temp bot debug logs file found, starting fresh")
            return

        try:
            import json
            with self._debug_logs_file.open("r") as f:
                data = json.load(f)

            # Convert string keys back to integers
            self._response_logs = {int(k): v for k, v in data.items()}
            _LOG.info("Loaded %d temp bot debug logs from disk", len(self._response_logs))
        except Exception as exc:
            _LOG.exception("Failed to load temp bot debug logs: %s", exc)
            self._response_logs = {}

    def _save_debug_logs(self) -> None:
        """Save debug logs to disk."""
        try:
            import json
            self._debug_logs_file.parent.mkdir(exist_ok=True)

            # Convert integer keys to strings for JSON serialization
            data = {str(k): v for k, v in self._response_logs.items()}

            with self._debug_logs_file.open("w") as f:
                json.dump(data, f, indent=2)

            _LOG.debug("Saved %d temp bot debug logs to disk", len(self._response_logs))
        except Exception as exc:
            _LOG.exception("Failed to save temp bot debug logs: %s", exc)

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

    # ==================== System Prompt ====================

    def _load_base_system_prompt(self) -> str:
        """Load base system prompt for temp bots."""
        try:
            # Use separate temp bot system prompt (no "Wendy" references)
            from pathlib import Path
            temp_bot_prompt_file = Path("config/temp_bot_system_prompt.txt")
            if temp_bot_prompt_file.exists():
                return temp_bot_prompt_file.read_text(encoding="utf-8").strip()
            else:
                _LOG.warning("temp_bot_system_prompt.txt not found, using default")
                return "You are a helpful AI assistant in a Discord chat."
        except Exception:
            _LOG.exception("Failed to load temp bot system prompt")
            return "You are a helpful AI assistant."

    def _register_commands(self) -> None:
        """Register temp bot commands."""
        # Commands will be registered separately
        pass

    # ==================== Message Handling ====================

    async def receive_message(
        self,
        message: discord.Message,
        history: list[ConversationTurn],
    ) -> dict[str, Any] | None:
        """Receive message event and potentially respond as a temp bot.

        Returns:
            dict with keys: message_id, text, webhook_id, bot_name
            Or None if no temp bot responds
        """
        _LOG.info(f"TempBotManager received message from {message.author}: {message.content[:50]}...")

        channel_id = message.channel.id
        temp_bots = get_temp_bots_for_channel(channel_id)

        if not temp_bots:
            # No temp bots in this channel
            _LOG.info("No temp bots in this channel")
            return None

        # Determine if we should respond and which bot
        selected_bot = self._select_responding_bot(message, temp_bots)
        if not selected_bot:
            _LOG.info("No temp bot selected to respond")
            return None

        webhook_id = selected_bot["webhook_id"]
        bot_name = selected_bot["name"]
        spawn_prompt = selected_bot["spawn_prompt"]
        spawn_message_id = selected_bot.get("spawn_message_id")

        _LOG.info(f"Temp bot '{bot_name}' will respond to message in channel {channel_id}")

        # Atomically reserve a reply slot before generation
        remaining, should_cleanup = decrement_temp_bot_replies(webhook_id)
        _LOG.info(f"Reserved reply slot for temp bot '{bot_name}', {remaining} remaining, should_cleanup={should_cleanup}")
        if remaining < 0:
            _LOG.warning(f"Temp bot '{bot_name}' depleted during reservation")
            return None

        # Mark this webhook as actively generating (prevents cleanup during generation)
        self._generating_webhooks.add(webhook_id)

        try:
            # Build current turn
            current_turn = self._extract_current_turn(message, history)
            if not current_turn:
                _LOG.warning("Failed to extract current turn from history")
                self._generating_webhooks.discard(webhook_id)
                return None

            # Filter history to only include messages after spawn
            filtered_history = self._filter_history_after_spawn(history[:-1], spawn_message_id)
            _LOG.info(f"Filtered history: {len(history[:-1])} turns -> {len(filtered_history)} turns (spawn_message_id={spawn_message_id})")

            # Translate history to this temp bot's perspective
            translated_history = self._translate_history(filtered_history, webhook_id)

            # Build temp bot system prompt (with replies remaining)
            temp_bot_system_prompt = self._build_temp_bot_system_prompt(bot_name, spawn_prompt, remaining)

            # Build conversation payload
            conversation = self._build_conversation_payload(
                translated_history, current_turn, temp_bot_system_prompt
            )

            # Cancel any existing generation in this channel (may be skipped if limits hit)
            was_cancelled = await self._cancel_generation(channel_id)

            # If there's still an active generation that wasn't cancelled, wait for it
            if not was_cancelled and channel_id in self._active_generations:
                _LOG.info("Existing generation in channel %s was not cancelled, skipping new generation", channel_id)
                # Refund since we reserved but won't generate
                increment_temp_bot_replies(webhook_id)
                self._generating_webhooks.discard(webhook_id)
                return None

            # Generate and send response (wrapped in task for cancellation support)
            job = GenerationJob(webhook_id=webhook_id)
            task = self.bot.loop.create_task(
                self._generate_and_send(
                    channel_id, webhook_id, bot_name, conversation, job, message.id, should_cleanup
                )
            )
            job.task = task
            self._active_generations[channel_id] = job

            # Wait for generation to complete
            # Note: _generate_and_send handles removing from _generating_webhooks in its finally block
            return await task
        except asyncio.CancelledError:
            _LOG.info(f"Temp bot '{bot_name}' generation cancelled")
            self._generating_webhooks.discard(webhook_id)
            return None
        except Exception:
            _LOG.exception(f"Failed to generate response for temp bot '{bot_name}'")
            self._generating_webhooks.discard(webhook_id)
            return None

    def _select_responding_bot(
        self, message: discord.Message, temp_bots: list[dict]
    ) -> dict | None:
        """Select which temp bot (if any) should respond to this message.

        Uses RESPONSE_PROBABILITY to determine if each bot wants to respond.
        """
        # Exclude the temp bot that just spoke (if message is from a temp bot)
        available_bots = temp_bots
        if message.author.bot and message.webhook_id:
            speaking_bot_webhook_id = message.webhook_id
            speaking_bot_name = next(
                (b["name"] for b in temp_bots if b["webhook_id"] == speaking_bot_webhook_id),
                "Unknown"
            )
            available_bots = [bot for bot in temp_bots if bot["webhook_id"] != speaking_bot_webhook_id]
            _LOG.info(
                f"Temp bot '{speaking_bot_name}' just spoke, excluded from response. "
                f"{len(available_bots)} temp bots available"
            )

        if not available_bots:
            return None

        # Roll probability for each bot independently
        willing_bots = [bot for bot in available_bots if random.random() < RESPONSE_PROBABILITY]

        if not willing_bots:
            # At least one temp bot must respond - force pick one
            _LOG.info(f"All {len(available_bots)} temp bots declined probability roll, forcing one to respond")
            willing_bots = [random.choice(available_bots)]

        # Pick a random bot from those willing to respond
        selected = random.choice(willing_bots)
        _LOG.info(f"{len(willing_bots)}/{len(available_bots)} bots willing to respond, selected '{selected['name']}'")
        return selected

    def _extract_current_turn(
        self, message: discord.Message, history: list[ConversationTurn]
    ) -> ModelTurn | None:
        """Extract current turn from history."""
        if not history:
            return None

        last_turn = history[-1]
        if last_turn.message_id != message.id:
            _LOG.warning("Last turn in history doesn't match current message")
            return None

        return ModelTurn(role=last_turn.role, text=last_turn.content, images=last_turn.images)

    # ==================== History Management ====================

    def _filter_history_after_spawn(
        self, history: list[ConversationTurn], spawn_message_id: int | None
    ) -> list[ConversationTurn]:
        """Filter history to only include messages after spawn_message_id.

        If spawn_message_id is None, return full history (bot joining existing conversation).
        """
        if spawn_message_id is None:
            # No spawn message ID means bot joins with full context
            return history

        # Find index of spawn message
        spawn_index = None
        for i, turn in enumerate(history):
            if turn.message_id == spawn_message_id:
                spawn_index = i
                break

        if spawn_index is None:
            # Spawn message not in history (e.g., !spawn commands are filtered out)
            # Filter by message ID: return all messages that came after spawn chronologically
            _LOG.info(f"Spawn message {spawn_message_id} not found in history, filtering by message ID")
            return [turn for turn in history if turn.message_id and turn.message_id > spawn_message_id]

        # Return only messages after the spawn message
        return history[spawn_index + 1:]

    # ==================== History Translation ====================

    def _translate_history(
        self, history: list[ConversationTurn], responding_webhook_id: int
    ) -> list[ConversationTurn]:
        """Translate raw history to this temp bot's perspective.

        Temp bot sees:
        - Own past messages (webhook_id matches) → "assistant"
        - All other messages (users, main bot, other temp bots) → "user"
        """
        translated = []
        for turn in history:
            # Check if this message is from this specific temp bot
            if turn.webhook_id == responding_webhook_id:
                # This is from current temp bot → "assistant"
                role = "assistant"
            else:
                # Everything else (users, main bot, other temp bots) → "user"
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

    def _build_temp_bot_system_prompt(
        self,
        bot_name: str,
        spawn_prompt: str,
        replies_remaining: int,
        recall_context: dict | None = None,
    ) -> str:
        """Build system prompt for temp bot with personality.

        Args:
            bot_name: The bot's display name
            spawn_prompt: The bot's personality/purpose
            replies_remaining: Number of replies remaining
            recall_context: Optional dict with keys:
                - previous_messages: list of bot's last messages
                - messages_missed: count of messages since bot left
                - conversation_summary: previous session summary
        """
        # Simple, clean personality section
        personality_suffix = (
            f"\n\n---\n"
            f"Your name: {bot_name}\n"
            f"Your personality: {spawn_prompt}\n"
            f"---\n\n"
        )

        # Add recall context if this is a returning bot
        if recall_context:
            messages_missed = recall_context.get("messages_missed", 0)
            previous_messages = recall_context.get("previous_messages", [])
            conversation_summary = recall_context.get("conversation_summary")

            personality_suffix += f"You've returned after being away ({messages_missed} messages were sent while you were gone).\n"

            if conversation_summary:
                personality_suffix += f"What you remember: {conversation_summary}\n"

            if previous_messages:
                personality_suffix += "Your last messages before you left:\n"
                for msg in previous_messages:
                    content = msg.get("content", "")
                    # Strip the <Author>: prefix if present
                    if content.startswith(f"<{bot_name}>:"):
                        content = content[len(f"<{bot_name}>:"):].strip()
                    # Truncate long messages
                    if len(content) > 200:
                        content = content[:200] + "..."
                    personality_suffix += f"- {content}\n"
            personality_suffix += "\n"

        personality_suffix += f"You have {replies_remaining} message(s) left before you automatically leave."

        return self.base_system_prompt + personality_suffix

    def _build_conversation_payload(
        self,
        history: list[ConversationTurn],
        current_turn: ModelTurn,
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """Build conversation payload for temp bot."""
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
            # Strip display name prefix from assistant messages
            # (LLM shouldn't see its own past messages with prefixes)
            text = turn.content
            if turn.role == "assistant":
                text = self._strip_display_name_prefix(text)

            # Strip arrival announcements from all messages
            # (Other bots shouldn't see "[BotName arrives for N messages]")
            text = self._strip_arrival_announcement(text)

            # Skip empty messages (might be arrival-only messages)
            if not text.strip():
                continue

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

    def _strip_display_name_prefix(self, text: str) -> str:
        """Strip display name prefix from text (e.g., '<DisplayName>: text' -> 'text')."""
        # Match pattern: <anything>: text
        match = re.match(r'^<[^>]+>:\s*', text)
        if match:
            return text[match.end():]
        return text

    def _strip_arrival_announcement(self, text: str) -> str:
        """Strip arrival/departure announcements from text.

        Strips:
        - '*[BotName arrives for N message(s)]*'
        - '*[BotName departs]*'
        - '*[BotName has depleted all replies and fades away]*'

        Returns the text with announcements removed, or empty string if
        the entire message was just an announcement.
        """
        # Pattern: *[SomeName arrives for N message(s)]*
        # Using .+? (non-greedy) to match bot name without consuming "arrives for"
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

    async def _build_turn_from_message(self, message: discord.Message) -> ConversationTurn | None:
        """Build a ConversationTurn from a Discord message."""
        try:
            from hollingsbot.cogs import chat_utils
            from hollingsbot.utils.discord_utils import get_display_name

            display = get_display_name(message.author)
            base_text = chat_utils.clean_mentions(message, self.bot).strip()

            # Collect images
            images = await chat_utils.collect_image_attachments(message)

            # Build content
            content = f"<{display}>: {base_text}"

            return ConversationTurn(
                role="user",
                content=content,
                images=images,
                message_id=message.id,
                author_id=message.author.id,
                author_name=display,
                webhook_id=message.webhook_id,
            )
        except Exception:
            _LOG.exception("Failed to build turn from message")
            return None

    # ==================== Generation ====================

    async def _generate_and_send(
        self,
        channel_id: int,
        webhook_id: int,
        bot_name: str,
        conversation: list[dict[str, Any]],
        job: GenerationJob,
        request_message_id: int | None = None,
        should_cleanup: bool = False,
    ) -> dict[str, Any] | None:
        """Generate temp bot response and send to Discord.

        Args:
            should_cleanup: If True, cleanup bot after this response (determined atomically at decrement time)

        Returns dict with message_id, text, webhook_id, bot_name or None on failure.
        """
        llm_debug: dict[str, Any] = {}
        response_text = ""
        stored_conversation = conversation.copy()

        try:
            _LOG.info(f"Temp bot '{bot_name}' starting generation...")

            # Store Celery async_result in job for cancellation
            response_text, llm_debug, stored_conversation = await self._generate_response_with_job(conversation, job)
            _LOG.info(f"Temp bot '{bot_name}' generated response: {response_text[:100]}...")

            if not response_text or not response_text.strip():
                _LOG.warning(f"Empty response from temp bot '{bot_name}'")
                return None

            # Check if bot wants to self-despawn
            should_despawn = "!despawn" in response_text.lower()
            if should_despawn:
                # Strip !despawn from the response
                response_text = response_text.replace("!despawn", "").replace("!DESPAWN", "").strip()
                _LOG.info(f"Temp bot '{bot_name}' requested self-despawn")

            # Wait for human to finish typing (if applicable)
            await self._wait_for_typing_to_clear(channel_id)

            # Send via webhook
            _LOG.info(f"Temp bot '{bot_name}' sending message via webhook {webhook_id}")
            webhook = await self.bot.fetch_webhook(webhook_id)
            sent_message = await webhook.send(response_text, username=bot_name, wait=True)
            _LOG.info(f"Temp bot '{bot_name}' sent message successfully")

            # Reset cancellation count on successful post
            self._cancellation_counts.pop(channel_id, None)

            # Store debug log for the response
            self._store_response_log(
                sent_message.id,
                stored_conversation,
                response_text,
                llm_debug,
                [],  # temp bots don't have tool calls
            )

            # Also store for request message if provided
            if request_message_id:
                self._store_response_log(
                    request_message_id,
                    stored_conversation,
                    response_text,
                    llm_debug,
                    [],
                )

            # Handle self-despawn or auto-cleanup
            if should_despawn:
                _LOG.info(f"Temp bot '{bot_name}' self-despawning")
                # Send departure message for self-despawn too
                try:
                    webhook = await self.bot.fetch_webhook(webhook_id)
                    await webhook.send(
                        f"*[{bot_name} departs]*",
                        username=bot_name,
                    )
                except Exception:
                    _LOG.warning(f"Failed to send self-despawn message for {bot_name}")
                await self._cleanup_temp_bot(webhook_id, bot_name, send_depletion_message=False)
            elif should_cleanup:
                # Bot depleted - cleanup (determined atomically at decrement time, no re-query needed)
                _LOG.info(f"Temp bot '{bot_name}' depleted, cleaning up")
                await self._cleanup_temp_bot(webhook_id, bot_name, send_depletion_message=True)

            return {
                "message_id": sent_message.id,
                "text": response_text,
                "webhook_id": webhook_id,
                "bot_name": bot_name,
            }

        except asyncio.CancelledError:
            # Store debug log even for cancelled generations (if we got a response)
            if response_text and request_message_id:
                llm_debug["status"] = "cancelled"
                self._store_response_log(
                    request_message_id,
                    stored_conversation,
                    response_text,
                    llm_debug,
                    [],
                )
                _LOG.info("Saved debug log for cancelled temp bot generation (message %s)", request_message_id)
            raise
        except Exception:
            _LOG.exception(f"Failed to generate response for temp bot '{bot_name}'")
            return None
        finally:
            # Remove from generating set (allows cleanup to proceed)
            self._generating_webhooks.discard(webhook_id)
            if self._active_generations.get(channel_id) is job:
                self._active_generations.pop(channel_id, None)

    async def _cancel_generation(self, channel_id: int) -> bool:
        """Cancel any active temp bot generation in the channel.

        This cancels both the local asyncio task and the remote Celery task.
        Note: Anthropic API calls already in flight will complete server-side
        and consume tokens, but we won't wait for or use the response.

        Returns:
            True if generation was cancelled, False if cancellation was skipped
            (due to hitting cancellation limit or minimum generation time).
        """
        job = self._active_generations.get(channel_id)
        if not job:
            return False

        if not job.task or job.task.done():
            return False

        # Check if we've hit the cancellation limit - if so, let it finish
        cancellation_count = self._cancellation_counts.get(channel_id, 0)
        if cancellation_count >= MAX_CANCELLATIONS_BEFORE_FORCE:
            _LOG.info(
                "Channel %s has %d cancellations, forcing response through (limit: %d)",
                channel_id, cancellation_count, MAX_CANCELLATIONS_BEFORE_FORCE
            )
            return False

        # Check if generation just started - give it time to complete
        elapsed = time.monotonic() - job.start_time
        if elapsed < MIN_GENERATION_TIME_BEFORE_CANCEL:
            _LOG.info(
                "Channel %s generation only %.1fs old, skipping cancel (min: %.1fs)",
                channel_id, elapsed, MIN_GENERATION_TIME_BEFORE_CANCEL
            )
            return False

        # Proceed with cancellation
        # Cancel the local asyncio task first
        job.task.cancel()

        # Revoke the Celery task immediately (don't wait for local task)
        # terminate=True sends SIGTERM to worker, stopping the API call ASAP
        if job.result:
            job.result.revoke(terminate=True)
            _LOG.info("Revoked temp bot Celery task for channel %s (terminate=True)", channel_id)

        # Now wait for local task cleanup
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(job.task, timeout=0.5)

        # Refund the reply count since the message wasn't sent
        if job.webhook_id:
            new_count = increment_temp_bot_replies(job.webhook_id)
            _LOG.info(
                "Refunded reply for cancelled generation (webhook %s), now has %d replies",
                job.webhook_id, new_count
            )

        # Increment cancellation count for this channel
        self._cancellation_counts[channel_id] = cancellation_count + 1

        self._active_generations.pop(channel_id, None)
        _LOG.info(
            "Cancelled active temp bot generation in channel %s (cancellation #%d)",
            channel_id, self._cancellation_counts[channel_id]
        )
        return True

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

    async def _generate_response_with_job(
        self, conversation: list[dict[str, Any]], job: GenerationJob
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        """Generate LLM response via Celery, storing async_result in job for cancellation.

        Returns (text, llm_debug, conversation).
        """
        provider = "anthropic"
        model = "claude-haiku-4-5"

        # Debug logging
        _LOG.info(f"Temp bot calling Celery with {len(conversation)} turns")
        for i, turn in enumerate(conversation):
            text_preview = turn.get('text', '')[:100].replace('\n', '\\n')
            _LOG.info(f"Turn {i}: role={turn.get('role')}, text_preview={text_preview}, images={len(turn.get('images', []))}")

        # Check for role alternation issues
        for i in range(1, len(conversation)):
            if conversation[i]['role'] == conversation[i-1]['role'] and conversation[i]['role'] != 'system':
                _LOG.warning(f"Role alternation issue: turns {i-1} and {i} both have role={conversation[i]['role']}")

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
            llm_debug = result.get("debug", {})
            stored_conversation = result.get("conversation", conversation)
        else:
            text = str(result)
            llm_debug = {}
            stored_conversation = conversation

        # Log for debugging
        if not text or not text.strip():
            _LOG.warning(f"Empty text from Celery: result={result}")

        return text, llm_debug, stored_conversation

    async def _generate_response(self, conversation: list[dict[str, Any]]) -> str:
        """Generate LLM response via Celery using Claude Haiku."""
        provider = "anthropic"
        model = "claude-haiku-4-5"

        # Debug logging
        _LOG.info(f"Temp bot calling Celery with {len(conversation)} turns")
        for i, turn in enumerate(conversation):
            text_preview = turn.get('text', '')[:100].replace('\n', '\\n')
            _LOG.info(f"Turn {i}: role={turn.get('role')}, text_preview={text_preview}, images={len(turn.get('images', []))}")

        # Check for role alternation issues
        for i in range(1, len(conversation)):
            if conversation[i]['role'] == conversation[i-1]['role'] and conversation[i]['role'] != 'system':
                _LOG.warning(f"Role alternation issue: turns {i-1} and {i} both have role={conversation[i]['role']}")

        async_result = generate_llm_chat_response.apply_async(
            (provider, model, conversation),
            kwargs={"temperature": 1.0},
        )

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

        # Extract text from result
        if isinstance(result, dict):
            text = str(result.get("text", ""))
        else:
            text = str(result)

        # Log for debugging
        if not text or not text.strip():
            _LOG.warning(f"Empty text from Celery: result={result}")

        return text

    # ==================== Cleanup ====================

    async def _generate_conversation_summary(
        self, channel_id: int, bot_name: str, spawn_prompt: str
    ) -> str | None:
        """Generate a conversation summary for a temp bot using Haiku.

        Args:
            channel_id: The channel where the bot was active
            bot_name: The bot's display name
            spawn_prompt: The bot's original personality/purpose

        Returns:
            Generated summary string, or None if generation failed
        """
        import sqlite3

        try:
            # Get conversation messages from cached_messages
            with sqlite3.connect(DB_PATH) as conn:
                # Find the bot's first and last message timestamps
                cur = conn.execute(
                    """
                    SELECT MIN(timestamp), MAX(timestamp)
                    FROM cached_messages
                    WHERE channel_id = ? AND author_name = ?
                    """,
                    (channel_id, bot_name),
                )
                row = cur.fetchone()
                if not row or row[0] is None:
                    _LOG.warning(f"No messages found for bot '{bot_name}' in channel {channel_id}")
                    return None

                first_ts, last_ts = row

                # Get all messages in that time range
                cur = conn.execute(
                    """
                    SELECT author_name, content
                    FROM cached_messages
                    WHERE channel_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                    """,
                    (channel_id, first_ts, last_ts),
                )
                messages = cur.fetchall()

            if not messages:
                return None

            # Format conversation
            lines = []
            for author, content in messages:
                if not content:
                    continue
                # Strip the <Author>: prefix if present
                if content.startswith(f"<{author}>:"):
                    content = content[len(f"<{author}>:"):].strip()
                # Truncate long messages
                if len(content) > 500:
                    content = content[:500] + "..."
                # Mark the temp bot's messages
                if author == bot_name:
                    lines.append(f"[{bot_name}]: {content}")
                else:
                    lines.append(f"{author}: {content}")

            conversation = "\n".join(lines)
            if len(conversation) < 50:
                _LOG.info(f"Conversation too short for bot '{bot_name}', skipping summary")
                return None

            # Truncate if too long
            if len(conversation) > 8000:
                conversation = conversation[:8000] + "\n... (truncated)"

            # Generate summary using Haiku
            prompt = f"""Summarize this Discord conversation involving a temporary bot named "{bot_name}".

The bot was spawned with this personality/purpose: "{spawn_prompt}"

Conversation:
{conversation}

Write a 3-5 sentence summary of what happened in this conversation. Focus on:
- What the bot's personality/character was like
- Key interactions or memorable moments
- How the conversation ended

Be concise and capture the essence of the bot's time in the chat."""

            generator = AnthropicTextGenerator(model="claude-haiku-4-5")
            summary = await generator.generate(prompt, temperature=0.7)
            return summary.strip() if summary else None

        except Exception:
            _LOG.exception(f"Failed to generate summary for bot '{bot_name}'")
            return None

    async def _save_conversation_summary(
        self, webhook_id: int, summary: str, append: bool = False
    ) -> None:
        """Save or append a conversation summary to the temp_bots table.

        Args:
            webhook_id: The bot's webhook ID
            summary: The summary text to save
            append: If True, append to existing summary (for recalled bots)
        """
        import sqlite3

        try:
            with sqlite3.connect(DB_PATH) as conn:
                if append:
                    # Get existing summary and append
                    cur = conn.execute(
                        "SELECT conversation_summary FROM temp_bots WHERE webhook_id = ?",
                        (webhook_id,),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        summary = f"{row[0]}\n\n---\n\n**Recalled session:**\n{summary}"

                conn.execute(
                    "UPDATE temp_bots SET conversation_summary = ? WHERE webhook_id = ?",
                    (summary, webhook_id),
                )
                conn.commit()
                _LOG.info(f"Saved conversation summary for webhook {webhook_id}")
        except Exception:
            _LOG.exception(f"Failed to save summary for webhook {webhook_id}")

    async def _cleanup_temp_bot(
        self,
        webhook_id: int,
        bot_name: str,
        send_depletion_message: bool = False,
        is_recall: bool = False,
    ) -> None:
        """Clean up a depleted temp bot.

        Args:
            webhook_id: The bot's webhook ID
            bot_name: The bot's display name
            send_depletion_message: Whether to send a departure message
            is_recall: If True, this bot was recalled (append to existing summary)
        """
        _LOG.info(f"Auto-despawning temp bot '{bot_name}' (depleted)")

        # Get bot info for summary generation before cleanup
        bot_info = get_temp_bot_by_webhook_id(webhook_id)
        channel_id = bot_info.get("channel_id") if bot_info else None
        spawn_prompt = bot_info.get("spawn_prompt", "") if bot_info else ""

        try:
            webhook = await self.bot.fetch_webhook(webhook_id)

            # Send depletion message if requested
            if send_depletion_message:
                try:
                    await webhook.send(
                        f"*[{bot_name} has depleted all replies and fades away]*",
                        username=bot_name,
                    )
                except Exception:
                    _LOG.warning(f"Failed to send depletion message for {bot_name}")

            await webhook.delete(reason="Temp bot depleted all replies")
            _LOG.info(f"Auto-despawned temp bot '{bot_name}'")
        except (discord.NotFound, discord.Forbidden):
            _LOG.warning(f"Failed to delete webhook {webhook_id}, removing from DB anyway")

        # Generate and save conversation summary before marking inactive
        if channel_id:
            _LOG.info(f"Generating conversation summary for '{bot_name}'...")
            summary = await self._generate_conversation_summary(
                channel_id, bot_name, spawn_prompt
            )
            if summary:
                await self._save_conversation_summary(webhook_id, summary, append=is_recall)

        # Mark bot as inactive (soft delete)
        delete_temp_bot(webhook_id)

    @tasks.loop(seconds=CLEANUP_INTERVAL_SECONDS)
    async def cleanup_task(self) -> None:
        """Periodically clean up depleted temp bots."""
        try:
            depleted_bots = get_depleted_temp_bots()
            if not depleted_bots:
                return

            _LOG.info(f"Found {len(depleted_bots)} depleted temp bot(s), cleaning up...")

            for bot_info in depleted_bots:
                webhook_id = bot_info["webhook_id"]
                bot_name = bot_info.get("name", "Unknown")

                # Skip bots that are actively generating a response
                if webhook_id in self._generating_webhooks:
                    _LOG.info(f"Skipping cleanup for '{bot_name}' - generation in progress")
                    continue

                # Check if this was a recalled bot (has existing summary)
                is_recall = bool(bot_info.get("conversation_summary"))
                await self._cleanup_temp_bot(
                    webhook_id, bot_name, send_depletion_message=True, is_recall=is_recall
                )

        except Exception:
            _LOG.exception("Error in temp bot cleanup task")

    @cleanup_task.before_loop
    async def before_cleanup_task(self) -> None:
        """Wait for bot to be ready before starting cleanup loop."""
        await self.bot.wait_until_ready()

    # ==================== Commands ====================

    async def _generate_bot_identity(self, topic: str) -> tuple[str, str | None]:
        """Generate a name and avatar prompt for a temp bot based on its purpose.

        Args:
            topic: The personality/purpose prompt for the bot

        Returns:
            Tuple of (bot_name, avatar_prompt). avatar_prompt may be None on failure.
        """
        try:
            identity_prompt = (
                f"Generate a name and avatar for a character based on this personality:\n\n"
                f"```\n{topic}\n```\n\n"
                "Requirements:\n"
                "1. NAME: An obscure 2-3 word name that subtly relates to the topic. "
                "Can be Firstname Lastname style or modern username style. Keep it subtle, not nerdy or cringe. "
                "Do NOT use the word Meridian.\n\n"
                "2. AVATAR: A prompt for an AI image generator to create a profile picture for this character. "
                "Be creative and descriptive. The image should visually represent the character's essence.\n\n"
                "Format your response EXACTLY like this:\n"
                "NAME: [the name]\n"
                "AVATAR: [the image generation prompt]"
            )

            conversation = [
                {"role": "system", "text": "You generate creative character identities.", "images": []},
                {"role": "user", "text": identity_prompt, "images": []}
            ]

            _LOG.info(f"Generating bot identity for topic: {topic[:50]}...")

            async_result = generate_llm_chat_response.apply_async(
                ("anthropic", "claude-opus-4-5", conversation),
            )

            start = time.monotonic()
            while True:
                if async_result.ready():
                    break
                if (time.monotonic() - start) > 30:
                    async_result.revoke(terminate=True)
                    _LOG.warning("Identity generation timed out, using fallback")
                    return generate_bot_name(), None
                await asyncio.sleep(0.5)

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(async_result.get, timeout=0.1),
            )

            if isinstance(result, dict):
                response_text = str(result.get("text", "")).strip()
            else:
                response_text = str(result).strip()

            if not response_text:
                _LOG.warning("Empty response from identity generation, using fallback")
                return generate_bot_name(), None

            # Parse NAME and AVATAR from response
            bot_name = None
            avatar_prompt = None

            for line in response_text.split('\n'):
                line = line.strip()
                if line.upper().startswith('NAME:'):
                    bot_name = line[5:].strip()
                elif line.upper().startswith('AVATAR:'):
                    avatar_prompt = line[7:].strip()

            # Validate name
            if not bot_name or len(bot_name) > 50:
                _LOG.warning(f"Invalid generated name: '{bot_name}', using fallback")
                bot_name = generate_bot_name()

            _LOG.info(f"Generated bot identity: name='{bot_name}', avatar_prompt={avatar_prompt[:50] if avatar_prompt else None}...")
            return bot_name, avatar_prompt

        except Exception:
            _LOG.exception("Failed to generate bot identity, using fallback")
            return generate_bot_name(), None

    async def handle_spawn_command(
        self,
        ctx: commands.Context,
        reply_count: int,
        *,
        initial_prompt: str,
        reply_message: discord.Message | None = None,
        include_context: bool = False,
    ) -> None:
        """Spawn a temporary LLM bot.

        Args:
            ctx: Command context
            reply_count: Number of replies before auto-despawn
            initial_prompt: The spawn prompt/personality
            reply_message: Message being replied to (if spawn used as reply)
            include_context: Whether to include previous 5 messages as context
        """
        # Validate
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send("This command only works in text channels.")
            return

        if reply_count < MIN_REPLY_COUNT:
            await ctx.send(f"Reply count must be at least {MIN_REPLY_COUNT}.")
            return

        if reply_count > MAX_REPLY_COUNT:
            await ctx.send(f"Reply count cannot exceed {MAX_REPLY_COUNT}.")
            return

        if not initial_prompt:
            await ctx.send("Initial prompt cannot be empty.")
            return

        if len(initial_prompt) > MAX_PROMPT_LENGTH:
            await ctx.send(f"Initial prompt must not exceed {MAX_PROMPT_LENGTH} characters.")
            return

        # Check temp bot limit for this channel
        existing_bots = get_temp_bots_for_channel(ctx.channel.id)
        if len(existing_bots) >= MAX_TEMP_BOTS_PER_CHANNEL:
            bot_names = ", ".join(f"**{b['name']}**" for b in existing_bots)
            await ctx.send(
                f"This channel already has {MAX_TEMP_BOTS_PER_CHANNEL} temp bots active: {bot_names}\n"
                f"Use `!despawn` to remove some before spawning more."
            )
            return

        # Check if the prompt matches an existing bot name - if so, recall them instead
        existing_bot = get_temp_bot_by_name(initial_prompt.strip(), channel_id=ctx.channel.id)
        if existing_bot and not existing_bot.get("is_active"):
            _LOG.info(f"Prompt matches existing bot '{existing_bot['name']}', recalling instead of creating new")
            await self.handle_recall_command(ctx, reply_count, bot_data=existing_bot)
            return

        # Generate bot name and avatar prompt together
        bot_name, avatar_prompt = await self._generate_bot_identity(initial_prompt)

        # Temp bots should only see messages from their spawn point forward
        spawn_message_id = ctx.message.id
        if existing_bots:
            _LOG.info(f"Spawning additional temp bot '{bot_name}' with context from spawn point ({len(existing_bots)} bots already active)")
        else:
            _LOG.info(f"Spawning first temp bot '{bot_name}' with context from spawn point")

        # Build initial context from reply and/or previous messages
        initial_context: list[ConversationTurn] = []

        # Get history for context building
        lock = self.coordinator._lock_for_channel(ctx.channel.id)
        async with lock:
            history = self.coordinator._history_for_channel(ctx.channel.id)
            history_snapshot = list(history)

        # Include previous 5 messages if -context flag used
        if include_context and history_snapshot:
            context_messages = history_snapshot[-5:]
            initial_context.extend(context_messages)
            _LOG.info(f"Including {len(context_messages)} previous messages as context for temp bot")

        # Include replied-to message if spawned as reply
        if reply_message:
            reply_turn = await self._build_turn_from_message(reply_message)
            if reply_turn:
                # Avoid duplicates if reply is already in context
                if not any(t.message_id == reply_turn.message_id for t in initial_context):
                    initial_context.insert(0, reply_turn)
                    _LOG.info(f"Including replied-to message {reply_message.id} as initial context")

        try:
            # Generate avatar image from the AI-generated prompt
            avatar_bytes = None
            if avatar_prompt:
                _LOG.info(f"Generating avatar for '{bot_name}' with prompt: {avatar_prompt[:80]}...")
                avatar_bytes = await generate_avatar(avatar_prompt)
                if avatar_bytes:
                    _LOG.info(f"Avatar generated for '{bot_name}' ({len(avatar_bytes)} bytes)")
                else:
                    _LOG.warning(f"Failed to generate avatar for '{bot_name}'")

            # Create webhook with avatar if available
            webhook = await ctx.channel.create_webhook(
                name=f"TempBot-{bot_name}",
                avatar=avatar_bytes,
                reason=f"Temporary LLM bot spawned by {ctx.author}",
            )

            # Register in database with spawn_message_id and avatar_bytes
            create_temp_bot(
                channel_id=ctx.channel.id,
                webhook_id=webhook.id,
                name=bot_name,
                avatar_url=None,
                spawn_prompt=initial_prompt,
                replies_remaining=reply_count,
                spawn_message_id=spawn_message_id,
                avatar_bytes=avatar_bytes,
            )

            _LOG.info(
                f"Spawned temp bot '{bot_name}' (webhook_id={webhook.id}) "
                f"in channel {ctx.channel.id}, {reply_count} replies remaining, "
                f"spawn_message_id={spawn_message_id}, initial_context={len(initial_context)} messages"
            )

            # Send initial response (with arrival message prepended)
            arrival_msg = f"*[{bot_name} arrives for {reply_count} message{'s' if reply_count != 1 else ''}]*"
            await self._send_initial_response(
                ctx.channel,
                webhook.id,
                bot_name,
                initial_prompt,
                arrival_msg,
                spawn_message_id,
                initial_context=initial_context,
            )

        except discord.Forbidden:
            await ctx.send("I don't have permission to create webhooks in this channel.")
        except discord.HTTPException as exc:
            _LOG.exception("Failed to create webhook")
            await ctx.send(f"Failed to create webhook: {exc}")

    async def _send_initial_response(
        self,
        channel: discord.TextChannel,
        webhook_id: int,
        bot_name: str,
        prompt: str,
        arrival_msg: str,
        spawn_message_id: int | None,
        initial_context: list[ConversationTurn] | None = None,
        recall_context: dict | None = None,
    ) -> None:
        """Send temp bot's initial response to spawn prompt with arrival message.

        Args:
            channel: Discord channel to send to
            webhook_id: Webhook ID for this temp bot
            bot_name: Name of the temp bot
            prompt: The spawn prompt/personality
            arrival_msg: The arrival announcement message
            spawn_message_id: Message ID where the bot was spawned
            initial_context: Optional list of context messages (from reply or -context flag)
            recall_context: Optional dict with recall info (previous_messages, messages_missed, conversation_summary)
        """
        try:
            # Decrement reply count first (atomic operation)
            remaining, should_cleanup = decrement_temp_bot_replies(webhook_id)
            if remaining < 0:
                _LOG.warning(f"Temp bot '{bot_name}' depleted before initial response")
                return

            # Use initial_context if provided, otherwise empty
            context_history = initial_context or []
            _LOG.info(f"Initial response using {len(context_history)} context messages")

            # Build current turn for the spawn prompt (only for this generation)
            current_turn = ModelTurn(role="user", text=prompt, images=[])

            # Translate history (context messages are all "user" from temp bot's perspective)
            translated_history = self._translate_history(context_history, webhook_id)

            # Build system prompt (with replies remaining and recall context)
            system_prompt = self._build_temp_bot_system_prompt(
                bot_name, prompt, remaining, recall_context=recall_context
            )

            # Build conversation
            conversation = self._build_conversation_payload(
                translated_history, current_turn, system_prompt
            )

            # Generate response
            response_text = await self._generate_response(conversation)
            if not response_text or not response_text.strip():
                _LOG.error("Empty initial response for temp bot")
                return

            # Check if bot wants to self-despawn in first message
            should_despawn = "!despawn" in response_text.lower()
            if should_despawn:
                # Strip !despawn from the response
                response_text = response_text.replace("!despawn", "").replace("!DESPAWN", "").strip()
                _LOG.info(f"Temp bot '{bot_name}' requested self-despawn in initial response")

            # Prepend arrival message to response
            full_response = f"{arrival_msg}\n\n{response_text}"

            # Send via webhook (coordinator will add to history automatically via on_message)
            webhook = await self.bot.fetch_webhook(webhook_id)
            await webhook.send(full_response, username=bot_name, wait=True)

            _LOG.info(f"Temp bot '{bot_name}' sent initial response ({remaining} replies remaining)")

            # Handle self-despawn or auto-cleanup
            if should_despawn:
                _LOG.info(f"Temp bot '{bot_name}' self-despawning after initial response")
                # Send departure message for self-despawn
                try:
                    await webhook.send(
                        f"*[{bot_name} departs]*",
                        username=bot_name,
                    )
                except Exception:
                    _LOG.warning(f"Failed to send self-despawn message for {bot_name}")
                await self._cleanup_temp_bot(webhook_id, bot_name, send_depletion_message=False)
            elif should_cleanup:
                # Bot depleted (determined atomically at decrement time)
                await self._cleanup_temp_bot(webhook_id, bot_name, send_depletion_message=True)

        except Exception:
            _LOG.exception(f"Failed to send initial response for temp bot '{bot_name}'")

    async def handle_despawn_command(self, ctx: commands.Context, name: str | None = None) -> None:
        """Manually remove temporary bots."""
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send("This command only works in text channels.")
            return

        temp_bots = get_temp_bots_for_channel(ctx.channel.id)

        if not temp_bots:
            await ctx.send("No temporary bots active in this channel.")
            return

        # Despawn all temp bots
        despawned = []
        failed = []

        for bot in temp_bots:
            webhook_id = bot["webhook_id"]
            bot_name = bot["name"]

            try:
                webhook = await self.bot.fetch_webhook(webhook_id)
                await webhook.delete(reason=f"Manual despawn by {ctx.author}")
                delete_temp_bot(webhook_id)
                despawned.append(bot_name)
                _LOG.info(
                    f"Despawned temp bot '{bot_name}' (webhook_id={webhook_id}) "
                    f"from channel {ctx.channel.id}"
                )
            except discord.NotFound:
                delete_temp_bot(webhook_id)
                despawned.append(bot_name)
            except (discord.Forbidden, discord.HTTPException):
                _LOG.exception(f"Failed to despawn {bot_name}")
                failed.append(bot_name)

        # Send result
        if despawned:
            await ctx.send(f"Despawned: {', '.join(f'**{name}**' for name in despawned)}")
        if failed:
            await ctx.send(f"Failed to despawn: {', '.join(f'**{name}**' for name in failed)}")

    async def handle_recall_command(
        self,
        ctx: commands.Context,
        reply_count: int,
        *,
        bot_data: dict,
    ) -> None:
        """Recall a previously spawned temp bot.

        Args:
            ctx: Command context
            reply_count: Number of replies before auto-despawn
            bot_data: The historical bot data from the database
        """
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send("This command only works in text channels.")
            return

        if reply_count < MIN_REPLY_COUNT:
            await ctx.send(f"Reply count must be at least {MIN_REPLY_COUNT}.")
            return

        if reply_count > MAX_REPLY_COUNT:
            await ctx.send(f"Reply count cannot exceed {MAX_REPLY_COUNT}.")
            return

        bot_name = bot_data["name"]
        spawn_prompt = bot_data["spawn_prompt"]
        avatar_bytes = bot_data.get("avatar_bytes")
        conversation_summary = bot_data.get("conversation_summary")

        # Temp bots should only see messages from their spawn point forward
        spawn_message_id = ctx.message.id

        # Build recall context - their previous messages and how many they missed
        previous_messages = get_temp_bot_previous_messages(ctx.channel.id, bot_name, limit=5)
        messages_missed = get_messages_since_bot_left(ctx.channel.id, bot_name)

        recall_context = {
            "previous_messages": previous_messages,
            "messages_missed": messages_missed,
            "conversation_summary": conversation_summary,
        }

        _LOG.info(
            f"Recalling temp bot '{bot_name}' with {reply_count} replies "
            f"(missed {messages_missed} messages, {len(previous_messages)} previous messages)"
        )

        try:
            # Create webhook with stored avatar if available
            webhook = await ctx.channel.create_webhook(
                name=f"TempBot-{bot_name}",
                avatar=avatar_bytes,
                reason=f"Temp bot recalled by {ctx.author}",
            )

            # Register in database
            create_temp_bot(
                channel_id=ctx.channel.id,
                webhook_id=webhook.id,
                name=bot_name,
                avatar_url=None,
                spawn_prompt=spawn_prompt,
                replies_remaining=reply_count,
                spawn_message_id=spawn_message_id,
                avatar_bytes=avatar_bytes,
            )

            _LOG.info(
                f"Recalled temp bot '{bot_name}' (webhook_id={webhook.id}) "
                f"in channel {ctx.channel.id}, {reply_count} replies"
            )

            # Send initial response (with return message and recall context)
            arrival_msg = f"*[{bot_name} returns for {reply_count} message{'s' if reply_count != 1 else ''}]*"
            await self._send_initial_response(
                ctx.channel,
                webhook.id,
                bot_name,
                spawn_prompt,
                arrival_msg,
                spawn_message_id,
                initial_context=None,
                recall_context=recall_context,
            )

        except discord.Forbidden:
            await ctx.send("I don't have permission to create webhooks in this channel.")
        except discord.HTTPException as exc:
            _LOG.exception("Failed to create webhook for recall")
            await ctx.send(f"Failed to recall bot: {exc}")
