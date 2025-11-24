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
from hollingsbot.prompt_db import (
    create_temp_bot,
    decrement_temp_bot_replies,
    delete_temp_bot,
    get_depleted_temp_bots,
    get_temp_bot_by_webhook_id,
    get_temp_bots_for_channel,
)
from hollingsbot.tasks import generate_llm_chat_response

_LOG = logging.getLogger(__name__)

# Configuration constants
MIN_REPLY_COUNT = 1
MAX_REPLY_COUNT = 20
MAX_PROMPT_LENGTH = 500
CLEANUP_INTERVAL_SECONDS = 60
RESPONSE_PROBABILITY = 0.5  # 50% chance temp bot responds

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

    def __init__(self):
        self.task: asyncio.Task | None = None
        self.result: Any = None


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
        remaining = decrement_temp_bot_replies(webhook_id)
        _LOG.info(f"Reserved reply slot for temp bot '{bot_name}', {remaining} remaining")
        if remaining < 0:
            _LOG.warning(f"Temp bot '{bot_name}' depleted during reservation")
            return None

        # Build current turn
        current_turn = self._extract_current_turn(message, history)
        if not current_turn:
            _LOG.warning("Failed to extract current turn from history")
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

        # Cancel any existing generation in this channel
        await self._cancel_generation(channel_id)

        # Generate and send response (wrapped in task for cancellation support)
        job = GenerationJob()
        task = self.bot.loop.create_task(
            self._generate_and_send(
                channel_id, webhook_id, bot_name, conversation, job, message.id
            )
        )
        job.task = task
        self._active_generations[channel_id] = job

        # Wait for generation to complete
        try:
            return await task
        except asyncio.CancelledError:
            _LOG.info(f"Temp bot '{bot_name}' generation cancelled")
            return None
        except Exception:
            _LOG.exception(f"Failed to generate response for temp bot '{bot_name}'")
            return None

    def _select_responding_bot(
        self, message: discord.Message, temp_bots: list[dict]
    ) -> dict | None:
        """Select which temp bot (if any) should respond to this message."""
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

        # Simple logic: Always pick a random temp bot to respond
        # TODO: Re-enable <no response> feature later
        # (They can decide themselves via <no response> if they don't want to)
        return random.choice(available_bots)

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

    def _build_temp_bot_system_prompt(self, bot_name: str, spawn_prompt: str, replies_remaining: int) -> str:
        """Build system prompt for temp bot with personality."""
        personality_suffix = (
            f"\n\nIMPORTANT: Your main purpose and personality trait is: {spawn_prompt}\n\n"
            f"You are {bot_name}, and this directive defines your core behavior and personality. "
            f"Embody this trait in all your responses.\n\n"
            f"You have {replies_remaining} message(s) remaining before you automatically despawn. "
            f"Use them wisely, or end early with !despawn if your purpose is fulfilled."
        )
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

    # ==================== Generation ====================

    async def _generate_and_send(
        self,
        channel_id: int,
        webhook_id: int,
        bot_name: str,
        conversation: list[dict[str, Any]],
        job: GenerationJob,
        request_message_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Generate temp bot response and send to Discord.

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
            else:
                # Check if bot is depleted and cleanup
                temp_bot_data = get_temp_bot_by_webhook_id(webhook_id)
                if temp_bot_data and temp_bot_data["replies_remaining"] <= 0:
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
            if self._active_generations.get(channel_id) is job:
                self._active_generations.pop(channel_id, None)

    async def _cancel_generation(self, channel_id: int) -> None:
        """Cancel any active temp bot generation in the channel.

        This cancels both the local asyncio task and the remote Celery task.
        Note: Anthropic API calls already in flight will complete server-side
        and consume tokens, but we won't wait for or use the response.
        """
        job = self._active_generations.get(channel_id)
        if not job:
            return

        if job.task and not job.task.done():
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

            self._active_generations.pop(channel_id, None)
            _LOG.info("Cancelled active temp bot generation in channel %s", channel_id)

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

    async def _cleanup_temp_bot(
        self, webhook_id: int, bot_name: str, send_depletion_message: bool = False
    ) -> None:
        """Clean up a depleted temp bot."""
        _LOG.info(f"Auto-despawning temp bot '{bot_name}' (depleted)")
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
        finally:
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
                try:
                    webhook = await self.bot.fetch_webhook(webhook_id)

                    # Send depletion message
                    try:
                        await webhook.send(
                            f"*[{bot_name} has depleted all replies and fades away]*",
                            username=bot_name,
                        )
                    except Exception:
                        _LOG.warning(f"Failed to send depletion message for {bot_name}")

                    await webhook.delete(reason="Temp bot depleted all replies")
                    _LOG.info(f"Deleted depleted webhook {webhook_id} (name: {bot_name})")
                except Exception:
                    _LOG.warning(f"Error cleaning up webhook {webhook_id}")
                finally:
                    delete_temp_bot(webhook_id)

        except Exception:
            _LOG.exception("Error in temp bot cleanup task")

    @cleanup_task.before_loop
    async def before_cleanup_task(self) -> None:
        """Wait for bot to be ready before starting cleanup loop."""
        await self.bot.wait_until_ready()

    # ==================== Commands ====================

    async def _generate_bot_name_from_topic(self, topic: str) -> str:
        """Generate an obscure, name-like identifier related to the topic using Claude Haiku."""
        try:
            name_prompt = (
                f"Generate a single obscure name (2-3 words) that is subtly related to this sentence:\n\n"
                f"```\n{topic}\n```\n\n"
                "The name should be:\n"
                "- Sound like a person, entity, or concept\n"
                "- Only tangentially related to the topic (not obvious)\n"
                "- Creative and unique\n"
                "- 2-3 words\n\n"
                "Respond with 2-3 sentences describing your thought process, then on a new line write ONLY the name you choose"
            )

            conversation = [
                {"role": "system", "text": "You generate creative, obscure names.", "images": []},
                {"role": "user", "text": name_prompt, "images": []}
            ]

            _LOG.info(f"Generating bot name for topic: {topic[:50]}...")

            async_result = generate_llm_chat_response.apply_async(
                ("anthropic", "claude-haiku-4-5", conversation),
                kwargs={"temperature": 1.0},
            )

            start = time.monotonic()
            while True:
                if async_result.ready():
                    break
                if (time.monotonic() - start) > 30:  # 30 second timeout for name generation
                    async_result.revoke(terminate=True)
                    _LOG.warning("Name generation timed out, using fallback")
                    return generate_bot_name()
                await asyncio.sleep(0.5)

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(async_result.get, timeout=0.1),
            )

            # Extract response text from result
            if isinstance(result, dict):
                response_text = str(result.get("text", "")).strip()
            else:
                response_text = str(result).strip()

            # Parse name from last line (take last 3 words max)
            if not response_text:
                _LOG.warning("Empty response from name generation, using fallback")
                return generate_bot_name()

            lines = response_text.strip().split('\n')
            last_line = lines[-1].strip()

            # Take only last 3 words from last line
            words = last_line.split()
            bot_name = ' '.join(words[-3:]) if len(words) >= 3 else last_line

            # Validate name
            if not bot_name or len(bot_name) > 50:
                _LOG.warning(f"Invalid generated name: '{bot_name}', using fallback")
                return generate_bot_name()

            _LOG.info(f"Generated bot name: {bot_name}")
            return bot_name

        except Exception:
            _LOG.exception("Failed to generate bot name from topic, using fallback")
            return generate_bot_name()

    async def handle_spawn_command(
        self,
        ctx: commands.Context,
        reply_count: int,
        *,
        initial_prompt: str,
    ) -> None:
        """Spawn a temporary LLM bot."""
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

        # Generate bot name using Claude Haiku
        bot_name = await self._generate_bot_name_from_topic(initial_prompt)

        # Check if there are already temp bots active in this channel
        existing_bots = get_temp_bots_for_channel(ctx.channel.id)

        # Temp bots should only see messages from their spawn point forward
        spawn_message_id = ctx.message.id
        if existing_bots:
            _LOG.info(f"Spawning additional temp bot '{bot_name}' with context from spawn point ({len(existing_bots)} bots already active)")
        else:
            _LOG.info(f"Spawning first temp bot '{bot_name}' with context from spawn point")

        try:
            # Create webhook
            webhook = await ctx.channel.create_webhook(
                name=f"TempBot-{bot_name}",
                reason=f"Temporary LLM bot spawned by {ctx.author}",
            )

            # Register in database with spawn_message_id
            create_temp_bot(
                channel_id=ctx.channel.id,
                webhook_id=webhook.id,
                name=bot_name,
                avatar_url=None,
                spawn_prompt=initial_prompt,
                replies_remaining=reply_count,
                spawn_message_id=spawn_message_id,
            )

            _LOG.info(
                f"Spawned temp bot '{bot_name}' (webhook_id={webhook.id}) "
                f"in channel {ctx.channel.id}, {reply_count} replies remaining, "
                f"spawn_message_id={spawn_message_id}"
            )

            # Send initial response (with arrival message prepended)
            arrival_msg = f"*[{bot_name} arrives for {reply_count} message{'s' if reply_count != 1 else ''}]*"
            await self._send_initial_response(ctx.channel, webhook.id, bot_name, initial_prompt, arrival_msg, spawn_message_id)

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
    ) -> None:
        """Send temp bot's initial response to spawn prompt with arrival message."""
        try:
            # Decrement reply count first
            remaining = decrement_temp_bot_replies(webhook_id)
            if remaining < 0:
                _LOG.warning(f"Temp bot '{bot_name}' depleted before initial response")
                return

            # Get current history (don't add spawn prompt to shared history)
            lock = self.coordinator._lock_for_channel(channel.id)
            async with lock:
                history = self.coordinator._history_for_channel(channel.id)
                snapshot = list(history)

            # Filter history to only include messages after spawn
            filtered_history = self._filter_history_after_spawn(snapshot, spawn_message_id)
            _LOG.info(f"Initial response filtered history: {len(snapshot)} turns -> {len(filtered_history)} turns (spawn_message_id={spawn_message_id})")

            # Build current turn for the spawn prompt (only for this generation)
            current_turn = ModelTurn(role="user", text=prompt, images=[])

            # Translate history
            translated_history = self._translate_history(filtered_history, webhook_id)

            # Build system prompt (with replies remaining)
            system_prompt = self._build_temp_bot_system_prompt(bot_name, prompt, remaining)

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
            sent_message = await webhook.send(full_response, username=bot_name, wait=True)

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
            elif remaining <= 0:
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
            except (discord.Forbidden, discord.HTTPException) as exc:
                _LOG.exception(f"Failed to despawn {bot_name}")
                failed.append(bot_name)

        # Send result
        if despawned:
            await ctx.send(f"Despawned: {', '.join(f'**{name}**' for name in despawned)}")
        if failed:
            await ctx.send(f"Failed to despawn: {', '.join(f'**{name}**' for name in failed)}")
