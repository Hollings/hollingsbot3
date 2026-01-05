"""Debug commands for inspecting bot state."""

import io
import json
import logging
import os
import re

import discord
from discord.ext import commands

_LOG = logging.getLogger(__name__)


class DebugCommands(commands.Cog):
    """Debug commands for development and troubleshooting."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.debug_channel_id = int(os.getenv("DEBUG_CHANNEL_ID", "1022951067593494589"))
        _LOG.info("DebugCommands cog initialized (debug_channel_id=%s)", self.debug_channel_id)

    @commands.command(name="dumphistory")
    async def dump_history(self, ctx: commands.Context, limit: int = 5):
        """Dump recent conversation history for this channel.

        Usage: !dumphistory [limit]
        Default limit is 5 messages.
        """
        # Get the chat coordinator
        coordinator = self.bot.get_cog("ChatCoordinator")
        if not coordinator:
            await ctx.send("ChatCoordinator not loaded")
            return

        channel_id = ctx.channel.id

        # Get history lock and read
        lock = coordinator._lock_for_channel(channel_id)
        async with lock:
            history = coordinator._history_for_channel(channel_id)
            recent = list(history)[-limit:]

        if not recent:
            await ctx.send("No history found for this channel")
            return

        # Format output
        lines = [f"**Last {len(recent)} messages in history:**\n"]
        for i, turn in enumerate(recent, 1):
            webhook_info = f" (webhook_id={turn.webhook_id})" if turn.webhook_id else ""
            content_preview = turn.content[:100].replace('\n', ' ')
            lines.append(
                f"{i}. **{turn.role}** | `{turn.author_name}`{webhook_info}\n"
                f"   {content_preview}{'...' if len(turn.content) > 100 else ''}\n"
            )

        output = "\n".join(lines)

        # Send in chunks if too long
        if len(output) > 2000:
            await ctx.send(output[:2000])
            await ctx.send(output[2000:])
        else:
            await ctx.send(output)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Listen for message forwards or links in the debug channel."""
        # Only process messages in the debug channel
        if message.channel.id != self.debug_channel_id:
            return

        # Ignore bot messages
        if message.author.bot:
            return

        message_id = None

        # Check if this is a message forward (snapshots)
        if hasattr(message, 'message_snapshots') and message.message_snapshots:
            try:
                # Get the first snapshot's message ID
                snapshot = message.message_snapshots[0]
                # MessageSnapshot has id directly
                if hasattr(snapshot, 'id'):
                    message_id = snapshot.id
                    _LOG.info("Detected message forward (snapshot): %s", message_id)
            except (IndexError, AttributeError) as exc:
                _LOG.warning("Failed to extract message ID from snapshot: %s", exc)

        # Check if this is a message reply
        if not message_id and message.reference and message.reference.message_id:
            message_id = message.reference.message_id
            _LOG.info("Detected message reply: %s", message_id)

        # Check if message contains a message link
        if not message_id and message.content:
            # Discord message link format: https://discord.com/channels/{guild_id}/{channel_id}/{message_id}
            link_match = re.search(
                r'https?://(?:ptb\.|canary\.)?discord(?:app)?\.com/channels/\d+/\d+/(\d+)',
                message.content,
            )
            if link_match:
                message_id = int(link_match.group(1))
                _LOG.info("Detected message link: %s", message_id)

        if not message_id:
            return

        # Get the ChatCoordinator to access WendyBot
        chat_coordinator = self.bot.get_cog("ChatCoordinator")
        if not chat_coordinator:
            await message.channel.send("ChatCoordinator not loaded")
            return

        # Try to get debug log from any bot
        log_data = None
        for bot in chat_coordinator.bots:
            if hasattr(bot, "get_response_log"):
                log_data = bot.get_response_log(message_id)
                if log_data:
                    bot_name = bot.__class__.__name__
                    _LOG.info(f"Found debug log in {bot_name}")
                    break

        if not log_data:
            await message.channel.send(f"No debug log found for message ID {message_id}")
            return

        # Format and send the debug files
        try:
            debug_file, io_file = self._format_debug_files(log_data)
            await message.channel.send(
                f"Debug logs for message {message_id}:",
                files=[debug_file, io_file],
            )
        except Exception as exc:
            _LOG.exception("Failed to format debug files: %s", exc)
            await message.channel.send(f"Failed to format debug logs: {exc}")

    def _truncate_base64_images(self, conversation: list) -> list:
        """Truncate base64 image data in conversation to prevent huge debug files."""
        import copy
        truncated = copy.deepcopy(conversation)
        for turn in truncated:
            if not isinstance(turn, dict):
                continue
            images = turn.get("images", [])
            if not isinstance(images, list):
                continue
            for img in images:
                if not isinstance(img, dict):
                    continue
                data_url = img.get("data_url", "")
                if isinstance(data_url, str) and ";base64," in data_url:
                    # Truncate: keep prefix and first 10 chars of base64 data
                    prefix, b64_data = data_url.split(";base64,", 1)
                    img["data_url"] = f"{prefix};base64,{b64_data[:10]}...[truncated]"
        return truncated

    def _format_debug_files(
        self, log_data: dict[str, object]
    ) -> tuple[discord.File, discord.File]:
        """Format debug data into two files: debug info and LLM I/O."""
        llm_debug = log_data.get("llm_debug", {})
        tool_debug = log_data.get("tool_debug", [])
        conversation = log_data.get("conversation", [])
        response_text = log_data.get("response_text", "")

        # Format debug info file
        debug_lines = ["=== LLM DEBUG INFO ===\n"]
        debug_lines.append(f"Provider: {llm_debug.get('provider', 'N/A')}")
        debug_lines.append(f"Model: {llm_debug.get('model', 'N/A')}")
        debug_lines.append(f"Status: {llm_debug.get('status', 'N/A')}")
        debug_lines.append(f"Duration: {llm_debug.get('duration', 'N/A')}s")
        debug_lines.append(f"Temperature: {llm_debug.get('temperature', 'N/A')}")

        # Daily taper info
        daily_count = llm_debug.get("daily_message_count")
        if daily_count is not None:
            debug_lines.append(f"\n=== DAILY TAPER ===")
            debug_lines.append(f"Daily Message Count: {daily_count}")
            debug_lines.append(f"Model Taper Phase: {llm_debug.get('model_taper_phase', 'N/A')}")
            debug_lines.append(f"Model Selected: {llm_debug.get('model_selected', 'N/A')}")
            l2_count = llm_debug.get('l2_summary_count', 'N/A')
            l2_max = llm_debug.get('l2_summary_max', 5)
            debug_lines.append(f"L2 Summaries: {l2_count}/{l2_max}")

        token_usage = llm_debug.get("token_usage")
        if token_usage:
            debug_lines.append(f"\nToken Usage: {json.dumps(token_usage, indent=2)}")

        error_msg = llm_debug.get("error_message")
        error_tb = llm_debug.get("error_traceback")
        if error_msg:
            debug_lines.append(f"\nError: {error_msg}")
        if error_tb:
            debug_lines.append(f"\nError Traceback:\n{error_tb}")

        if tool_debug:
            debug_lines.append("\n\n=== TOOL CALLS ===\n")
            for i, tool_call in enumerate(tool_debug, 1):
                debug_lines.append(f"\n--- Tool Call {i} ---")
                debug_lines.append(f"Tool: {tool_call.get('tool_name', 'N/A')}")
                debug_lines.append(f"Raw Args: {tool_call.get('raw_args', 'N/A')}")
                result = tool_call.get('result')
                error = tool_call.get('error')
                if error:
                    debug_lines.append(f"Error: {error}")
                elif result:
                    debug_lines.append(f"Result: {result}")

        debug_content = "\n".join(debug_lines)
        debug_file = discord.File(
            io.BytesIO(debug_content.encode("utf-8")),
            filename="debug.txt",
        )

        # Format LLM I/O file
        io_lines = []

        # Extract and display system prompt separately
        system_prompt = None
        if conversation and isinstance(conversation, list) and len(conversation) > 0:
            first_msg = conversation[0]
            if isinstance(first_msg, dict) and first_msg.get("role") == "system":
                system_prompt = first_msg.get("text", "")

        if system_prompt:
            io_lines.append("=== SYSTEM PROMPT ===\n")
            io_lines.append(system_prompt)
            io_lines.append("\n\n")

        io_lines.append("=== FULL LLM INPUT (JSON) ===\n")
        truncated_conversation = self._truncate_base64_images(conversation)
        io_lines.append(json.dumps(truncated_conversation, indent=2, ensure_ascii=False))
        io_lines.append("\n\n=== LLM OUTPUT ===\n")
        io_lines.append(response_text)

        io_content = "\n".join(io_lines)
        io_file = discord.File(
            io.BytesIO(io_content.encode("utf-8")),
            filename="llm_io.txt",
        )

        return debug_file, io_file


async def setup(bot: commands.Bot) -> None:
    """Load the debug commands cog."""
    await bot.add_cog(DebugCommands(bot))
