"""Feature request automation cog using Claude Code CLI."""

import asyncio
import io
import logging
import os
from pathlib import Path

import discord
from discord.ext import commands

from hollingsbot import prompt_db

_LOG = logging.getLogger(__name__)

# Environment configuration
FEATURE_REQUEST_CHANNEL_IDS = [
    int(x.strip())
    for x in os.getenv("FEATURE_REQUEST_CHANNEL_IDS", "").split(",")
    if x.strip()
]
CLAUDE_CODE_CLI_PATH = os.getenv("CLAUDE_CODE_CLI_PATH", "claude")


class FeatureRequests(commands.Cog):
    """Automated feature implementation using Claude Code CLI."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._allowed_channels = set(FEATURE_REQUEST_CHANNEL_IDS)
        self._pending_implementations: set[asyncio.Task] = set()
        _LOG.info(
            f"FeatureRequests cog initialized (allowed channels: {self._allowed_channels})"
        )

    def _is_allowed_channel(self, channel_id: int) -> bool:
        """Check if a channel is allowed for feature requests."""
        # If no channels configured, deny all
        if not self._allowed_channels:
            return False
        return channel_id in self._allowed_channels

    async def _run_claude_code(
        self,
        prompt: str,
        *,
        timeout: int = 600,
    ) -> tuple[str, bool]:
        """Run Claude Code CLI with the given prompt.

        Returns (output, success)
        """
        try:
            _LOG.info(f"Running Claude Code with prompt: {prompt[:200]}...")
            proc = await asyncio.create_subprocess_exec(
                CLAUDE_CODE_CLI_PATH,
                "-p",
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path.cwd()),
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                error_text = stderr.decode("utf-8", errors="replace")
                output += f"\n\n=== STDERR ===\n{error_text}"

            success = proc.returncode == 0
            _LOG.info(f"Claude Code completed with return code: {proc.returncode}")

            return output, success

        except asyncio.TimeoutError:
            _LOG.error(f"Claude Code timed out after {timeout}s")
            return f"Error: Claude Code timed out after {timeout}s", False
        except FileNotFoundError:
            _LOG.error("Claude Code CLI not found at: %s", CLAUDE_CODE_CLI_PATH)
            return (
                "Error: Claude Code CLI not found. Please ensure 'claude' is installed and accessible.",
                False,
            )
        except Exception as e:
            _LOG.exception("Failed to run Claude Code")
            return f"Error running Claude Code: {e}", False

    async def _auto_commit_git(self) -> tuple[str, bool]:
        """Auto-commit any dirty git state before implementation.

        Returns (message, success)
        """
        try:
            # Check git status
            proc = await asyncio.create_subprocess_exec(
                "git",
                "status",
                "--porcelain",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return f"Failed to check git status: {stderr.decode()}", False

            status_output = stdout.decode().strip()

            # If nothing to commit, we're good
            if not status_output:
                return "Git working directory is clean", True

            # Stage all changes
            proc = await asyncio.create_subprocess_exec(
                "git",
                "add",
                "-A",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if proc.returncode != 0:
                return "Failed to stage changes", False

            # Commit with auto-save message
            proc = await asyncio.create_subprocess_exec(
                "git",
                "commit",
                "-m",
                "WIP: auto-save before feature request",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                # Check if it failed because there was nothing to commit after staging
                # (this can happen with gitignored files)
                if b"nothing to commit" in stdout or b"nothing to commit" in stderr:
                    return "Git working directory is clean", True
                return f"Failed to commit changes: {stderr.decode()}", False

            return "Auto-committed dirty git state", True

        except Exception as e:
            _LOG.exception("Failed to auto-commit git")
            return f"Error: {e}", False

    @commands.command(name="request")
    async def request_feature(self, ctx: commands.Context, *, description: str) -> None:
        """Request automated feature implementation via Claude Code.

        Usage: !request <description of feature>

        Example: !request make the bot reply to messages that its responding to
        """
        # Check channel permissions
        if not self._is_allowed_channel(ctx.channel.id):
            await ctx.send(
                "This command can only be used in designated feature request channels."
            )
            return

        # Validate description
        if len(description.strip()) < 10:
            await ctx.send(
                "Please provide a more detailed description (at least 10 characters)."
            )
            return

        _LOG.info(
            f"Feature request from {ctx.author} ({ctx.author.id}): {description}"
        )

        # Create database record
        request_id = prompt_db.create_feature_request(
            channel_id=ctx.channel.id,
            message_id=ctx.message.id,
            author_id=ctx.author.id,
            request_description=description,
            status="pending_questions",
        )

        # Send initial status message
        status_msg = await ctx.send("Exploring codebase and preparing questions...")

        # Phase 1: Explore and ask questions
        phase1_prompt = f"""Explore the codebase to understand the current architecture and implementation.

Then, based on this feature request:
"{description}"

Ask 2-3 clarifying questions that would help you implement this feature correctly.
Keep the questions concise and focused on implementation details.

Output format:
1. [Question 1]
2. [Question 2]
3. [Question 3] (optional)
"""

        try:
            await status_msg.edit(content="Exploring codebase...")
            output, success = await self._run_claude_code(phase1_prompt, timeout=300)

            if not success:
                await status_msg.edit(
                    content="Failed to generate questions. See attached log."
                )
                await self._send_log_file(
                    ctx.channel, request_id, output, prefix="error"
                )
                prompt_db.update_feature_request(request_id, status="failed")
                return

            # Extract questions from output (try to find numbered list)
            questions = self._extract_questions(output)

            # Reply with questions
            questions_text = (
                f"**Feature Request #{request_id}**\n\n"
                f"I have some questions about your request:\n\n{questions}\n\n"
                f"Please reply to this message with your answers."
            )

            questions_msg = await ctx.send(questions_text)

            # Update database with questions message ID and conversation log
            prompt_db.update_feature_request(
                request_id,
                questions_message_id=questions_msg.id,
                conversation_log=output,
                status="awaiting_user_reply",
            )

            # Delete status message
            await status_msg.delete()

            _LOG.info(f"Feature request #{request_id}: Questions sent")

        except Exception as e:
            _LOG.exception(f"Error in Phase 1 for request #{request_id}")
            await status_msg.edit(
                content=f"An error occurred while processing your request: {e}"
            )
            prompt_db.update_feature_request(request_id, status="failed")

    def _extract_questions(self, output: str) -> str:
        """Extract questions from Claude Code output.

        Try to find a numbered list, otherwise return the last few paragraphs.
        """
        lines = output.strip().split("\n")

        # Look for numbered questions
        questions = []
        for line in lines:
            stripped = line.strip()
            # Match patterns like "1.", "2)", "1:", etc.
            if stripped and any(
                stripped.startswith(f"{i}{sep}")
                for i in range(1, 10)
                for sep in [".", ")", ":"]
            ):
                questions.append(stripped)

        if questions:
            return "\n".join(questions)

        # Fallback: return last 500 chars
        return output[-500:] if len(output) > 500 else output

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Listen for user replies to questions."""
        # Ignore bot messages
        if message.author.bot:
            return

        # Check if this is a reply
        if not message.reference:
            return

        # Get the replied-to message
        try:
            replied_msg_id = message.reference.message_id
            if replied_msg_id is None:
                return

            # Check if this is a reply to a questions message
            request = prompt_db.get_feature_request_by_questions_message_id(
                replied_msg_id
            )

            if not request:
                return

            # Verify the reply is from the original requester
            if request["author_id"] != message.author.id:
                await message.reply(
                    "Only the original requester can answer these questions.",
                    mention_author=False,
                )
                return

            # Verify status is awaiting reply
            if request["status"] != "awaiting_user_reply":
                await message.reply(
                    "This feature request is no longer awaiting a reply.",
                    mention_author=False,
                )
                return

            _LOG.info(
                f"Feature request #{request['id']}: Received user reply, starting implementation"
            )

            # Update status
            prompt_db.update_feature_request(request["id"], status="implementing")

            # Start implementation in background
            task = asyncio.create_task(
                self._implement_feature(request, message.content)
            )
            self._pending_implementations.add(task)
            task.add_done_callback(self._pending_implementations.discard)

        except Exception as e:
            _LOG.exception("Error handling reply to feature request questions")
            await message.reply(
                f"An error occurred while processing your reply: {e}",
                mention_author=False,
            )

    async def _implement_feature(
        self, request: dict, user_answers: str
    ) -> None:
        """Phase 2: Implement the feature with user answers."""
        request_id = request["id"]
        channel = self.bot.get_channel(request["channel_id"])

        if channel is None:
            _LOG.error(f"Could not find channel {request['channel_id']}")
            return

        try:
            # Send status message
            status_msg = await channel.send(
                f"**Feature Request #{request_id}**: Starting implementation..."
            )

            # Auto-commit git state
            await status_msg.edit(
                content=f"**Feature Request #{request_id}**: Checking git status..."
            )
            git_msg, git_success = await self._auto_commit_git()

            if not git_success:
                await status_msg.edit(
                    content=f"**Feature Request #{request_id}**: Failed to prepare git state: {git_msg}"
                )
                prompt_db.update_feature_request(request_id, status="failed")
                return

            _LOG.info(f"Feature request #{request_id}: {git_msg}")

            # Phase 2 prompt
            phase2_prompt = f"""Implement this feature request:

**Original request:** {request['request_description']}

**User answers to clarifying questions:**
{user_answers}

**Instructions:**
1. Explore the codebase to understand the current implementation
2. Implement the requested feature according to the user's answers
3. Follow existing code patterns and conventions
4. Test your implementation if possible
5. Provide a clear summary of what you implemented

Please proceed with the implementation.
"""

            # Update status
            await status_msg.edit(
                content=f"**Feature Request #{request_id}**: Implementing feature..."
            )

            # Run Claude Code with longer timeout for implementation
            full_log = request.get("conversation_log", "")
            full_log += "\n\n=== USER ANSWERS ===\n" + user_answers + "\n\n"
            full_log += "=== IMPLEMENTATION PHASE ===\n"

            output, success = await self._run_claude_code(phase2_prompt, timeout=900)
            full_log += output

            # Update conversation log
            prompt_db.update_feature_request(
                request_id, conversation_log=full_log
            )

            # Send results
            if success:
                await status_msg.edit(
                    content=f"**Feature Request #{request_id}**: Implementation complete!"
                )
                prompt_db.update_feature_request(request_id, status="completed")

                # Send conversation log
                await self._send_log_file(
                    channel, request_id, full_log, prefix="completed"
                )

                _LOG.info(f"Feature request #{request_id}: Implementation completed")
            else:
                await status_msg.edit(
                    content=f"**Feature Request #{request_id}**: Implementation failed. See attached log."
                )
                prompt_db.update_feature_request(request_id, status="failed")

                # Send partial log
                await self._send_log_file(channel, request_id, full_log, prefix="failed")

                _LOG.warning(f"Feature request #{request_id}: Implementation failed")

        except Exception as e:
            _LOG.exception(f"Error implementing feature request #{request_id}")
            try:
                await channel.send(
                    f"**Feature Request #{request_id}**: An error occurred during implementation: {e}"
                )
            except:
                pass
            prompt_db.update_feature_request(request_id, status="failed")

    async def _send_log_file(
        self, channel: discord.abc.Messageable, request_id: int, log: str, prefix: str = "log"
    ) -> None:
        """Send the conversation log as a text file."""
        try:
            # Create summary at the end
            summary = f"""
=== FEATURE REQUEST #{request_id} SUMMARY ===
Status: {prefix}
Total output length: {len(log)} characters

This file contains the complete conversation log between the system and Claude Code
for this automated feature request.
"""
            full_content = log + "\n\n" + summary

            filename = f"feature_request_{request_id}_{prefix}.txt"
            file = discord.File(
                io.BytesIO(full_content.encode("utf-8")), filename=filename
            )

            await channel.send(f"Conversation log for feature request #{request_id}:", file=file)
        except Exception as e:
            _LOG.exception("Failed to send log file")
            await channel.send(f"Failed to attach log file: {e}")


async def setup(bot: commands.Bot) -> None:
    """Load the FeatureRequests cog."""
    await bot.add_cog(FeatureRequests(bot))
