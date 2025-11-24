"""Track typing indicators to enable typing-aware bot responses."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import discord

_LOG = logging.getLogger(__name__)


class TypingTracker:
    """Tracks user typing indicators across channels.

    This allows bots to wait for users to finish typing before sending responses,
    preventing bots from interrupting users who are composing messages.
    """

    def __init__(self):
        """Initialize the typing tracker."""
        # Maps channel_id -> (user_id, timestamp)
        self._typing_state: Dict[int, Tuple[int, datetime]] = {}
        # Typing indicators last ~10 seconds on Discord
        self._typing_timeout = timedelta(seconds=10)

    async def on_typing(
        self, channel: discord.abc.Messageable, user: discord.User | discord.Member, when: datetime
    ) -> None:
        """Handle typing event from Discord.

        Args:
            channel: The channel where typing occurred
            user: The user who is typing
            when: When the typing started
        """
        channel_id = channel.id
        self._typing_state[channel_id] = (user.id, when)
        _LOG.debug(
            "User %s (%d) started typing in channel %d at %s",
            user.name,
            user.id,
            channel_id,
            when,
        )

    def is_human_typing(self, channel_id: int, bot_user_id: int) -> bool:
        """Check if a human (non-bot) user is currently typing in the channel.

        Args:
            channel_id: The channel to check
            bot_user_id: The bot's user ID (to exclude from typing checks)

        Returns:
            True if a human is actively typing (within last 10 seconds), False otherwise
        """
        if channel_id not in self._typing_state:
            return False

        user_id, timestamp = self._typing_state[channel_id]

        # Ignore typing from the bot itself
        if user_id == bot_user_id:
            return False

        # Check if typing state is recent (within timeout)
        now = datetime.now(timezone.utc)
        age = now - timestamp
        is_recent = age < self._typing_timeout

        if not is_recent:
            # Clean up stale state
            del self._typing_state[channel_id]
            _LOG.debug("Expired stale typing state for channel %d (age: %s)", channel_id, age)

        return is_recent

    def clear_channel(self, channel_id: int) -> None:
        """Clear typing state for a channel.

        Useful when a message is sent to reset the typing indicator.

        Args:
            channel_id: The channel to clear
        """
        if channel_id in self._typing_state:
            del self._typing_state[channel_id]
            _LOG.debug("Cleared typing state for channel %d", channel_id)
