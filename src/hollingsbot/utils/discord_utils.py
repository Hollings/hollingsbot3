"""Discord utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import discord


def get_display_name(user: discord.User | discord.Member) -> str:
    """Get user's display name with proper fallback hierarchy.

    Priority: server nickname > global display name > username

    Args:
        user: Discord user or member object

    Returns:
        The display name to use for this user
    """
    # For Member objects, check if they have a server nickname (nick)
    if hasattr(user, 'nick') and user.nick:
        return user.nick

    # Fall back to global display name (cross-server display name)
    if hasattr(user, 'global_name') and user.global_name:
        return user.global_name

    # Final fallback to username
    return user.name
