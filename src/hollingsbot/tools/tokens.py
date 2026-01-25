"""Token tool for giving tokens to users."""

from __future__ import annotations

import logging
import random
import re

from hollingsbot.prompt_db import (
    get_user_token_balance,
    give_user_token,
    resolve_user_by_display_name,
)

from .parser import get_current_context

_LOG = logging.getLogger(__name__)

# Message templates for giving tokens. {user} = mention
_TOKEN_MESSAGES = [
    "A token for {user}!",
    "{user} earned a token!",
    "Token for {user}.",
    "{user} gets a token!",
    "One token for {user}!",
    "{user}, here's a token.",
    "Token awarded to {user}!",
    "{user} receives a token!",
]


def _parse_user_identifier(user: str) -> tuple[int | None, str]:
    """Parse a user identifier and return (user_id, identifier_type).

    Accepts:
        - Discord mention: <@123456> or <@!123456>
        - Raw user ID: 123456
        - Display name: SomeName

    Returns:
        (user_id, "mention" | "id" | "name") or (None, "unknown") if parsing fails
    """
    user = user.strip()

    # Check for mention format: <@123456> or <@!123456>
    mention_match = re.match(r"^<@!?(\d+)>$", user)
    if mention_match:
        return int(mention_match.group(1)), "mention"

    # Check for raw numeric ID
    if user.isdigit():
        return int(user), "id"

    # Assume it's a display name
    return None, "name"


def give_token(user: str) -> str:
    """Give one token to a user.

    Args:
        user: User identifier - can be:
            - Discord mention (@username or <@123456>)
            - User ID (numeric)
            - Display name (will search message history)

    Returns:
        Result message indicating success or failure
    """
    user_id, id_type = _parse_user_identifier(user)
    context = get_current_context()

    # If display name, try to resolve it
    if id_type == "name":
        channel_id = context.get("channel_id")
        user_id = resolve_user_by_display_name(user, channel_id)

    # If user is invalid, credit to Wendy herself
    if user_id is None:
        bot_user_id = context.get("bot_user_id")
        if bot_user_id:
            try:
                new_balance = give_user_token(bot_user_id)
                _LOG.info("Invalid user '%s', gave token to Wendy instead (balance: %d)", user, new_balance)
                return f"Couldn't find '{user}', so Wendy gets to have it. She now has {new_balance} token(s)."
            except Exception as exc:
                _LOG.exception("Failed to give token to Wendy")
                return f"Failed to give token: {exc}"
        return f"Invalid user identifier: {user}"

    # Give the token
    try:
        new_balance = give_user_token(user_id)
        _LOG.info("Gave token to user %d, new balance: %d", user_id, new_balance)
        message = random.choice(_TOKEN_MESSAGES).format(user=f"<@{user_id}>")
        return message
    except Exception as exc:
        _LOG.exception("Failed to give token to user %d", user_id)
        return f"Failed to give token: {exc}"


def check_tokens(user: str) -> str:
    """Check a user's token balance.

    Args:
        user: User identifier - can be mention, ID, or display name

    Returns:
        Result message with the user's token balance
    """
    user_id, id_type = _parse_user_identifier(user)

    # If display name, try to resolve it
    if id_type == "name":
        context = get_current_context()
        channel_id = context.get("channel_id")
        user_id = resolve_user_by_display_name(user, channel_id)

        if user_id is None:
            return f"Could not find user '{user}' in message history."

    if user_id is None:
        return f"Invalid user identifier: {user}"

    try:
        balance = get_user_token_balance(user_id)
        return f"<@{user_id}> has {balance} token(s)."
    except Exception as exc:
        _LOG.exception("Failed to check tokens for user %d", user_id)
        return f"Failed to check tokens: {exc}"
