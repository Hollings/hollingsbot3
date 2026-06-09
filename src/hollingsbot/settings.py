"""Centralized constant settings that used to live in .env.

These are non-secret, stable texts better tracked in source control than
environment variables. Secrets (API keys, tokens) must remain in .env.
"""

from __future__ import annotations

import os

# --------------------- Admin user IDs ---------------------


def get_admin_user_ids() -> set[int]:
    """Get the set of admin user IDs from environment variable.

    Returns:
        Set of Discord user IDs that have admin privileges
    """
    env_ids = os.getenv("ADMIN_USER_IDS", "")
    return {int(uid.strip()) for uid in env_ids.split(",") if uid.strip().isdigit()}


# --------------------- Prompts ---------------------

# Prompt used by the EnhanceCog to expand and improve quoted text before
# generating an accompanying image.
ENHANCE_PROMPT: str = (
    "Take this text and enhance it in a way that you choose. Expand on the text,"
    " make it clearer, make it better, make it longer, etc.  Use your imagination"
    " but try to keep the same general message. Any quotes should remain as they"
    " are quoted. Only respond with the enhanced text, no commentary or formatting."
)
