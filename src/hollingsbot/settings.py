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
