"""Centralized constant settings that used to live in .env.

These are non-secret, stable texts better tracked in source control than
environment variables. Secrets (API keys, tokens) must remain in .env.
"""

from __future__ import annotations

import os

# --------------------- ID parsing helpers ---------------------


def parse_id_set(raw: str | None) -> set[int]:
    """Parse a comma-separated string of Discord IDs into a set of ints.

    Non-numeric tokens and surrounding whitespace are ignored. Equivalent to the
    inline ``{int(x.strip()) for x in raw.split(",") if x.strip().isdigit()}``
    pattern that was duplicated across several modules.
    """
    if not raw:
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip().isdigit()}


# --------------------- Admin user IDs ---------------------


def get_admin_user_ids() -> set[int]:
    """Get the set of admin user IDs from environment variable.

    Returns:
        Set of Discord user IDs that have admin privileges
    """
    return parse_id_set(os.getenv("ADMIN_USER_IDS", ""))
