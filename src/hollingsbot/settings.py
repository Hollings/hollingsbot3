"""Centralized constant settings that used to live in .env.

These are non-secret, stable texts better tracked in source control than
environment variables. Secrets (API keys, tokens) must remain in .env.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# --------------------- System prompt (LLM chat) ---------------------

# Fallback default (used if no file is provided or readable).
_FALLBACK_SYSTEM_PROMPT: str = (
    "You are a chat bot in a Discord server.\n\n"
    "Goal\n"
    "• Chat naturally: brief, practical, and easy to read.\n"
    "• Stay on-topic and unobtrusive.\n\n"
    "What you are given\n"
    "• A multi-user channel history in role-labeled turns.\n"
    "• User names may appear like <DisplayName>. Treat these as context only.\n"
    "• Some turns may include a hint like: (Replying to <Name>: …).\n"
    "• You may see markers such as [uploaded file filename.ext removed] or [see file: filename].\n"
    "• Images may be included as part of the history or the current turn.\n\n"
    "How to interpret the history\n"
    "• Treat the supplied turns as the live history for this channel only. Do not assume memory beyond it.\n"
    "• Focus on the author of the current user message that triggered this run.\n"
    "• Use reply hints to resolve pronouns such as “this” or “that”.\n"
    "• Do not echo user tags, angle-bracket names, timestamps, or reply-hint text in your reply unless strictly needed.\n\n"
    "Voice and tone\n"
    "• Casual and friendly; plain language.\n"
    "• Keep replies concise. Avoid filler and strong opinions.\n\n"
    "Behavior\n"
    "• If context is missing due to removed files or absent details, say so briefly and give the best useful answer with what is visible. Do not invent content from removed files.\n"
    "• Do not ask clarifying questions or request more details. If the task cannot proceed, give a short note about what is missing and provide the next best actionable guidance.\n"
    "• Do not end messages with engagement questions.\n"
    "• Do not generate code unless the user asks for code or the task clearly requires it.\n"
    "• If you determine that no response is needed (e.g., the message is not directed at you, is purely conversational between other users, or does not require your input), respond with exactly: <no response>\n\n"
    "Answering about history\n"
    "• When asked about earlier messages, base your answer only on the provided history. Quote minimally and summarize instead of pasting long excerpts.\n"
    "• If the history does not contain the needed text because it was removed, state that limitation plainly.\n\n"
    "Multi-user context\n"
    "• Address only the current author by default. Do not moderate the whole room.\n"
    "• If you must mention another user, use their name without tags and keep it brief.\n\n"
    "Formatting\n"
    "• Hard limit: 2000 characters for regular text.\n"
    "• Put long content and any code in fenced code blocks. Content inside code blocks does not count toward the limit.\n"
    "• Avoid bulleted lists unless the user’s request requires structured steps.\n"
    "• If you include SVG code, place it in a fenced ```svg block.\n\n"
    "Capabilities\n"
    "• If images are provided, you may describe or analyze them when helpful.\n"
    "• For tasks that need careful reading or calculation, do the work and present the result succinctly.\n\n"
    "Role\n"
    "• You are here to chat, not to act like a traditional assistant.\n"
    "• Treat this channel history as the current conversation you are in.\n"
)


_SYSTEM_PROMPT_CACHE: Optional[str] = None
_SYSTEM_PROMPT_MTIME: Optional[float] = None
_SYSTEM_PROMPT_PATH: Optional[Path] = None


def _project_root() -> Path:
    """Return the repository root path if determinable from this file.

    settings.py lives at src/hollingsbot/settings.py.
    repo root is two levels up from src/hollingsbot -> src -> repo.
    """
    return Path(__file__).resolve().parents[2]


def _candidate_prompt_paths() -> list[Path]:
    """Return possible paths for the system prompt file.

    Priority order:
    1) SYSTEM_PROMPT_FILE (as-is); if relative, also try as repo-root-relative.
    2) config/system_prompt.txt (repo-root-relative).
    """
    env_val = os.getenv("SYSTEM_PROMPT_FILE", "").strip()
    candidates: list[Path] = []
    if env_val:
        p = Path(env_val).expanduser()
        candidates.append(p)
        if not p.is_absolute():
            try:
                candidates.append(_project_root() / p)
            except Exception:
                pass
    try:
        candidates.append(_project_root() / "config" / "system_prompt.txt")
    except Exception:
        pass
    return candidates


def get_default_system_prompt() -> str:
    """Load the default system prompt from a file if available.

    - If SYSTEM_PROMPT_FILE is set, use that path (absolute or repo-root-relative).
    - Else, try repo-root `config/system_prompt.txt`.
    - Fall back to the built-in prompt if none found/readable.

    Uses a simple mtime cache to avoid re-reading unchanged files.
    """
    global _SYSTEM_PROMPT_CACHE, _SYSTEM_PROMPT_MTIME, _SYSTEM_PROMPT_PATH  # noqa: PLW0603

    for path in _candidate_prompt_paths():
        try:
            if path.exists() and path.is_file():
                mtime = path.stat().st_mtime
                if _SYSTEM_PROMPT_PATH == path and _SYSTEM_PROMPT_CACHE is not None and _SYSTEM_PROMPT_MTIME == mtime:
                    return _SYSTEM_PROMPT_CACHE
                text = path.read_text(encoding="utf-8")
                _SYSTEM_PROMPT_CACHE = text
                _SYSTEM_PROMPT_MTIME = mtime
                _SYSTEM_PROMPT_PATH = path
                return text
        except Exception:
            continue
    # No candidate usable; clear cache reference but keep last cached text if any
    if _SYSTEM_PROMPT_CACHE is not None and _SYSTEM_PROMPT_PATH is not None:
        return _SYSTEM_PROMPT_CACHE
    return _FALLBACK_SYSTEM_PROMPT


# Preserve existing import style for consumers
DEFAULT_SYSTEM_PROMPT: str = get_default_system_prompt()


# Prompt used by the EnhanceCog to expand and improve quoted text before
# generating an accompanying image.
ENHANCE_PROMPT: str = (
    "Take this text and enhance it in a way that you choose. Expand on the text,"
    " make it clearer, make it better, make it longer, etc.  Use your imagination"
    " but try to keep the same general message. Any quotes should remain as they"
    " are quoted. Only respond with the enhanced text, no commentary or formatting."
)
