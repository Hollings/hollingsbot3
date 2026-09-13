"""Post-processing for raw text completions before they go to Discord."""

from __future__ import annotations

from hollingsbot.cogs.chat_utils import DISCORD_MESSAGE_LIMIT


def trim_completion(full_text: str, prompt_len: int, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    """Fit a raw completion into one Discord message and drop the dangling last line.

    Raw (base-style) models stop at ``max_tokens`` mid-sentence, so the final
    line is almost always a fragment. Cut back to the last newline, but only
    one that lies *after* the prompt: a single-line continuation is kept whole,
    and the prompt itself is never eaten.

    ``full_text`` is ``prompt + completion`` and ``prompt_len`` is ``len(prompt)``.
    """
    text = full_text[:limit]
    cut = text.rfind("\n")
    if cut > prompt_len:
        text = text[:cut]
    return text.rstrip()
