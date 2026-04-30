"""LLM-driven identity generation for temp bots.

Asks Claude (via the Celery text queue) to invent a name and an avatar prompt
that suit a given personality topic. Falls back to ``generate_bot_name`` if the
LLM call fails, times out, or returns something unusable.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time

from hollingsbot.tasks import generate_llm_chat_response

from .names import generate_bot_name

_LOG = logging.getLogger(__name__)

# How long to wait for the identity-generation Celery task before giving up.
_IDENTITY_TIMEOUT_SECONDS = 30
# Discord's username length limit is 80, but we constrain harder for sanity.
_MAX_GENERATED_NAME_LENGTH = 50

_IDENTITY_SYSTEM_PROMPT = "You generate creative character identities."

_IDENTITY_USER_PROMPT_TEMPLATE = (
    "Generate a name and avatar for a character based on this personality:\n\n"
    "```\n{topic}\n```\n\n"
    "Requirements:\n"
    "1. NAME: An obscure 2-3 word name that subtly relates to the topic. "
    "Can be Firstname Lastname style or modern username style. Keep it subtle, not nerdy or cringe. "
    "Do NOT use the word Meridian.\n\n"
    "2. AVATAR: A prompt for an AI image generator to create a profile picture for this character. "
    "Be creative and descriptive. The image should visually represent the character's essence.\n\n"
    "Format your response EXACTLY like this:\n"
    "NAME: [the name]\n"
    "AVATAR: [the image generation prompt]"
)


def _parse_identity_response(response_text: str) -> tuple[str | None, str | None]:
    """Pull NAME and AVATAR fields out of the LLM response."""
    bot_name: str | None = None
    avatar_prompt: str | None = None
    for raw_line in response_text.split("\n"):
        line = raw_line.strip()
        if line.upper().startswith("NAME:"):
            bot_name = line[5:].strip()
        elif line.upper().startswith("AVATAR:"):
            avatar_prompt = line[7:].strip()
    return bot_name, avatar_prompt


async def generate_bot_identity(topic: str) -> tuple[str, str | None]:
    """Generate a name and avatar prompt for a temp bot based on its purpose.

    Args:
        topic: The personality/purpose prompt for the bot.

    Returns:
        Tuple of ``(bot_name, avatar_prompt)``. ``bot_name`` always falls back to
        :func:`generate_bot_name` if anything goes wrong; ``avatar_prompt`` may be
        ``None`` if the LLM didn't supply one.
    """
    try:
        identity_prompt = _IDENTITY_USER_PROMPT_TEMPLATE.format(topic=topic)

        conversation = [
            {"role": "system", "text": _IDENTITY_SYSTEM_PROMPT, "images": []},
            {"role": "user", "text": identity_prompt, "images": []},
        ]

        _LOG.info("Generating bot identity for topic: %s...", topic[:50])

        async_result = generate_llm_chat_response.apply_async(
            ("anthropic", "claude-opus-4-5", conversation),
        )

        start = time.monotonic()
        while True:
            if async_result.ready():
                break
            if (time.monotonic() - start) > _IDENTITY_TIMEOUT_SECONDS:
                async_result.revoke(terminate=True)
                _LOG.warning("Identity generation timed out, using fallback")
                return generate_bot_name(), None
            await asyncio.sleep(0.5)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            functools.partial(async_result.get, timeout=0.1),
        )

        if isinstance(result, dict):
            response_text = str(result.get("text", "")).strip()
        else:
            response_text = str(result).strip()

        if not response_text:
            _LOG.warning("Empty response from identity generation, using fallback")
            return generate_bot_name(), None

        bot_name, avatar_prompt = _parse_identity_response(response_text)

        if not bot_name or len(bot_name) > _MAX_GENERATED_NAME_LENGTH:
            _LOG.warning("Invalid generated name: '%s', using fallback", bot_name)
            bot_name = generate_bot_name()

        _LOG.info(
            "Generated bot identity: name='%s', avatar_prompt=%s...",
            bot_name,
            avatar_prompt[:50] if avatar_prompt else None,
        )
        return bot_name, avatar_prompt

    except Exception:
        _LOG.exception("Failed to generate bot identity, using fallback")
        return generate_bot_name(), None
