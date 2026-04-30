"""Random name and departure phrase helpers for temp bots.

Pure functions — no side effects, no I/O. The fallback name generator is used
when the LLM-based identity generator (see ``identity.py``) fails or times out.
"""

from __future__ import annotations

import random

# Name generation - occult, abstract names
_ADJECTIVES = [
    "Veiled",
    "Forgotten",
    "Obscured",
    "Liminal",
    "Spectral",
    "Cryptic",
    "Arcane",
    "Fractured",
    "Ephemeral",
    "Aberrant",
    "Eldritch",
    "Nameless",
    "Silent",
    "Distant",
    "Hollow",
    "Echoing",
    "Wandering",
    "Shrouded",
    "Hidden",
    "Fading",
]

_NOUNS = [
    "Cipher",
    "Sigil",
    "Phantom",
    "Revenant",
    "Threshold",
    "Abyss",
    "Echo",
    "Vestige",
    "Whisper",
    "Shadow",
    "Oracle",
    "Glyph",
    "Omen",
    "Ritual",
    "Fragment",
    "Veil",
    "Specter",
    "Herald",
    "Watcher",
    "Void",
]

# Departure phrases - used when a temp bot leaves the conversation
_DEPARTURE_PHRASES = [
    "departs",
    "leaves",
    "slips away into the night",
    "fades from view",
    "vanishes without a word",
    "steps out of the room",
    "takes their leave",
    "disappears into the crowd",
    "slips quietly out the door",
    "drifts off into silence",
    "retreats into the shadows",
    "turns and walks away",
    "closes the door behind them",
    "waves and wanders off",
    "mutters something and disappears",
    "yawns and vanishes",
    "shrugs and slips away",
    "exits stage left",
    "blinks out of existence",
    "remembers a prior engagement",
    "tips their hat and leaves",
    "wanders off in search of snacks",
    "quietly excuses themselves",
    "slips out unnoticed",
    "is gone",
    "leaves the conversation",
    "ducks out",
    "evaporates mid-sentence",
]


def generate_bot_name() -> str:
    """Generate a random occult bot name."""
    return f"{random.choice(_ADJECTIVES)} {random.choice(_NOUNS)}"


def departure_message(bot_name: str) -> str:
    """Return a random departure announcement for a temp bot."""
    return f"*[{bot_name} {random.choice(_DEPARTURE_PHRASES)}]*"
