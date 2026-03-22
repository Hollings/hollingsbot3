"""Tamagotchi helper/utility functions."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from .constants import AGE_STAGES, PERSONALITY_TRAITS, WEATHER


def _clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


def _get_age_days(pet):
    return (time.time() - pet["born_at"]) / 86400


def _get_age_stage(pet):
    days = _get_age_days(pet)
    for max_days, label, decay_mult, sick_vuln in AGE_STAGES:
        if days < max_days:
            return label, decay_mult, sick_vuln
    return "Senior", 1.3, 1.8


def _get_tdee(pet):
    """Total Daily Energy Expenditure in calories, based on weight/age/fitness."""
    base = 200
    weight_factor = pet["weight"] * 15
    fitness_factor = pet["fitness"] * 0.5
    age_label, _, _ = _get_age_stage(pet)
    age_mult = {"Baby": 1.5, "Child": 1.2, "Teen": 1.1, "Adult": 1.0, "Senior": 0.8}
    return (base + weight_factor + fitness_factor) * age_mult.get(age_label, 1.0)


def _get_trait_modifier(pet, key, default=1.0):
    """Get a personality trait modifier value."""
    traits = json.loads(pet["traits"]) if isinstance(pet["traits"], str) else pet["traits"]
    for trait_name in traits:
        trait = PERSONALITY_TRAITS.get(trait_name, {})
        if key in trait:
            return trait[key]
    return default


def _has_trait(pet, trait_name):
    traits = json.loads(pet["traits"]) if isinstance(pet["traits"], str) else pet["traits"]
    return trait_name in traits


def _is_nighttime():
    """Check if it's nighttime (10 PM - 6 AM UTC)."""
    hour = datetime.now(timezone.utc).hour
    return hour >= 22 or hour < 6


def _bar(value, width=10, reverse=False):
    """Render a text progress bar. If reverse=True, lower is better (like boredom/nails)."""
    display = 100 - value if reverse else value
    filled = int(display / 100 * width)
    empty = width - filled
    bar_str = "[" + "#" * filled + "-" * empty + "]"
    pct = int(value)
    warning = ""
    if not reverse:
        if value <= 15:
            warning = "  !! CRITICAL"
        elif value <= 30:
            warning = "  !! LOW"
    else:
        if value >= 85:
            warning = "  !! CRITICAL"
        elif value >= 70:
            warning = "  !! HIGH"
    return f"{bar_str} {pct}%{warning}"


def _mini_bar(value, width=4):
    """Tiny bar for vitamins."""
    filled = int(value / 100 * width)
    return "#" * filled + "-" * (width - filled)


def _get_feeling(pet):
    """Get an overall feeling string from mood and other stats."""
    mood = pet["mood"]
    if mood >= 80:
        return "Ecstatic"
    if mood >= 60:
        return "Content"
    if mood >= 45:
        return "Okay"
    if mood >= 30:
        return "Anxious"
    if mood >= 15:
        return "Miserable"
    return "Despairing"


def _get_effective_room_temp(pet):
    """Room temp modified by weather."""
    weather = WEATHER.get(pet["current_weather"], {})
    return pet["room_temp"] + weather.get("temp_mod", 0)
