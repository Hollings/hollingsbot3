"""Tamagotchi status display builder."""

from __future__ import annotations

import json

from .constants import AILMENTS, DAILY_TARGETS
from .helpers import (
    _bar,
    _get_age_days,
    _get_age_stage,
    _get_effective_room_temp,
    _get_feeling,
    _get_tdee,
    _mini_bar,
)


def _build_status(pet, ailments):
    """Build the massive status display."""
    age_days = _get_age_days(pet)
    age_label, _, _ = _get_age_stage(pet)
    tdee = _get_tdee(pet)
    eff_room_temp = _get_effective_room_temp(pet)
    feeling = _get_feeling(pet)

    is_dead = pet["died_at"] is not None
    if is_dead:
        lines = [
            "```",
            "========================================",
            f"  RIP {pet['name'].upper()}",
            f"  Died on day {int(age_days)} from {pet['cause_of_death']}",
            "  Use !pet bury to lay them to rest",
            "========================================",
            "```",
        ]
        return "\n".join(lines)

    sleeping_str = " (SLEEPING)" if pet["is_sleeping"] else ""
    clothing_name = pet["current_clothing"] or "nothing"

    lines = [
        "```",
        "========================================",
        f"  {pet['name'].upper()} ({age_label}, Day {int(age_days)}) - {feeling}{sleeping_str}",
        f"  Weight: {pet['weight']:.1f} kg | Temp: {pet['body_temp']:.1f}C | Wearing: {clothing_name.title()}",
        f"  Money: {pet['money']:.0f} coins | Weather: {pet['current_weather'].title()}, ~{eff_room_temp:.0f}C",
        "========================================",
        "",
        "-- VITAL NEEDS --",
        f"  Hunger:     {_bar(pet['hunger'])}",
        f"  Thirst:     {_bar(pet['thirst'])}",
        f"  Energy:     {_bar(pet['energy'])}",
        f"  Happiness:  {_bar(pet['happiness'])}",
        f"  Hygiene:    {_bar(pet['hygiene'])}",
        f"  Bladder:    {_bar(pet['bladder'])}",
        f"  Social:     {_bar(pet['social'])}",
        f"  Comfort:    {_bar(pet['comfort'])}",
        f"  Boredom:    {_bar(pet['boredom'], reverse=True)}",
        "",
        "-- BODY & GROOMING --",
        f"  Fitness:    {_bar(pet['fitness'])}",
        f"  Teeth:      {_bar(pet['teeth_health'])}",
        f"  Nails:      {_bar(pet['nail_length'], reverse=True)}",
        f"  Fur:        {_bar(pet['fur_condition'])}",
        f"  Skin:       {_bar(pet['skin_condition'])}",
        "",
        f"-- NUTRITION TODAY (TDEE: {tdee:.0f} cal) --",
        f"  Calories:   {pet['calories_today']:.0f}/{tdee:.0f} cal",
        f"  Protein:    {_bar(min(100, pet['protein_today'] / DAILY_TARGETS['protein'] * 100))}",
        f"  Carbs:      {_bar(min(100, pet['carbs_today'] / DAILY_TARGETS['carbs'] * 100))}",
        f"  Fat:        {_bar(min(100, pet['fat_today'] / DAILY_TARGETS['fat'] * 100))}",
        f"  Fiber:      {_bar(min(100, pet['fiber_today'] / DAILY_TARGETS['fiber'] * 100))}",
        f"  Vitamins:   A[{_mini_bar(min(100, pet['vit_a_today']))}]"
        f" B[{_mini_bar(min(100, pet['vit_b_today']))}]"
        f" C[{_mini_bar(min(100, pet['vit_c_today']))}]"
        f" D[{_mini_bar(min(100, pet['vit_d_today']))}]"
        f" E[{_mini_bar(min(100, pet['vit_e_today']))}]",
        "",
        "-- MENTAL --",
        f"  Intelligence: {_bar(pet['intelligence'])}",
        f"  Creativity:   {_bar(pet['creativity'])}",
        f"  Trust:        {_bar(pet['trust'])}",
        f"  Mood:         {_bar(pet['mood'])}",
        f"  Discipline:   {_bar(pet['discipline'])}",
        "",
        "-- ENVIRONMENT --",
        f"  Room:       {_bar(pet['room_cleanliness'])}",
        f"  Thermostat: {pet['room_temp']:.0f}C | Lights: {'ON' if pet['lights_on'] else 'OFF'}",
        f"  Decor:      {_bar(pet['decoration_score'])}  ({(pet['decoration_color'] or 'none').title()})",
        f"  Poops:      {pet['poop_count']}{' (!! clean me)' if pet['poop_count'] > 0 else ''}",
    ]

    # Ailments
    if ailments:
        lines.append("")
        lines.append("-- AILMENTS --")
        for a in ailments:
            info = AILMENTS.get(a["ailment"], {})
            cure = info.get("cure")
            cure_hint = f" - try: !pet medicine {cure}" if cure else f" - {info.get('desc', '')}"
            lines.append(f"  * {a['ailment'].title()} (severity: {a['severity']:.0f}%){cure_hint}")

    # Tricks
    tricks = json.loads(pet["tricks_known"]) if isinstance(pet["tricks_known"], str) else pet["tricks_known"]
    if tricks:
        lines.append("")
        lines.append("-- KNOWN TRICKS --")
        trick_strs = []
        for t, prof in tricks.items():
            if prof >= 100:
                trick_strs.append(f"{t.title()} (mastered)")
            else:
                trick_strs.append(f"{t.title()} ({prof:.0f}%)")
        lines.append(f"  {', '.join(trick_strs)}")

    lines.append("========================================")
    lines.append("```")
    return "\n".join(lines)
