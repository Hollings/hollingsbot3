"""Tamagotchi background tick logic."""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone

from .constants import (
    AILMENT_ESCALATION_THRESHOLD,
    AILMENT_SEVERITY_RATE,
    AILMENTS,
    BASE_DECAY,
    CLOTHING,
    HEATSTROKE_CLEAR_TEMP,
    HEATSTROKE_TEMP,
    HYPOTHERMIA_CLEAR_TEMP,
    HYPOTHERMIA_TEMP,
    IDEAL_BODY_TEMP,
    LETHAL_BODY_TEMP_HIGH,
    LETHAL_BODY_TEMP_LOW,
    LETHAL_WEIGHT_HIGH,
    LETHAL_WEIGHT_LOW,
    LONELINESS_CLEAR_THRESHOLD,
    MALNUTRITION_CLEAR_THRESHOLD,
    MALNUTRITION_WEIGHT_THRESHOLD,
    OBESITY_CLEAR_THRESHOLD,
    OBESITY_WEIGHT_THRESHOLD,
    OLD_AGE_DEATH_CHANCE_PER_MIN,
    OLD_AGE_START_DAY,
    POOP_BLADDER_THRESHOLD,
    POOP_CHANCE_PER_MIN,
    SICKNESS_CHANCE_COLD,
    SICKNESS_CHANCE_DEPRESSION,
    SICKNESS_CHANCE_FLEAS,
    SICKNESS_CHANCE_LONELINESS,
    SICKNESS_CHANCE_SUNBURN,
    SICKNESS_CHANCE_TOOTH_DECAY,
    TOOTH_DECAY_CLEAR_THRESHOLD,
    WEATHER,
    WEATHER_CHANGE_INTERVAL,
)
from .database import _add_ailment, _get_ailments, _remove_ailment, _save_pet
from .helpers import (
    _clamp,
    _get_age_days,
    _get_age_stage,
    _get_effective_room_temp,
    _get_tdee,
    _get_trait_modifier,
    _is_nighttime,
)

MAX_DECAY_SECONDS = 4 * 60 * 60  # Cap decay at 4 hours max between interactions


def _tick_pet(conn, pet):
    """Process decay since last interaction. Called on every player command.

    Elapsed time is capped at MAX_DECAY_SECONDS so players who go offline
    for a day don't come back to a dead pet.
    """
    now = time.time()
    elapsed_seconds = now - pet["last_tick"]
    elapsed_seconds = min(elapsed_seconds, MAX_DECAY_SECONDS)
    elapsed_minutes = elapsed_seconds / 60.0

    if elapsed_minutes < 0.1:
        return []

    events = []
    pet["last_tick"] = now

    # -- Check if nutrition day needs reset --
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if pet["nutrition_day"] != today:
        pet["nutrition_day"] = today
        pet["calories_today"] = 0
        pet["protein_today"] = 0
        pet["carbs_today"] = 0
        pet["fat_today"] = 0
        pet["fiber_today"] = 0
        pet["vit_a_today"] = 0
        pet["vit_b_today"] = 0
        pet["vit_c_today"] = 0
        pet["vit_d_today"] = 0
        pet["vit_e_today"] = 0
        pet["foods_eaten_today"] = "{}"
        pet["toy_uses_today"] = "{}"

    # -- Weather change --
    weather_age = now - (pet["weather_changed_at"] or 0)
    if weather_age > WEATHER_CHANGE_INTERVAL:
        new_weather = random.choice(list(WEATHER.keys()))
        pet["current_weather"] = new_weather
        pet["weather_changed_at"] = now

    # -- Get modifiers --
    _age_label, age_decay_mult, sick_vuln = _get_age_stage(pet)
    is_sleeping = bool(pet["is_sleeping"])
    is_night = _is_nighttime()

    # -- Stat decay --
    for stat, base_rate in BASE_DECAY.items():
        rate = base_rate * age_decay_mult

        # Personality modifiers
        if stat == "hunger":
            rate *= _get_trait_modifier(pet, "hunger_decay")
        elif stat == "social":
            rate *= _get_trait_modifier(pet, "social_decay")
        elif stat == "hygiene":
            rate *= _get_trait_modifier(pet, "hygiene_decay")
        elif stat == "fitness":
            rate *= _get_trait_modifier(pet, "fitness_decay")
        elif stat == "boredom":
            rate *= _get_trait_modifier(pet, "boredom_rate")
        elif stat == "energy":
            rate *= _get_trait_modifier(pet, "energy_decay")
            # Night owl / early bird
            if is_night:
                rate *= _get_trait_modifier(pet, "energy_night")
            else:
                rate *= _get_trait_modifier(pet, "energy_day")
        elif stat == "mood":
            if rate < 0:  # Mood recovery (when positive actions happen)
                rate *= _get_trait_modifier(pet, "mood_recovery")

        # Sleeping modifiers
        if is_sleeping:
            if stat == "energy":
                lights_penalty = 0.5 if pet["lights_on"] else 1.0
                rate = 2.0 * lights_penalty  # Energy RECOVERS during sleep
            elif stat in ("hunger", "thirst"):
                rate *= 0.1  # Metabolism slows significantly during sleep
            elif stat == "bladder":
                rate *= 0.3  # Slower bladder during sleep
            elif stat == "boredom":
                rate = 0  # Don't get bored while sleeping
            elif stat in ("social", "happiness", "comfort"):
                rate *= 0.3  # Slow decay while asleep

        # Poop penalty on room cleanliness
        if stat == "room_cleanliness" and pet["poop_count"] > 0:
            rate *= 1 + pet["poop_count"] * 0.5

        # Apply
        if stat in pet:
            pet[stat] = _clamp(pet[stat] + rate * elapsed_minutes)

    # -- Weather mood effect --
    weather_info = WEATHER.get(pet["current_weather"], {})
    weather_mood = weather_info.get("mood_mod", 0)
    if weather_mood != 0:
        pet["mood"] = _clamp(pet["mood"] + weather_mood * 0.01 * elapsed_minutes)

    # -- Body temperature regulation (exponential approach) --
    effective_room_temp = _get_effective_room_temp(pet)
    clothing = CLOTHING.get(pet["current_clothing"], {})
    warmth = clothing.get("warmth", 0)
    # Equilibrium temp: body strongly self-regulates, room has mild influence
    clothing_offset = warmth * 1.0
    equilibrium = (IDEAL_BODY_TEMP * 7 + effective_room_temp + clothing_offset) / 8
    # Exponential approach: body temp moves toward equilibrium
    # Rate: ~5% of the gap per minute (works correctly for any elapsed time)
    import math

    approach_rate = 1 - math.exp(-0.02 * elapsed_minutes)
    gap = equilibrium - pet["body_temp"]
    pet["body_temp"] = round(pet["body_temp"] + gap * approach_rate, 1)
    pet["body_temp"] = _clamp(pet["body_temp"], 30.0, 45.0)

    # -- Poop check --
    poop_prob = min(0.9, POOP_CHANCE_PER_MIN * elapsed_minutes)
    if not is_sleeping and pet["bladder"] < POOP_BLADDER_THRESHOLD and random.random() < poop_prob:
        pet["poop_count"] += 1
        pet["bladder"] = _clamp(pet["bladder"] + 60)
        events.append(f"{pet['name']} pooped on the floor!")

    # -- Weight management --
    tdee = _get_tdee(pet)
    if pet["calories_today"] > 0:
        cal_balance = (pet["calories_today"] - tdee) / tdee
        weight_change = cal_balance * 0.001 * elapsed_minutes
        pet["weight"] = max(0.5, pet["weight"] + weight_change)

    # -- Sickness checks (probabilities capped at 0.9) --
    ailments = _get_ailments(conn, pet["user_id"])
    ailment_names = {a["ailment"] for a in ailments}

    if pet["hygiene"] < 25 and pet["fur_condition"] < 30 and "fleas" not in ailment_names:
        if random.random() < min(0.9, SICKNESS_CHANCE_FLEAS * sick_vuln * elapsed_minutes):
            if _add_ailment(conn, pet["user_id"], "fleas"):
                events.append(f"{pet['name']} caught fleas!")

    if pet["hygiene"] < 20 and _get_effective_room_temp(pet) < 15 and "cold" not in ailment_names:
        if random.random() < min(0.9, SICKNESS_CHANCE_COLD * sick_vuln * elapsed_minutes):
            if _add_ailment(conn, pet["user_id"], "cold"):
                events.append(f"{pet['name']} caught a cold!")

    if pet["teeth_health"] < 20 and "tooth decay" not in ailment_names:
        if random.random() < min(0.9, SICKNESS_CHANCE_TOOTH_DECAY * sick_vuln * elapsed_minutes):
            if _add_ailment(conn, pet["user_id"], "tooth decay"):
                events.append(f"{pet['name']} has tooth decay!")

    if pet["happiness"] < 15 and "depression" not in ailment_names:
        if random.random() < min(0.9, SICKNESS_CHANCE_DEPRESSION * sick_vuln * elapsed_minutes):
            if _add_ailment(conn, pet["user_id"], "depression"):
                events.append(f"{pet['name']} has fallen into depression...")

    if pet["social"] < 10 and "loneliness" not in ailment_names:
        if random.random() < min(0.9, SICKNESS_CHANCE_LONELINESS * sick_vuln * elapsed_minutes):
            if _add_ailment(conn, pet["user_id"], "loneliness"):
                events.append(f"{pet['name']} is painfully lonely...")

    if pet["body_temp"] > HEATSTROKE_TEMP and "heatstroke" not in ailment_names:
        if _add_ailment(conn, pet["user_id"], "heatstroke"):
            events.append(f"{pet['name']} has HEATSTROKE!")

    if pet["body_temp"] < HYPOTHERMIA_TEMP and "hypothermia" not in ailment_names:
        if _add_ailment(conn, pet["user_id"], "hypothermia"):
            events.append(f"{pet['name']} has HYPOTHERMIA!")

    if pet["weight"] > OBESITY_WEIGHT_THRESHOLD and "obesity" not in ailment_names:
        if _add_ailment(conn, pet["user_id"], "obesity"):
            events.append(f"{pet['name']} is now obese!")

    if pet["weight"] < MALNUTRITION_WEIGHT_THRESHOLD and "malnutrition" not in ailment_names:
        if _add_ailment(conn, pet["user_id"], "malnutrition"):
            events.append(f"{pet['name']} is malnourished!")

    weather = pet["current_weather"]
    if weather == "hot" and pet["current_clothing"] not in ("nothing", "sunhat", "t-shirt"):
        if "sunburn" not in ailment_names and random.random() < min(0.9, SICKNESS_CHANCE_SUNBURN * elapsed_minutes):
            if _add_ailment(conn, pet["user_id"], "sunburn"):
                events.append(f"{pet['name']} got sunburned!")

    # -- Ailment progression --
    for ailment_row in ailments:
        aname = ailment_row["ailment"]
        ailment_info = AILMENTS.get(aname)
        if not ailment_info:
            continue

        for stat, penalty in ailment_info["penalties"].items():
            if stat in pet:
                pet[stat] = _clamp(pet[stat] + penalty * elapsed_minutes)

        new_severity = min(100, ailment_row["severity"] + AILMENT_SEVERITY_RATE * elapsed_minutes)
        conn.execute(
            "UPDATE tamagotchi_ailments SET severity = ? WHERE user_id = ? AND ailment = ?",
            (new_severity, pet["user_id"], aname),
        )
        conn.commit()

        if new_severity >= AILMENT_ESCALATION_THRESHOLD and ailment_info.get("escalates_to"):
            esc = ailment_info["escalates_to"]
            if esc not in ailment_names:
                if _add_ailment(conn, pet["user_id"], esc):
                    events.append(f"{pet['name']}'s {aname} has escalated to {esc}!")
                    ailment_names.add(esc)

    # -- Auto-heal ailments when conditions normalize --
    if pet["body_temp"] <= HEATSTROKE_CLEAR_TEMP and "heatstroke" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "heatstroke")
        events.append(f"{pet['name']}'s heatstroke has resolved.")

    if pet["body_temp"] >= HYPOTHERMIA_CLEAR_TEMP and "hypothermia" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "hypothermia")
        events.append(f"{pet['name']}'s hypothermia has resolved.")

    if pet["weight"] <= OBESITY_CLEAR_THRESHOLD and "obesity" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "obesity")
        events.append(f"{pet['name']} is no longer obese!")

    if pet["weight"] >= MALNUTRITION_CLEAR_THRESHOLD and "malnutrition" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "malnutrition")
        events.append(f"{pet['name']} is no longer malnourished!")

    if pet["teeth_health"] >= TOOTH_DECAY_CLEAR_THRESHOLD and "tooth decay" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "tooth decay")
        events.append(f"{pet['name']}'s teeth are better!")

    if pet["social"] >= LONELINESS_CLEAR_THRESHOLD and "loneliness" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "loneliness")

    # -- Trick decay (forget unused tricks) --
    tricks = json.loads(pet["tricks_known"]) if isinstance(pet["tricks_known"], str) else pet["tricks_known"]
    practice = (
        json.loads(pet["last_trick_practice"])
        if isinstance(pet["last_trick_practice"], str)
        else pet["last_trick_practice"]
    )
    tricks_changed = False
    for trick_name in list(tricks.keys()):
        last_practiced = practice.get(trick_name, pet["born_at"])
        days_since = (now - last_practiced) / 86400
        if days_since > 5:
            decay = (days_since - 5) * 2 * elapsed_minutes / 60  # Slow decay
            tricks[trick_name] = max(0, tricks[trick_name] - decay)
            if tricks[trick_name] <= 0:
                del tricks[trick_name]
                if trick_name in practice:
                    del practice[trick_name]
                events.append(f"{pet['name']} forgot how to {trick_name}!")
            tricks_changed = True
    if tricks_changed:
        pet["tricks_known"] = json.dumps(tricks)
        pet["last_trick_practice"] = json.dumps(practice)

    # -- Vitamin D mood penalty --
    if pet["vit_d_today"] < 20 and pet["nutrition_day"] == today:
        pet["mood"] = _clamp(pet["mood"] - 0.05 * elapsed_minutes)

    # -- Depression debuff: all positive changes halved (handled in commands) --

    # -- Death checks --
    death_cause = None

    # Lethal vitals: must be at zero on TWO consecutive interactions to die.
    # First time at zero: set zero_vital_since as a flag.
    # Second time (next interaction) still at zero: death.
    # Happiness is NOT lethal (causes depression instead).
    vitals = ["hunger", "thirst", "energy"]
    any_zero = any(pet[v] <= 0 for v in vitals)
    if any_zero:
        if pet["zero_vital_since"] is None:
            # First time at zero - warn, set flag, but don't kill yet
            pet["zero_vital_since"] = now
            zero_stats = [v for v in vitals if pet[v] <= 0]
            events.append(f"!! WARNING: {pet['name']}'s {zero_stats[0]} is at ZERO! Act now or they die! !!")
        else:
            # Was already at zero last interaction - fatal
            zero_stats = [v for v in vitals if pet[v] <= 0]
            death_cause = f"fatal {zero_stats[0]} deprivation"
    else:
        pet["zero_vital_since"] = None

    # Extreme body temp
    if pet["body_temp"] >= LETHAL_BODY_TEMP_HIGH:
        death_cause = "extreme heatstroke"
    elif pet["body_temp"] <= LETHAL_BODY_TEMP_LOW:
        death_cause = "extreme hypothermia"

    # Extreme weight
    if pet["weight"] <= LETHAL_WEIGHT_LOW:
        death_cause = "starvation"
    elif pet["weight"] >= LETHAL_WEIGHT_HIGH:
        death_cause = "morbid obesity"

    # Old age
    age_days = _get_age_days(pet)
    if age_days > OLD_AGE_START_DAY:
        death_chance = (age_days - OLD_AGE_START_DAY) * OLD_AGE_DEATH_CHANCE_PER_MIN * elapsed_minutes
        if random.random() < death_chance:
            death_cause = "old age"

    if death_cause:
        pet["died_at"] = now
        pet["cause_of_death"] = death_cause
        events.append(f"!! {pet['name']} has DIED from {death_cause}. !!")

    _save_pet(conn, pet)
    return events
