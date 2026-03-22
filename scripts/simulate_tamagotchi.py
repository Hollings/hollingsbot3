"""Tamagotchi economy & balance simulator.

Runs headless simulations of the pet lifecycle with different player strategies
to test whether the economy is fair, if pets survive, and how money flows.

Usage:
    python scripts/simulate_tamagotchi.py
    python scripts/simulate_tamagotchi.py --runs 50 --strategy smart
    python scripts/simulate_tamagotchi.py --strategy all --days 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import tempfile

# Add src to path so we can import the tamagotchi modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# We reimplement tick with a controllable clock instead of time.time()
from hollingsbot.cogs.tamagotchi.constants import (
    AILMENTS,
    BASE_DECAY,
    CLOTHING_PRICES,
    FOOD_PRICES,
    FOODS,
    IDEAL_BODY_TEMP,
    MEDICINE_PRICES,
    MEDICINES,
    PERSONALITY_TRAITS,
    TOY_PRICES,
    TOYS,
    WEATHER,
    WEATHER_CHANGE_INTERVAL,
    WORK_COOLDOWN,
    WORK_PAY_MAX,
    WORK_PAY_MIN,
)
from hollingsbot.cogs.tamagotchi.database import (
    _add_ailment,
    _get_ailments,
    _get_pet,
    _remove_ailment,
    _save_pet,
)
from hollingsbot.cogs.tamagotchi.helpers import (
    _clamp,
    _get_age_days,
    _get_age_stage,
    _get_effective_room_temp,
    _get_tdee,
    _get_trait_modifier,
)

# ---------------------------------------------------------------------------
# Simulation tick - same as real tick but with injectable "now" timestamp
# ---------------------------------------------------------------------------
MAX_DECAY_SECONDS = 4 * 60 * 60  # Match the real game cap


def sim_tick(conn, pet, now):
    """Process one tick with a controllable timestamp."""
    elapsed_seconds = now - pet["last_tick"]
    elapsed_seconds = min(elapsed_seconds, MAX_DECAY_SECONDS)
    elapsed_minutes = elapsed_seconds / 60.0

    if elapsed_minutes < 0.1:
        return []

    events = []
    pet["last_tick"] = now

    # Nutrition day reset (simplified - use day number from born_at)
    day_num = int((now - pet["born_at"]) / 86400)
    day_key = str(day_num)
    if pet["nutrition_day"] != day_key:
        pet["nutrition_day"] = day_key
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

    # Weather change
    weather_age = now - (pet["weather_changed_at"] or 0)
    if weather_age > WEATHER_CHANGE_INTERVAL:
        pet["current_weather"] = random.choice(list(WEATHER.keys()))
        pet["weather_changed_at"] = now

    # Get modifiers
    _age_label, age_decay_mult, sick_vuln = _get_age_stage(pet)
    is_sleeping = bool(pet["is_sleeping"])
    hour_of_day = int((now % 86400) / 3600)
    is_night = hour_of_day >= 22 or hour_of_day < 6

    # Stat decay
    for stat, base_rate in BASE_DECAY.items():
        rate = base_rate * age_decay_mult

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
            if is_night:
                rate *= _get_trait_modifier(pet, "energy_night")
            else:
                rate *= _get_trait_modifier(pet, "energy_day")
        elif stat == "mood":
            if rate < 0:
                rate *= _get_trait_modifier(pet, "mood_recovery")

        if is_sleeping:
            if stat == "energy":
                lights_penalty = 0.5 if pet["lights_on"] else 1.0
                rate = 2.0 * lights_penalty
            elif stat in ("hunger", "thirst"):
                rate *= 0.1
            elif stat == "bladder":
                rate *= 0.3
            elif stat == "boredom":
                rate = 0
            elif stat in ("social", "happiness", "comfort"):
                rate *= 0.3

        if stat == "room_cleanliness" and pet["poop_count"] > 0:
            rate *= 1 + pet["poop_count"] * 0.5

        if stat in pet:
            pet[stat] = _clamp(pet[stat] + rate * elapsed_minutes)

    # Weather mood
    weather_info = WEATHER.get(pet["current_weather"], {})
    weather_mood = weather_info.get("mood_mod", 0)
    if weather_mood != 0:
        pet["mood"] = _clamp(pet["mood"] + weather_mood * 0.01 * elapsed_minutes)

    # Body temp (exponential approach to equilibrium)
    effective_room_temp = _get_effective_room_temp(pet)
    from hollingsbot.cogs.tamagotchi.constants import CLOTHING as CLOTHING_DATA

    clothing = CLOTHING_DATA.get(pet["current_clothing"], {})
    warmth = clothing.get("warmth", 0)
    clothing_offset = warmth * 1.0
    equilibrium = (IDEAL_BODY_TEMP * 7 + effective_room_temp + clothing_offset) / 8
    import math

    approach_rate = 1 - math.exp(-0.02 * elapsed_minutes)
    gap = equilibrium - pet["body_temp"]
    pet["body_temp"] = round(pet["body_temp"] + gap * approach_rate, 1)
    pet["body_temp"] = _clamp(pet["body_temp"], 30.0, 45.0)

    # Poop
    if not is_sleeping and pet["bladder"] < 20 and random.random() < min(0.9, 0.3 * elapsed_minutes):
        pet["poop_count"] += 1
        pet["bladder"] = _clamp(pet["bladder"] + 60)

    # Weight
    tdee = _get_tdee(pet)
    if pet["calories_today"] > 0:
        cal_balance = (pet["calories_today"] - tdee) / tdee
        weight_change = cal_balance * 0.001 * elapsed_minutes
        pet["weight"] = max(0.5, pet["weight"] + weight_change)

    # Sickness
    ailments = _get_ailments(conn, pet["user_id"])
    ailment_names = {a["ailment"] for a in ailments}

    if pet["hygiene"] < 25 and pet["fur_condition"] < 30 and "fleas" not in ailment_names:
        if random.random() < 0.05 * sick_vuln * elapsed_minutes:
            _add_ailment(conn, pet["user_id"], "fleas")

    if pet["hygiene"] < 20 and effective_room_temp < 15 and "cold" not in ailment_names:
        if random.random() < 0.04 * sick_vuln * elapsed_minutes:
            _add_ailment(conn, pet["user_id"], "cold")

    if pet["teeth_health"] < 20 and "tooth decay" not in ailment_names:
        if random.random() < 0.03 * sick_vuln * elapsed_minutes:
            _add_ailment(conn, pet["user_id"], "tooth decay")

    if pet["happiness"] < 15 and "depression" not in ailment_names:
        if random.random() < 0.02 * sick_vuln * elapsed_minutes:
            _add_ailment(conn, pet["user_id"], "depression")

    if pet["social"] < 10 and "loneliness" not in ailment_names:
        if random.random() < 0.03 * sick_vuln * elapsed_minutes:
            _add_ailment(conn, pet["user_id"], "loneliness")

    if pet["body_temp"] > 40 and "heatstroke" not in ailment_names:
        _add_ailment(conn, pet["user_id"], "heatstroke")

    if pet["body_temp"] < 35 and "hypothermia" not in ailment_names:
        _add_ailment(conn, pet["user_id"], "hypothermia")

    if pet["weight"] > 12 and "obesity" not in ailment_names:
        _add_ailment(conn, pet["user_id"], "obesity")

    if pet["weight"] < 2.5 and "malnutrition" not in ailment_names:
        _add_ailment(conn, pet["user_id"], "malnutrition")

    # Ailment progression
    for ailment_row in ailments:
        aname = ailment_row["ailment"]
        ailment_info = AILMENTS.get(aname)
        if not ailment_info:
            continue
        for stat, penalty in ailment_info["penalties"].items():
            if stat in pet:
                pet[stat] = _clamp(pet[stat] + penalty * elapsed_minutes)
        new_severity = min(100, ailment_row["severity"] + 0.5 * elapsed_minutes)
        conn.execute(
            "UPDATE tamagotchi_ailments SET severity = ? WHERE user_id = ? AND ailment = ?",
            (new_severity, pet["user_id"], aname),
        )
        conn.commit()
        if new_severity >= 80 and ailment_info.get("escalates_to"):
            esc = ailment_info["escalates_to"]
            if esc not in ailment_names:
                _add_ailment(conn, pet["user_id"], esc)
                ailment_names.add(esc)

    # Auto-heal
    if pet["body_temp"] <= 39.5 and "heatstroke" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "heatstroke")
    if pet["body_temp"] >= 36 and "hypothermia" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "hypothermia")
    if pet["weight"] <= 11.5 and "obesity" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "obesity")
    if pet["weight"] >= 3.0 and "malnutrition" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "malnutrition")
    if pet["teeth_health"] >= 50 and "tooth decay" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "tooth decay")
    if pet["social"] >= 40 and "loneliness" in ailment_names:
        _remove_ailment(conn, pet["user_id"], "loneliness")

    # Death checks
    death_cause = None
    vitals = ["hunger", "thirst", "energy"]
    any_zero = any(pet[v] <= 0 for v in vitals)
    if any_zero:
        if pet["zero_vital_since"] is None:
            pet["zero_vital_since"] = now  # First time: warning
        else:
            zero_stats = [v for v in vitals if pet[v] <= 0]
            death_cause = f"fatal {zero_stats[0]} deprivation"  # Second time: death
    else:
        pet["zero_vital_since"] = None

    if pet["body_temp"] >= 43:
        death_cause = "extreme heatstroke"
    elif pet["body_temp"] <= 32:
        death_cause = "extreme hypothermia"
    if pet["weight"] <= 1.0:
        death_cause = "starvation"
    elif pet["weight"] >= 20:
        death_cause = "morbid obesity"

    age_days = _get_age_days(pet)
    if age_days > 30:
        death_chance = (age_days - 30) * 0.0005 * elapsed_minutes
        if random.random() < death_chance:
            death_cause = "old age"

    if death_cause:
        pet["died_at"] = now
        pet["cause_of_death"] = death_cause
        events.append(death_cause)

    _save_pet(conn, pet)
    return events


# ---------------------------------------------------------------------------
# Player actions (simulate what a player would do)
# ---------------------------------------------------------------------------


def action_feed(conn, pet, food_name):
    food = FOODS[food_name]
    price = FOOD_PRICES.get(food_name, 10)
    if pet["money"] < price:
        return False, "broke"
    pet["money"] -= price
    pet["hunger"] = _clamp(pet["hunger"] + food["hunger_restore"])
    pet["calories_today"] += food["calories"]
    pet["protein_today"] += food["protein"]
    pet["carbs_today"] += food["carbs"]
    pet["fat_today"] += food["fat"]
    pet["fiber_today"] += food["fiber"]
    pet["vit_a_today"] += food["vit_a"]
    pet["vit_b_today"] += food["vit_b"]
    pet["vit_c_today"] += food["vit_c"]
    pet["vit_d_today"] += food["vit_d"]
    pet["vit_e_today"] += food["vit_e"]
    pet["bladder"] = _clamp(pet["bladder"] - 8)

    # Check allergy
    if pet.get("allergy") and food_name == pet["allergy"]:
        _add_ailment(conn, pet["user_id"], "allergic reaction")

    _save_pet(conn, pet)
    return True, "fed"


def action_water(conn, pet):
    pet["thirst"] = _clamp(pet["thirst"] + 30)
    pet["bladder"] = _clamp(pet["bladder"] - 10)
    _save_pet(conn, pet)


def action_work(conn, pet, now):
    cooldown_left = (pet["last_work_at"] or 0) + WORK_COOLDOWN - now
    if cooldown_left > 0:
        return False
    pay = random.randint(WORK_PAY_MIN, WORK_PAY_MAX)
    pet["money"] += pay
    pet["last_work_at"] = now
    pet["energy"] = _clamp(pet["energy"] - 10)
    _save_pet(conn, pet)
    return True


def action_sleep(conn, pet, now):
    if pet["is_sleeping"]:
        return
    pet["is_sleeping"] = 1
    pet["sleep_started_at"] = now
    pet["lights_on"] = 0
    _save_pet(conn, pet)


def action_wake(conn, pet):
    if not pet["is_sleeping"]:
        return
    pet["is_sleeping"] = 0
    pet["lights_on"] = 1
    _save_pet(conn, pet)


def action_toilet(conn, pet):
    pet["bladder"] = _clamp(pet["bladder"] + 50)
    _save_pet(conn, pet)


def action_clean(conn, pet):
    pet["hygiene"] = _clamp(pet["hygiene"] + 40)
    _save_pet(conn, pet)


def action_brush_teeth(conn, pet):
    pet["teeth_health"] = _clamp(pet["teeth_health"] + 20)
    _save_pet(conn, pet)


def action_brush_fur(conn, pet):
    pet["fur_condition"] = _clamp(pet["fur_condition"] + 25)
    _save_pet(conn, pet)


def action_trim_nails(conn, pet):
    pet["nail_length"] = _clamp(pet["nail_length"] - 40)
    _save_pet(conn, pet)


def action_scoop(conn, pet):
    pet["poop_count"] = 0
    pet["room_cleanliness"] = _clamp(pet["room_cleanliness"] + 10)
    _save_pet(conn, pet)


def action_clean_room(conn, pet):
    pet["room_cleanliness"] = _clamp(pet["room_cleanliness"] + 35)
    _save_pet(conn, pet)


def action_talk(conn, pet):
    pet["social"] = _clamp(pet["social"] + 15)
    pet["happiness"] = _clamp(pet["happiness"] + 5)
    pet["mood"] = _clamp(pet["mood"] + 5)
    pet["trust"] = _clamp(pet["trust"] + 2)
    _save_pet(conn, pet)


def action_play(conn, pet, toy_name):
    price = TOY_PRICES.get(toy_name, 5)
    if pet["money"] < price:
        return False
    pet["money"] -= price
    toy = TOYS[toy_name]
    for stat, boost in toy.items():
        if stat in ("max_uses", "desc"):
            continue
        if stat in pet:
            pet[stat] = _clamp(pet[stat] + boost)
    pet["social"] = _clamp(pet["social"] + 5)
    _save_pet(conn, pet)
    return True


def action_exercise(conn, pet):
    pet["fitness"] = _clamp(pet["fitness"] + 8)
    pet["energy"] = _clamp(pet["energy"] - 15)
    pet["hunger"] = _clamp(pet["hunger"] - 10)
    pet["thirst"] = _clamp(pet["thirst"] - 8)
    pet["boredom"] = _clamp(pet["boredom"] - 10)
    pet["weight"] = max(0.5, pet["weight"] - 0.02)
    _save_pet(conn, pet)


def action_medicine(conn, pet, med_name):
    price = MEDICINE_PRICES.get(med_name, 15)
    if pet["money"] < price:
        return False, "broke"
    pet["money"] -= price
    med = MEDICINES[med_name]
    ailments = _get_ailments(conn, pet["user_id"])
    ailment_names = {a["ailment"] for a in ailments}
    cured = []
    for curable in med["cures"]:
        if curable in ailment_names:
            _remove_ailment(conn, pet["user_id"], curable)
            cured.append(curable)
    _save_pet(conn, pet)
    return True, cured


def action_dress(conn, pet, clothing_name):
    price = CLOTHING_PRICES.get(clothing_name, 5)
    if pet["money"] < price:
        return False
    pet["money"] -= price
    pet["current_clothing"] = clothing_name
    _save_pet(conn, pet)
    return True


def action_thermostat(conn, pet, temp):
    pet["room_temp"] = temp
    _save_pet(conn, pet)


# ---------------------------------------------------------------------------
# Player strategies
# ---------------------------------------------------------------------------


def pick_cheapest_food():
    """Pick the cheapest food that provides decent nutrition."""
    return min(FOOD_PRICES, key=FOOD_PRICES.get)


def pick_best_value_food(pet):
    """Pick food that balances cost, hunger, and nutrition gaps."""
    scores = {}
    for name, food in FOODS.items():
        price = FOOD_PRICES.get(name, 10)
        hunger_value = food["hunger_restore"] / max(price, 1)
        # Bonus for filling nutrition gaps
        nutrient_bonus = 0
        if pet["protein_today"] < 30:
            nutrient_bonus += food["protein"] * 0.5
        if pet["vit_d_today"] < 50:
            nutrient_bonus += food["vit_d"] * 0.3
        if pet["vit_c_today"] < 50:
            nutrient_bonus += food["vit_c"] * 0.3
        scores[name] = hunger_value + nutrient_bonus / max(price, 1)
    return max(scores, key=scores.get)


def strategy_smart(conn, pet, now, hour):
    """Smart player: checks in every ~30 min, makes good decisions."""
    is_sleeping = bool(pet["is_sleeping"])

    # Sleep schedule: sleep at 22:00, wake at 06:00
    if hour >= 22 or hour < 6:
        if not is_sleeping and pet["happiness"] > 20:
            # Gorge before bed
            for _ in range(4):
                if pet["hunger"] >= 85:
                    break
                action_feed(conn, pet, pick_cheapest_food())
            action_water(conn, pet)
            action_water(conn, pet)
            action_water(conn, pet)
            action_brush_teeth(conn, pet)
            action_sleep(conn, pet, now)
            return
    elif is_sleeping:
        # Wake up, but only if energy is decent
        if pet["energy"] > 40:
            action_wake(conn, pet)
        else:
            return  # Keep sleeping until energy recovers

    if is_sleeping:
        return

    # Nap if energy is critically low (even during the day), but not if happiness is dire
    if pet["energy"] < 20 and pet["happiness"] > 25:
        action_sleep(conn, pet, now)
        return

    # Handle ailments FIRST (depression is a silent killer)
    ailments = _get_ailments(conn, pet["user_id"])
    for a in ailments:
        ailment_info = AILMENTS.get(a["ailment"], {})
        cure = ailment_info.get("cure")
        if cure and cure in MEDICINES:
            action_medicine(conn, pet, cure)

    # Work whenever cooldown is up - top priority for income
    if pet["energy"] > 25:
        action_work(conn, pet, now)

    # Critical needs first - feed/water BEFORE energy runs out
    if pet["bladder"] < 30:
        action_toilet(conn, pet)

    if pet["poop_count"] > 0:
        action_scoop(conn, pet)

    # Feed until hunger is reasonable (don't over-spend)
    for _ in range(3):
        if pet["hunger"] >= 55:
            break
        food = pick_best_value_food(pet) if pet["money"] > 20 else pick_cheapest_food()
        ok, _ = action_feed(conn, pet, food)
        if not ok:
            break

    # Water is free
    if pet["thirst"] < 60:
        action_water(conn, pet)
        action_water(conn, pet)  # Double water since it's free

    if pet["hygiene"] < 40:
        action_clean(conn, pet)

    # Keep social and happiness up
    if pet["social"] < 50 or pet["happiness"] < 50:
        action_talk(conn, pet)

    if pet["boredom"] > 50 or pet["happiness"] < 40:
        action_play(conn, pet, "ball")

    if pet["room_cleanliness"] < 50:
        action_clean_room(conn, pet)

    if pet["teeth_health"] < 60:
        action_brush_teeth(conn, pet)

    if pet["fur_condition"] < 50:
        action_brush_fur(conn, pet)

    if pet["nail_length"] > 60:
        action_trim_nails(conn, pet)

    if pet["fitness"] < 30 and pet["energy"] > 40:
        action_exercise(conn, pet)

    # Boost happiness if low
    if pet["happiness"] < 35:
        action_talk(conn, pet)  # extra talk
        action_play(conn, pet, "music box")  # happiness +10

    # Weather-appropriate clothing
    weather = pet["current_weather"]
    eff_temp = _get_effective_room_temp(pet)
    if eff_temp < 12:
        action_dress(conn, pet, "coat")
    elif eff_temp < 18:
        action_dress(conn, pet, "sweater")
    elif weather == "rainy":
        action_dress(conn, pet, "raincoat")
    elif eff_temp > 28:
        action_dress(conn, pet, "sunhat")
    else:
        action_dress(conn, pet, "t-shirt")

    # Always keep thermostat reasonable
    action_thermostat(conn, pet, 22)


def strategy_casual(conn, pet, now, hour):
    """Casual player: checks in 3-4 times a day, does the basics."""
    is_sleeping = bool(pet["is_sleeping"])

    if hour >= 23 or hour < 7:
        if not is_sleeping:
            action_sleep(conn, pet, now)
        return
    elif is_sleeping:
        action_wake(conn, pet)

    if is_sleeping:
        return

    action_work(conn, pet, now)

    if pet["hunger"] < 30:
        action_feed(conn, pet, pick_cheapest_food())

    if pet["thirst"] < 30:
        action_water(conn, pet)

    if pet["bladder"] < 20:
        action_toilet(conn, pet)

    if pet["poop_count"] > 2:
        action_scoop(conn, pet)

    if pet["hygiene"] < 25:
        action_clean(conn, pet)

    if pet["social"] < 20:
        action_talk(conn, pet)


def strategy_neglectful(conn, pet, now, hour):
    """Neglectful player: checks in once a day, does minimum."""
    is_sleeping = bool(pet["is_sleeping"])

    if hour >= 23 or hour < 8:
        if not is_sleeping:
            action_sleep(conn, pet, now)
        return
    elif is_sleeping:
        action_wake(conn, pet)

    if is_sleeping:
        return

    action_work(conn, pet, now)

    if pet["hunger"] < 15:
        action_feed(conn, pet, pick_cheapest_food())

    if pet["thirst"] < 15:
        action_water(conn, pet)


STRATEGIES = {
    "smart": (strategy_smart, 30),  # checks every 30 min
    "casual": (strategy_casual, 120),  # checks every 2 hours
    "neglectful": (strategy_neglectful, 360),  # checks every 6 hours
}


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def create_pet(conn, user_id, now):
    """Create a pet in the database."""
    traits = random.sample(list(PERSONALITY_TRAITS.keys()), k=random.randint(2, 3))
    food_list = list(FOODS.keys())
    loves = random.sample(food_list, k=2)
    remaining = [f for f in food_list if f not in loves]
    hates = random.sample(remaining, k=2)
    remaining2 = [f for f in remaining if f not in hates]
    allergy = random.choice(remaining2) if remaining2 else None

    conn.execute(
        """INSERT INTO tamagotchi_pets (
            user_id, name, born_at, last_tick, nutrition_day,
            traits, food_loves, food_hates, allergy, favorite_color,
            weather_changed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            "SimPet",
            now,
            now,
            "0",
            json.dumps(traits),
            json.dumps(loves),
            json.dumps(hates),
            allergy,
            random.choice(["red", "blue", "green"]),
            now,
        ),
    )
    conn.commit()


def run_simulation(strategy_name, max_days=30, verbose=False):
    """Run a single pet simulation. Returns stats dict."""
    strategy_fn, check_interval_min = STRATEGIES[strategy_name]

    # Create temp database
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Create tables
    # Monkey-patch DB_PATH for init
    import hollingsbot.cogs.tamagotchi.database as db_mod
    from hollingsbot.cogs.tamagotchi.database import _init_db

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = db_path
    _init_db()
    db_mod.DB_PATH = old_path

    import time as _time

    import hollingsbot.cogs.tamagotchi.helpers as helpers_mod

    start_time = _time.time()
    user_id = 1
    create_pet(conn, user_id, start_time)

    now = start_time
    check_interval = check_interval_min * 60

    # Monkey-patch time.time in helpers & database modules so age/ailment timestamps use sim clock
    _real_time = _time.time
    sim_clock = [start_time]  # Mutable container so lambda sees updates
    fake_time_mod = type("FakeTime", (), {"time": staticmethod(lambda: sim_clock[0])})()
    helpers_mod.time = fake_time_mod
    db_mod.time = fake_time_mod

    # Tracking
    money_history = []
    ailment_log = []
    total_spent = 0.0
    total_earned = 0.0
    actions_taken = 0
    work_count = 0

    day = 0
    alive = True

    while alive and day < max_days:
        # Advance time to next check-in (no background ticks - just like the real game)
        now += check_interval
        sim_clock[0] = now

        pet = _get_pet(conn, user_id)
        if not pet:
            break

        # Tick decay on interaction (same as real game)
        sim_tick(conn, pet, now)
        pet = _get_pet(conn, user_id)

        if pet["died_at"] is not None:
            alive = False
            if verbose:
                elapsed_hrs = (now - start_time) / 3600
                print(
                    f"  DIED at day {elapsed_hrs / 24:.1f}: {pet['cause_of_death']} | "
                    f"H:{pet['hunger']:.0f} T:{pet['thirst']:.0f} E:{pet['energy']:.0f} "
                    f"Hp:{pet['happiness']:.0f} | Money:{pet['money']:.0f} | "
                    f"Sleeping:{pet['is_sleeping']}"
                )
            break

        # Player strategy
        hour = int((now % 86400) / 3600)
        old_money = pet["money"]
        old_work = pet.get("last_work_at", 0)

        strategy_fn(conn, pet, now, hour)
        pet = _get_pet(conn, user_id)

        money_diff = pet["money"] - old_money
        if money_diff > 0:
            total_earned += money_diff
        else:
            total_spent += abs(money_diff)

        if pet.get("last_work_at", 0) != old_work:
            work_count += 1
        actions_taken += 1

        # Daily snapshot
        new_day = int((now - start_time) / 86400)
        if new_day > day:
            day = new_day
            money_history.append(pet["money"])
            ailments = _get_ailments(conn, user_id)
            if ailments:
                for a in ailments:
                    ailment_log.append(a["ailment"])

            if verbose and day % 5 == 0:
                age_label, _, _ = _get_age_stage(pet)
                ails = [a["ailment"] for a in ailments]
                print(
                    f"  Day {day:>3} ({age_label:<6}) | "
                    f"Money: {pet['money']:>6.0f} | "
                    f"H:{pet['hunger']:>4.0f} T:{pet['thirst']:>4.0f} E:{pet['energy']:>4.0f} "
                    f"Hp:{pet['happiness']:>4.0f} Hy:{pet['hygiene']:>4.0f} | "
                    f"W:{pet['weight']:>4.1f}kg | "
                    f"Ailments: {', '.join(ails) if ails else 'none'}"
                )

    pet = _get_pet(conn, user_id)
    conn.close()

    # Restore real time modules
    import time as real_time_mod

    helpers_mod.time = real_time_mod
    db_mod.time = real_time_mod

    try:
        os.unlink(db_path)
    except OSError:
        pass

    cause = pet["cause_of_death"] if pet and pet["died_at"] else None
    lifespan = (now - start_time) / 86400

    # Count ailment occurrences
    ailment_counts = {}
    for a in ailment_log:
        ailment_counts[a] = ailment_counts.get(a, 0) + 1

    return {
        "alive": alive,
        "lifespan_days": lifespan,
        "cause_of_death": cause,
        "total_earned": total_earned,
        "total_spent": total_spent,
        "final_money": pet["money"] if pet else 0,
        "money_history": money_history,
        "work_count": work_count,
        "actions_taken": actions_taken,
        "ailment_counts": ailment_counts,
        "final_weight": pet["weight"] if pet else 0,
    }


def run_batch(strategy_name, num_runs=20, max_days=30, verbose=False):
    """Run multiple simulations and aggregate results."""
    results = []
    for i in range(num_runs):
        if verbose:
            print(f"\n--- Run {i + 1}/{num_runs} ({strategy_name}) ---")
        r = run_simulation(strategy_name, max_days=max_days, verbose=verbose)
        results.append(r)

    # Aggregate
    lifespans = [r["lifespan_days"] for r in results]
    survived = sum(1 for r in results if r["alive"])
    deaths = {}
    for r in results:
        cause = r["cause_of_death"]
        if cause:
            deaths[cause] = deaths.get(cause, 0) + 1

    avg_earned = sum(r["total_earned"] for r in results) / num_runs
    avg_spent = sum(r["total_spent"] for r in results) / num_runs
    avg_final_money = sum(r["final_money"] for r in results) / num_runs

    all_ailments = {}
    for r in results:
        for a, count in r["ailment_counts"].items():
            all_ailments[a] = all_ailments.get(a, 0) + count

    avg_money_by_day = []
    max_days_seen = max(len(r["money_history"]) for r in results) if results else 0
    for d in range(max_days_seen):
        values = [r["money_history"][d] for r in results if d < len(r["money_history"])]
        if values:
            avg_money_by_day.append(sum(values) / len(values))

    print(f"\n{'=' * 60}")
    print(f"  STRATEGY: {strategy_name.upper()}")
    print(f"  {num_runs} runs, max {max_days} days each")
    print(f"{'=' * 60}")
    print(f"  Survival rate: {survived}/{num_runs} ({survived / num_runs * 100:.0f}%)")
    print(f"  Avg lifespan:  {sum(lifespans) / len(lifespans):.1f} days")
    print(f"  Min lifespan:  {min(lifespans):.1f} days")
    print(f"  Max lifespan:  {max(lifespans):.1f} days")
    print()
    print("  -- ECONOMY --")
    print(f"  Avg earned:     {avg_earned:.0f} coins total")
    print(f"  Avg spent:      {avg_spent:.0f} coins total")
    print(f"  Avg final bal:  {avg_final_money:.0f} coins")
    if avg_money_by_day:
        print("  Money trend (daily avg):")
        for d in range(0, len(avg_money_by_day), 5):
            bar_len = int(avg_money_by_day[d] / 5)
            bar = "#" * min(bar_len, 40)
            print(f"    Day {d + 1:>3}: {avg_money_by_day[d]:>6.0f} |{bar}")

    if deaths:
        print()
        print("  -- CAUSES OF DEATH --")
        for cause, count in sorted(deaths.items(), key=lambda x: -x[1]):
            print(f"    {cause:<35} {count:>3} ({count / num_runs * 100:.0f}%)")

    if all_ailments:
        print()
        print("  -- AILMENT FREQUENCY (total across all runs) --")
        for ailment, count in sorted(all_ailments.items(), key=lambda x: -x[1])[:10]:
            print(f"    {ailment:<25} {count:>4}")

    print(f"{'=' * 60}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Simulate tamagotchi pet lifecycle")
    parser.add_argument("--strategy", default="all", choices=["smart", "casual", "neglectful", "all"])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    strategies = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    for strat in strategies:
        run_batch(strat, num_runs=args.runs, max_days=args.days, verbose=args.verbose)


if __name__ == "__main__":
    main()
