"""Tamagotchi Discord cog with all commands."""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import time
from datetime import datetime, timezone

from discord.ext import commands

from .constants import (
    AILMENTS,
    CLOTHING,
    CLOTHING_PRICES,
    COLORS,
    DECORATE_PRICE,
    FOOD_PRICES,
    FOODS,
    MEDICINE_PRICES,
    MEDICINES,
    PERSONALITY_TRAITS,
    TOY_PRICES,
    TOYS,
    TRICKS,
    WORK_COOLDOWN,
    WORK_PAY_MAX,
    WORK_PAY_MIN,
)
from .database import (
    DB_PATH,
    _add_ailment,
    _get_ailments,
    _get_pet,
    _init_db,
    _remove_ailment,
    _save_pet,
)
from .display import _build_status
from .helpers import (
    _clamp,
    _get_age_days,
    _get_effective_room_temp,
    _get_trait_modifier,
)
from .tick import _tick_pet

_LOG = logging.getLogger(__name__)

__all__ = ["TamagotchiCog"]


class TamagotchiCog(commands.Cog):
    """Stats only decay when a player runs a !pet command (on-interaction tick).
    No background loop - players who go offline won't come back to a dead pet.
    Decay is capped at 4 hours max between interactions.
    """

    def __init__(self, bot):
        self.bot = bot
        _init_db()

    # -- Helper to get pet or send error --
    async def _get_pet_or_error(self, ctx):
        conn = sqlite3.connect(DB_PATH)
        pet = _get_pet(conn, ctx.author.id)
        if pet is None:
            await ctx.send("You don't have a pet! Use `!pet adopt <name>` to get one.")
            conn.close()
            return None, None
        if pet["died_at"] is not None:
            await ctx.send(
                f"**{pet['name']}** is dead (cause: {pet['cause_of_death']}). "
                f"Use `!pet bury` to lay them to rest and adopt a new one."
            )
            conn.close()
            return None, None
        return conn, pet

    def _has_depression(self, conn, user_id):
        row = conn.execute(
            "SELECT 1 FROM tamagotchi_ailments WHERE user_id = ? AND ailment = 'depression'",
            (user_id,),
        ).fetchone()
        return row is not None

    def _depression_mult(self, conn, user_id):
        """Returns 0.5 if depressed (positive actions halved), else 1.0."""
        return 0.5 if self._has_depression(conn, user_id) else 1.0

    # -----------------------------------------------------------------------
    # COMMANDS
    # -----------------------------------------------------------------------

    @commands.command(name="pet")
    async def pet_cmd(self, ctx, *, args: str = ""):
        """Main pet command router."""
        parts = args.strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else "status"
        rest = parts[1] if len(parts) > 1 else ""

        dispatch = {
            "status": self._cmd_status,
            "adopt": self._cmd_adopt,
            "feed": self._cmd_feed,
            "water": self._cmd_water,
            "play": self._cmd_play,
            "sleep": self._cmd_sleep,
            "wake": self._cmd_wake,
            "clean": self._cmd_clean,
            "brush": self._cmd_brush,
            "trim": self._cmd_trim,
            "toilet": self._cmd_toilet,
            "scoop": self._cmd_scoop,
            "medicine": self._cmd_medicine,
            "dress": self._cmd_dress,
            "teach": self._cmd_teach,
            "perform": self._cmd_perform,
            "exercise": self._cmd_exercise,
            "talk": self._cmd_talk,
            "praise": self._cmd_praise,
            "scold": self._cmd_scold,
            "lights": self._cmd_lights,
            "decorate": self._cmd_decorate,
            "thermostat": self._cmd_thermostat,
            "foods": self._cmd_foods,
            "toys": self._cmd_toys,
            "tricks": self._cmd_tricks,
            "ailments": self._cmd_ailments,
            "work": self._cmd_work,
            "balance": self._cmd_balance,
            "start": self._cmd_adopt,
            "bury": self._cmd_bury,
            "help": self._cmd_help,
        }

        handler = dispatch.get(sub)
        if handler:
            await handler(ctx, rest)
        else:
            await ctx.send(f"Unknown command `{sub}`. Use `!pet help` for a list of commands.")

    async def _cmd_status(self, ctx, _rest):
        conn = sqlite3.connect(DB_PATH)
        pet = _get_pet(conn, ctx.author.id)
        if pet is None:
            await ctx.send("You don't have a pet! Use `!pet adopt <name>` to get one.")
            conn.close()
            return
        # Run a tick first to update stats
        events = _tick_pet(conn, pet)
        pet = _get_pet(conn, ctx.author.id)  # Re-fetch after tick
        ailments = _get_ailments(conn, ctx.author.id)
        conn.close()
        status = _build_status(pet, ailments)
        if events:
            event_text = "\n".join(events)
            await ctx.send(f"{event_text}\n{status}")
        else:
            await ctx.send(status)

    async def _cmd_adopt(self, ctx, name):
        if not name:
            await ctx.send("Usage: `!pet adopt <name>`")
            return
        name = name.strip()[:30]

        conn = sqlite3.connect(DB_PATH)
        existing = _get_pet(conn, ctx.author.id)
        if existing and existing["died_at"] is None:
            await ctx.send(f"You already have a living pet named **{existing['name']}**!")
            conn.close()
            return
        if existing and existing["died_at"] is not None:
            await ctx.send(f"You still have **{existing['name']}**'s remains. Use `!pet bury` first.")
            conn.close()
            return

        now = time.time()
        # Generate random personality
        trait_list = random.sample(list(PERSONALITY_TRAITS.keys()), k=random.randint(2, 3))
        food_list = list(FOODS.keys())
        loves = random.sample(food_list, k=2)
        remaining = [f for f in food_list if f not in loves]
        hates = random.sample(remaining, k=2)
        remaining2 = [f for f in remaining if f not in hates]
        allergy = random.choice(remaining2) if remaining2 else None
        fav_color = random.choice(COLORS)

        conn.execute(
            """INSERT INTO tamagotchi_pets (
                user_id, name, born_at, last_tick, nutrition_day,
                traits, food_loves, food_hates, allergy, favorite_color,
                weather_changed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ctx.author.id,
                name,
                now,
                now,
                datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                json.dumps(trait_list),
                json.dumps(loves),
                json.dumps(hates),
                allergy,
                fav_color,
                now,
            ),
        )
        conn.commit()
        conn.close()

        await ctx.send(
            f"You adopted **{name}**! They're a tiny baby blob.\n"
            f"Use `!pet` to check on them and `!pet help` for commands.\n"
            f"They have secret personality traits, food preferences, and an allergy. Good luck figuring them out!"
        )

    async def _cmd_feed(self, ctx, food_name):
        if not food_name:
            await ctx.send("Usage: `!pet feed <food>` - see `!pet foods` for options")
            return
        food_name = food_name.strip().lower()
        food = FOODS.get(food_name)
        if not food:
            await ctx.send(f"Unknown food `{food_name}`. Use `!pet foods` to see options.")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        # Run tick first
        _tick_pet(conn, pet)
        pet = _get_pet(conn, ctx.author.id)

        # Refusal checks
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping! Wake them up first.")
            conn.close()
            return
        price = FOOD_PRICES.get(food_name, 10)
        if pet["money"] < price:
            await ctx.send(f"Can't afford {food_name} ({price} coins)! You have {pet['money']:.0f}. Use `!pet work`.")
            conn.close()
            return
        if pet["bladder"] < 10:
            await ctx.send(f"**{pet['name']}** refuses to eat - they need the toilet first!")
            conn.close()
            return

        msgs = []
        mult = self._depression_mult(conn, ctx.author.id)

        # Deduct cost
        pet["money"] -= price

        # Apply nutrition
        pet["hunger"] = _clamp(pet["hunger"] + food["hunger_restore"] * mult)
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

        # Happiness from food
        happiness_boost = food["happiness_boost"] * mult

        # Check if it's a loved/hated food
        loves = json.loads(pet["food_loves"]) if isinstance(pet["food_loves"], str) else pet["food_loves"]
        hates = json.loads(pet["food_hates"]) if isinstance(pet["food_hates"], str) else pet["food_hates"]

        if food_name in loves:
            happiness_boost *= 2
            msgs.append(f"**{pet['name']}** LOVES {food_name}!")
        elif food_name in hates:
            hate_mult = _get_trait_modifier(pet, "food_mood_penalty", 1.0)
            happiness_boost = -abs(happiness_boost) * hate_mult
            msgs.append(f"**{pet['name']}** HATES {food_name}! Mood tanked.")

        pet["happiness"] = _clamp(pet["happiness"] + happiness_boost)
        pet["mood"] = _clamp(pet["mood"] + happiness_boost * 0.5)

        # Track food repetition
        foods_today = (
            json.loads(pet["foods_eaten_today"])
            if isinstance(pet["foods_eaten_today"], str)
            else pet["foods_eaten_today"]
        )
        foods_today[food_name] = foods_today.get(food_name, 0) + 1
        pet["foods_eaten_today"] = json.dumps(foods_today)

        if foods_today[food_name] >= 3:
            pet["boredom"] = _clamp(pet["boredom"] + 10)
            msgs.append(f"**{pet['name']}** is getting bored of {food_name}...")
            if foods_today[food_name] >= 4:
                if random.random() < 0.4:
                    _add_ailment(conn, ctx.author.id, "indigestion")
                    msgs.append(f"**{pet['name']}** got indigestion from eating too much {food_name}!")

        # Allergy check
        if pet["allergy"] and food_name == pet["allergy"]:
            _add_ailment(conn, ctx.author.id, "allergic reaction")
            msgs.append(f"**{pet['name']}** is having an ALLERGIC REACTION to {food_name}!")

        # Feeding with low hygiene = food poisoning risk
        if pet["hygiene"] < 20 and random.random() < 0.3:
            _add_ailment(conn, ctx.author.id, "food poisoning")
            msgs.append(f"**{pet['name']}** got food poisoning! (too dirty to eat safely)")

        _save_pet(conn, pet)
        conn.close()

        base_msg = f"Fed **{pet['name']}** some {food_name}! (-{price} coins, +{food['hunger_restore']} hunger, {food['calories']} cal)"
        if msgs:
            await ctx.send(base_msg + "\n" + "\n".join(msgs))
        else:
            await ctx.send(base_msg)

    async def _cmd_water(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        _tick_pet(conn, pet)
        pet = _get_pet(conn, ctx.author.id)
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping!")
            conn.close()
            return

        mult = self._depression_mult(conn, ctx.author.id)
        pet["thirst"] = _clamp(pet["thirst"] + 30 * mult)
        pet["bladder"] = _clamp(pet["bladder"] - 10)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"**{pet['name']}** drank some water! (+30 thirst)")

    async def _cmd_play(self, ctx, toy_name):
        if not toy_name:
            await ctx.send("Usage: `!pet play <toy>` - see `!pet toys` for options")
            return
        toy_name = toy_name.strip().lower()
        toy = TOYS.get(toy_name)
        if not toy:
            await ctx.send(f"Unknown toy `{toy_name}`. Use `!pet toys` to see options.")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping!")
            conn.close()
            return

        price = TOY_PRICES.get(toy_name, 5)
        if pet["money"] < price:
            await ctx.send(f"Can't afford {toy_name} ({price} coins)! You have {pet['money']:.0f}. Use `!pet work`.")
            conn.close()
            return
        pet["money"] -= price

        mult = self._depression_mult(conn, ctx.author.id)
        msgs = []

        # Check toy uses today
        toy_uses = (
            json.loads(pet["toy_uses_today"]) if isinstance(pet["toy_uses_today"], str) else pet["toy_uses_today"]
        )
        uses = toy_uses.get(toy_name, 0)
        diminishing = max(0.2, 1.0 - uses * 0.25)  # 100%, 75%, 50%, 25%, 20%...
        if uses >= 3:
            msgs.append(f"**{pet['name']}** is getting bored of the {toy_name}... (diminishing returns)")

        toy_uses[toy_name] = uses + 1
        pet["toy_uses_today"] = json.dumps(toy_uses)

        # Apply stat boosts
        for stat, boost in toy.items():
            if stat in ("max_uses", "desc"):
                continue
            if stat in pet:
                effective = boost * mult * diminishing
                pet[stat] = _clamp(pet[stat] + effective)

        pet["social"] = _clamp(pet["social"] + 5 * mult)

        _save_pet(conn, pet)
        conn.close()
        base = f"**{pet['name']}** played with the {toy_name}!"
        if msgs:
            await ctx.send(base + "\n" + "\n".join(msgs))
        else:
            await ctx.send(base)

    async def _cmd_sleep(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is already sleeping!")
            conn.close()
            return

        pet["is_sleeping"] = 1
        pet["sleep_started_at"] = time.time()

        msgs = [f"**{pet['name']}** is going to sleep..."]
        if pet["lights_on"]:
            msgs.append("The lights are still on! Sleep quality will suffer. (!pet lights off)")
        if pet["current_clothing"] != "pajamas":
            msgs.append("Not wearing pajamas... comfort won't recover as well.")
        if pet["teeth_health"] < 50:
            msgs.append("Didn't brush teeth before bed! Teeth health will keep declining.")

        _save_pet(conn, pet)
        conn.close()
        await ctx.send("\n".join(msgs))

    async def _cmd_wake(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if not pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is already awake!")
            conn.close()
            return

        pet["is_sleeping"] = 0
        sleep_duration = time.time() - (pet["sleep_started_at"] or time.time())
        sleep_hours = sleep_duration / 3600

        msgs = [f"**{pet['name']}** woke up! (slept {sleep_hours:.1f} hours)"]
        if sleep_hours < 2:
            pet["mood"] = _clamp(pet["mood"] - 15)
            msgs.append("Woken up too early! Mood dropped.")
        elif sleep_hours > 8:
            pet["comfort"] = _clamp(pet["comfort"] + 10)
            msgs.append("Well rested!")

        _save_pet(conn, pet)
        conn.close()
        await ctx.send("\n".join(msgs))

    async def _cmd_clean(self, ctx, rest):
        # "clean room" is separate
        if rest.strip().lower() == "room":
            await self._cmd_clean_room(ctx, "")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping!")
            conn.close()
            return

        mult = self._depression_mult(conn, ctx.author.id)
        pet["hygiene"] = _clamp(pet["hygiene"] + 40 * mult)
        pet["comfort"] = _clamp(pet["comfort"] + 10 * mult)
        pet["skin_condition"] = _clamp(pet["skin_condition"] + 5 * mult)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"**{pet['name']}** had a bath! Squeaky clean. (+40 hygiene)")

    async def _cmd_brush(self, ctx, rest):
        what = rest.strip().lower()
        if what not in ("teeth", "fur"):
            await ctx.send("Usage: `!pet brush teeth` or `!pet brush fur`")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping!")
            conn.close()
            return

        mult = self._depression_mult(conn, ctx.author.id)
        if what == "teeth":
            pet["teeth_health"] = _clamp(pet["teeth_health"] + 20 * mult)
            _save_pet(conn, pet)
            conn.close()
            await ctx.send(f"**{pet['name']}**'s teeth are sparkling! (+20 teeth health)")
        else:
            pet["fur_condition"] = _clamp(pet["fur_condition"] + 25 * mult)
            pet["hygiene"] = _clamp(pet["hygiene"] + 5 * mult)
            _save_pet(conn, pet)
            conn.close()
            await ctx.send(f"**{pet['name']}**'s fur is silky smooth! (+25 fur condition)")

    async def _cmd_trim(self, ctx, rest):
        if rest.strip().lower() != "nails" and rest.strip():
            await ctx.send("Usage: `!pet trim nails`")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        old_nails = pet["nail_length"]
        pet["nail_length"] = _clamp(pet["nail_length"] - 40)
        pet["comfort"] = _clamp(pet["comfort"] + 5)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"**{pet['name']}**'s nails are trimmed! ({old_nails:.0f}% -> {pet['nail_length']:.0f}%)")

    async def _cmd_toilet(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping!")
            conn.close()
            return

        old_bladder = pet["bladder"]
        pet["bladder"] = _clamp(pet["bladder"] + 50)
        pet["comfort"] = _clamp(pet["comfort"] + 10)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"**{pet['name']}** used the toilet! (+50 bladder, {old_bladder:.0f}% -> {pet['bladder']:.0f}%)")

    async def _cmd_scoop(self, ctx, rest):
        # Also handle "scoop poop"
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        if pet["poop_count"] == 0:
            await ctx.send("No poops to clean up!")
            conn.close()
            return

        count = pet["poop_count"]
        pet["poop_count"] = 0
        pet["room_cleanliness"] = _clamp(pet["room_cleanliness"] + 10)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"Scooped {count} poop(s)! Room is a bit cleaner.")

    async def _cmd_clean_room(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        pet["room_cleanliness"] = _clamp(pet["room_cleanliness"] + 35)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"Cleaned **{pet['name']}**'s room! (+35 cleanliness)")

    async def _cmd_medicine(self, ctx, med_name):
        if not med_name:
            await ctx.send("Usage: `!pet medicine <type>` - see `!pet ailments` for what you need")
            return
        med_name = med_name.strip().lower()
        med = MEDICINES.get(med_name)
        if not med:
            await ctx.send(f"Unknown medicine `{med_name}`. Available: {', '.join(MEDICINES.keys())}")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        price = MEDICINE_PRICES.get(med_name, 15)
        if pet["money"] < price:
            await ctx.send(f"Can't afford {med_name} ({price} coins)! You have {pet['money']:.0f}. Use `!pet work`.")
            conn.close()
            return

        pet["money"] -= price
        ailments = _get_ailments(conn, ctx.author.id)
        ailment_names = {a["ailment"] for a in ailments}

        cured = []
        for curable in med["cures"]:
            if curable in ailment_names:
                _remove_ailment(conn, ctx.author.id, curable)
                cured.append(curable)

        _save_pet(conn, pet)
        conn.close()
        if cured:
            await ctx.send(f"Gave **{pet['name']}** {med_name}! (-{price} coins) Cured: {', '.join(cured)}")
        else:
            await ctx.send(
                f"Gave **{pet['name']}** {med_name} (-{price} coins), but they don't have any ailments it treats."
            )

    async def _cmd_dress(self, ctx, clothing_name):
        if not clothing_name:
            await ctx.send(f"Usage: `!pet dress <clothing>`\nOptions: {', '.join(CLOTHING.keys())}")
            return
        clothing_name = clothing_name.strip().lower()
        if clothing_name not in CLOTHING:
            await ctx.send(f"Unknown clothing. Options: {', '.join(CLOTHING.keys())}")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        price = CLOTHING_PRICES.get(clothing_name, 5)
        if pet["money"] < price:
            await ctx.send(
                f"Can't afford {clothing_name} ({price} coins)! You have {pet['money']:.0f}. Use `!pet work`."
            )
            conn.close()
            return
        pet["money"] -= price

        pet["current_clothing"] = clothing_name
        info = CLOTHING[clothing_name]

        cost_str = f" (-{price} coins)" if price > 0 else ""
        msgs = [f"**{pet['name']}** is now wearing: {clothing_name.title()} (warmth: {info['warmth']:+d}C){cost_str}"]

        weather = pet["current_weather"]
        if info["best_weather"] and weather not in info["best_weather"]:
            msgs.append(f"Hmm, {clothing_name} might not be ideal for {weather} weather...")

        _save_pet(conn, pet)
        conn.close()
        await ctx.send("\n".join(msgs))

    async def _cmd_teach(self, ctx, trick_name):
        if not trick_name:
            await ctx.send(f"Usage: `!pet teach <trick>`\nAvailable: {', '.join(TRICKS)}")
            return
        trick_name = trick_name.strip().lower()
        if trick_name not in TRICKS:
            await ctx.send(f"Unknown trick. Available: {', '.join(TRICKS)}")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping!")
            conn.close()
            return

        if pet["trust"] < 30:
            await ctx.send(
                f"**{pet['name']}** doesn't trust you enough to learn! (trust: {pet['trust']:.0f}%, need 30%)"
            )
            conn.close()
            return

        tricks = json.loads(pet["tricks_known"]) if isinstance(pet["tricks_known"], str) else pet["tricks_known"]
        practice = (
            json.loads(pet["last_trick_practice"])
            if isinstance(pet["last_trick_practice"], str)
            else pet["last_trick_practice"]
        )

        current = tricks.get(trick_name, 0)
        if current >= 100:
            await ctx.send(f"**{pet['name']}** has already mastered {trick_name}!")
            conn.close()
            return

        # Learning amount based on intelligence, trust, discipline
        base_learn = 8
        intel_bonus = pet["intelligence"] / 100 * 5
        trust_bonus = pet["trust"] / 100 * 3
        learn_rate = _get_trait_modifier(pet, "learning_rate", 1.0)
        mult = self._depression_mult(conn, ctx.author.id)
        total_learn = (base_learn + intel_bonus + trust_bonus) * learn_rate * mult

        new_prof = min(100, current + total_learn)
        tricks[trick_name] = new_prof
        practice[trick_name] = time.time()
        pet["tricks_known"] = json.dumps(tricks)
        pet["last_trick_practice"] = json.dumps(practice)
        pet["intelligence"] = _clamp(pet["intelligence"] + 2 * mult)
        pet["boredom"] = _clamp(pet["boredom"] - 5)

        _save_pet(conn, pet)
        conn.close()

        if new_prof >= 100:
            await ctx.send(f"**{pet['name']}** MASTERED {trick_name}!")
        else:
            await ctx.send(
                f"Training {trick_name}... **{pet['name']}** is at {new_prof:.0f}% proficiency (+{total_learn:.0f})"
            )

    async def _cmd_perform(self, ctx, trick_name):
        if not trick_name:
            await ctx.send("Usage: `!pet perform <trick>`")
            return
        trick_name = trick_name.strip().lower()

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping!")
            conn.close()
            return

        tricks = json.loads(pet["tricks_known"]) if isinstance(pet["tricks_known"], str) else pet["tricks_known"]
        prof = tricks.get(trick_name, 0)

        if prof <= 0:
            await ctx.send(f"**{pet['name']}** doesn't know {trick_name}! Use `!pet teach {trick_name}`")
            conn.close()
            return

        # Success chance based on proficiency
        if random.random() * 100 < prof:
            mult = self._depression_mult(conn, ctx.author.id)
            pet["happiness"] = _clamp(pet["happiness"] + 8 * mult)
            pet["social"] = _clamp(pet["social"] + 5 * mult)
            pet["mood"] = _clamp(pet["mood"] + 5 * mult)
            practice = (
                json.loads(pet["last_trick_practice"])
                if isinstance(pet["last_trick_practice"], str)
                else pet["last_trick_practice"]
            )
            practice[trick_name] = time.time()
            pet["last_trick_practice"] = json.dumps(practice)
            _save_pet(conn, pet)
            conn.close()
            await ctx.send(f"**{pet['name']}** performed {trick_name} perfectly!")
        else:
            pet["mood"] = _clamp(pet["mood"] - 3)
            _save_pet(conn, pet)
            conn.close()
            await ctx.send(f"**{pet['name']}** tried to {trick_name} but failed... (proficiency: {prof:.0f}%)")

    async def _cmd_exercise(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping!")
            conn.close()
            return

        msgs = []
        mult = self._depression_mult(conn, ctx.author.id)

        if pet["energy"] < 20:
            if random.random() < 0.4:
                pet["fitness"] = _clamp(pet["fitness"] - 10)
                msgs.append(f"**{pet['name']}** pushed too hard and got injured! (-10 fitness)")
            else:
                msgs.append(f"**{pet['name']}** is too tired to exercise properly.")
                mult *= 0.3

        fitness_gain = 8 * mult
        energy_cost = 15
        pet["fitness"] = _clamp(pet["fitness"] + fitness_gain)
        pet["energy"] = _clamp(pet["energy"] - energy_cost)
        pet["hunger"] = _clamp(pet["hunger"] - 10)
        pet["thirst"] = _clamp(pet["thirst"] - 8)
        pet["boredom"] = _clamp(pet["boredom"] - 10)
        pet["hygiene"] = _clamp(pet["hygiene"] - 5)  # Sweaty

        # Weight loss from exercise
        pet["weight"] = max(0.5, pet["weight"] - 0.02)

        base = f"**{pet['name']}** exercised! (+{fitness_gain:.0f} fitness, -{energy_cost} energy)"
        _save_pet(conn, pet)
        conn.close()
        if msgs:
            await ctx.send("\n".join(msgs) + "\n" + base)
        else:
            await ctx.send(base)

    async def _cmd_talk(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return
        if pet["is_sleeping"]:
            await ctx.send(f"**{pet['name']}** is sleeping! Shh!")
            conn.close()
            return

        mult = self._depression_mult(conn, ctx.author.id)
        pet["social"] = _clamp(pet["social"] + 15 * mult)
        pet["happiness"] = _clamp(pet["happiness"] + 5 * mult)
        pet["mood"] = _clamp(pet["mood"] + 5 * mult)
        pet["boredom"] = _clamp(pet["boredom"] - 5)
        pet["trust"] = _clamp(pet["trust"] + 2 * mult)

        _save_pet(conn, pet)
        conn.close()
        await ctx.send(
            f"You talked to **{pet['name']}**! They seem to appreciate the company. (+15 social, +5 happiness)"
        )

    async def _cmd_praise(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        mult = self._depression_mult(conn, ctx.author.id)
        pet["trust"] = _clamp(pet["trust"] + 5 * mult)
        pet["mood"] = _clamp(pet["mood"] + 8 * mult)
        pet["happiness"] = _clamp(pet["happiness"] + 5 * mult)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"You praised **{pet['name']}**! (+5 trust, +8 mood)")

    async def _cmd_scold(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        learn_rate = _get_trait_modifier(pet, "learning_rate", 1.0)
        pet["discipline"] = _clamp(pet["discipline"] + 10 * learn_rate)
        pet["trust"] = _clamp(pet["trust"] - 5)
        pet["mood"] = _clamp(pet["mood"] - 10)
        pet["happiness"] = _clamp(pet["happiness"] - 3)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"You scolded **{pet['name']}**. (+10 discipline, -5 trust, -10 mood)")

    async def _cmd_lights(self, ctx, rest):
        rest = rest.strip().lower()
        if rest not in ("on", "off"):
            await ctx.send("Usage: `!pet lights on` or `!pet lights off`")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        pet["lights_on"] = 1 if rest == "on" else 0
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"Lights turned **{rest.upper()}**.")

    async def _cmd_decorate(self, ctx, color):
        if not color:
            await ctx.send(f"Usage: `!pet decorate <color>`\nColors: {', '.join(COLORS)}")
            return
        color = color.strip().lower()
        if color not in COLORS:
            await ctx.send(f"Unknown color. Options: {', '.join(COLORS)}")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        if pet["money"] < DECORATE_PRICE:
            await ctx.send(
                f"Can't afford decoration ({DECORATE_PRICE} coins)! You have {pet['money']:.0f}. Use `!pet work`."
            )
            conn.close()
            return
        pet["money"] -= DECORATE_PRICE

        boost = 15
        if color == pet["favorite_color"]:
            boost = 35
            msg = f"Decorated **{pet['name']}**'s room in {color}! They LOVE it!! (+{boost} decor)"
        else:
            msg = f"Decorated **{pet['name']}**'s room in {color}. (+{boost} decor)"

        pet["decoration_score"] = _clamp(pet["decoration_score"] + boost)
        pet["decoration_color"] = color
        pet["mood"] = _clamp(pet["mood"] + boost * 0.3)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(msg)

    async def _cmd_thermostat(self, ctx, temp_str):
        if not temp_str:
            await ctx.send("Usage: `!pet thermostat <temp>` (in celsius, 10-35)")
            return
        try:
            temp = float(temp_str.strip())
        except ValueError:
            await ctx.send("Please provide a number (celsius).")
            return
        if temp < 10 or temp > 35:
            await ctx.send("Thermostat range is 10-35 C.")
            return

        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        pet["room_temp"] = temp
        eff = _get_effective_room_temp(pet)
        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"Thermostat set to {temp:.0f}C. (Effective room temp with weather: {eff:.0f}C)")

    async def _cmd_foods(self, ctx, _rest):
        lines = ["```", "-- AVAILABLE FOODS --"]
        for name, food in FOODS.items():
            price = FOOD_PRICES.get(name, "?")
            lines.append(
                f"  {name:<12} {price:>2}c | {food['calories']:>3} cal | "
                f"P:{food['protein']:>2} C:{food['carbs']:>2} F:{food['fat']:>2} Fi:{food['fiber']:>2} | "
                f"{food['desc']}"
            )
        lines.append("")
        lines.append("  Daily targets: ~300 cal, P:50 C:80 F:30 Fi:20")
        lines.append("```")
        await ctx.send("\n".join(lines))

    async def _cmd_toys(self, ctx, _rest):
        lines = ["```", "-- AVAILABLE TOYS --"]
        for name, toy in TOYS.items():
            price = TOY_PRICES.get(name, "?")
            effects = []
            for stat, val in toy.items():
                if stat in ("max_uses", "desc"):
                    continue
                sign = "+" if val > 0 else ""
                effects.append(f"{stat}:{sign}{val}")
            lines.append(f"  {name:<14} {price}c | {', '.join(effects)}")
            lines.append(f"                 {toy['desc']}")
        lines.append("```")
        await ctx.send("\n".join(lines))

    async def _cmd_tricks(self, ctx, _rest):
        conn = sqlite3.connect(DB_PATH)
        pet = _get_pet(conn, ctx.author.id)
        conn.close()
        if not pet:
            await ctx.send("You don't have a pet!")
            return

        tricks = json.loads(pet["tricks_known"]) if isinstance(pet["tricks_known"], str) else pet["tricks_known"]
        if not tricks:
            await ctx.send(
                f"**{pet['name']}** doesn't know any tricks yet! Use `!pet teach <trick>`\nAvailable: {', '.join(TRICKS)}"
            )
            return

        lines = [f"**{pet['name']}**'s tricks:"]
        for t, prof in tricks.items():
            status = "MASTERED" if prof >= 100 else f"{prof:.0f}%"
            lines.append(f"  {t.title()}: {status}")
        lines.append(f"\nTeachable: {', '.join(t for t in TRICKS if t not in tricks)}")
        await ctx.send("\n".join(lines))

    async def _cmd_ailments(self, ctx, _rest):
        conn = sqlite3.connect(DB_PATH)
        pet = _get_pet(conn, ctx.author.id)
        if not pet:
            await ctx.send("You don't have a pet!")
            conn.close()
            return
        ailments = _get_ailments(conn, ctx.author.id)
        conn.close()

        if not ailments:
            await ctx.send(f"**{pet['name']}** is healthy! No ailments.")
            return

        lines = [f"**{pet['name']}**'s ailments:"]
        for a in ailments:
            info = AILMENTS.get(a["ailment"], {})
            cure = info.get("cure")
            cure_str = f"Cure: `!pet medicine {cure}`" if cure else info.get("desc", "No medicine cure")
            lines.append(f"  **{a['ailment'].title()}** - severity: {a['severity']:.0f}% - {cure_str}")
        lines.append(f"\nAvailable medicines: {', '.join(MEDICINES.keys())}")
        await ctx.send("\n".join(lines))

    async def _cmd_work(self, ctx, _rest):
        conn, pet = await self._get_pet_or_error(ctx)
        if not pet:
            return

        now = time.time()
        cooldown_left = (pet["last_work_at"] or 0) + WORK_COOLDOWN - now
        if cooldown_left > 0:
            mins = int(cooldown_left / 60)
            await ctx.send(f"**{pet['name']}** is tired from working! Try again in {mins} minutes.")
            conn.close()
            return

        pay = random.randint(WORK_PAY_MIN, WORK_PAY_MAX)
        pet["money"] += pay
        pet["last_work_at"] = now
        pet["energy"] = _clamp(pet["energy"] - 10)
        pet["boredom"] = _clamp(pet["boredom"] - 5)

        jobs = [
            "delivered packages",
            "washed dishes",
            "walked dogs",
            "sorted mail",
            "mowed lawns",
            "sold lemonade",
            "busked on the corner",
            "did data entry",
            "babysat kittens",
            "organized a closet",
            "painted a fence",
            "tutored math",
        ]
        job = random.choice(jobs)

        _save_pet(conn, pet)
        conn.close()
        await ctx.send(f"**{pet['name']}** {job} and earned **{pay} coins**! (Balance: {pet['money']:.0f})")

    async def _cmd_balance(self, ctx, _rest):
        conn = sqlite3.connect(DB_PATH)
        pet = _get_pet(conn, ctx.author.id)
        conn.close()
        if not pet:
            await ctx.send("You don't have a pet!")
            return
        await ctx.send(f"**{pet['name']}** has **{pet['money']:.0f} coins**.")

    async def _cmd_bury(self, ctx, _rest):
        conn = sqlite3.connect(DB_PATH)
        pet = _get_pet(conn, ctx.author.id)
        if not pet:
            await ctx.send("You don't have a pet to bury.")
            conn.close()
            return
        if pet["died_at"] is None:
            await ctx.send(f"**{pet['name']}** is still alive! You monster.")
            conn.close()
            return

        age_days = _get_age_days(pet)
        conn.execute("DELETE FROM tamagotchi_ailments WHERE user_id = ?", (ctx.author.id,))
        conn.execute("DELETE FROM tamagotchi_pets WHERE user_id = ?", (ctx.author.id,))
        conn.commit()
        conn.close()
        await ctx.send(
            f"**{pet['name']}** has been laid to rest. They lived for {age_days:.1f} days.\n"
            f"Use `!pet adopt <name>` to welcome a new friend."
        )

    async def _cmd_help(self, ctx, _rest):
        help_text = """```
-- PET COMMANDS --
  !pet / !pet status     Show all stats
  !pet start <name>      Adopt a new pet
  !pet work              Earn coins (1hr cooldown)
  !pet balance           Check your coins
  !pet feed <food>       Feed your pet ($)
  !pet water             Give water (free)
  !pet play <toy>        Play with a toy ($)
  !pet sleep / !pet wake Sleep management
  !pet clean             Bathe your pet (free)
  !pet clean room        Clean the room (free)
  !pet brush teeth/fur   Grooming (free)
  !pet trim nails        Trim nails (free)
  !pet toilet            Let pet use bathroom
  !pet scoop poop        Clean up poop
  !pet medicine <type>   Give medicine ($)
  !pet dress <clothing>  Change clothes ($)
  !pet teach <trick>     Teach a trick (free)
  !pet perform <trick>   Perform a trick
  !pet exercise          Exercise (free)
  !pet talk              Chat with your pet
  !pet praise / scold    Discipline
  !pet lights on/off     Toggle room lights
  !pet decorate <color>  Decorate room ($)
  !pet thermostat <temp> Set room temperature
  !pet foods             List foods + prices
  !pet toys              List toys + prices
  !pet tricks            List known tricks
  !pet ailments          Show ailments & cures
  !pet bury              Bury a dead pet

-- TIPS --
  * You need MONEY for food, toys, clothes,
    medicine, and decoration. Use !pet work
  * Stats decay over time - check regularly!
  * Your pet has SECRET personality traits,
    food preferences, and an allergy
  * Sleep with lights off and pajamas on
  * Weather affects room temperature
  * Depression halves all positive actions
  * Your pet WILL die if neglected
```"""
        await ctx.send(help_text)
