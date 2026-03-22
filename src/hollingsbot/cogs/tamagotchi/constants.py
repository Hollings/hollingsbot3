"""Tamagotchi constants: foods, toys, clothing, weather, ailments, prices, etc."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# FOODS - name -> {calories, protein, carbs, fat, fiber, vit_a..vit_e, happiness_boost}
# Nutritional values are per serving. Daily targets: ~300 cal, 50g protein,
# 80g carbs, 30g fat, 20g fiber, 100% each vitamin.
# ---------------------------------------------------------------------------
FOODS: dict[str, dict] = {
    "kibble": {
        "calories": 45,
        "protein": 8,
        "carbs": 12,
        "fat": 3,
        "fiber": 4,
        "vit_a": 10,
        "vit_b": 10,
        "vit_c": 5,
        "vit_d": 5,
        "vit_e": 10,
        "happiness_boost": 2,
        "hunger_restore": 20,
        "desc": "Basic balanced meal",
    },
    "steak": {
        "calories": 120,
        "protein": 25,
        "carbs": 0,
        "fat": 12,
        "fiber": 0,
        "vit_a": 5,
        "vit_b": 30,
        "vit_c": 0,
        "vit_d": 5,
        "vit_e": 3,
        "happiness_boost": 15,
        "hunger_restore": 35,
        "desc": "Juicy and delicious",
    },
    "fish": {
        "calories": 70,
        "protein": 18,
        "carbs": 0,
        "fat": 4,
        "fiber": 0,
        "vit_a": 5,
        "vit_b": 15,
        "vit_c": 0,
        "vit_d": 40,
        "vit_e": 5,
        "happiness_boost": 10,
        "hunger_restore": 25,
        "desc": "Rich in vitamin D",
    },
    "salad": {
        "calories": 20,
        "protein": 2,
        "carbs": 5,
        "fat": 0,
        "fiber": 8,
        "vit_a": 30,
        "vit_b": 5,
        "vit_c": 25,
        "vit_d": 0,
        "vit_e": 10,
        "happiness_boost": 1,
        "hunger_restore": 10,
        "desc": "Leafy and healthy",
    },
    "cheese": {
        "calories": 80,
        "protein": 7,
        "carbs": 1,
        "fat": 14,
        "fiber": 0,
        "vit_a": 10,
        "vit_b": 5,
        "vit_c": 0,
        "vit_d": 3,
        "vit_e": 2,
        "happiness_boost": 8,
        "hunger_restore": 15,
        "desc": "Rich and fatty",
    },
    "bread": {
        "calories": 55,
        "protein": 3,
        "carbs": 22,
        "fat": 2,
        "fiber": 3,
        "vit_a": 0,
        "vit_b": 10,
        "vit_c": 0,
        "vit_d": 0,
        "vit_e": 2,
        "happiness_boost": 4,
        "hunger_restore": 18,
        "desc": "Carb-heavy staple",
    },
    "fruit": {
        "calories": 30,
        "protein": 1,
        "carbs": 12,
        "fat": 0,
        "fiber": 5,
        "vit_a": 10,
        "vit_b": 3,
        "vit_c": 35,
        "vit_d": 0,
        "vit_e": 5,
        "happiness_boost": 6,
        "hunger_restore": 12,
        "desc": "Sweet and nutritious",
    },
    "eggs": {
        "calories": 60,
        "protein": 12,
        "carbs": 1,
        "fat": 8,
        "fiber": 0,
        "vit_a": 8,
        "vit_b": 25,
        "vit_c": 0,
        "vit_d": 10,
        "vit_e": 5,
        "happiness_boost": 5,
        "hunger_restore": 18,
        "desc": "Protein-packed",
    },
    "rice": {
        "calories": 50,
        "protein": 2,
        "carbs": 25,
        "fat": 0,
        "fiber": 1,
        "vit_a": 0,
        "vit_b": 5,
        "vit_c": 0,
        "vit_d": 0,
        "vit_e": 0,
        "happiness_boost": 2,
        "hunger_restore": 20,
        "desc": "Plain but filling",
    },
    "candy": {
        "calories": 90,
        "protein": 0,
        "carbs": 40,
        "fat": 3,
        "fiber": 0,
        "vit_a": 0,
        "vit_b": 0,
        "vit_c": 0,
        "vit_d": 0,
        "vit_e": 0,
        "happiness_boost": 25,
        "hunger_restore": 8,
        "desc": "Pure sugar rush",
    },
    "sushi": {
        "calories": 75,
        "protein": 15,
        "carbs": 14,
        "fat": 3,
        "fiber": 1,
        "vit_a": 5,
        "vit_b": 10,
        "vit_c": 2,
        "vit_d": 30,
        "vit_e": 5,
        "happiness_boost": 18,
        "hunger_restore": 22,
        "desc": "Fancy and balanced",
    },
    "pizza": {
        "calories": 130,
        "protein": 8,
        "carbs": 28,
        "fat": 14,
        "fiber": 2,
        "vit_a": 5,
        "vit_b": 5,
        "vit_c": 3,
        "vit_d": 2,
        "vit_e": 2,
        "happiness_boost": 20,
        "hunger_restore": 30,
        "desc": "Greasy comfort food",
    },
    "broccoli": {
        "calories": 15,
        "protein": 3,
        "carbs": 4,
        "fat": 0,
        "fiber": 6,
        "vit_a": 20,
        "vit_b": 8,
        "vit_c": 40,
        "vit_d": 0,
        "vit_e": 15,
        "happiness_boost": -2,
        "hunger_restore": 8,
        "desc": "Extremely healthy, tastes bad",
    },
    "milk": {
        "calories": 40,
        "protein": 6,
        "carbs": 8,
        "fat": 5,
        "fiber": 0,
        "vit_a": 8,
        "vit_b": 15,
        "vit_c": 0,
        "vit_d": 20,
        "vit_e": 2,
        "happiness_boost": 5,
        "hunger_restore": 10,
        "desc": "Nutritious drink",
    },
    "bugs": {
        "calories": 25,
        "protein": 14,
        "carbs": 1,
        "fat": 2,
        "fiber": 2,
        "vit_a": 3,
        "vit_b": 20,
        "vit_c": 0,
        "vit_d": 0,
        "vit_e": 3,
        "happiness_boost": -5,
        "hunger_restore": 12,
        "desc": "Crunchy. Sustainable.",
    },
}

# ---------------------------------------------------------------------------
# TOYS
# ---------------------------------------------------------------------------
TOYS: dict[str, dict] = {
    "ball": {"boredom": -25, "fitness": 8, "happiness": 5, "energy": -8, "max_uses": 30, "desc": "Bouncy fun"},
    "puzzle": {
        "boredom": -20,
        "intelligence": 12,
        "happiness": 3,
        "energy": -3,
        "max_uses": 20,
        "desc": "Brain teaser",
    },
    "squeaky toy": {"boredom": -22, "happiness": 12, "energy": -5, "max_uses": 15, "desc": "SQUEAK SQUEAK"},
    "rope": {"boredom": -18, "fitness": 10, "social": 8, "energy": -10, "max_uses": 25, "desc": "Tug of war"},
    "plushie": {"boredom": -15, "comfort": 12, "social": 5, "max_uses": 40, "desc": "Soft and cuddly"},
    "art kit": {"boredom": -20, "creativity": 15, "happiness": 5, "max_uses": 12, "desc": "Express yourself"},
    "book": {"boredom": -18, "intelligence": 15, "creativity": 8, "energy": -2, "max_uses": 10, "desc": "Knowledge!"},
    "music box": {"boredom": -15, "happiness": 10, "comfort": 8, "max_uses": 50, "desc": "Soothing melodies"},
}

# ---------------------------------------------------------------------------
# CLOTHING - name -> {warmth, best_weather, desc}
# warmth: degrees celsius of insulation added to body temp regulation
# ---------------------------------------------------------------------------
CLOTHING: dict[str, dict] = {
    "nothing": {"warmth": 0, "best_weather": ["hot", "sunny"], "desc": "Au naturel"},
    "t-shirt": {"warmth": 1, "best_weather": ["sunny", "cloudy", "windy"], "desc": "Casual"},
    "sweater": {"warmth": 4, "best_weather": ["cloudy", "cold", "windy"], "desc": "Cozy knit"},
    "coat": {"warmth": 7, "best_weather": ["cold", "snowy", "stormy"], "desc": "Heavy warmth"},
    "raincoat": {"warmth": 2, "best_weather": ["rainy", "stormy"], "desc": "Waterproof"},
    "sunhat": {"warmth": -1, "best_weather": ["sunny", "hot"], "desc": "Sun protection"},
    "scarf": {"warmth": 3, "best_weather": ["cold", "windy", "snowy"], "desc": "Neck warmer"},
    "pajamas": {"warmth": 2, "best_weather": [], "desc": "Sleepytime only"},
}

# ---------------------------------------------------------------------------
# WEATHER - name -> {temp_modifier, mood_modifier, desc}
# temp_modifier affects room temperature (added to thermostat)
# ---------------------------------------------------------------------------
WEATHER: dict[str, dict] = {
    "sunny": {"temp_mod": 3, "mood_mod": 5, "desc": "Bright and clear"},
    "cloudy": {"temp_mod": 0, "mood_mod": -1, "desc": "Overcast skies"},
    "rainy": {"temp_mod": -2, "mood_mod": -5, "desc": "Wet and dreary"},
    "snowy": {"temp_mod": -8, "mood_mod": 2, "desc": "Winter wonderland"},
    "hot": {"temp_mod": 8, "mood_mod": -3, "desc": "Scorching heat"},
    "cold": {"temp_mod": -6, "mood_mod": -4, "desc": "Biting chill"},
    "windy": {"temp_mod": -3, "mood_mod": -2, "desc": "Howling gusts"},
    "stormy": {"temp_mod": -4, "mood_mod": -8, "desc": "Thunder and lightning"},
}

# ---------------------------------------------------------------------------
# AILMENTS - name -> {stat_penalties (per tick), cure, desc, escalates_to}
# ---------------------------------------------------------------------------
AILMENTS: dict[str, dict] = {
    "cold": {
        "penalties": {"energy": -0.05, "happiness": -0.03, "comfort": -0.05},
        "cure": "cold medicine",
        "desc": "Sniffly and sneezy",
        "escalates_to": "flu",
    },
    "flu": {
        "penalties": {"energy": -0.1, "happiness": -0.05, "hunger": -0.05, "comfort": -0.05},
        "cure": "antibiotics",
        "desc": "Severely ill",
        "escalates_to": None,
    },
    "food poisoning": {
        "penalties": {"energy": -0.08, "happiness": -0.05, "hunger": -0.05, "comfort": -0.08},
        "cure": "antacid",
        "desc": "Stomach in revolt",
        "escalates_to": None,
    },
    "allergic reaction": {
        "penalties": {"happiness": -0.05, "comfort": -0.05, "skin_condition": -0.03},
        "cure": "antihistamine",
        "desc": "Itchy and swollen",
        "escalates_to": None,
    },
    "depression": {
        "penalties": {"happiness": -0.04, "social": -0.03, "energy": -0.03, "boredom": 0.03},
        "cure": "antidepressant",
        "desc": "A deep sadness",
        "escalates_to": None,
    },
    "insomnia": {
        "penalties": {"energy": -0.05, "mood": -0.03, "happiness": -0.03},
        "cure": "antidepressant",
        "desc": "Cannot sleep properly",
        "escalates_to": "depression",
    },
    "obesity": {
        "penalties": {"fitness": -0.03, "energy": -0.03, "comfort": -0.03},
        "cure": None,
        "desc": "Dangerously overweight (needs diet + exercise)",
        "escalates_to": None,
    },
    "malnutrition": {
        "penalties": {"energy": -0.08, "fitness": -0.03, "happiness": -0.04, "fur_condition": -0.02},
        "cure": "vitamins",
        "desc": "Wasting away",
        "escalates_to": None,
    },
    "tooth decay": {
        "penalties": {"happiness": -0.03, "hunger": -0.02, "comfort": -0.03},
        "cure": None,
        "desc": "Painful teeth (brush them!)",
        "escalates_to": None,
    },
    "fleas": {
        "penalties": {"hygiene": -0.05, "comfort": -0.05, "happiness": -0.03, "skin_condition": -0.03},
        "cure": "flea treatment",
        "desc": "Itchy little critters",
        "escalates_to": None,
    },
    "sunburn": {
        "penalties": {"comfort": -0.05, "happiness": -0.03, "skin_condition": -0.04},
        "cure": "aloe vera",
        "desc": "Lobster red",
        "escalates_to": None,
    },
    "heatstroke": {
        "penalties": {"energy": -0.15, "comfort": -0.1, "happiness": -0.08},
        "cure": None,
        "desc": "DANGEROUSLY HOT (cool down NOW!)",
        "escalates_to": None,
    },
    "hypothermia": {
        "penalties": {"energy": -0.15, "comfort": -0.1, "happiness": -0.08},
        "cure": None,
        "desc": "DANGEROUSLY COLD (warm up NOW!)",
        "escalates_to": None,
    },
    "indigestion": {
        "penalties": {"comfort": -0.04, "happiness": -0.02, "hunger": -0.02},
        "cure": "antacid",
        "desc": "Ate too much or too fast",
        "escalates_to": None,
    },
    "loneliness": {
        "penalties": {"happiness": -0.04, "mood": -0.05, "social": -0.03},
        "cure": None,
        "desc": "Desperately needs companionship (talk to it!)",
        "escalates_to": "depression",
    },
}

MEDICINES: dict[str, dict] = {
    "cold medicine": {"cures": ["cold"], "desc": "For colds and sniffles"},
    "antibiotics": {"cures": ["flu"], "desc": "Strong stuff for serious illness"},
    "antacid": {"cures": ["food poisoning", "indigestion"], "desc": "Settles the stomach"},
    "antihistamine": {"cures": ["allergic reaction"], "desc": "Stops allergic reactions"},
    "antidepressant": {"cures": ["depression", "insomnia"], "desc": "Mood stabilizer"},
    "flea treatment": {"cures": ["fleas"], "desc": "Kills parasites"},
    "aloe vera": {"cures": ["sunburn"], "desc": "Soothing gel"},
    "vitamins": {"cures": ["malnutrition"], "desc": "Nutritional supplement"},
}

# ---------------------------------------------------------------------------
# TRICKS
# ---------------------------------------------------------------------------
TRICKS = ["sit", "shake", "roll over", "speak", "dance", "fetch", "high five", "play dead", "backflip"]

# ---------------------------------------------------------------------------
# PERSONALITY TRAITS - name -> {stat_multipliers}
# Multipliers are applied to decay rates. >1 = faster decay, <1 = slower decay
# ---------------------------------------------------------------------------
PERSONALITY_TRAITS: dict[str, dict] = {
    "picky eater": {"desc": "Hated foods cause 2x mood penalty", "food_mood_penalty": 2.0},
    "night owl": {"desc": "Better energy at night, worse during day", "energy_night": 0.5, "energy_day": 1.5},
    "early bird": {"desc": "Better energy during day, worse at night", "energy_night": 1.5, "energy_day": 0.5},
    "clingy": {"desc": "Social need decays much faster", "social_decay": 2.0},
    "independent": {"desc": "Social need decays slower", "social_decay": 0.5},
    "glutton": {"desc": "Gets hungry faster", "hunger_decay": 1.5},
    "finicky": {"desc": "Gets dirty faster", "hygiene_decay": 2.0},
    "couch potato": {"desc": "Fitness drops faster, but less tiring", "fitness_decay": 2.0, "energy_decay": 0.7},
    "adventurous": {"desc": "Gets bored much faster", "boredom_rate": 2.0},
    "grumpy": {"desc": "Mood recovers half as fast", "mood_recovery": 0.5},
    "cheerful": {"desc": "Mood recovers twice as fast", "mood_recovery": 2.0},
    "stubborn": {"desc": "Harder to train and discipline", "learning_rate": 0.5},
}

# ---------------------------------------------------------------------------
# AGE STAGES - (max_days, label, decay_multiplier, sickness_vulnerability)
# ---------------------------------------------------------------------------
AGE_STAGES = [
    (2, "Baby", 1.2, 1.2),  # Slightly faster decay, moderate sickness risk
    (5, "Child", 1.0, 1.0),  # Baseline
    (10, "Teen", 0.9, 0.8),  # Slightly hardier
    (20, "Adult", 0.8, 0.7),  # Peak health
    (999, "Senior", 1.3, 1.8),  # Fragile
]

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "teal", "white", "black"]

# ---------------------------------------------------------------------------
# BASE DECAY RATES per minute (negative = stat goes down, positive = stat goes up)
# ---------------------------------------------------------------------------
BASE_DECAY = {
    "hunger": -0.35,  # ~5 hours from full to zero
    "thirst": -0.45,  # ~4 hours from full to zero
    "energy": -0.15,  # ~11 hours from full to zero
    "happiness": -0.06,  # very slow bleed (~28 hours full to zero)
    "hygiene": -0.08,  # ~20 hours
    "bladder": -0.30,  # ~5.5 hours
    "social": -0.06,  # very slow
    "comfort": -0.04,  # very slow
    "boredom": 0.15,  # boredom INCREASES, ~11 hours to max
    "fitness": -0.01,  # glacial
    "teeth_health": -0.02,  # ~3 days to critical
    "nail_length": 0.01,  # nails GROW slowly
    "fur_condition": -0.015,
    "skin_condition": -0.005,
    "room_cleanliness": -0.03,
    "mood": -0.05,
    "intelligence": -0.005,
    "creativity": -0.005,
    "discipline": -0.01,
}

# Daily nutritional targets
DAILY_TARGETS = {
    "protein": 50,
    "carbs": 80,
    "fat": 30,
    "fiber": 20,
    "vit_a": 100,
    "vit_b": 100,
    "vit_c": 100,
    "vit_d": 100,
    "vit_e": 100,
}

IDEAL_BODY_TEMP = 38.0
WEATHER_CHANGE_INTERVAL = 3 * 60 * 60  # 3 hours in seconds
WORK_COOLDOWN = 3600  # 1 hour in seconds
WORK_PAY_MIN = 20
WORK_PAY_MAX = 50

# ---------------------------------------------------------------------------
# PROBABILITIES & THRESHOLDS (per minute, multiplied by elapsed_minutes in tick)
# ---------------------------------------------------------------------------
POOP_CHANCE_PER_MIN = 0.3  # Chance to poop when bladder < 20
POOP_BLADDER_THRESHOLD = 20

SICKNESS_CHANCE_FLEAS = 0.05  # When hygiene < 25 and fur < 30
SICKNESS_CHANCE_COLD = 0.04  # When hygiene < 20 and room temp < 15
SICKNESS_CHANCE_TOOTH_DECAY = 0.03  # When teeth_health < 20
SICKNESS_CHANCE_DEPRESSION = 0.02  # When happiness < 15
SICKNESS_CHANCE_LONELINESS = 0.03  # When social < 10
SICKNESS_CHANCE_SUNBURN = 0.03  # When hot weather + wrong clothing

OBESITY_WEIGHT_THRESHOLD = 12.0
MALNUTRITION_WEIGHT_THRESHOLD = 2.5
OBESITY_CLEAR_THRESHOLD = 11.5
MALNUTRITION_CLEAR_THRESHOLD = 3.0
HEATSTROKE_TEMP = 40.0
HYPOTHERMIA_TEMP = 35.0
HEATSTROKE_CLEAR_TEMP = 39.5
HYPOTHERMIA_CLEAR_TEMP = 36.0
TOOTH_DECAY_CLEAR_THRESHOLD = 50.0
LONELINESS_CLEAR_THRESHOLD = 40.0

LETHAL_BODY_TEMP_HIGH = 43.0
LETHAL_BODY_TEMP_LOW = 32.0
LETHAL_WEIGHT_LOW = 1.0
LETHAL_WEIGHT_HIGH = 20.0

OLD_AGE_START_DAY = 30
OLD_AGE_DEATH_CHANCE_PER_MIN = 0.0005  # Multiplied by (days - 30)

AILMENT_SEVERITY_RATE = 0.5  # Severity increase per minute
AILMENT_ESCALATION_THRESHOLD = 80  # Severity at which ailment escalates

# ---------------------------------------------------------------------------
# PRICES - tuned so the player is always somewhat poor
# Work pays 10-30 coins (avg ~20) per hour.
# Daily income ~100-200 coins. Daily food cost ~40-80. Medicine 10-30.
# ---------------------------------------------------------------------------
FOOD_PRICES = {
    "kibble": 8,
    "steak": 35,
    "fish": 20,
    "salad": 6,
    "cheese": 15,
    "bread": 8,
    "fruit": 10,
    "eggs": 12,
    "rice": 6,
    "candy": 12,
    "sushi": 40,
    "pizza": 25,
    "broccoli": 5,
    "milk": 8,
    "bugs": 3,
}
TOY_PRICES = {
    "ball": 5,
    "puzzle": 10,
    "squeaky toy": 8,
    "rope": 6,
    "plushie": 8,
    "art kit": 15,
    "book": 12,
    "music box": 10,
}
CLOTHING_PRICES = {
    "nothing": 0,
    "t-shirt": 5,
    "sweater": 12,
    "coat": 18,
    "raincoat": 14,
    "sunhat": 8,
    "scarf": 10,
    "pajamas": 6,
}
MEDICINE_PRICES = {
    "cold medicine": 15,
    "antibiotics": 30,
    "antacid": 10,
    "antihistamine": 15,
    "antidepressant": 35,
    "flea treatment": 20,
    "aloe vera": 10,
    "vitamins": 12,
}
DECORATE_PRICE = 20
