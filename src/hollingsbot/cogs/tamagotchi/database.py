"""Tamagotchi database operations."""

from __future__ import annotations

import os
import sqlite3
import time

DB_PATH = os.getenv("PROMPT_DB_PATH", "/data/hollingsbot.db")


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tamagotchi_pets (
            user_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            born_at REAL NOT NULL,
            died_at REAL,
            cause_of_death TEXT,
            last_tick REAL NOT NULL,
            is_sleeping INTEGER DEFAULT 0,
            sleep_started_at REAL,

            hunger REAL DEFAULT 80.0,
            thirst REAL DEFAULT 80.0,
            energy REAL DEFAULT 90.0,
            happiness REAL DEFAULT 70.0,
            hygiene REAL DEFAULT 85.0,
            bladder REAL DEFAULT 80.0,
            social REAL DEFAULT 60.0,
            comfort REAL DEFAULT 75.0,
            boredom REAL DEFAULT 10.0,

            weight REAL DEFAULT 5.0,
            body_temp REAL DEFAULT 38.0,
            fitness REAL DEFAULT 50.0,

            teeth_health REAL DEFAULT 95.0,
            nail_length REAL DEFAULT 10.0,
            fur_condition REAL DEFAULT 85.0,
            skin_condition REAL DEFAULT 85.0,

            nutrition_day TEXT,
            calories_today REAL DEFAULT 0.0,
            protein_today REAL DEFAULT 0.0,
            carbs_today REAL DEFAULT 0.0,
            fat_today REAL DEFAULT 0.0,
            fiber_today REAL DEFAULT 0.0,
            vit_a_today REAL DEFAULT 0.0,
            vit_b_today REAL DEFAULT 0.0,
            vit_c_today REAL DEFAULT 0.0,
            vit_d_today REAL DEFAULT 0.0,
            vit_e_today REAL DEFAULT 0.0,
            foods_eaten_today TEXT DEFAULT '{}',

            intelligence REAL DEFAULT 30.0,
            creativity REAL DEFAULT 30.0,
            trust REAL DEFAULT 50.0,
            mood REAL DEFAULT 60.0,
            discipline REAL DEFAULT 40.0,

            room_cleanliness REAL DEFAULT 90.0,
            room_temp REAL DEFAULT 20.0,
            lights_on INTEGER DEFAULT 1,
            decoration_score REAL DEFAULT 0.0,
            decoration_color TEXT,
            poop_count INTEGER DEFAULT 0,
            current_clothing TEXT DEFAULT 'nothing',

            traits TEXT DEFAULT '[]',
            food_loves TEXT DEFAULT '[]',
            food_hates TEXT DEFAULT '[]',
            allergy TEXT,
            favorite_color TEXT,

            tricks_known TEXT DEFAULT '{}',
            last_trick_practice TEXT DEFAULT '{}',

            toy_uses_today TEXT DEFAULT '{}',
            current_weather TEXT DEFAULT 'sunny',
            weather_changed_at REAL,
            zero_vital_since REAL,

            money REAL DEFAULT 100.0,
            last_work_at REAL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tamagotchi_ailments (
            user_id INTEGER NOT NULL,
            ailment TEXT NOT NULL,
            severity REAL DEFAULT 10.0,
            contracted_at REAL NOT NULL,
            PRIMARY KEY (user_id, ailment)
        )
    """)
    # Migrate existing tables
    for col, default in [("money REAL", "50.0"), ("last_work_at REAL", "0")]:
        try:
            conn.execute(f"ALTER TABLE tamagotchi_pets ADD COLUMN {col} DEFAULT {default}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def _get_pet(conn, user_id):
    """Get pet row as dict, or None."""
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM tamagotchi_pets WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def _save_pet(conn, pet):
    """Save a pet dict back to DB."""
    cols = [k for k in pet if k != "user_id"]
    sets = ", ".join(f"{c} = ?" for c in cols)
    vals = [pet[c] for c in cols]
    vals.append(pet["user_id"])
    conn.execute(f"UPDATE tamagotchi_pets SET {sets} WHERE user_id = ?", vals)
    conn.commit()


def _get_ailments(conn, user_id):
    """Get list of active ailments for a pet."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM tamagotchi_ailments WHERE user_id = ?", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def _add_ailment(conn, user_id, ailment_name):
    """Add an ailment if not already present."""
    try:
        conn.execute(
            "INSERT INTO tamagotchi_ailments (user_id, ailment, severity, contracted_at) VALUES (?, ?, 10.0, ?)",
            (user_id, ailment_name, time.time()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Already has this ailment


def _remove_ailment(conn, user_id, ailment_name):
    conn.execute(
        "DELETE FROM tamagotchi_ailments WHERE user_id = ? AND ailment = ?",
        (user_id, ailment_name),
    )
    conn.commit()
