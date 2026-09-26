"""Tests for hollingsbot.jev.spawns: where `!spawn jev N` sent Jev and how many replies it has left."""

from __future__ import annotations

import contextlib
import sqlite3

from hollingsbot.jev.spawns import JevSpawns


def test_a_visit_counts_down_and_ends_at_zero(temp_db):
    spawns = JevSpawns(temp_db)
    spawns.start(1, 2, by="Hollings")
    assert spawns.active() == {1: 2}
    assert spawns.use_reply(1) == 1
    assert spawns.use_reply(1) == 0
    assert spawns.active() == {}
    assert spawns.use_reply(1) is None  # not visiting any more


def test_spawning_again_resets_the_count(temp_db):
    spawns = JevSpawns(temp_db)
    spawns.start(1, 5, by="Hollings")
    spawns.use_reply(1)
    spawns.start(1, 3, by="Jackson")
    assert spawns.active() == {1: 3}


def test_end_early_and_channels_are_independent(temp_db):
    spawns = JevSpawns(temp_db)
    spawns.start(1, 5, by="Hollings")
    spawns.start(2, 4, by="Hollings")
    assert spawns.end(1) is True
    assert spawns.end(1) is False
    assert spawns.active() == {2: 4}


def test_visits_survive_a_restart(temp_db):
    JevSpawns(temp_db).start(7, 10, by="Hollings")
    assert JevSpawns(temp_db).active() == {7: 10}


def test_each_jev_has_its_own_visits(temp_db):
    jev, jev2 = JevSpawns(temp_db), JevSpawns(temp_db, bot="Jev2")
    jev.start(1, 5, by="Hollings")
    jev2.start(1, 3, by="Hollings")  # the same channel
    assert jev.use_reply(1) == 4
    assert jev2.end(1) is True
    assert (jev.active(), jev2.active()) == ({1: 4}, {})


def test_visits_from_the_one_jev_layout_carry_over_as_jevs(temp_db):
    with contextlib.closing(sqlite3.connect(temp_db)) as conn:
        conn.execute(
            "CREATE TABLE jev_spawns (channel_id INTEGER PRIMARY KEY, replies_left INTEGER NOT NULL, "
            "spawned_by TEXT, spawned_at TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO jev_spawns VALUES (9, 6, 'Hollings', '2026-09-26T00:35:58+00:00')")
        conn.commit()

    assert JevSpawns(temp_db).active() == {9: 6}
    assert JevSpawns(temp_db, bot="Jev2").active() == {}
    with contextlib.closing(sqlite3.connect(temp_db)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "jev_spawns" not in tables
