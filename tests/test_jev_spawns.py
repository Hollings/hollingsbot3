"""Tests for hollingsbot.jev.spawns: where `!spawn jev N` sent Jev and how many replies it has left."""

from __future__ import annotations

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
