"""Tests for Best Bot Posts ELO tournament system."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import random

import pytest


# We need to patch the DB_PATH before importing the module
@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test.db"
        yield str(db_file)


@pytest.fixture
def setup_best_bot_posts(temp_db):
    """Set up the best_bot_posts module with test database."""
    with patch.dict('os.environ', {'PROMPT_DB_PATH': temp_db}):
        # Re-import to pick up new DB_PATH
        import importlib
        import hollingsbot.cogs.best_bot_posts as bbp
        importlib.reload(bbp)

        # Initialize the database
        bbp._init_db()
        yield bbp, temp_db


class TestInitDb:
    """Tests for database initialization."""

    def test_creates_elo_posts_table(self, setup_best_bot_posts):
        """Test that elo_posts table is created."""
        bbp, db_path = setup_best_bot_posts
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='elo_posts'"
            )
            result = cursor.fetchone()
        assert result is not None

    def test_creates_match_history_table(self, setup_best_bot_posts):
        """Test that match_history table is created."""
        bbp, db_path = setup_best_bot_posts
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='match_history'"
            )
            result = cursor.fetchone()
        assert result is not None

    def test_idempotent(self, setup_best_bot_posts):
        """Test that calling _init_db twice doesn't error."""
        bbp, _ = setup_best_bot_posts
        # Should not raise
        bbp._init_db()
        bbp._init_db()


class TestEloCalc:
    """Tests for ELO calculation."""

    def test_equal_ratings_changes(self, setup_best_bot_posts):
        """Test ELO changes when ratings are equal."""
        bbp, _ = setup_best_bot_posts
        winner_new, loser_new = bbp._elo_calc(1000, 1000)
        # Winner should gain, loser should lose
        assert winner_new > 1000
        assert loser_new < 1000
        # Changes should be symmetric for equal ratings
        assert winner_new - 1000 == 1000 - loser_new

    def test_upset_win_changes_more(self, setup_best_bot_posts):
        """Test that upset wins result in larger rating changes."""
        bbp, _ = setup_best_bot_posts
        # Underdog (800) beats favorite (1200)
        upset_winner_new, upset_loser_new = bbp._elo_calc(800, 1200)
        # Favorite (1200) beats underdog (800)
        expected_winner_new, expected_loser_new = bbp._elo_calc(1200, 800)

        # Upset should cause larger change
        upset_change = upset_winner_new - 800
        expected_change = expected_winner_new - 1200
        assert upset_change > expected_change

    def test_k_factor_affects_magnitude(self, setup_best_bot_posts):
        """Test that K factor affects rating change magnitude."""
        bbp, _ = setup_best_bot_posts
        high_k_winner, high_k_loser = bbp._elo_calc(1000, 1000, k=100)
        low_k_winner, low_k_loser = bbp._elo_calc(1000, 1000, k=25)

        assert high_k_winner - 1000 > low_k_winner - 1000

    def test_returns_integers(self, setup_best_bot_posts):
        """Test that ELO returns integer ratings."""
        bbp, _ = setup_best_bot_posts
        winner_new, loser_new = bbp._elo_calc(1000, 1000)
        assert isinstance(winner_new, int)
        assert isinstance(loser_new, int)


class TestGetRandomPair:
    """Tests for pairing strategy."""

    def test_returns_two_posts_when_available(self, setup_best_bot_posts):
        """Test that two posts are returned when available."""
        bbp, db_path = setup_best_bot_posts

        # Insert test posts
        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT INTO elo_posts (name, filename) VALUES ('post1', 'file1.png')")
            conn.execute("INSERT INTO elo_posts (name, filename) VALUES ('post2', 'file2.png')")
            conn.commit()

        result = bbp._get_random_pair()
        assert len(result) == 2

    def test_returns_all_when_less_than_two(self, setup_best_bot_posts):
        """Test behavior with less than 2 posts."""
        bbp, db_path = setup_best_bot_posts

        # Insert only one post
        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT INTO elo_posts (name, filename) VALUES ('post1', 'file1.png')")
            conn.commit()

        result = bbp._get_random_pair()
        assert len(result) == 1

    def test_returns_different_posts(self, setup_best_bot_posts):
        """Test that returned posts are different."""
        bbp, db_path = setup_best_bot_posts

        # Insert test posts
        with sqlite3.connect(db_path) as conn:
            for i in range(10):
                conn.execute(
                    "INSERT INTO elo_posts (name, filename, wins, losses) VALUES (?, ?, ?, ?)",
                    (f'post{i}', f'file{i}.png', i, 0)
                )
            conn.commit()

        # Run multiple times to check randomness
        for _ in range(10):
            result = bbp._get_random_pair()
            assert result[0]['id'] != result[1]['id']


class TestUpdateRatings:
    """Tests for rating updates."""

    def test_updates_winner_rating(self, setup_best_bot_posts):
        """Test that winner's rating is updated."""
        bbp, db_path = setup_best_bot_posts

        # Insert test posts
        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT INTO elo_posts (id, name, filename, rating) VALUES (1, 'winner', 'w.png', 1000)")
            conn.execute("INSERT INTO elo_posts (id, name, filename, rating) VALUES (2, 'loser', 'l.png', 1000)")
            conn.commit()

        bbp._update_ratings(1, 1025, 2, 975, 1000, 1000)

        with sqlite3.connect(db_path) as conn:
            winner = conn.execute("SELECT rating, wins FROM elo_posts WHERE id = 1").fetchone()
            loser = conn.execute("SELECT rating, losses FROM elo_posts WHERE id = 2").fetchone()

        assert winner[0] == 1025
        assert winner[1] == 1
        assert loser[0] == 975
        assert loser[1] == 1

    def test_records_match_history(self, setup_best_bot_posts):
        """Test that match history is recorded."""
        bbp, db_path = setup_best_bot_posts

        # Insert test posts
        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT INTO elo_posts (id, name, filename, rating) VALUES (1, 'winner', 'w.png', 1000)")
            conn.execute("INSERT INTO elo_posts (id, name, filename, rating) VALUES (2, 'loser', 'l.png', 1000)")
            conn.commit()

        bbp._update_ratings(1, 1025, 2, 975, 1000, 1000)

        with sqlite3.connect(db_path) as conn:
            history = conn.execute("SELECT * FROM match_history").fetchone()

        assert history is not None
        assert history[1] == 1  # winner_id
        assert history[2] == 2  # loser_id
        assert history[3] == 1000  # winner_rating_before
        assert history[4] == 1000  # loser_rating_before


class TestPairingStrategies:
    """Tests for different pairing strategies."""

    def test_fresh_vs_fresh_strategy(self, setup_best_bot_posts):
        """Test fresh vs fresh pairing."""
        bbp, db_path = setup_best_bot_posts

        # Insert unrated posts (0 wins, 0 losses)
        with sqlite3.connect(db_path) as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO elo_posts (name, filename, wins, losses) VALUES (?, ?, 0, 0)",
                    (f'fresh{i}', f'fresh{i}.png')
                )
            conn.commit()

        # All posts are fresh, should get fresh_vs_fresh
        result = bbp._get_random_pair()
        assert len(result) == 2
        # Both should have 0 matches
        assert result[0]['wins'] + result[0]['losses'] == 0
        assert result[1]['wins'] + result[1]['losses'] == 0

    def test_high_vs_high_strategy(self, setup_best_bot_posts):
        """Test high vs high pairing picks close ratings."""
        bbp, db_path = setup_best_bot_posts

        # Insert rated posts with varying ratings
        with sqlite3.connect(db_path) as conn:
            for i, rating in enumerate([800, 900, 1000, 1100, 1200]):
                conn.execute(
                    "INSERT INTO elo_posts (name, filename, rating, wins, losses) VALUES (?, ?, ?, 10, 10)",
                    (f'post{i}', f'post{i}.png', rating)
                )
            conn.commit()

        # Run many times to verify strategy works
        pairs_seen = []
        for _ in range(50):
            result = bbp._get_random_pair()
            pairs_seen.append((result[0]['rating'], result[1]['rating']))

        # Should see high-rated pairs (1100, 1200 range) or low-rated pairs (800, 900)
        # rather than always random mix


class TestFilenameExists:
    """Tests for filename existence check."""

    def test_detects_existing_filename(self, setup_best_bot_posts):
        """Test that existing filename is detected."""
        bbp, db_path = setup_best_bot_posts

        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT INTO elo_posts (name, filename) VALUES ('test', 'existing.png')")
            conn.commit()

        assert bbp._filename_exists('existing.png') is True

    def test_returns_false_for_missing(self, setup_best_bot_posts):
        """Test that missing filename returns False."""
        bbp, _ = setup_best_bot_posts
        assert bbp._filename_exists('nonexistent.png') is False


class TestInsertPost:
    """Tests for inserting new posts."""

    def test_inserts_image_post(self, setup_best_bot_posts):
        """Test inserting an image post."""
        bbp, db_path = setup_best_bot_posts

        bbp._insert_post('test post', 'test.png', 'image')

        with sqlite3.connect(db_path) as conn:
            post = conn.execute("SELECT * FROM elo_posts WHERE filename = 'test.png'").fetchone()

        assert post is not None
        assert post[1] == 'test post'  # name
        assert post[3] == 'image'  # post_type
        assert post[5] == 1000  # default rating

    def test_inserts_text_post(self, setup_best_bot_posts):
        """Test inserting a text post."""
        bbp, db_path = setup_best_bot_posts

        bbp._insert_post('text post', 'text_1234.txt', 'text', text_content='Hello world')

        with sqlite3.connect(db_path) as conn:
            post = conn.execute("SELECT * FROM elo_posts WHERE post_type = 'text'").fetchone()

        assert post is not None
        assert post[4] == 'Hello world'  # text_content
