"""Tests for Best Bot Posts tournament system."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

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
    with patch.dict("os.environ", {"PROMPT_DB_PATH": temp_db}):
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
        _bbp, db_path = setup_best_bot_posts
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='elo_posts'")
            result = cursor.fetchone()
        assert result is not None

    def test_creates_match_history_table(self, setup_best_bot_posts):
        """Test that match_history table is created."""
        _bbp, db_path = setup_best_bot_posts
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='match_history'")
            result = cursor.fetchone()
        assert result is not None

    def test_idempotent(self, setup_best_bot_posts):
        """Test that calling _init_db twice doesn't error."""
        bbp, _ = setup_best_bot_posts
        # Should not raise
        bbp._init_db()
        bbp._init_db()


class TestFilenameExists:
    """Tests for filename existence check."""

    def test_detects_existing_filename(self, setup_best_bot_posts):
        """Test that existing filename is detected."""
        bbp, db_path = setup_best_bot_posts

        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT INTO elo_posts (name, filename) VALUES ('test', 'existing.png')")
            conn.commit()

        assert bbp._filename_exists("existing.png") is True

    def test_returns_false_for_missing(self, setup_best_bot_posts):
        """Test that missing filename returns False."""
        bbp, _ = setup_best_bot_posts
        assert bbp._filename_exists("nonexistent.png") is False


class TestInsertPost:
    """Tests for inserting new posts."""

    def test_inserts_image_post(self, setup_best_bot_posts):
        """Test inserting an image post."""
        bbp, db_path = setup_best_bot_posts

        bbp._insert_post("test post", "test.png", "image")

        with sqlite3.connect(db_path) as conn:
            post = conn.execute("SELECT * FROM elo_posts WHERE filename = 'test.png'").fetchone()

        assert post is not None
        assert post[1] == "test post"  # name
        assert post[3] == "image"  # post_type
        assert post[5] == 1000  # default rating

    def test_inserts_text_post(self, setup_best_bot_posts):
        """Test inserting a text post."""
        bbp, db_path = setup_best_bot_posts

        bbp._insert_post("text post", "text_1234.txt", "text", text_content="Hello world")

        with sqlite3.connect(db_path) as conn:
            post = conn.execute("SELECT * FROM elo_posts WHERE post_type = 'text'").fetchone()

        assert post is not None
        assert post[4] == "Hello world"  # text_content
