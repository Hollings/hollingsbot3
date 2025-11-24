"""Tests for time window boundary calculations."""

import pytest
from datetime import datetime, timezone
from prototype_summarization.time_windows import (
    get_current_window_boundary,
    get_window_at_time,
    calculate_summary_level,
)


class TestGetWindowAtTime:
    """Test window boundary calculation for specific timestamps."""

    def test_window_at_1_00(self):
        """Test window starting at 1:00."""
        # 1:00 PM UTC
        timestamp = int(datetime(2025, 1, 7, 13, 0, 0, tzinfo=timezone.utc).timestamp())
        start, end = get_window_at_time(timestamp)

        expected_start = int(datetime(2025, 1, 7, 13, 0, 0, tzinfo=timezone.utc).timestamp())
        expected_end = int(datetime(2025, 1, 7, 13, 29, 59, tzinfo=timezone.utc).timestamp())

        assert start == expected_start
        assert end == expected_end

    def test_window_at_1_30(self):
        """Test window starting at 1:30."""
        # 1:30 PM UTC
        timestamp = int(datetime(2025, 1, 7, 13, 30, 0, tzinfo=timezone.utc).timestamp())
        start, end = get_window_at_time(timestamp)

        expected_start = int(datetime(2025, 1, 7, 13, 30, 0, tzinfo=timezone.utc).timestamp())
        expected_end = int(datetime(2025, 1, 7, 13, 59, 59, tzinfo=timezone.utc).timestamp())

        assert start == expected_start
        assert end == expected_end

    def test_window_at_2_00(self):
        """Test window starting at 2:00."""
        timestamp = int(datetime(2025, 1, 7, 14, 0, 0, tzinfo=timezone.utc).timestamp())
        start, end = get_window_at_time(timestamp)

        expected_start = int(datetime(2025, 1, 7, 14, 0, 0, tzinfo=timezone.utc).timestamp())
        expected_end = int(datetime(2025, 1, 7, 14, 29, 59, tzinfo=timezone.utc).timestamp())

        assert start == expected_start
        assert end == expected_end

    def test_window_mid_period_rounds_down(self):
        """Test that 1:15 falls in the 1:00-1:29 window."""
        # 1:15 PM UTC
        timestamp = int(datetime(2025, 1, 7, 13, 15, 0, tzinfo=timezone.utc).timestamp())
        start, end = get_window_at_time(timestamp)

        expected_start = int(datetime(2025, 1, 7, 13, 0, 0, tzinfo=timezone.utc).timestamp())
        expected_end = int(datetime(2025, 1, 7, 13, 29, 59, tzinfo=timezone.utc).timestamp())

        assert start == expected_start
        assert end == expected_end

    def test_window_at_1_29_59(self):
        """Test that 1:29:59 is still in the 1:00-1:29 window."""
        timestamp = int(datetime(2025, 1, 7, 13, 29, 59, tzinfo=timezone.utc).timestamp())
        start, end = get_window_at_time(timestamp)

        expected_start = int(datetime(2025, 1, 7, 13, 0, 0, tzinfo=timezone.utc).timestamp())
        expected_end = int(datetime(2025, 1, 7, 13, 29, 59, tzinfo=timezone.utc).timestamp())

        assert start == expected_start
        assert end == expected_end

    def test_window_crosses_day_boundary(self):
        """Test window calculation at midnight."""
        # 11:30 PM UTC
        timestamp = int(datetime(2025, 1, 7, 23, 30, 0, tzinfo=timezone.utc).timestamp())
        start, end = get_window_at_time(timestamp)

        expected_start = int(datetime(2025, 1, 7, 23, 30, 0, tzinfo=timezone.utc).timestamp())
        expected_end = int(datetime(2025, 1, 7, 23, 59, 59, tzinfo=timezone.utc).timestamp())

        assert start == expected_start
        assert end == expected_end

    def test_window_at_midnight(self):
        """Test window at exactly midnight."""
        timestamp = int(datetime(2025, 1, 8, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        start, end = get_window_at_time(timestamp)

        expected_start = int(datetime(2025, 1, 8, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        expected_end = int(datetime(2025, 1, 8, 0, 29, 59, tzinfo=timezone.utc).timestamp())

        assert start == expected_start
        assert end == expected_end


class TestGetCurrentWindowBoundary:
    """Test getting the current 30-minute window boundary."""

    def test_returns_tuple_of_ints(self):
        """Test that function returns two integers."""
        start, end = get_current_window_boundary()
        assert isinstance(start, int)
        assert isinstance(end, int)

    def test_window_is_30_minutes(self):
        """Test that window is approximately 30 minutes (1800 seconds)."""
        start, end = get_current_window_boundary()
        duration = end - start
        # Should be 29 minutes 59 seconds = 1799 seconds
        assert 1799 <= duration <= 1800

    def test_start_is_on_hour_or_half_hour(self):
        """Test that window starts on :00 or :30."""
        start, _ = get_current_window_boundary()
        dt = datetime.fromtimestamp(start, tz=timezone.utc)
        assert dt.minute in (0, 30)
        assert dt.second == 0


class TestCalculateSummaryLevel:
    """Test calculating summary level from time span."""

    def test_single_window_is_level_1(self):
        """Test that a single 30-min window is level 1."""
        start = int(datetime(2025, 1, 7, 13, 0, 0, tzinfo=timezone.utc).timestamp())
        end = int(datetime(2025, 1, 7, 13, 29, 59, tzinfo=timezone.utc).timestamp())
        assert calculate_summary_level(start, end) == 1

    def test_one_hour_is_level_2(self):
        """Test that ~1 hour span is level 2."""
        start = int(datetime(2025, 1, 7, 13, 0, 0, tzinfo=timezone.utc).timestamp())
        end = int(datetime(2025, 1, 7, 13, 59, 59, tzinfo=timezone.utc).timestamp())
        assert calculate_summary_level(start, end) == 2

    def test_two_hours_is_level_3(self):
        """Test that ~2 hour span is level 3."""
        start = int(datetime(2025, 1, 7, 13, 0, 0, tzinfo=timezone.utc).timestamp())
        end = int(datetime(2025, 1, 7, 14, 59, 59, tzinfo=timezone.utc).timestamp())
        assert calculate_summary_level(start, end) == 3

    def test_four_hours_is_level_4(self):
        """Test that ~4 hour span is level 4."""
        start = int(datetime(2025, 1, 7, 13, 0, 0, tzinfo=timezone.utc).timestamp())
        end = int(datetime(2025, 1, 7, 16, 59, 59, tzinfo=timezone.utc).timestamp())
        assert calculate_summary_level(start, end) == 4

    def test_with_gaps_calculates_by_actual_span(self):
        """Test that level is based on total time span, not number of windows."""
        # 2.5 hour gap (1:00-3:30) should still be level 3 (2-4 hour range)
        start = int(datetime(2025, 1, 7, 13, 0, 0, tzinfo=timezone.utc).timestamp())
        end = int(datetime(2025, 1, 7, 15, 30, 0, tzinfo=timezone.utc).timestamp())
        assert calculate_summary_level(start, end) == 3
