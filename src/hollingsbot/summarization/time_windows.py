"""Time window boundary calculations for progressive summarization."""

import math
from datetime import datetime, timezone


def get_window_at_time(timestamp: int) -> tuple[int, int]:
    """
    Get the 30-minute window boundaries for a given timestamp.

    Windows are aligned to wall clock: :00-:29 and :30-:59.

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        tuple: (start_timestamp, end_timestamp) where end is inclusive
               (last second of the window)
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    # Determine if we're in the :00-:29 or :30-:59 window
    if dt.minute < 30:
        # First half of hour: :00-:29
        start_dt = dt.replace(minute=0, second=0, microsecond=0)
        end_dt = dt.replace(minute=29, second=59, microsecond=0)
    else:
        # Second half of hour: :30-:59
        start_dt = dt.replace(minute=30, second=0, microsecond=0)
        end_dt = dt.replace(minute=59, second=59, microsecond=0)

    return int(start_dt.timestamp()), int(end_dt.timestamp())


def get_current_window_boundary() -> tuple[int, int]:
    """
    Get the current 30-minute window boundary based on current time.

    Returns:
        tuple: (start_timestamp, end_timestamp) for the current window
    """
    now = int(datetime.now(tz=timezone.utc).timestamp())
    return get_window_at_time(now)


def calculate_summary_level(start: int, end: int) -> int:
    """
    Calculate what summary level a time span represents.

    Levels are based on approximate time spans:
    - Level 1: ~30 min (single window)
    - Level 2: ~1 hour (2 windows)
    - Level 3: ~2 hours (4 windows)
    - Level 4: ~4 hours (8 windows)
    - etc. (exponential)

    Args:
        start: Start timestamp
        end: End timestamp

    Returns:
        Summary level (1, 2, 3, ...)
    """
    duration_seconds = end - start
    duration_hours = duration_seconds / 3600

    # Level 1 = 0.5 hours, Level 2 = 1 hour, Level 3 = 2 hours, Level 4 = 4 hours
    # Formula: level = log2(duration_hours / 0.5) + 1
    # But we need to round to handle gaps and slight variations

    if duration_hours < 0.75:  # Less than 45 min -> Level 1
        return 1
    elif duration_hours < 1.5:  # Less than 1.5 hours -> Level 2
        return 2
    elif duration_hours < 3:  # Less than 3 hours -> Level 3
        return 3
    elif duration_hours < 6:  # Less than 6 hours -> Level 4
        return 4
    else:
        # For very long spans, use logarithmic calculation
        return max(1, int(math.log2(duration_hours / 0.5)) + 1)
