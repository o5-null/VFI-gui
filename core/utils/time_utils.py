"""Time and progress utilities for VFI-gui.

Provides time formatting and progress calculation functions.
"""

from typing import Optional


def format_time_remaining(current: int, total: int, fps: float) -> str:
    """Format estimated time remaining.

    Args:
        current: Current progress count (e.g., frames processed)
        total: Total count (e.g., total frames)
        fps: Current processing rate (frames per second)

    Returns:
        Formatted time string (HH:MM:SS) or placeholder if unavailable
    """
    if fps <= 0 or total <= 0:
        return "--:--:--"

    remaining_frames = total - current
    remaining_seconds = remaining_frames / fps

    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    seconds = int(remaining_seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS string.

    Args:
        seconds: Total seconds

    Returns:
        Formatted time string (HH:MM:SS)
    """
    if seconds < 0:
        return "00:00:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def calculate_progress_percentage(current: int, total: int) -> int:
    """Calculate progress percentage.

    Args:
        current: Current progress count
        total: Total count

    Returns:
        Percentage (0-100)
    """
    if total <= 0:
        return 0
    return min(100, int((current / total) * 100))


def estimate_completion_time(
    elapsed_seconds: float,
    progress_percent: int,
) -> Optional[float]:
    """Estimate total completion time based on elapsed time and progress.

    Args:
        elapsed_seconds: Time elapsed so far
        progress_percent: Current progress percentage (0-100)

    Returns:
        Estimated total seconds to completion, or None if progress is 0
    """
    if progress_percent <= 0:
        return None

    # Total = elapsed * (100 / progress)
    return elapsed_seconds * (100 / progress_percent)