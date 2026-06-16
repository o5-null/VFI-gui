"""Tests for core/utils/time_utils.py — time formatting and ETA."""

from __future__ import annotations

from core.utils.time_utils import (
    calculate_progress_percentage,
    estimate_completion_time,
    format_seconds,
    format_time_remaining,
)


class TestFormatTimeRemaining:
    """format_time_remaining edge cases."""

    def test_normal_case(self):
        """100/200 frames at 30 fps → ~3.33s → 00:00:03."""
        result = format_time_remaining(100, 200, 30.0)
        assert result == "00:00:03"

    def test_zero_fps(self):
        """Zero fps returns placeholder."""
        assert format_time_remaining(50, 100, 0.0) == "--:--:--"

    def test_zero_total(self):
        """Zero total frames returns placeholder."""
        assert format_time_remaining(0, 0, 30.0) == "--:--:--"

    def test_negative_fps(self):
        """Negative fps is treated as invalid."""
        assert format_time_remaining(50, 100, -1.0) == "--:--:--"

    def test_no_progress(self):
        """Current=0, total=100 → full duration."""
        result = format_time_remaining(0, 100, 10.0)
        assert result == "00:00:10"

    def test_large_values(self):
        """5000/10000 frames at 30 fps → 166s → 00:02:46."""
        result = format_time_remaining(5000, 10000, 30.0)
        assert result == "00:02:46"

    def test_hour_boundary(self):
        """Crosses the 1-hour boundary."""
        # 7200 frames left at 2 fps = 3600s = 1 hour
        result = format_time_remaining(0, 7200, 2.0)
        assert result == "01:00:00"


class TestFormatSeconds:
    """format_seconds edge cases."""

    def test_zero(self):
        assert format_seconds(0) == "00:00:00"

    def test_negative(self):
        assert format_seconds(-5) == "00:00:00"

    def test_exact_hour(self):
        assert format_seconds(3600) == "01:00:00"

    def test_exact_minute(self):
        assert format_seconds(60) == "00:01:00"

    def test_complex(self):
        assert format_seconds(3661) == "01:01:01"

    def test_float_truncation(self):
        assert format_seconds(90.7) == "00:01:30"


class TestCalculateProgressPercentage:
    """calculate_progress_percentage edge cases."""

    def test_zero_total(self):
        assert calculate_progress_percentage(50, 0) == 0

    def test_halfway(self):
        assert calculate_progress_percentage(50, 100) == 50

    def test_complete(self):
        assert calculate_progress_percentage(100, 100) == 100

    def test_over_cap(self):
        assert calculate_progress_percentage(150, 100) == 100

    def test_no_progress(self):
        assert calculate_progress_percentage(0, 100) == 0


class TestEstimateCompletionTime:
    """estimate_completion_time edge cases."""

    def test_normal(self):
        # 30s elapsed at 50% progress → 60s total
        result = estimate_completion_time(30.0, 50)
        assert result is not None
        assert abs(result - 60.0) < 0.01

    def test_zero_progress(self):
        assert estimate_completion_time(30.0, 0) is None

    def test_just_started(self):
        # 5s at 1% → 500s total
        result = estimate_completion_time(5.0, 1)
        assert result is not None
        assert abs(result - 500.0) < 0.01

    def test_hundred_percent(self):
        result = estimate_completion_time(120.0, 100)
        assert result is not None
        assert abs(result - 120.0) < 0.01
