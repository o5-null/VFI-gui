"""Tests for core/io/frame_lifecycle.py — write-once + consumer tracking."""

from __future__ import annotations

from core.io.frame_lifecycle import FrameLifecycle


class TestFrameLifecycle:
    """FrameLifecycle write-once and consumer tracking invariants."""

    def test_register_and_can_write(self):
        """A freshly registered frame can be written."""
        lifecycle = FrameLifecycle()
        lifecycle.register(0, "subtask_a")
        assert lifecycle.can_write(0) is True

    def test_mark_written(self):
        """After mark_written, can_write returns False."""
        lifecycle = FrameLifecycle()
        lifecycle.register(0, "subtask_a")
        lifecycle.mark_written(0)
        assert lifecycle.can_write(0) is False

    def test_double_write_prevention(self):
        """mark_written twice is idempotent (no error)."""
        lifecycle = FrameLifecycle()
        lifecycle.register(0, "subtask_a")
        lifecycle.mark_written(0)
        lifecycle.mark_written(0)  # should not raise
        assert lifecycle.can_write(0) is False

    def test_can_release_single_consumer(self):
        """Frame with one consumer is releasable after that consumer completes."""
        lifecycle = FrameLifecycle()
        lifecycle.register(0, "subtask_a")

        assert lifecycle.can_release(0, "subtask_a") is True
        assert lifecycle.get_consumer_count(0) == 0

    def test_can_release_multi_consumer(self):
        """Frame with multiple consumers is NOT releasable until all complete."""
        lifecycle = FrameLifecycle()
        lifecycle.register(0, "subtask_a")
        lifecycle.register(0, "subtask_b")

        # Only one consumer released — frame is NOT releasable
        assert lifecycle.can_release(0, "subtask_a") is False
        assert lifecycle.get_consumer_count(0) == 1

        # Both consumers released — frame IS releasable
        assert lifecycle.can_release(0, "subtask_b") is True
        assert lifecycle.get_consumer_count(0) == 0

    def test_can_release_unregistered_frame(self):
        """Releasing a frame with no consumers is a no-op (returns True)."""
        lifecycle = FrameLifecycle()
        assert lifecycle.can_release(5, "nonexistent") is True

    def test_discard_on_release(self):
        """Releasing a non-registered consumer does not error."""
        lifecycle = FrameLifecycle()
        lifecycle.register(0, "subtask_a")
        # Release with wrong ID — should be ignored
        lifecycle.can_release(0, "wrong_id")
        assert lifecycle.get_consumer_count(0) == 1  # original still there

    def test_register_multiple_frames(self):
        """Multiple frames can be tracked independently."""
        lifecycle = FrameLifecycle()
        lifecycle.register(0, "task_a")
        lifecycle.register(0, "task_b")
        lifecycle.register(1, "task_a")

        assert lifecycle.get_consumer_count(0) == 2
        assert lifecycle.get_consumer_count(1) == 1

    def test_complex_lifecycle_scenario(self):
        """Simulate a realistic streaming scenario."""
        lifecycle = FrameLifecycle()

        # Frame 0 is consumed by subtask_a (as frame0 of pair 0-1)
        # Frame 1 is consumed by subtask_a (as frame1) and subtask_b (as frame0 of pair 1-2)
        lifecycle.register(0, "subtask_a")
        lifecycle.register(1, "subtask_a")
        lifecycle.register(1, "subtask_b")

        # Write frame 0
        lifecycle.mark_written(0)
        assert lifecycle.can_write(0) is False

        # subtask_a completes — frame 0 released, frame 1 still held by subtask_b
        assert lifecycle.can_release(0, "subtask_a") is True
        assert lifecycle.can_release(1, "subtask_a") is False
        assert lifecycle.get_consumer_count(1) == 1  # subtask_b still active

        # subtask_b completes — frame 1 released
        assert lifecycle.can_release(1, "subtask_b") is True
        assert lifecycle.get_consumer_count(1) == 0
